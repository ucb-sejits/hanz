from hanz.liveness_analysis import perform_liveness_analysis

from ctree.frontend import get_ast
import ast
import inspect


class CFGBuilder(ast.NodeTransformer):
    def __init__(self):
        self.funcs = []
        self.tmp = -1
        self.curr_target = None

    def _gen_tmp(self):
        self.tmp += 1
        return "_t{}".format(self.tmp)

    def visit_FunctionDecl(self, node):
        new_body = []
        for statement in node.body:
            result = self.visit(statement)
            if isinstance(result, list):
                new_body.extend(result)
            else:
                new_body.append(result)
        node.body = new_body
        return node

    def visit_Call(self, node):
        args = []
        for arg in node.args:
            if isinstance(arg, ast.Name):
                args.append(arg)
            else:
                raise NotImplementedError()
        node.args = args
        return [ast.Assign([ast.Name(self.curr_target, ast.Store())], node)]

    def visit_BinOp(self, node):
        operands = ()
        ret = []
        for operand in (node.right, node.left):
            if isinstance(operand, ast.Name):
                operands += (operand, )
            else:
                old_target = self.curr_target
                self.curr_target = self._gen_tmp()
                ret.extend(self.visit(operand))
                operands += (ast.Name(self.curr_target, ast.Load()), )
                self.curr_target = old_target
        node.right = operands[0]
        node.left = operands[1]
        ret.append(ast.Assign([ast.Name(self.curr_target, ast.Store())], node))
        return ret

    def visit_Assign(self, node):
        old_target = self.curr_target
        self.curr_target = node.targets[0].id
        ret = self.visit(node.value)
        self.curr_target = old_target
        return ret

    def visit_Return(self, node):
        if isinstance(node.value, ast.Name):
            return node
        elif isinstance(node.value, ast.Tuple):
            raise NotImplementedError()
        tmp = self._gen_tmp()
        self.curr_target = tmp
        value = self.visit(node.value)
        node.value = ast.Name(tmp, ast.Load())
        return value + [node]


op2str = {
    ast.Add: "+",
    ast.Mult: "*",
    ast.Sub: "-",
    ast.Div: "/"
}


def dump_op(op):
    if isinstance(op, ast.Call):
        return "{}({})".format(op.func.id, ", ".join([arg.id for arg in
                                                      op.args]))
    elif isinstance(op, ast.BinOp):
        return "{} {} {}".format(op.left.id, op2str[op.op.__class__],
                                 op.right.id)
    else:
        raise NotImplementedError(op)


class BasicBlock(object):

    """Docstring for BasicBlock. """

    def __init__(self):
        """TODO: to be defined1. """
        self.statements = []
        self.live_ins = set()
        self.live_outs = set()

    def add_statement(self, statement):
        self.statements.append(statement)

    def __getitem__(self, index):
        return self.statements[index]

    def __len__(self):
        return len(self.statements)

    def dump(self, tab):
        output = tab + self.__class__.__name__ + "\n"
        tab += "  "
        output += tab + "live ins: {}\n".format(", ".join(self.live_ins))
        output += tab + "live outs: {}\n".format(", ".join(self.live_outs))
        output += tab + "body:\n"
        tab += "  "
        for expr in self.statements:
            if isinstance(expr, ast.Assign):
                output += tab + "{} = {}\n".format(
                    expr.targets[0].id, dump_op(expr.value))
            elif isinstance(expr, ast.Return):
                output += tab + "return {}\n".format(expr.value.id)
        return output


class ComposableBasicBlock(BasicBlock):
    pass


class NonComposableBasicBlock(BasicBlock):
    pass


class ControlFlowGraph(object):
    """
    A datastructure to represent the control flow graph of a function as a
    graph of basic blocks.
    """
    def __init__(self, func):
        """
        :param ast.FunctionDef func:
        """
        self.name = func.name
        self.params = func.args
        self.graph = CFGBuilder().visit(func)

    def __str__(self):
        output = ""
        tab = ""
        for block in self.graph.body:
            if isinstance(block, BasicBlock):
                output += block.dump(tab)
            else:
                raise NotImplementedError(block)
        return output

    def compile_to_fn(self, env):
        # TODO: Should we create a new module/funcdef or just reuse the one
        # passed in
        tree = ast.Module(
            [self.graph]
        )
        ast.fix_missing_locations(tree)
        exec(compile(tree, filename=self.name, mode="exec"), env, env)
        return env[self.name]

    def append_composable(self, new_block, specializer, statement,
                          symbol_table, args):
        if len(new_block) < 1 or \
                not isinstance(new_block[-1], ComposableBasicBlock):
            new_block.append(ComposableBasicBlock())
        new_block[-1].add_statement(statement)
        target = statement.targets[0].id
        symbol_table[target] = specializer.eval_symbolically(*args)

    def append_noncomposable(self, new_block, statement):
        if len(new_block) < 1 or \
                not isinstance(new_block[-1], NonComposableBasicBlock):
            new_block.append(NonComposableBasicBlock())
        new_block[-1].add_statement(statement)

    binop2attr = {
        ast.Add: "__add__",
        ast.Mult: "__mul__",
        ast.Sub: "__sub__",
        ast.Div: "__div__"
    }

    binop2rattr = {
        ast.Add: "__radd__",
        ast.Mult: "__rmul__",
        ast.Sub: "__rsub__",
        ast.Div: "__rdiv__"
    }

    def separate_composable_blocks(self, block, symbol_table):
        new_block = []
        for statement in block:
            if isinstance(statement, ast.Assign):
                if isinstance(statement.value, ast.Call):
                    # TODO: Could be attribute
                    func = symbol_table[statement.value.func.id]
                    if hasattr(func, '_specializer'):
                        args = (symbol_table[arg.id]
                                for arg in statement.value.args)
                        self.append_composable(new_block, func._specializer,
                                               statement, symbol_table, args)
                    else:
                        new_block.append(statement)
                elif isinstance(statement.value, ast.BinOp):
                    attr = self.binop2attr[statement.value.op.__class__]
                    rattr = self.binop2rattr[statement.value.op.__class__]
                    if statement.value.left.id in symbol_table and \
                            statement.value.right.id in symbol_table:
                        left = symbol_table[statement.value.left.id]
                        right = symbol_table[statement.value.right.id]
                        left_func = getattr(left, attr)
                        right_func = getattr(right, rattr)
                        if hasattr(left_func, 'specialized_dispatch'):
                            func = left_func.fn(left, right)
                            self.append_composable(new_block,
                                                   func._specializer,
                                                   statement, symbol_table,
                                                   (left, right))
                        elif hasattr(right_func, 'specialized_dispatch'):
                            func = right_func.fn(right, left)
                            self.append_composable(new_block,
                                                   func._specializer,
                                                   statement, symbol_table,
                                                   (right, left))
                        else:
                            self.append_noncomposable(new_block, statement)
                    else:
                        self.append_noncomposable(new_block, statement)
                else:
                    self.append_noncomposable(new_block, statement)
            elif isinstance(statement, ast.Return):
                self.append_noncomposable(new_block, statement)
            else:
                raise NotImplementedError(statement)
        return new_block

    def find_composable_blocks(self, symbol_table):
        self.graph.body = self.separate_composable_blocks(self.graph.body,
                                                          symbol_table)

    def liveness_analysis(self):
        perform_liveness_analysis(self.graph.body)
        print(self)


def compose(fn):
    tree = get_ast(fn)
    cfg = ControlFlowGraph(tree.body[0])
    frame = inspect.stack()[1][0]
    symbol_table = frame.f_locals
    symbol_table.update(frame.f_globals)
    symbol_table.update(frame.f_back.f_locals)
    symbol_table.update(frame.f_back.f_globals)

    def wrapped(*args, **kwargs):
        # Make a mutable copy of closure
        env = dict(symbol_table)
        # Load parameters into environment
        for index, arg in enumerate(tree.body[0].args.args):
            env[arg.id] = args[index]
        cfg.find_composable_blocks(env)
        cfg.liveness_analysis()
        fn = cfg.compile_to_fn(env)
        return fn(*args, **kwargs)
    return wrapped
