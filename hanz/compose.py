from ctree.frontend import get_ast
import ast
import inspect


def gen_tmp():
    gen_tmp.tmp += 1
    return "_t{}".format(gen_tmp.tmp)

gen_tmp.tmp = -1


def unpack(expr):
    def visit(expr, curr_target=None):
        if isinstance(expr, ast.Return):
            if isinstance(expr.value, (ast.Name, ast.Tuple)):
                body = (expr, )
            else:
                tmp = gen_tmp()
                body = visit(expr.value, ast.Name(tmp, ast.Store()))
                body += (ast.Return(ast.Name(tmp, ast.Load())), )
        elif isinstance(expr, ast.Name):
            return expr
        elif isinstance(expr, ast.BinOp):
            body = ()
            operands = []

            if isinstance(expr.left, ast.Num):
                body += (ast.Assign([curr_target], expr), )
            else:
                for operand in [expr.left, expr.right]:
                    if isinstance(operand, (ast.Name, ast.Num)):
                        operands += (operand, )
                    else:
                        tmp = gen_tmp()
                        body += visit(operand,
                                      ast.Name(tmp, ast.Store()))
                        operands.append(ast.Name(tmp, ast.Load()))
                expr.left = operands[0]
                expr.right = operands[1]
                if curr_target is not None:
                    body += (ast.Assign([curr_target], expr), )
        elif isinstance(expr, ast.Assign):
            target = expr.targets[0]
            if isinstance(target, ast.Tuple):
                body = reduce(lambda x, y: x + y,
                              map(visit, expr.value.elts, target.elts), ())
            else:
                body = visit(expr.value, target)
        elif isinstance(expr, ast.Call):
            body = ()
            args = []
            for arg in expr.args:
                val = visit(arg)
                if isinstance(val, tuple):
                    tmp = gen_tmp()
                    val = visit(arg, ast.Name(tmp, ast.Store))
                    body += val
                    args.append(ast.Name(tmp, ast.Load()))
                elif isinstance(val, (ast.Name, ast.Num)):
                    args.append(val)
                else:
                    raise Exception("Call argument returned\
                                     unsupported type {}".format(type(val)))
            if curr_target is not None:
                body += (ast.Assign(
                    [curr_target],
                    ast.Call(visit(expr.func), args, [], None, None)
                ), )
            else:
                body += (ast.Call(visit(expr.func), args, [], None, None), )
        elif isinstance(expr, ast.Expr):
            return (ast.Expr(visit(expr.value)[0]), )
        else:
            raise Exception("Unsupported expression {}".format(expr))
        return body

    return visit(expr)


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


class ComposableBlock(object):
    def __init__(self):
        self.statements = []

    def add_statement(self, statement):
        self.statements.append(statement)

    def dump(self, tab):
        output = tab + "ComposableBlock\n"
        tab += "    "
        for expr in self.statements:
            if isinstance(expr, ast.Assign):
                output += tab + "{} = {}\n".format(
                    expr.targets[0].id, dump_op(expr.value))
        return output


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
        body = map(unpack, func.body)
        self.basic_blocks = [list(reduce(lambda x, y: x + y, body, ()))]

    def __str__(self):
        output = ""
        tab = "    "
        for index, block in enumerate(self.basic_blocks):
            output += "BasicBlock{}\n".format(index)
            for expr in block:
                if isinstance(expr, ast.Assign):
                    output += tab + "{} = {}\n".format(
                        expr.targets[0].id, dump_op(expr.value))
                elif isinstance(expr, ast.Return):
                    output += tab + "return {}\n".format(expr.value.id)
                elif isinstance(expr, ComposableBlock):
                    output += expr.dump(tab)
        return output

    def compile_to_fn(self, env):
        # TODO: Should we create a new module/funcdef or just reuse the one
        # passed in
        tree = ast.Module(
            [ast.FunctionDef(self.name, self.params,
                             self.basic_blocks[0], [])]
        )
        ast.fix_missing_locations(tree)
        exec(compile(tree, filename=self.name, mode="exec"), env, env)
        return env[self.name]

    def append_composable(self, new_block, specializer, statement,
                          symbol_table, args):
        if len(new_block) < 1 or \
                not isinstance(new_block[-1], ComposableBlock):
            new_block.append(ComposableBlock())
        new_block[-1].add_statement(statement)
        target = statement.targets[0].id
        symbol_table[target] = specializer.eval_symbolically(*args)

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

    def separate_composable_ops(self, block, symbol_table):
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
                            new_block.append(statement)
                    else:
                        new_block.append(statement)
                else:
                    new_block.append(statement)
            elif isinstance(statement, ast.Return):
                new_block.append(statement)
            else:
                raise NotImplementedError(statement)
        return new_block

    def find_composable_blocks(self, symbol_table):
        new_basic_blocks = []
        for block in self.basic_blocks:
            new_block = self.separate_composable_ops(block, symbol_table)
            new_basic_blocks.append(new_block)
        self.basic_blocks = new_basic_blocks
        print(self)
        raise NotImplementedError()


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
        fn = cfg.compile_to_fn(env)
        return fn(*args, **kwargs)
    return wrapped
