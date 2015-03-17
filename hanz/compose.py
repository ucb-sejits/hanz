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

    op2str = {
        ast.Add: "+",
        ast.Mult: "*",
        ast.Sub: "-",
        ast.Div: "/"
    }

    def dump_op(self, op):
        if isinstance(op, ast.Call):
            return "{}({})".format(op.func.id, ", ".join([arg.id for arg in
                                                          op.args]))
        elif isinstance(op, ast.BinOp):
            return "{} {} {}".format(op.left.id, self.op2str[op.op.__class__],
                                     op.right.id)
        else:
            raise NotImplementedError(op)

    def __str__(self):
        output = ""
        tab = "    "
        for index, block in enumerate(self.basic_blocks):
            output += "BasicBlock{}\n".format(index)
            for expr in block:
                if isinstance(expr, ast.Assign):
                    output += tab + "{} = {}\n".format(
                        expr.targets[0].id, self.dump_op(expr.value))
                elif isinstance(expr, ast.Return):
                    output += tab + "return {}\n".format(expr.value.id)
        return output


def compile_cfg(graph, env):
    tree = ast.Module(
        [ast.FunctionDef(graph.name, graph.params,
                         graph.basic_blocks[0], [])]
    )
    ast.fix_missing_locations(tree)
    exec(compile(tree, filename="file", mode="exec"), env, env)
    return env[graph.name]


def compose(fn):
    tree = get_ast(fn)
    cfg = ControlFlowGraph(tree.body[0])
    frame = inspect.stack()[1][0]
    symbol_table = frame.f_locals
    symbol_table.update(frame.f_globals)
    symbol_table.update(frame.f_back.f_locals)
    symbol_table.update(frame.f_back.f_globals)
    print(cfg)

    def wrapped(*args, **kwargs):
        fn = compile_cfg(cfg, symbol_table)
        return fn(*args, **kwargs)
    return wrapped
