import ast


class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.gen = set()
        self.kill = set()

    def visit_Call(self, node):
        for arg in node.args:
            self.visit(arg)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if node.id not in self.kill:
                self.gen.add(node.id)
        else:
            self.kill.add(node.id)


def perform_liveness_analysis(basic_blocks):
    for index, block in enumerate(reversed(basic_blocks)):
        analyzer = Analyzer()
        for statement in block:
            analyzer.visit(statement)
        if index == 0:
            block.live_outs = set()
        else:
            block.live_outs = set().union(
                *(b.live_ins for b in basic_blocks[-index:]))
        block.live_ins = analyzer.gen.union(
            block.live_outs.difference(analyzer.kill))
