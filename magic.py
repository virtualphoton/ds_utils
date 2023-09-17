import ast
from _ast import Call, Module
from collections import defaultdict
from itertools import count

from IPython import get_ipython
from IPython.core.magic import register_cell_magic
from IPython.display import clear_output, display
from ipywidgets import Output


class MyOutput(Output):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_displayed = False


_outputs = defaultdict(MyOutput)


def _reprint(*args, html_id, **kwargs):
    output = _outputs[html_id]
    if not output.is_displayed:
        output.is_displayed = True
        display(output)

    with output:
        output.clear_output()
        print(*args, **kwargs)


class PrintTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.count = count()
        super().__init__()

    def visit_Call(self, node: Call) -> Call:
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            node.func.id = "_reprint"
            node.keywords.append(ast.keyword(
                arg="html_id",
                value=ast.Constant(next(self.count))
            ))

        return super().generic_visit(node)

    def __eq__(self, other):
        return self is other


class SoftCtrlCTransformer(ast.NodeTransformer):
    def visit_Module(self, node: Module) -> Module:
        try_ = ast.parse("""
try:
    pass
except KeyboardInterrupt:
    print("interrupted by keyboard")
""")
        try_.body[0].body = node.body

        return super().generic_visit(try_)

    def __eq__(self, other):
        return self is other


ipy = get_ipython()


@register_cell_magic
def reprint(_, cell):
    tr = PrintTransformer()
    _outputs.clear()
    clear_output()
    ipy.ast_transformers.append(tr)
    ipy.run_cell(cell)
    if tr in ipy.ast_transformers:
        ipy.ast_transformers.remove(tr)


@register_cell_magic
def soft_ctrl_c(_, cell):
    tr = SoftCtrlCTransformer()
    _outputs.clear()
    clear_output()
    ipy.ast_transformers.append(tr)
    ipy.run_cell(cell)
    if tr in ipy.ast_transformers:
        ipy.ast_transformers.remove(tr)


def display_no_widgets(plotter):
    clear_output()
    display(plotter.draw_no_widget())
    print("\n".join(output.outputs[0]["text"] for output in _outputs.values() if output.outputs))
