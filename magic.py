import ast
import inspect
from _ast import (
    Call,
    Expr,
    FunctionDef,
    Module,
    Name,
)
from collections import defaultdict
from itertools import count
from types import FunctionType, CodeType
from typing import Any, Iterable

from IPython import get_ipython
from IPython.core.magic import register_cell_magic
from IPython.display import clear_output, display
from ipywidgets import Output
from tqdm.auto import tqdm


class MyOutput(Output):
    def __init__(self):
        super().__init__()
        display(self)
        
class ReusableTqdm:
    def __init__(self):
        self.tqdm = tqdm([None])
    
    def __call__(self, iterable: Iterable, **kwargs):
        try:
            self.tqdm.total = len(iterable)
        except TypeError:
            self.tqdm.total = None
        
        for key, val in kwargs.items():
            setattr(self.tqdm, key, val)
        
        self.tqdm.reset()
        if self.tqdm.total is not None:
            self.tqdm.container.children[1].max = self.tqdm.total
        
        for entry in iterable:
            yield entry
            self.tqdm.update()
            self.tqdm.refresh()
    
_OUTPUTS = defaultdict(MyOutput)
_TQDMS = defaultdict(ReusableTqdm)

def _reprint(*args, _caller_id, **kwargs):
    if None in _caller_id:
        # function was called without magic applied to all levels
        return print(*args, **kwargs)    
    
    output = _OUTPUTS[_caller_id]

    with output:
        output.clear_output()
        print(*args, **kwargs)

def _retqdm(*args, _caller_id, **kwargs):
    if None in _caller_id:
        # function was called without magic applied to all levels
        yield from tqdm(*args, **kwargs)
        return
    
    tqdm_creator = _TQDMS[_caller_id]
    
    yield from tqdm_creator(*args, **kwargs)
    

###########################################################

reprintable_funcs = {"print", "tqdm"}

def get_expr_ast(line: str):
    res = ast.parse(line).body[0]
    return res.value if isinstance(res, Expr) else res

def copy_data(dest, src):
    ast.copy_location(dest, src)
    ast.fix_missing_locations(dest)

class PrintTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.count = count()
        super().__init__()

    def visit_Call(self, node: Call) -> Call:
        func = node.func
        if isinstance(func, Name) and func.id in reprintable_funcs:
            if func.id in ["print", "tqdm"]:
                node.func = get_expr_ast(f"reprint.{func.id}")
                copy_data(node.func, node)
            node.keywords.append(ast.keyword(
                arg="_caller_id",
                value=get_expr_ast(f"_caller_id + ({next(self.count)}, )")
            ))
            
            copy_data(node.keywords[-1], node)

        return super().generic_visit(node)
    
    def visit_Module(self, node: Module) -> Module:
        # crutch for module-level func callers
        node.body.insert(0, get_expr_ast(f'_caller_id = ()'))
        
        for statement in node.body:
            if isinstance(statement, FunctionDef):
                reprintable_funcs.add(statement.name)
                statement.args.kwonlyargs.append(ast.arg(arg="_caller_id"))
                statement.args.kw_defaults.append(get_expr_ast("(None,)"))
                
                copy_data(statement.args.kwonlyargs[-1], statement.args)
                copy_data(statement.args.kw_defaults[-1], statement.args)
                
        return super().generic_visit(node)

    def __eq__(self, other):
        return self is other


ipy = get_ipython()


@register_cell_magic
def reprint(f, cell = None):
    tr = PrintTransformer()
    if cell is not None:
        # magic call
        _OUTPUTS.clear()
        _TQDMS.clear()
        clear_output()
        ipy.ast_transformers.append(tr)
        ipy.run_cell(cell)
        if tr in ipy.ast_transformers:
            ipy.ast_transformers.remove(tr)
    else:
        # decorator call
        # https://devmessias.github.io/post/python_ast_metaprogramming_with_introspection_and_decorators/#creating-a-new-function-at-runtime
        source = inspect.getsource(f)
        source = "\n".join(line
                           for line in source.split("\n")
                           if not line.startswith("@reprint"))
        tree = ast.parse(source)
        tree = tr.visit(tree)
        code_obj = compile(tree, f.__code__.co_filename, 'exec')
        function_code = [c for c in code_obj.co_consts if isinstance(c, CodeType)][0]
        transformed_func = FunctionType(function_code, f.__globals__, argdefs=f.__defaults__)
        
        kw_default = (f.__kwdefaults__ or {}) | {"_caller_id" : (None,)}
        return lambda *args, **kwargs: transformed_func(*args, **(kw_default | kwargs))

reprint.print = _reprint
reprint.tqdm = _retqdm
    
def display_no_widgets(plotter, outs=None):
    clear_output()
    display(plotter.draw_no_widget())
    if outs is None:
        outs = _OUTPUTS.values()
    print("\n".join(output.outputs[0]["text"] for output in outs if output.outputs))

def get_outputs():
    return _OUTPUTS.values()

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

@register_cell_magic
def soft_ctrl_c(_, cell):
    tr = SoftCtrlCTransformer()
    ipy.ast_transformers.append(tr)
    ipy.run_cell(cell)
    if tr in ipy.ast_transformers:
        ipy.ast_transformers.remove(tr)
