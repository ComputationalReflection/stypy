"""
This is a Python source code generator visitor, that transform an AST into valid source code. It is used to create
type annotated programs and type inference programs when their AST is finally created.

Adapted from: http://svn.python.org/view/python/trunk/Demo/parser/unparse.py
"""

import ast
import sys
from cStringIO import StringIO

import manual_module_types_generation
from stypy.visitor.type_inference.visitor_utils.stypy_functions import default_module_type_store_var_name

# Large float and imaginary literals get turned into infinities in the AST.
# We unparse those infinities to INFSTR.
INFSTR = "1e" + repr(sys.float_info.max_10_exp + 1)


def interleave(inter, f, seq):
    """
    Call f on each item in seq, calling inter() in between.
    """
    seq = iter(seq)
    try:
        f(next(seq))
    except StopIteration:
        pass
    else:
        for x in seq:
            inter()
            f(x)


class PythonSrcGeneratorVisitor(ast.NodeVisitor):
    """
    Methods in this class recursively traverse an AST and
    output source code for the abstract syntax; original formatting
    is disregarded.
    """

    def __init__(self, tree, file_path, verbose=False):
        self.output = StringIO()
        self.future_imports = []
        self._indent = 0
        self._indent_str = "    "
        self.output.write("")
        self.output.flush()
        self.tree = tree
        self.verbose = verbose
        self.file_path = file_path
        self.code_generation_started = False
        self.generates_code = manual_module_types_generation.module_generates_type_inference_code(file_path)

    def generate_code(self):
        self.visit(self.tree)
        src = self.output.getvalue()
        src += manual_module_types_generation.get_manual_member_types_for_module(self.file_path, True)

        return src

    def fill(self, text=""):
        """
        Indent a piece of text, according to the current indentation level
        """
        if self.verbose:
            sys.stdout.write("\n" + self._indent_str * self._indent + text)
        self.output.write("\n" + self._indent_str * self._indent + text)

    def write(self, text):
        """
        Append a piece of text to the current line.
        """
        if self.verbose:
            sys.stdout.write(text)
        self.output.write(text)

    def enter(self):
        """
        Print ':', and increase the indentation.
        """
        self.write(":")
        self._indent += 1

    def leave(self):
        """
        Decrease the indentation level.
        """
        self._indent -= 1

    def visit(self, tree):
        """
        General visit method, calling the appropriate visit method for type T
        """
        if self.code_generation_started and not self.generates_code:
            return

        if isinstance(tree, list):
            for t in tree:
                self.visit(t)
            return

        if type(tree) is tuple:
            print (tree)

        meth = getattr(self, "visit_" + tree.__class__.__name__)
        meth(tree)

    # ############## Unparsing methods ######################
    # There should be one method per concrete grammar type #
    # Constructors should be grouped by sum type. Ideally, #
    # this would follow the order in the grammar, but      #
    # currently doesn't.                                   #
    # #######################################################

    def visit_Module(self, tree):
        for stmt in tree.body:
            self.visit(stmt)

    # stmt
    def visit_Expr(self, tree):
        self.fill()
        self.visit(tree.value)

    def visit_Import(self, t):
        self.fill("import ")
        interleave(lambda: self.write(", "), self.visit, t.names)
        self.write("\n")

    def visit_ImportFrom(self, t):
        # A from __future__ import may affect unparsing, so record it.
        if t.module and t.module == '__future__':
            self.future_imports.extend(n.name for n in t.names)

        self.fill("from ")
        self.write("." * t.level)
        if t.module:
            self.write(t.module)
        self.write(" import ")
        interleave(lambda: self.write(", "), self.visit, t.names)
        self.write("\n")

    def visit_Assign(self, t):
        self.fill()
        for target in t.targets:
            self.visit(target)
            self.write(" = ")
        self.visit(t.value)
        if not self.code_generation_started:
            # This defines the beginning of code generation, just after module_type_store is created
            if len(t.targets) == 1 and type(t.targets[0]) is ast.Name and \
                            t.targets[0].id == default_module_type_store_var_name:
                if not self.generates_code:
                    self.write(
                        "\n\n# *********************************************************************************"
                        "*****************\n")
                    self.write(
                        "# THIS FILE DO NOT CONTAIN TYPE INFERENCE CODE. ITS TYPES ARE MANUALLY INSERTED "
                        "INTO ITS TYPE STORE\n")
                    self.write(
                        "# **************************************************************************************"
                        "************\n\n")
                self.write(manual_module_types_generation.get_manual_member_types_for_module(self.file_path, False))
                self.code_generation_started = True

    def visit_AugAssign(self, t):
        self.fill()
        self.visit(t.target)
        self.write(" " + self.binop[t.op.__class__.__name__] + "= ")
        self.visit(t.value)

    def visit_Return(self, t):
        self.fill("return")
        if t.value:
            self.write(" ")
            self.visit(t.value)

    def visit_Pass(self, t):
        self.fill("pass")

    def visit_Break(self, t):
        self.fill("break")

    def visit_Continue(self, t):
        self.fill("continue")

    def visit_Delete(self, t):
        self.fill("del ")
        interleave(lambda: self.write(", "), self.visit, t.targets)

    def visit_Assert(self, t):
        self.fill("assert ")
        self.visit(t.test)
        if t.msg:
            self.write(", ")
            self.visit(t.msg)

    def visit_Exec(self, t):
        self.fill("exec ")
        self.visit(t.body)
        if t.globals:
            self.write(" in ")
            self.visit(t.globals)
        if t.locals:
            self.write(", ")
            self.visit(t.locals)

    def visit_Print(self, t):
        self.fill("print ")
        do_comma = False
        if t.dest:
            self.write(">>")
            self.visit(t.dest)
            do_comma = True
        for e in t.values:
            if do_comma:
                self.write(", ")
            else:
                do_comma = True
            self.visit(e)
        if not t.nl:
            self.write(",")

    def visit_Global(self, t):
        self.fill("global ")
        interleave(lambda: self.write(", "), self.write, t.names)

    def visit_Yield(self, t):
        self.write("(")
        self.write("yield")
        if t.value:
            self.write(" ")
            self.visit(t.value)
        self.write(")")

    def visit_Raise(self, t):
        self.fill('raise ')
        if t.type:
            self.visit(t.type)
        if t.inst:
            self.write(", ")
            self.visit(t.inst)
        if t.tback:
            self.write(", ")
            self.visit(t.tback)

    def visit_TryExcept(self, t):
        self.fill("try")
        self.enter()
        self.visit(t.body)
        self.leave()

        for ex in t.handlers:
            self.visit(ex)
        if t.orelse:
            self.fill("else")
            self.enter()
            self.visit(t.orelse)
            self.leave()

    def visit_TryFinally(self, t):
        if len(t.body) == 1 and isinstance(t.body[0], ast.TryExcept):
            # try-except-finally
            self.visit(t.body)
        else:
            self.fill("try")
            self.enter()
            self.visit(t.body)
            self.leave()

        self.fill("finally")
        self.enter()
        self.visit(t.finalbody)
        self.leave()

    def visit_ExceptHandler(self, t):
        self.fill("except")
        if t.type:
            self.write(" ")
            self.visit(t.type)
        if t.name:
            self.write(" as ")
            self.visit(t.name)
        self.enter()
        self.visit(t.body)
        self.leave()

    def visit_ClassDef(self, t):
        self.write("\n")
        for deco in t.decorator_list:
            self.fill("@")
            self.visit(deco)
        self.fill("class " + t.name)
        if t.bases:
            self.write("(")
            for a in t.bases:
                self.visit(a)
                self.write(", ")
            self.write(")")
        self.enter()
        self.visit(t.body)
        self.leave()

    def visit_FunctionDef(self, t):
        self.write("\n")
        for deco in t.decorator_list:
            self.fill("@")
            self.visit(deco)
        self.fill("def " + t.name + "(")
        self.visit(t.args)
        self.write(")")
        self.enter()
        self.visit(t.body)
        self.write("\n")
        self.leave()

    def visit_For(self, t):
        self.fill("for ")
        self.visit(t.target)
        self.write(" in ")
        self.visit(t.iter)
        self.enter()
        self.visit(t.body)
        self.leave()
        if t.orelse:
            self.fill("else")
            self.enter()
            self.visit(t.orelse)
            self.leave()

    def visit_If(self, t):
        self.write("\n")
        self.fill("if ")
        self.visit(t.test)
        self.enter()
        self.visit(t.body)
        self.leave()
        # collapse nested ifs into equivalent elifs.
        while (t.orelse and len(t.orelse) == 1 and
                   isinstance(t.orelse[0], ast.If)):
            t = t.orelse[0]
            self.fill("elif ")
            self.visit(t.test)
            self.enter()
            self.visit(t.body)
            self.leave()
        # final else
        if t.orelse:
            self.fill("else")
            self.enter()
            self.visit(t.orelse)
            self.leave()
        self.write("\n")

    def visit_While(self, t):
        self.fill("while ")
        self.visit(t.test)
        self.enter()
        self.visit(t.body)
        self.leave()
        if t.orelse:
            self.fill("else")
            self.enter()
            self.visit(t.orelse)
            self.leave()

    def visit_With(self, t):
        self.fill("with ")
        self.visit(t.context_expr)
        if t.optional_vars:
            self.write(" as ")
            self.visit(t.optional_vars)
        self.enter()
        self.visit(t.body)
        self.leave()

    # expr
    def visit_Str(self, tree):
        # if from __future__ import unicode_literals is in effect,
        # then we want to output string literals using a 'b' prefix
        # and unicode literals with no prefix.
        if "unicode_literals" not in self.future_imports:
            self.write(repr(tree.s))
        elif isinstance(tree.s, str):
            self.write("b" + repr(tree.s))
        elif isinstance(tree.s, unicode):
            self.write(repr(tree.s).lstrip("u"))
        else:
            assert False, "shouldn't get here"

    def visit_Name(self, t):
        self.write(t.id)

    def visit_Repr(self, t):
        self.write("`")
        self.visit(t.value)
        self.write("`")

    def visit_Num(self, t):
        repr_n = repr(t.n)
        # Parenthesize negative numbers, to avoid turning (-1)**2 into -1**2.
        if repr_n.startswith("-"):
            self.write("(")
        # Substitute overflowing decimal literal for AST infinities.
        self.write(repr_n.replace("inf", INFSTR))
        if repr_n.startswith("-"):
            self.write(")")

    def visit_List(self, t):
        self.write("[")
        interleave(lambda: self.write(", "), self.visit, t.elts)
        self.write("]")

    def visit_ListComp(self, t):
        self.write("[")
        self.visit(t.elt)
        for gen in t.generators:
            self.visit(gen)
        self.write("]")

    def visit_GeneratorExp(self, t):
        self.write("(")
        self.visit(t.elt)
        for gen in t.generators:
            self.visit(gen)
        self.write(")")

    def visit_SetComp(self, t):
        self.write("{")
        self.visit(t.elt)
        for gen in t.generators:
            self.visit(gen)
        self.write("}")

    def visit_DictComp(self, t):
        self.write("{")
        self.visit(t.key)
        self.write(": ")
        self.visit(t.value)
        for gen in t.generators:
            self.visit(gen)
        self.write("}")

    def visit_comprehension(self, t):
        self.write(" for ")
        self.visit(t.target)
        self.write(" in ")
        self.visit(t.iter)
        for if_clause in t.ifs:
            self.write(" if ")
            self.visit(if_clause)

    def visit_IfExp(self, t):
        self.write("(")
        self.visit(t.body)
        self.write(" if ")
        self.visit(t.test)
        self.write(" else ")
        self.visit(t.orelse)
        self.write(")")

    def visit_Set(self, t):
        assert (t.elts)  # should be at least one element
        self.write("{")
        interleave(lambda: self.write(", "), self.visit, t.elts)
        self.write("}")

    def visit_Dict(self, t):
        self.write("{")

        def write_pair(pair):
            (k, v) = pair
            self.visit(k)
            self.write(": ")
            self.visit(v)

        interleave(lambda: self.write(", "), write_pair, zip(t.keys, t.values))
        self.write("}")

    def visit_Tuple(self, t):
        self.write("(")
        if len(t.elts) == 1:
            (elt,) = t.elts
            self.visit(elt)
            self.write(",")
        else:
            interleave(lambda: self.write(", "), self.visit, t.elts)
        self.write(")")

    unop = {"Invert": "~", "Not": "not", "UAdd": "+", "USub": "-"}

    def visit_UnaryOp(self, t):
        self.write("(")
        self.write(self.unop[t.op.__class__.__name__])
        self.write(" ")
        # If we're applying unary minus to a number, parenthesize the number.
        # This is necessary: -2147483648 is different from -(2147483648) on
        # a 32-bit machine (the first is an int, the second a long), and
        # -7j is different from -(7j).  (The first has real part 0.0, the second
        # has real part -0.0.)
        if isinstance(t.op, ast.USub) and isinstance(t.operand, ast.Num):
            self.write("(")
            self.visit(t.operand)
            self.write(")")
        else:
            self.visit(t.operand)
        self.write(")")

    binop = {"Add": "+", "Sub": "-", "Mult": "*", "Div": "/", "Mod": "%",
             "LShift": "<<", "RShift": ">>", "BitOr": "|", "BitXor": "^", "BitAnd": "&",
             "FloorDiv": "//", "Pow": "**"}

    def visit_BinOp(self, t):
        self.write("(")
        self.visit(t.left)
        self.write(" " + self.binop[t.op.__class__.__name__] + " ")
        self.visit(t.right)
        self.write(")")

    cmpops = {"Eq": "==", "NotEq": "!=", "Lt": "<", "LtE": "<=", "Gt": ">", "GtE": ">=",
              "Is": "is", "IsNot": "is not", "In": "in", "NotIn": "not in"}

    def visit_Compare(self, t):
        self.write("(")
        self.visit(t.left)
        for o, e in zip(t.ops, t.comparators):
            self.write(" " + self.cmpops[o.__class__.__name__] + " ")
            self.visit(e)
        self.write(")")

    boolops = {ast.And: 'and', ast.Or: 'or'}

    def visit_BoolOp(self, t):
        self.write("(")
        s = " %s " % self.boolops[t.op.__class__]
        interleave(lambda: self.write(s), self.visit, t.values)
        self.write(")")

    def visit_Attribute(self, t):
        self.visit(t.value)
        # Special case: 3.__abs__() is a syntax error, so if t.value
        # is an integer literal then we need to either parenthesize
        # it or add an extra space to get 3 .__abs__().
        if isinstance(t.value, ast.Num) and isinstance(t.value.n, int):
            self.write(" ")
        self.write(".")
        self.write(t.attr)

    def visit_Call(self, t):
        self.visit(t.func)
        self.write("(")
        comma = False
        for e in t.args:
            if comma:
                self.write(", ")
            else:
                comma = True
            self.visit(e)
        for e in t.keywords:
            if comma:
                self.write(", ")
            else:
                comma = True
            self.visit(e)
        if t.starargs:
            if comma:
                self.write(", ")
            else:
                comma = True
            self.write("*")
            self.visit(t.starargs)
        if t.kwargs:
            if comma:
                self.write(", ")
            else:
                comma = True
            self.write("**")
            self.visit(t.kwargs)
        self.write(")")

    def visit_Subscript(self, t):
        self.visit(t.value)
        self.write("[")
        self.visit(t.slice)
        self.write("]")

    # slice
    def visit_Ellipsis(self, t):
        self.write("...")

    def visit_Index(self, t):
        self.visit(t.value)

    def visit_Slice(self, t):
        if t.lower:
            self.visit(t.lower)
        self.write(":")
        if t.upper:
            self.visit(t.upper)
        if t.step:
            self.write(":")
            self.visit(t.step)

    def visit_ExtSlice(self, t):
        interleave(lambda: self.write(', '), self.visit, t.dims)

    # others
    def visit_arguments(self, t):
        first = True
        # normal arguments
        defaults = [None] * (len(t.args) - len(t.defaults)) + t.defaults
        for a, d in zip(t.args, defaults):
            if first:
                first = False
            else:
                self.write(", ")
            self.visit(a),
            if d:
                self.write("=")
                self.visit(d)

        # varargs
        if t.vararg:
            if first:
                first = False
            else:
                self.write(", ")
            self.write("*")
            self.write(t.vararg)

        # kwargs
        if t.kwarg:
            if first:
                first = False
            else:
                self.write(", ")
            self.write("**" + t.kwarg)

    def visit_keyword(self, t):
        self.write(t.arg)
        self.write("=")
        self.visit(t.value)

    def visit_Lambda(self, t):
        self.write("(")
        self.write("lambda ")
        self.visit(t.args)
        self.write(": ")
        self.visit(t.body)
        self.write(")")

    def visit_alias(self, t):
        self.write(t.name)
        if t.asname:
            self.write(" as " + t.asname)

