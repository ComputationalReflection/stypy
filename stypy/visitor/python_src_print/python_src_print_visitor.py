import ast
import cStringIO
import sys


class PrintVisitor(ast.NodeVisitor):
    """
    Pretty-prints an AST node and its children with the following features:

    - Indentation levels
    - Non-private properties of each AST Node and their values.
    - Highlights line numbers and column offsets in those nodes in which these data is available.
    - Highlights representations of some special properties (such as context)

    This visitor is mainly used for debugging purposes, as printing an AST do not have any practical utility to the
    stypy end user
    """

    def __init__(self, ast_node, admitted_types=list(), max_output_level=0, indent='    ', output=sys.stdout):
        """
        Walk AST tree and print each matched node contents.
            ast_node: Root node
            admitted_types: Process only node types in the list
            max_output_level: limit output to the given depth (0 (default) prints all levels)
            indent: Chars to print within indentation levels
            output: where to print the ast (stdout as default)
        """
        self.current_depth = 0
        self.admitted_types = admitted_types
        self.max_output_level = max_output_level
        self.output = output
        self.indent = indent
        self.last_line_no = 0
        self.last_col_offset = 0
        self.visit(ast_node)  # Go!

    def visit(self, node):
        nodetype = type(node)

        if len(self.admitted_types) == 0 or nodetype in self.admitted_types:
            self.__print_node(node)

        self.current_depth += 1

        if self.max_output_level == 0 or self.current_depth <= self.max_output_level:
            self.generic_visit(node)

        self.current_depth -= 1

    def __print_node(self, node):
        node_class_name_txt = node.__class__.__name__

        if not hasattr(node, "col_offset"):
            col = self.last_col_offset
        else:
            self.last_col_offset = node.col_offset
            col = node.col_offset

        col_offset = ", Column: " + str(col) + "]"

        if not hasattr(node, "lineno"):
            line = self.last_line_no
        else:
            self.last_line_no = node.lineno
            line = node.lineno

        lineno = "[Line: " + str(line)

        node_printable_properties = filter(lambda prop: not (prop.startswith("_") or
                                                             (prop == "lineno") or (prop == "col_offset")), dir(node))

        printable_properties_txt = reduce(lambda x, y: x + ", " + y + ": " + str(getattr(node, y)),
                                          node_printable_properties, "")

        if not (printable_properties_txt == ""):
            printable_properties_txt = ": " + printable_properties_txt
        else:
            node_class_name_txt = "(" + node_class_name_txt + ")"

        txt = " " + node_class_name_txt + printable_properties_txt

        self.output.write(lineno + col_offset + self.indent * self.current_depth + txt + "\n")


"""
Interface with the PrintVisitor from external modules
"""


def print_ast(root_node):
    """
    Prints an AST (or any AST node and its children)
    :param root_node: Base node to print
    :return:
    """
    PrintVisitor(root_node)


def dump_ast(root_node):
    """
    Prints an AST (or any AST node and its children) to a string
    :param root_node: Base node to print
    :return: str
    """
    str_output = cStringIO.StringIO()
    PrintVisitor(root_node, output=str_output)
    txt = str_output.getvalue()
    str_output.close()

    return txt
