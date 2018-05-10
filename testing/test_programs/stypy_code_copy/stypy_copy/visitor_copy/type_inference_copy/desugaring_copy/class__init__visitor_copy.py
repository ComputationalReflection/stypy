from stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy import *
from stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import core_language_copy
import ast

class ClassInitVisitor(ast.NodeTransformer):
    """
    This transformer ensures that every declared class has an __init__ method. If the original class do not declare
    one, an empty one is added. This is needed to avoid code generation errors
    """

    def __search_init_method(self, node):
        if hasattr(node, 'name'):
            return node.name == '__init__'
        else:
            return False

    def visit_ClassDef(self, node):
        # Test if we have an __init__ method
        init_method = filter(lambda n: self.__search_init_method(n), node.body)
        if len(init_method) > 0:
            return node

        # If no __init__ method is declared, declare an empty one
        function_def_arguments = ast.arguments()

        function_def_arguments.args = [core_language_copy.create_Name('self')]

        function_def = ast.FunctionDef()
        function_def.lineno = node.lineno
        function_def.col_offset = node.col_offset
        function_def.name = '__init__'

        function_def.args = function_def_arguments
        function_def_arguments.kwarg = None
        function_def_arguments.vararg = None
        function_def_arguments.defaults = []
        function_def.decorator_list = []

        function_def.body = []

        function_def.body.append(ast.Pass())

        node.body.append(function_def)

        return node
