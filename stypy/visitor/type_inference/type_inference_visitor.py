#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ast

import statement_visitor
from stypy.visitor.type_inference.visitor_utils import stypy_functions


class TypeInferenceGeneratorVisitor(ast.NodeVisitor):
    """
    This visitor is responsible of generating type inference code AST Tree from standard Pyhon code contained in a
    .py file. It just process the Module node, generating a fixed prefix code nodes, creating and running a
    StatementVisitor object and appending a fixed postfix code at the end to form the full AST tree of the type
    inference program created from a Python source code file.
    """

    def __init__(self, file_name, original_code=None):
        """
        Initializes the visitor.
        :param file_name: File name of the source code whose ast will be parsed. This is needed for localization,
        needed to report errors precisely.
        :param original_code: If present, it includes the original file source code as a comment at the beggining of
        the file. This can be useful for debugging purposes.
        """
        self.type_inference_ast = ast.Module()
        self.file_name = file_name
        self.original_code = original_code

    @classmethod
    def get_postfix_src_code(cls):
        """
        All generated type inference programs has this code at the end, to capture generated TypeErrors and TypeWarnings
        in known variables
        :return:
        """
        return "\n{0} = stypy.errors.type_error.StypyTypeError.get_error_msgs()\n{1} = " \
               "stypy.errors.type_warning.TypeWarning.get_warning_msgs()\n" \
            .format(stypy_functions.default_type_error_var_name, stypy_functions.default_type_warning_var_name)

    def generic_visit(self, node):
        """
        Initiates the type inference visit process
        :param node:
        :return:
        """
        new_stmts = list()

        # Add encoding
        new_stmts.append(stypy_functions.create_src_comment("-*- coding: utf-8 -*-"))
        new_stmts.append(stypy_functions.create_blank_line())

        # Add the source code of the original program as a comment, if provided
        # if not self.original_code is None:
        new_stmts.append(stypy_functions.create_original_code_comment(self.file_name, self.original_code))

        # Writes the instruction: from stypy.type_inference_programs_imports import *
        new_stmts.append(stypy_functions.create_src_comment("Import the stypy library necessary elements"))
        new_stmts.append(stypy_functions.create_import_stypy())

        # Writes the instruction: module_type_store = TypeStore(__file__)
        new_stmts.append(stypy_functions.create_src_comment("Create the module type store"))
        new_stmts.append(stypy_functions.create_type_store())

        new_stmts.append(stypy_functions.create_program_section_src_comment("Begin of the type inference program"))

        # Visit the current module source code beginning with an Statement visitor
        statements_visitor = statement_visitor.StatementVisitor(self.file_name)

        new_stmts.extend(statements_visitor.visit(node, []))

        new_stmts.append(stypy_functions.create_program_section_src_comment("End of the type inference program"))
        return ast.Module(new_stmts)
