import copy

from stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy import *
from stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import core_language_copy, stypy_functions_copy, functions_copy
import ast

"""
This visitor decompose several forms of multiple assignments into single assignments, that can be properly processed
by stypy. There are various forms of assignments in Python that involve multiple elements, such as:

a, b = c, d
a, b = [c, d]
a, b = function_that_returns_tuple()
a, b = function_that_returns_list()

This visitor reduces the complexity of dealing with assignments when generating type inference code
"""


def multiple_value_call_assignment_handler(target, value, assign_stmts, node, id_str):
    """
    Handles code that uses a multiple assignment with a call to a function in the right part of the assignment. The
    code var1, var2 = original_call(...) is transformed into:

    temp = original_call(...)
    var1 = temp[0]
    var2 = temp[1]
    ...

    This way we only perform the call once.

    :param target: tuple or list of variables
    :param value: call
    :param assign_stmts: statements holder list
    :param node: current AST node
    :param id_str: Type of assignment that we are processing (to create variables)
    :return:
    """
    target_stmts, value_var = stypy_functions_copy.create_temp_Assign(value, node.lineno, node.col_offset,
                                                                 "{0}_assignment".format(id_str))
    assign_stmts.append(target_stmts)

    value_var_to_load = copy.deepcopy(value_var)
    value_var_to_load.ctx = ast.Load()

    for i in range(len(target.elts)):
        # Assign values to each element.
        getitem_att = core_language_copy.create_attribute(value_var_to_load, '__getitem__', context=ast.Load(),
                                                     line=node.lineno,
                                                     column=node.col_offset)
        item_call = functions_copy.create_call(getitem_att, [core_language_copy.create_num(i, node.lineno, node.col_offset)])
        temp_stmts, temp_value = stypy_functions_copy.create_temp_Assign(item_call, node.lineno, node.col_offset,
                                                                    "{0}_assignment".format(id_str))
        assign_stmts.append(temp_stmts)

        temp_stmts = core_language_copy.create_Assign(target.elts[i], temp_value)
        assign_stmts.append(temp_stmts)


def multiple_value_assignment_handler(target, value, assign_stmts, node, id_str):
    """
    Code to handle assignments like a, b = c, d. This code is converted to:
    a = c
    b = d

    Length of left and right part is cheched to make sure we are dealing with a valid assignment (an error is produced
    otherwise)

    :param target: tuple or list of variables
    :param value:  tuple or list of variables
    :param assign_stmts: statements holder list
    :param node: current AST node
    :param id_str: Type of assignment that we are processing (to create variables)
    :return:
    """
    if len(target.elts) == len(value.elts):
        temp_var_names = []

        for i in range(len(value.elts)):
            temp_stmts, temp_value = stypy_functions_copy.create_temp_Assign(value.elts[i], node.lineno, node.col_offset,
                                                                        "{0}_assignment".format(id_str))
            assign_stmts.append(temp_stmts)
            temp_var_names.append(temp_value)
        for i in range(len(target.elts)):
            temp_stmts = core_language_copy.create_Assign(target.elts[i], temp_var_names[i])
            assign_stmts.append(temp_stmts)
    else:
        TypeError(stypy_functions_copy.create_localization(node.lineno, node.col_offset),
                  "Multi-value assignments with {0}s must have the same amount of elements on both assignment sides".
                  format(id_str))


def single_assignment_handler(target, value, assign_stmts, node, id_str):
    """
    Handles single statements for hte visitor. No change is produced in the code
    :param target: Variable
    :param value: Value to assign
    :param assign_stmts: statements holder list
    :param node: current AST node
    :param id_str: Type of assignment that we are processing (to create variables)
    :return:
    """
    temp_stmts = core_language_copy.create_Assign(target, value)
    assign_stmts.append(temp_stmts)


class MultipleAssignmentsDesugaringVisitor(ast.NodeTransformer):
    # Table of functions that determines what assignment handler is going to be executed for an assignment. Each
    # key is a function that, if evaluated to true, execute the associated value function that adds the necessary
    # statements to handle the call
    __assignment_handlers = {
        (lambda target, value: isinstance(target, ast.Tuple) and (isinstance(value, ast.Tuple) or
                                                                  isinstance(value, ast.List))): (
            "tuple", multiple_value_assignment_handler),

        (lambda target, value: isinstance(target, ast.List) and (isinstance(value, ast.Tuple) or
                                                                 isinstance(value, ast.List))): (
            "list", multiple_value_assignment_handler),

        (lambda target, value: (isinstance(target, ast.List) or isinstance(target, ast.Tuple)) and (
            isinstance(value, ast.Call))): ("call", multiple_value_call_assignment_handler),

        lambda target, value: isinstance(target, ast.Name):
            ("assignment", single_assignment_handler),

        lambda target, value: isinstance(target, ast.Subscript):
            ("assignment", single_assignment_handler),

        lambda target, value: isinstance(target, ast.Attribute):
            ("assignment", single_assignment_handler),
    }

    # ######################################### MAIN MODULE #############################################

    def visit_Assign(self, node):
        assign_stmts = []
        value = node.value
        reversed_targets = node.targets
        reversed_targets.reverse()
        assign_stmts.append(stypy_functions_copy.create_blank_line())
        if len(reversed_targets) > 1:
            assign_stmts.append(
                stypy_functions_copy.create_src_comment("Multiple assigment of {0} elements.".format(len(reversed_targets))))
        else:
            assign_stmts.append(stypy_functions_copy.create_src_comment(
                "Assignment to a {0} from a {1}".format(type(reversed_targets[0]).__name__,
                                                        type(value).__name__)))

        for assign_num in range(len(reversed_targets)):
            target = reversed_targets[assign_num]
            # Function guard is true? execute handler
            for handler_func_guard in self.__assignment_handlers:
                if handler_func_guard(target, value):
                    id_str, handler_func = self.__assignment_handlers[handler_func_guard]
                    handler_func(target, value, assign_stmts, node, id_str)
                    assign_stmts = stypy_functions_copy.flatten_lists(assign_stmts)
                    value = target
                    break

        if len(assign_stmts) > 0:
            return assign_stmts
        return node
