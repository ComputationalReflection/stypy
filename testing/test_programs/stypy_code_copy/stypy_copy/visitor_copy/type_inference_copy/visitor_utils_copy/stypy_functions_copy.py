import ast

import core_language_copy
import functions_copy
import operators_copy
from ....code_generation_copy.type_inference_programs_copy.python_operators_copy import operator_name_to_symbol
from ....stypy_parameters_copy import ENABLE_CODING_ADVICES
from ....reporting_copy.module_line_numbering_copy import ModuleLineNumbering

"""
This file contains helper functions_copy to generate type inference code. These functions_copy refer to common language elements
such as assignments, numbers, strings and so on.
"""

default_function_ret_var_name = "__stypy_ret_value"
default_module_type_store_var_name = "type_store"
default_type_error_var_name = "module_errors"
default_type_warning_var_name = "module_warnings"
default_lambda_var_name = "__temp_lambda_"

# ############################################# TEMP VARIABLE CREATION ##############################################

"""
Keeps the global count of temp_<x> variables created during type inference code creation.
"""
__temp_variable_counter = 0


def __new_temp():
    global __temp_variable_counter
    __temp_variable_counter += 1
    return __temp_variable_counter


def __new_temp_str(descriptive_var_name):
    return "__temp_" + descriptive_var_name + str(__new_temp())


def new_temp_Name(right_hand_side=True, descriptive_var_name="", lineno=0, col_offset=0):
    """
    Creates an AST Name node with a suitable name for a new temp variable. If descriptive_var_name has a value, then
    this value is added to the variable predefined name
    """
    return core_language_copy.create_Name(__new_temp_str(descriptive_var_name), right_hand_side, lineno, col_offset)


def create_temp_Assign(right_hand_side, line, column, descriptive_var_name=""):
    """
    Creates an assignmen to a newly created temp variable
    """
    left_hand_side = new_temp_Name(right_hand_side=False, descriptive_var_name=descriptive_var_name, lineno=line,
                                   col_offset=column)
    right_hand_side.ctx = ast.Load()
    left_hand_side.ctx = ast.Store()
    assign_statement = ast.Assign([left_hand_side], right_hand_side)
    return assign_statement, left_hand_side


# ################################# TEMP LAMBDA FUNCTION NAME CREATION ##############################################

"""
Keeps the global count of temp_<x> variables created during type inference code creation.
"""
__temp_lambda_counter = 0


def __new_temp_lambda():
    global __temp_lambda_counter
    __temp_lambda_counter += 1
    return __temp_lambda_counter


def new_temp_lambda_str(descriptive_var_name=""):
    """
    Creates a new name for a lambda function. If descriptive_var_name has a value, then
    this value is added to the variable predefined name
    """
    return default_lambda_var_name + descriptive_var_name + str(__new_temp_lambda())


# ################################################### COMMENTS ####################################################

def __create_src_comment(comment_txt):
    comment_node = core_language_copy.create_Name(comment_txt)
    comment_expr = ast.Expr()
    comment_expr.value = comment_node

    return comment_expr


def is_blank_line(node):
    """
    Determines if a node represent a blank source code line
    """
    if isinstance(node, ast.Expr):
        if isinstance(node.value, ast.Name):
            if node.value.id == "":
                return True

    return False


def create_blank_line():
    """
    Creates a blank line in the source code
    """
    return __create_src_comment("")


def is_src_comment(node):
    """
    Determines if a node represent a Python comment
    """
    if isinstance(node, ast.Expr):
        if isinstance(node.value, ast.Name):
            if node.value.id.startswith("#"):
                return True

    return False


def create_src_comment(comment_txt, lineno=0):
    """
    Creates a Python comment with comment_txt
    """
    if lineno != 0:
        line_str = " (line {0})".format(lineno)
    else:
        line_str = ""

    return __create_src_comment("# " + comment_txt + line_str)


def create_program_section_src_comment(comment_txt):
    """
    Creates a Python comment with comment_txt and additional characters to mark code blocks
    """
    return __create_src_comment("\n################## " + comment_txt + " ##################\n")


def create_begin_block_src_comment(comment_txt):
    """
    Creates a Python comment with comment_txt to init a block of code
    """
    return __create_src_comment("\n# " + comment_txt)


def create_end_block_src_comment(comment_txt):
    """
    Creates a Python comment with comment_txt to finish a block of code
    """
    return __create_src_comment("# " + comment_txt + "\n")


def create_original_code_comment(file_name, original_code):
    """
    Creates a Python block comment with the original source file code
    """
    # Remove block comments, as this code will be placed in a block comment
    original_code = original_code.replace("\"\"\"", "'''")

    numbered_original_code = ModuleLineNumbering.put_line_numbers_to_module_code(file_name, original_code)

    comment_txt = core_language_copy.create_Name(
        "\"\"\"\nORIGINAL PROGRAM SOURCE CODE:\n" + numbered_original_code + "\n\"\"\"\n")
    initial_comment = ast.Expr()
    initial_comment.value = comment_txt

    return initial_comment


# ####################################### MISCELLANEOUS STYPY UTILITY FUNCTIONS ########################################

def flatten_lists(*args):
    """
    Recursive function to convert a list of lists into a single "flattened" list, mostly used to streamline lists
    of instructions that can contain other instruction lists
    """
    if len(args) == 0:
        return []
    if isinstance(args[0], list):
        arguments = args[0] + list(args[1:])
        return flatten_lists(*arguments)
    return [args[0]] + flatten_lists(*args[1:])


def create_print_var(variable):
    """
    Creates a node to print a variable
    """
    node = ast.Print()
    node.nl = True
    node.dest = None
    node.values = [core_language_copy.create_Name(variable)]

    return node


def assign_line_and_column(dest_node, src_node):
    """
    Assign to dest_node the same source line and column of src_node
    """
    dest_node.lineno = src_node.lineno
    dest_node.col_offset = src_node.col_offset


def create_localization(line, col):
    """
    Creates AST Nodes that creates a new Localization instance
    """
    linen = core_language_copy.create_num(line)
    coln = core_language_copy.create_num(col)
    file_namen = core_language_copy.create_Name('__file__')
    loc_namen = core_language_copy.create_Name('stypy.python_lib.python_types.type_inference.localization.Localization')
    loc_call = functions_copy.create_call(loc_namen, [file_namen, linen, coln])

    return loc_call


def create_import_stypy():
    """
    Creates AST Nodes that encode "from stypy import *"
    """
    alias = core_language_copy.create_alias('*')
    importfrom = core_language_copy.create_importfrom("stypy", alias)

    return importfrom


def create_print_errors():
    """
    Creates AST Nodes that encode "ErrorType.print_error_msgs()"
    """
    attribute = core_language_copy.create_attribute("ErrorType", "print_error_msgs")
    expr = ast.Expr()
    expr.value = functions_copy.create_call(attribute, [])

    return expr


def create_default_return_variable():
    """
    Creates AST Nodes that adds the default return variable to a function. Functions of generated type inference
     programs only has a return clause
    """
    assign_target = core_language_copy.create_Name(default_function_ret_var_name, False)
    assign = core_language_copy.create_Assign(assign_target, core_language_copy.create_Name("None"))

    return assign


def create_store_return_from_function(lineno, col_offset):
    set_type_of_comment = create_src_comment("Storing return type", lineno)
    set_type_of_method = core_language_copy.create_attribute(default_module_type_store_var_name,
                                                        "store_return_type_of_current_context")

    return_var_name = core_language_copy.create_Name(default_function_ret_var_name)
    set_type_of_call = functions_copy.create_call_expression(set_type_of_method,
                                                        [return_var_name])

    return flatten_lists(set_type_of_comment, set_type_of_call)


def create_return_from_function(lineno, col_offset):
    """
    Creates an AST node to return from a function
    """
    return_ = ast.Return()
    return_var_name = core_language_copy.create_Name(default_function_ret_var_name)
    return_.value = return_var_name

    return flatten_lists(return_)


def get_descritive_element_name(node):
    """
    Gets the name of an AST Name node or an AST Attribute node
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr

    return ""


def create_pass_node():
    """
    Creates an AST Pass node
    """
    return ast.Pass()


def assign_as_return_type(value):
    """
    Creates AST nodes to store in default_function_ret_var_name a possible return type
    """
    default_function_ret_var = core_language_copy.create_Name(default_function_ret_var_name)
    return core_language_copy.create_Assign(default_function_ret_var, value)


def create_unsupported_feature_call(localization, feature_name, feature_desc, lineno, col_offset):
    """
    Creates AST nodes to call to the unsupported_python_feature function
    """
    unsupported_feature_func = core_language_copy.create_Name('unsupported_python_feature',
                                                         line=lineno,
                                                         column=col_offset)
    unsupported_feature = core_language_copy.create_str(feature_name)
    unsupported_description = core_language_copy.create_str(
        feature_desc)
    return functions_copy.create_call_expression(unsupported_feature_func,
                                            [localization, unsupported_feature,
                                             unsupported_description])


# TODO: Remove?
# def needs_self_object_information(context, node):
#     if type(context[-1]) is ast.Call:
#         call = context[-1]
#         if type(node) is ast.Attribute:
#             if type(node.value) is ast.Name:
#                 if node.value.id == "type_store":
#                     return False
#         if type(node) is ast.Name:
#             if node.id == "type_store":
#                 return False
#             introspection_funcs = dir(runtime_type_inspection)
#             if node.id in introspection_funcs:
#                 return False
#     else:
#         return False
#
#     return True

# ################################## GET/SET TYPE AND MEMBERS FUNCTIONS ############################################

"""
Functions to get / set the type of variables
"""


def create_add_alias(alias_name, var_name, lineno, col_offset):
    get_type_of_comment = create_src_comment("Adding an alias")
    get_type_of_method = core_language_copy.create_attribute(default_module_type_store_var_name,
                                                        "add_alias", line=lineno,
                                                        column=col_offset)

    get_type_of_call = functions_copy.create_call_expression(get_type_of_method, [alias_name, var_name])

    return flatten_lists(get_type_of_comment, get_type_of_call)


def create_get_type_of(var_name, lineno, col_offset, test_unreferenced=True):
    get_type_of_comment = create_src_comment("Getting the type of '{0}'".format(var_name), lineno)
    get_type_of_method = core_language_copy.create_attribute(default_module_type_store_var_name,
                                                        "get_type_of", line=lineno,
                                                        column=col_offset)
    localization = create_localization(lineno, col_offset)
    if test_unreferenced:
        get_type_of_call = functions_copy.create_call(get_type_of_method, [localization, core_language_copy.create_str(var_name)])
    else:
        get_type_of_call = functions_copy.create_call(get_type_of_method, [localization, core_language_copy.create_str(var_name),
                                                                      core_language_copy.create_Name('False')])

    assign_stmts, temp_assign = create_temp_Assign(get_type_of_call, lineno, col_offset)

    return flatten_lists(get_type_of_comment, assign_stmts), temp_assign


def create_set_type_of(var_name, new_value, lineno, col_offset):
    set_type_of_comment = create_src_comment("Type assignment", lineno)
    set_type_of_method = core_language_copy.create_attribute(default_module_type_store_var_name, "set_type_of")

    localization = create_localization(lineno, col_offset)

    set_type_of_call = functions_copy.create_call_expression(set_type_of_method,
                                                        [localization, core_language_copy.create_str(var_name, lineno,
                                                                                                col_offset), new_value])

    return flatten_lists(set_type_of_comment, set_type_of_call)


def create_get_type_of_member(owner_var, member_name, lineno, col_offset, test_unreferenced=True):
    comment = create_src_comment("Obtaining the member '{0}' of a type".format(member_name), lineno)
    localization = create_localization(lineno, col_offset)
    # TODO: Remove?
    # get_type_of_member_func = core_language_copy.create_Name('get_type_of_member')
    # get_type_of_member_call = functions_copy.create_call(get_type_of_member_func, [localization, owner_var,
    #                                                                           core_language_copy.create_str(
    #                                                                               member_name)])

    get_type_of_member_func = core_language_copy.create_attribute(owner_var, 'get_type_of_member')
    if not test_unreferenced:
        get_type_of_member_call = functions_copy.create_call(get_type_of_member_func, [localization,
                                                                                  core_language_copy.create_str(
                                                                                      member_name),
                                                                                  core_language_copy.create_Name('False')])
    else:
        get_type_of_member_call = functions_copy.create_call(get_type_of_member_func, [localization,
                                                                                  core_language_copy.create_str(
                                                                                      member_name)])

    member_stmts, member_var = create_temp_Assign(get_type_of_member_call, lineno, col_offset)

    return flatten_lists(comment, member_stmts), member_var


def create_set_type_of_member(owner_var, member_name, value, lineno, col_offset):
    comment = create_src_comment("Setting the type of the member '{0}' of a type".format(member_name), lineno)
    localization = create_localization(lineno, col_offset)
    # TODO: Remove?
    # set_type_of_member_func = core_language_copy.create_Name('set_type_of_member')
    # set_type_of_member_call = functions_copy.create_call_expression(set_type_of_member_func, [localization, onwer_var,
    #                                                                           core_language_copy.create_str(
    #                                                                               member_name), value])

    set_type_of_member_func = core_language_copy.create_attribute(owner_var, 'set_type_of_member')
    set_type_of_member_call = functions_copy.create_call_expression(set_type_of_member_func, [localization,
                                                                                         core_language_copy.create_str(
                                                                                             member_name), value])

    return flatten_lists(comment, set_type_of_member_call)


def create_add_stored_type(owner_var, index, value, lineno, col_offset):
    comment = create_src_comment("Storing an element on a container", lineno)
    localization = create_localization(lineno, col_offset)

    add_type_func = core_language_copy.create_attribute(owner_var, 'add_key_and_value_type')
    param_tuple = ast.Tuple()
    param_tuple.elts = [index, value]
    set_type_of_member_call = functions_copy.create_call_expression(add_type_func, [localization, param_tuple])

    return flatten_lists(comment, set_type_of_member_call)


# ############################################# TYPE STORE FUNCTIONS ##############################################

"""
Code to deal with type store related functions_copy, assignments, cloning and other operations needed for the SSA algorithm
implementation
"""

# Keeps the global count of type_store_<x> variables created during type inference code creation.
__temp_type_store_counter = 0


def __new_temp_type_store():
    global __temp_type_store_counter
    __temp_type_store_counter += 1
    return __temp_type_store_counter


def __new_type_store_name_str():
    return "__temp_type_store" + str(__new_temp_type_store())


def __new_temp_type_store_Name(right_hand_side=True):
    if right_hand_side:
        return ast.Name(id=__new_type_store_name_str(), ctx=ast.Load())
    return ast.Name(id=__new_type_store_name_str(), ctx=ast.Store())


def create_type_store(type_store_name=default_module_type_store_var_name):
    call_arg = core_language_copy.create_Name("__file__")
    call_func = core_language_copy.create_Name("TypeStore")
    call = functions_copy.create_call(call_func, call_arg)
    assign_target = core_language_copy.create_Name(type_store_name, False)
    assign = core_language_copy.create_Assign(assign_target, call)

    return assign


def create_temp_type_store_Assign(right_hand_side):
    left_hand_side = __new_temp_type_store_Name(right_hand_side=False)
    assign_statement = ast.Assign([left_hand_side], right_hand_side)
    return assign_statement, left_hand_side


def create_clone_type_store():
    attribute = core_language_copy.create_attribute("type_store", "clone_type_store")
    clone_call = functions_copy.create_call(attribute, [])

    return create_temp_type_store_Assign(clone_call)


def create_set_unreferenced_var_check(state):
    if ENABLE_CODING_ADVICES:
        attribute = core_language_copy.create_attribute("type_store", "set_check_unreferenced_vars")
        call_ = functions_copy.create_call_expression(attribute, [core_language_copy.create_Name(str(state))])

        return call_
    else:
        return []


def create_set_type_store(type_store_param, clone=True):
    attribute = core_language_copy.create_attribute("type_store", "set_type_store")

    if clone:
        clone_param = core_language_copy.create_Name("True")
    else:
        clone_param = core_language_copy.create_Name("False")

    set_call = functions_copy.create_call(attribute, [type_store_param, clone_param])

    set_expr = ast.Expr()
    set_expr.value = set_call

    return set_expr


def create_join_type_store(join_func_name, type_stores_to_join):
    join_func = core_language_copy.create_Name(join_func_name)
    join_call = functions_copy.create_call(join_func, type_stores_to_join)

    left_hand_side = __new_temp_type_store_Name(right_hand_side=False)
    join_statement = ast.Assign([left_hand_side], join_call)

    return join_statement, left_hand_side


# ############################################# OPERATOR FUNCTIONS ##############################################


def create_binary_operator(operator_name, left_op, rigth_op, lineno, col_offset):
    """
    Creates AST Nodes to model a binary operator

    :param operator_name: Name of the operator
    :param left_op: Left operand (AST Node)
    :param rigth_op: Right operand (AST Node)
    :param lineno: Line
    :param col_offset: Column
    :return: List of instructions
    """
    operator_symbol = operator_name_to_symbol(operator_name)
    op_name = core_language_copy.create_str(operator_symbol)
    operation_comment = create_src_comment("Applying the '{0}' binary operator".format(operator_symbol), lineno)
    operator_call, result_var = create_temp_Assign(
        operators_copy.create_binary_operator(op_name, left_op, rigth_op, lineno, col_offset), lineno, col_offset)

    return flatten_lists(operation_comment, operator_call), result_var


def create_unary_operator(operator_name, left_op, lineno, col_offset):
    """
    Creates AST Nodes to model an unary operator

    :param operator_name: Name of the operator
    :param left_op: operand (AST Node)
    :param lineno: Line
    :param col_offset: Column
    :return: List of instructions
    """
    operator_symbol = operator_name_to_symbol(operator_name)
    op_name = core_language_copy.create_str(operator_symbol)
    operation_comment = create_src_comment("Applying the '{0}' unary operator".format(operator_symbol), lineno)
    operator_call, result_var = create_temp_Assign(
        operators_copy.create_unary_operator(op_name, left_op, lineno, col_offset), lineno, col_offset)

    return flatten_lists(operation_comment, operator_call), result_var
