#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ast
import itertools

import core_language
import functions
import operators
from stypy.reporting.module_line_numbering import ModuleLineNumbering
from stypy.stypy_parameters import ENABLE_CODING_ADVICES
from stypy.type_inference_programs.python_operators import operator_name_to_symbol
from stypy.type_inference_programs.python_operators import operator_symbol_to_name

"""
This file contains helper functions to generate type inference code. These functions refer to common language elements
such as assignments, numbers, strings and so on.
"""

"""
Names of known stypy variables
"""
default_operator_call_name = "python_operator"
default_function_ret_var_name = "stypy_return_type"
default_module_type_store_var_name = "module_type_store"
default_type_error_var_name = "module_errors"
default_type_warning_var_name = "module_warnings"
default_temp_var_name = ""  # "_stypy_"
default_lambda_var_name = "_stypy_temp_lambda_"
default_temp_type_store_var_name = "_stypy_temp_type_store"
default_import_function = "import_module"
default_import_from_function = "import_from_module"
auto_var_name = "__stypy_auto_var"


# ############################################# TEMP VARIABLE CREATION ##############################################

"""
Keeps the global count of temp_<x> variables created during type inference code creation.
"""
__temp_variable_counter = 0


def __new_temp():
    """
    Creates a new temp variable number
    :return:
    """
    global __temp_variable_counter
    __temp_variable_counter += 1
    return __temp_variable_counter


def __new_temp_str(descriptive_var_name):
    """
    Creates a new temp variable
    :param descriptive_var_name:
    :return:
    """
    if descriptive_var_name == "":
        return default_temp_var_name + str(__new_temp())
    if descriptive_var_name.startswith("__"):
        # this handles problems with private variables, that are renamed by python runtime and therefore cannot be
        # addressed using its real name
        descriptive_var_name = descriptive_var_name[2:]

    return default_temp_var_name + descriptive_var_name + "_" + str(__new_temp())


def new_temp_Name(right_hand_side=True, descriptive_var_name="", lineno=0, col_offset=0):
    """
    Creates an AST Name node with a suitable name for a new temp variable. If descriptive_var_name has a value, then
    this value is added to the variable predefined name
    """
    return core_language.create_Name(__new_temp_str(descriptive_var_name), right_hand_side, lineno, col_offset)


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
    """
    Creates a new temp lambda function number
    :return:
    """
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
    """
    Creates a new comment in the source with the provided text
    :param comment_txt:
    :return:
    """
    comment_node = core_language.create_Name(comment_txt)
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
    return [create_blank_line(),
            __create_src_comment("# ################# " + comment_txt + " ##################\n")
            ]


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
    if original_code is not None:
        # Remove block comments, as this code will be placed in a block comment
        original_code = original_code.replace("\"\"\"", "'''")

        numbered_original_code = ModuleLineNumbering.put_line_numbers_to_module_code(file_name, original_code)

        comment_txt = core_language.create_Name(
            "\"\"\"\nORIGINAL PROGRAM SOURCE CODE:\n" + numbered_original_code + "\n\"\"\"\n")
    else:
        comment_txt = core_language.create_Name("")

    initial_comment = ast.Expr()
    initial_comment.value = comment_txt

    return initial_comment


# ####################################### MISCELLANEOUS STYPY UTILITY FUNCTIONS ########################################

def __turn_to_list(elem):
    """
    Converts the passed element to a list if it is not already a list
    :param elem:
    :return:
    """
    if not isinstance(elem, list):
        return [elem]
    return elem


def flatten_lists(*args):
    """
    Recursive function to convert a list of lists into a single "flattened" list, mostly used to streamline lists
    of instructions that can contain other instruction lists
    """

    if len(args) == 0:
        return []
    turn_elements_to_lists = map(lambda elem: __turn_to_list(elem), args)
    return list(itertools.chain.from_iterable(turn_elements_to_lists))


def create_print_var(variable):
    """
    Creates a node to print a variable
    """
    node = ast.Print()
    node.nl = True
    node.dest = None
    node.values = [core_language.create_Name(variable)]

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
    linen = core_language.create_num(line)
    coln = core_language.create_num(col)
    file_namen = core_language.create_Name('__file__')

    module1 = core_language.create_attribute('stypy', 'reporting')
    module2 = core_language.create_attribute(module1, 'localization')
    loc_namen = core_language.create_attribute(module2, 'Localization')

    loc_call = functions.create_call(loc_namen, [file_namen, linen, coln])

    return loc_call


def update_localization(line, col):
    """
    Creates AST Nodes that updates the current localization
    """
    current_loc_stmts = create_localization(line, col)

    module1 = core_language.create_attribute('stypy', 'reporting')
    module2 = core_language.create_attribute(module1, 'localization')
    loc_namen = core_language.create_attribute(module2, 'Localization')
    set_current_call = core_language.create_attribute(loc_namen, 'set_current')

    loc_call = functions.create_call_expression(set_current_call, [current_loc_stmts])

    return loc_call


def create_import_stypy():
    """
    Creates AST Nodes that encode "from stypy import *"
    """
    alias = core_language.create_alias('*')
    importfrom = core_language.create_importfrom("stypy.type_inference_programs.type_inference_programs_imports", alias)

    return importfrom


def create_print_errors():
    """
    Creates AST Nodes that encode "ErrorType.print_error_msgs()"
    """
    attribute = core_language.create_attribute("StypyTypeError", "print_error_msgs")
    expr = ast.Expr()
    expr.value = functions.create_call(attribute, [])

    return expr


def create_default_return_variable():
    """
    Creates AST Nodes that adds the default return variable to a function. Functions of generated type inference
     programs only has a return clause
    """
    return_val_instr = create_set_type_of(default_function_ret_var_name, core_language.create_Name("None"), 0, 0)
    return return_val_instr


def create_store_return_from_function(f_name, lineno, col_offset):
    """
    Creates the return source code of any type inference function
    :param f_name:
    :param lineno:
    :param col_offset:
    :return:
    """
    set_type_of_comment = create_src_comment(
        "Storing the return type of function '{0}' in the type store".format(f_name),
    )
    set_type_of_method = core_language.create_attribute(default_module_type_store_var_name,
                                                        "store_return_type_of_current_context")

    return_val_instr, val_var = create_get_type_of(default_function_ret_var_name, lineno, col_offset)
    set_type_of_call = functions.create_call_expression(set_type_of_method,
                                                        [val_var])

    return_comment = create_src_comment("Return type of the function '{0}'".format(f_name))

    return_ = ast.Return()
    return_.value = val_var

    context_unset = functions.create_context_unset()

    return flatten_lists(create_blank_line(),
                         set_type_of_comment,
                         return_val_instr,
                         set_type_of_call,
                         create_blank_line(),
                         context_unset,
                         create_blank_line(),
                         return_comment,
                         return_)


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


def assign_as_return_type(value, lineno, col_offset):
    """
    Creates AST nodes to store in default_function_ret_var_name a possible return type
    """
    localization = create_localization(lineno, col_offset)
    default_function_ret_var = core_language.create_str(default_function_ret_var_name)
    set_type_of_member_func = core_language.create_attribute(default_module_type_store_var_name, 'set_type_of')

    return functions.create_call_expression(set_type_of_member_func, [localization, default_function_ret_var, value])


def create_unsupported_feature_call(localization, feature_name, feature_desc, lineno, col_offset):
    """
    Creates AST nodes to call to the unsupported_python_feature function
    """
    unsupported_feature_func = core_language.create_Name('unsupported_python_feature',
                                                         line=lineno,
                                                         column=col_offset)
    unsupported_feature = core_language.create_str(feature_name)
    unsupported_description = core_language.create_str(
        feature_desc)
    return functions.create_call_expression(unsupported_feature_func,
                                            [localization, unsupported_feature,
                                             unsupported_description])


# ################################## GET/SET TYPE AND MEMBERS FUNCTIONS ############################################

"""
Functions to get / set the type of variables
"""


def create_add_alias(alias_name, var_name, lineno, col_offset):
    """
    Creates code to add an alias to the type store
    :param alias_name:
    :param var_name:
    :param lineno:
    :param col_offset:
    :return:
    """
    get_type_of_comment = create_src_comment("Adding an alias")
    get_type_of_method = core_language.create_attribute(default_module_type_store_var_name,
                                                        "add_alias", line=lineno,
                                                        column=col_offset)

    get_type_of_call = functions.create_call_expression(get_type_of_method, [alias_name, var_name])

    return flatten_lists(get_type_of_comment, get_type_of_call)


def create_get_type_of(var_name, lineno, col_offset, test_unreferenced=True):
    """
    Creates code to get the type of a variable
    :param var_name:
    :param lineno:
    :param col_offset:
    :param test_unreferenced:
    :return:
    """
    get_type_of_comment = create_src_comment("Getting the type of '{0}'".format(var_name), lineno)
    get_type_of_method = core_language.create_attribute(default_module_type_store_var_name,
                                                        "get_type_of", line=lineno,
                                                        column=col_offset)
    localization = create_localization(lineno, col_offset)
    if test_unreferenced:
        get_type_of_call = functions.create_call(get_type_of_method, [localization, core_language.create_str(var_name)])
    else:
        get_type_of_call = functions.create_call(get_type_of_method, [localization, core_language.create_str(var_name),
                                                                      core_language.create_Name('False')])

    assign_stmts, temp_assign = create_temp_Assign(get_type_of_call, lineno, col_offset, descriptive_var_name=var_name)

    return flatten_lists(get_type_of_comment, assign_stmts), temp_assign


def create_set_type_of(var_name, new_value, lineno, col_offset):
    """
    Creates code to set the type of a variable
    :param var_name:
    :param new_value:
    :param lineno:
    :param col_offset:
    :return:
    """
    set_type_of_comment = create_src_comment("Assigning a type to the variable '{0}'".format(var_name), lineno)
    set_type_of_method = core_language.create_attribute(default_module_type_store_var_name, "set_type_of")

    localization = create_localization(lineno, col_offset)

    set_type_of_call = functions.create_call_expression(set_type_of_method,
                                                        [localization, core_language.create_str(var_name, lineno,
                                                                                                col_offset), new_value])

    return flatten_lists(set_type_of_comment, set_type_of_call)


def create_get_type_of_member(owner_var, member_name, lineno, col_offset, test_unreferenced=True):
    """
    Creates code to get the type of an object member
    :param owner_var:
    :param member_name:
    :param lineno:
    :param col_offset:
    :param test_unreferenced:
    :return:
    """
    comment = create_src_comment("Obtaining the member '{0}' of a type".format(member_name), lineno)
    localization = create_localization(lineno, col_offset)

    get_type_of_member_func = core_language.create_attribute(default_module_type_store_var_name, 'get_type_of_member')
    if not test_unreferenced:
        get_type_of_member_call = functions.create_call(get_type_of_member_func, [localization, owner_var,
                                                                                  core_language.create_str(
                                                                                      member_name),
                                                                                  core_language.create_Name('False')])
    else:
        get_type_of_member_call = functions.create_call(get_type_of_member_func, [localization, owner_var,
                                                                                  core_language.create_str(
                                                                                      member_name)])

    member_stmts, member_var = create_temp_Assign(get_type_of_member_call, lineno, col_offset, member_name)

    return flatten_lists(comment, member_stmts), member_var


def create_set_type_of_member(owner_var, member_name, value, lineno, col_offset):
    """
    Creates code to set the type of an object member
    :param owner_var:
    :param member_name:
    :param value:
    :param lineno:
    :param col_offset:
    :return:
    """
    comment = create_src_comment("Setting the type of the member '{0}' of a type".format(member_name), lineno)
    localization = create_localization(lineno, col_offset)

    set_type_of_member_func = core_language.create_attribute(default_module_type_store_var_name, 'set_type_of_member')
    set_type_of_member_call = functions.create_call_expression(set_type_of_member_func, [localization, owner_var,
                                                                                         core_language.create_str(
                                                                                             member_name), value])

    return flatten_lists(comment, set_type_of_member_call)


def create_add_stored_type(owner_var, index, value, lineno, col_offset):
    """
    Create code to add an element to a container
    :param owner_var:
    :param index:
    :param value:
    :param lineno:
    :param col_offset:
    :return:
    """
    comment = create_src_comment("Storing an element on a container", lineno)
    localization = create_localization(lineno, col_offset)

    add_type_func = core_language.create_Name('set_contained_elements_type')
    param_tuple = ast.Tuple()
    param_tuple.elts = [index, value]
    set_type_of_member_call = functions.create_call_expression(add_type_func, [localization, owner_var, param_tuple])

    return flatten_lists(comment, set_type_of_member_call)


def create_get_type_instance_of(type_name, lineno, col_offset):
    """
    Create code to get an instance of the passed type
    :param type_name:
    :param lineno:
    :param col_offset:
    :return:
    """
    get_func = core_language.create_Name("get_builtin_python_type_instance")
    param1 = core_language.create_str(type_name)
    localization = create_localization(lineno, col_offset)
    get_list_call = functions.create_call(get_func, [localization, param1])
    return create_temp_Assign(get_list_call, lineno, col_offset, type_name.replace(".", "_"))


# ############################################# TYPE STORE FUNCTIONS ##############################################

"""
Code to deal with type store related functions, assignments, cloning and other operations needed for the SSA algorithm
implementation
"""

# Keeps the global count of type_store_<x> variables created during type inference code creation.
__temp_type_store_counter = 0


def __new_temp_type_store():
    """
    Create a new temp type store variable number
    :return:
    """
    global __temp_type_store_counter
    __temp_type_store_counter += 1
    return __temp_type_store_counter


def __new_type_store_name_str():
    """
    Create a new temp type store variable
    :return:
    """
    return default_temp_type_store_var_name + str(__new_temp_type_store())


def __new_temp_type_store_Name(right_hand_side=True):
    """
    Create a new temp type store ast.Name node
    :return:
    """
    if right_hand_side:
        return ast.Name(id=__new_type_store_name_str(), ctx=ast.Load())
    return ast.Name(id=__new_type_store_name_str(), ctx=ast.Store())


def create_type_store(type_store_name=default_module_type_store_var_name):
    """
    Creates code to create a new type store
    :param type_store_name:
    :return:
    """
    call_arg = core_language.create_Name("None")
    call_arg2 = core_language.create_Name("__file__")
    call_func = core_language.create_Name("Context")
    call = functions.create_call(call_func, [call_arg, call_arg2])
    assign_target = core_language.create_Name(type_store_name, False)
    assign = core_language.create_Assign(assign_target, call)

    return assign


def create_temp_type_store_Assign(right_hand_side):
    """
    Creates a new type store variable assignment
    :param right_hand_side:
    :return:
    """
    left_hand_side = core_language.create_Name(default_module_type_store_var_name, False)
    assign_statement = ast.Assign([left_hand_side], right_hand_side)
    return assign_statement, left_hand_side


# ############################################# SSA ##############################################

def create_open_ssa_context(context_name):
    """
    Creates code to open a new SSA context
    :param context_name:
    :return:
    """
    ssa_context_class = core_language.create_Name("SSAContext")
    attribute = core_language.create_attribute(ssa_context_class, "create_ssa_context")
    clone_call = functions.create_call(attribute, core_language.create_Name(default_module_type_store_var_name),
                                       [core_language.create_str(context_name)])

    return create_temp_type_store_Assign(clone_call)


def create_open_ssa_branch(branch_name):
    """
    Creates code to open a new SSA branch
    :param branch_name:
    :return:
    """
    attribute = core_language.create_attribute(default_module_type_store_var_name, "open_ssa_branch")
    clone_call = functions.create_call_expression(attribute, [core_language.create_str(branch_name)])

    return clone_call


def create_join_ssa_context():
    """
    Creates code to close a SSA context
    :return:
    """
    attribute = core_language.create_attribute(default_module_type_store_var_name, "join_ssa_context")
    clone_call = functions.create_call(attribute, [])

    return create_temp_type_store_Assign(clone_call)


# ############################################# OPERATOR FUNCTIONS ##############################################


def create_binary_operator(operator_symbol, left_op, rigth_op, lineno, col_offset, on_aug_assign=False):
    """
    Creates AST Nodes to model a binary operator

    :param operator_symbol: Name of the operator
    :param left_op: Left operand (AST Node)
    :param rigth_op: Right operand (AST Node)
    :param lineno: Line
    :param col_offset: Column
    :param on_aug_assign: Tells if we are into an augment assignment or not
    :return: List of instructions
    """
    operator_symbol = operator_name_to_symbol(operator_symbol)
    if on_aug_assign:
        operator_symbol += "="

    op_name = core_language.create_str(operator_symbol)
    operation_comment = create_src_comment("Applying the binary operator '{0}'".format(operator_symbol), lineno)
    operator_call, result_var = create_temp_Assign(
        operators.create_binary_operator(op_name, left_op, rigth_op, lineno, col_offset), lineno, col_offset,
        descriptive_var_name="result_" + operator_symbol_to_name(operator_symbol))

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
    op_name = core_language.create_str(operator_symbol)
    operation_comment = create_src_comment("Applying the '{0}' unary operator".format(operator_symbol), lineno)
    operator_call, result_var = create_temp_Assign(
        operators.create_unary_operator(op_name, left_op, lineno, col_offset), lineno, col_offset,
        descriptive_var_name="result_" + operator_symbol_to_name(operator_symbol))

    return flatten_lists(operation_comment, operator_call), result_var


def create_set_unreferenced_var_check(state):
    """
    Creates code to check for unreferenced variables
    :param state:
    :return:
    """
    if ENABLE_CODING_ADVICES:
        attribute = core_language.create_attribute(default_module_type_store_var_name, "set_check_unreferenced_vars")
        call_ = functions.create_call_expression(attribute, [core_language.create_Name(str(state))])

        return call_
    else:
        return []


def extract_name_node_values(sub_ast_tree):
    """
    Extract the names contained in a tuple or name expression (used in for loop processing)
    :param sub_ast_tree:
    :return:
    """
    names = []
    if isinstance(sub_ast_tree, ast.Tuple):
        for name in sub_ast_tree.elts:
            if isinstance(name, ast.Name):
                names.append(name.id)
            else:
                names += extract_name_node_values(name)
    if isinstance(sub_ast_tree, ast.Name):
        names.append(sub_ast_tree.id)

    return names
