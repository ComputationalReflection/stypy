from ...stypy_copy import type_store_copy
from ...stypy_copy.errors_copy.type_error_copy import TypeError
from ...stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy.stypy_functions_copy import default_lambda_var_name

"""
Several functions to help printing elements with a readable style on error reports
"""

# These variables are used by stypy, therefore they should not be printed.
private_variable_names = ["__temp_tuple_assignment", "__temp_list_assignment",
                          "__temp_lambda", "__temp_call_assignment"]


def format_function_name(fname):
    """
    Prints a function name, considering lambda functions.
    :param fname: Function name
    :return: Proper function name
    """
    if default_lambda_var_name in fname:
        return "lambda function"
    return fname


def is_private_variable_name(var_name):
    """
    Determines if a variable is a stypy private variable
    :param var_name: Variable name
    :return: bool
    """
    for private_name in private_variable_names:
        if private_name in var_name:
            return True

    return False


def get_type_str(type_):
    """
    Get the abbreviated str representation of a type for printing friendly error messages
    :param type_: Type
    :return: str
    """
    if isinstance(type_, TypeError):
        return "TypeError"

    # Is this a type store? Then it is a non-python library module
    if type(type_) == type_store_copy.typestore.TypeStore:
        return "External module '" + type_.program_name + "'"
    return str(type_)


def get_param_position(source_code, param_number):
    """
    Get the offset of a parameter within a source code line that specify a method header. This is used to mark
    parameters with type errors when reporting them.

    :param source_code: Source code (method header)
    :param param_number: Number of parameter
    :return: Offset of the parameter in the source line
    """
    try:
        split_str = source_code.split(',')
        if param_number >= len(split_str):
            return 0

        if param_number == 0:
            name_and_first = split_str[0].split('(')
            offset = len(name_and_first[0]) + 1

            blank_offset = 0
            for car in name_and_first[1]:
                if car == " ":
                    blank_offset += 1
        else:
            offset = 0
            for i in range(param_number):
                offset += len(split_str[i]) + 1  # The comma also counts

            blank_offset = 0
            for car in split_str[param_number]:
                if car == " ":
                    blank_offset += 1

        return offset + blank_offset
    except:
        return -1
