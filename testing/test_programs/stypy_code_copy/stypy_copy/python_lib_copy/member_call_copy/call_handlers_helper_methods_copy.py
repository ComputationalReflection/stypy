from ...errors_copy.type_error_copy import TypeError
from ...errors_copy.type_warning_copy import TypeWarning
from ...python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType
from ...python_lib_copy.python_types_copy.type_inference_copy import union_type_copy
from ...reporting_copy import print_utils_copy, module_line_numbering_copy

"""
Several functions that help call handler management in various ways. Moved here to limit the size of Python files.
"""


def exist_a_type_error_within_parameters(*arg_types, **kwargs_types):
    """
    Is there at least a type error among the call parameters?
    :param arg_types: Call arguments
    :param kwargs_types: Call keyword arguments
    :return: bool
    """
    t_e_args = filter(lambda elem: isinstance(elem, TypeError), arg_types)
    if len(t_e_args) > 0:
        return True

    t_e_kwargs = filter(lambda elem: isinstance(elem, TypeError), kwargs_types.values())
    if len(t_e_kwargs) > 0:
        return True

    return False


def strip_undefined_type_from_union_type(union):
    """
    Remove undefined types from a union type
    :param union:
    :return:
    """
    ret_union = None

    for type_ in union.types:
        if not isinstance(type_, UndefinedType):
            ret_union = union_type_copy.UnionType.add(ret_union, type_)

    return ret_union


def check_undefined_type_within_parameters(localization, call_description, *arg_types, **kwargs_types):
    """
    When calling a callable element, the type of some parameters might be undefined (not initialized
    to any value in the preceding code). This function check this fact and substitute the Undefined
    parameters by suitable type errors. It also creates warnings if the undefined type is inside a
    union type, removing the undefined type from the union afterwards. It does the same with keyword arguments.

    :param localization: Caller information
    :param call_description: A textual description of the call (to generate errors)
    :param arg_types: Call arguments
    :param kwargs_types: Call keyword arguments
    :return: arguments, keyword arguments tuple with the undefined types removed or substituted by TypeErrors depending
    on if they are into union types or not
    """
    arg_types_list = list(arg_types)

    # Process arguments
    for i in range(len(arg_types_list)):
        if isinstance(arg_types_list[i], union_type_copy.UnionType):
            # Is an undefined type inside this union type?
            exist_undefined = len(filter(lambda elem: isinstance(elem, UndefinedType), arg_types[i].types)) > 0
            if exist_undefined:
                # Compose a type warning with the full description of the problem.
                offset = print_utils_copy.get_param_position(
                    module_line_numbering_copy.ModuleLineNumbering.get_line_from_module_code(
                        localization.file_name, localization.line), i)
                if offset is not -1:  # Sometimes offsets of the offending parameters cannot be obtained
                    clone_loc = localization.clone()
                    clone_loc.column = offset
                else:
                    clone_loc = localization
                TypeWarning.instance(clone_loc, "{0}: Argument {1} could be undefined".format(call_description,
                                                                                              i + 1))
            # Remove undefined type from the union type
            arg_types_list[i] = strip_undefined_type_from_union_type(arg_types[i])
            continue
        else:
            # Undefined types outside union types are treated as Type errors.
            if isinstance(arg_types[i], UndefinedType):
                offset = print_utils_copy.get_param_position(
                    module_line_numbering_copy.ModuleLineNumbering.get_line_from_module_code(
                        localization.file_name, localization.line), i)
                if offset is not -1:  # Sometimes offsets of the offending parameters cannot be obtained
                    clone_loc = localization.clone()
                    clone_loc.column = offset
                else:
                    clone_loc = localization

                arg_types_list[i] = TypeError(clone_loc, "{0}: Argument {1} is not defined".format(call_description,
                                                                                                   i + 1))
                continue

        arg_types_list[i] = arg_types[i]

    # Process keyword arguments (the same processing as argument lists)
    final_kwargs = {}
    for key, value in kwargs_types.items():
        if isinstance(value, union_type_copy.UnionType):
            exist_undefined = filter(lambda elem: isinstance(elem, UndefinedType), value.types)
            if exist_undefined:
                TypeWarning.instance(localization,
                                     "{0}: Keyword argument {1} could be undefined".format(call_description,
                                                                                           key))
            final_kwargs[key] = strip_undefined_type_from_union_type(value)
            continue
        else:
            if isinstance(value, UndefinedType):
                final_kwargs[key] = TypeError(localization,
                                              "{0}: Keyword argument {1} is not defined".format(call_description,
                                                                                                key))
                continue
        final_kwargs[key] = value

    return tuple(arg_types_list), final_kwargs


# ########################################## PRETTY-PRINTING FUNCTIONS ##########################################


def __type_error_str(arg):
    """
    Helper function of the following one.
    If arg is a type error, this avoids printing all the TypeError information and only prints the name. This is
    convenient when pretty-printing calls and its passed parameters to report errors, because if we print the full
    error information (the same one that is returned by stypy at the end) the message will be unclear.
    :param arg:
    :return:
    """
    if isinstance(arg, TypeError):
        return "TypeError"
    else:
        return str(arg)


def __format_type_list(*arg_types, **kwargs_types):
    """
    Pretty-print passed parameter list
    :param arg_types:
    :param kwargs_types:
    :return:
    """
    arg_str_list = map(lambda elem: __type_error_str(elem), arg_types[0])
    arg_str = ""
    for arg in arg_str_list:
        arg_str += arg + ", "

    if len(arg_str) > 0:
        arg_str = arg_str[:-2]

    kwarg_str_list = map(lambda elem: __type_error_str(elem), kwargs_types)
    kwarg_str = ""
    for arg in kwarg_str_list:
        kwarg_str += arg + ", "

    if len(kwarg_str) > 0:
        kwarg_str = kwarg_str[:-1]
        kwarg_str = '{' + kwarg_str + '}'

    return arg_str, kwarg_str


def __format_callable(callable_):
    """
    Pretty-print a callable entity
    :param callable_:
    :return:
    """
    if hasattr(callable_, "__name__"):
        return callable_.__name__
    else:
        return str(callable_)


def format_call(callable_, arg_types, kwarg_types):
    """
    Pretty-print calls and its passed parameters, for error reporting, using the previously defined functions
    :param callable_:
    :param arg_types:
    :param kwarg_types:
    :return:
    """
    arg_str, kwarg_str = __format_type_list(arg_types, kwarg_types.values())
    callable_str = __format_callable(callable_)
    if len(kwarg_str) == 0:
        return "\t" + callable_str + "(" + arg_str + ")"
    else:
        return "\t" + callable_str + "(" + arg_str + ", " + kwarg_str + ")"
