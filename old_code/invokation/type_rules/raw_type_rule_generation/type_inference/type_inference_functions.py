import ast, collections
from stypy import ErrorType, TypeWarning
from stypy.visitor.type_inference.visitor_utils import core_language, data_structures
from stypy.type_expert_system.types.library.python_wrappers.python_data_structures import PythonIndexableDataStructure
from stypy.type_expert_system.types.python.python_language_interface import get_builtin_type


def __assign_arguments(localization, type_store, declared_arguments_list, type_of_args):
    if len(declared_arguments_list) < len(type_of_args):
        min_len = len(declared_arguments_list)
    else:
        min_len = len(type_of_args)

    # Assign arguments
    for i in range(min_len):
        type_store.set_type_of(localization, declared_arguments_list[i], type_of_args[i])


def __process_call_type_or_args(fname, localization, type_of_self, declared_argument_name_list, call_varargs, call_kwargs, defaults):
    # First: Calculate the value of the passed parameters. They can be calculated by extracting them from
    # the parameter list or the keyword list. Values in both lists are not allowed for a parameter
    call_type_of_args = []  # Per-call passed args
    cont = 0
    error_msg = ""
    found_error = False
    arg_count = 1
    if not type_of_self is None:
        call_varargs = [type_of_self] + list(call_varargs)
    else:
        call_varargs = list(call_varargs)

    for name in declared_argument_name_list:
        # Exist an argument on that position in the passed args
        if len(call_varargs) > cont:
            call_type_of_args.append(call_varargs[cont])
            # If there is a keyword with the same name of the argument, and a value is found for it in the arg list,
            # Python reports an error
            if name in call_kwargs and not name in defaults:
                found_error = True
                msg = "'{0}' got multiple values for keyword argument '{1}'; ".format(fname, name)
                error_msg += msg
                call_type_of_args.append(ErrorType(localization, msg))

            # One the argument is processed, we delete it to not to consider it as an extra vararg
            del call_varargs[cont]
        else:
            # If no argument is passed in the list, search for a compatible keyword argument
            if name in call_kwargs:
                call_type_of_args.append(call_kwargs[name])
                # Remove the argument from the kwargs dict as we don't want it to appear as an extra kwarg
                del call_kwargs[name]
            else:
                found_error = True
                msg = "Insufficient number of arguments for function '{0}': Cannot find a value for argument" \
                      " number {1} ('{2}'); ".format(fname, arg_count, name)
                error_msg += msg
                call_type_of_args.append(ErrorType(localization, msg))

        arg_count += 1

    if found_error:
        return call_type_of_args, error_msg, True, call_varargs

    return call_type_of_args, "", False, call_varargs


def process_argument_values(localization, type_of_self, type_store, fname,
                            declared_argument_name_list,
                            declared_varargs_var,
                            declared_kwargs_var,
                            declared_defaults,
                            call_varargs=list(),  # List of arguments to unpack (if present)
                            call_kwargs=dict(),
                            allow_argument_keywords=True):  # Dictonary of keyword arguments to unpack (if present)

    found_error = False
    error_msg = ""
    if not allow_argument_keywords and len(call_kwargs) > 0:
        for arg in call_kwargs:
            if not arg in declared_defaults:
                found_error = True
                error_msg += "'{0}' takes no keyword arguments; ".format(fname)
                break

    # If a defaults dict is provided, we include those values not already present in the kwargs dict to assign its
    # values to the corresponding parameters when checking
    # Defaults can be provided in the form of a list (no name - value pairing) or in the form of a dict (name-value
    # pairing done). In the first case we build a dictionary first
    if type(declared_defaults) is list or type(declared_defaults) is tuple:
        defaults_dict = {}
        declared_argument_name_list.reverse()
        declared_defaults.reverse()
        cont = 0
        for value in declared_defaults:
            defaults_dict[declared_argument_name_list[cont]] = value
            cont = cont + 1

        declared_argument_name_list.reverse()
        declared_defaults.reverse()
    else:
        defaults_dict = declared_defaults

    # Assign defaults to those values not present in the passed parameters
    for elem in defaults_dict:
        if not elem in call_kwargs:
            call_kwargs[elem] = defaults_dict[elem]

    #print "Initial varargs = ", call_varargs
    call_type_of_args, error, found_error_on_call_args, call_varargs = __process_call_type_or_args(fname, localization, type_of_self,
                                                                                                   declared_argument_name_list,
                                                                                                   call_varargs,
                                                                                                   call_kwargs,
                                                                                                   defaults_dict)

    if found_error_on_call_args:  # Arg. arity error, return it
        error_msg += error

    found_error |= found_error_on_call_args

    # Assign arguments
    __assign_arguments(localization, type_store, declared_argument_name_list, call_type_of_args)

    #Delete left defaults to not to consider them extra parameters (if there is any left means that the
    # function has extra parameters and some of them have defaults and have not been passed
    left_kwargs = call_kwargs.keys()
    for name in left_kwargs:
        if name in defaults_dict:
            del call_kwargs[name]

    #var (star) args are composed by excess of args in the call once standard arguments are processed
    if not declared_varargs_var is None:
        # Var args is a list of all the rest of the passed types
        excess_arguments = get_builtin_type("list")
        excess_arguments.add_types_from_list(excess_arguments, call_varargs)
        type_store.set_type_of(localization, declared_varargs_var, excess_arguments)
    else:
        #print "Rest of args = ", call_varargs
        if len(call_varargs) > 0:
            found_error = True
            error_msg += "'{0}' got {1} more arguments than expected; ".format(fname, str(len(call_varargs)))


    # keyword args are composed by the contents of the keywords list left from the argument processing
    if not declared_kwargs_var is None:
        kwargs_variable = get_builtin_type("dict")

        for name in call_kwargs:
            kwargs_variable.add_index_and_value_type(kwargs_variable, (str, call_kwargs[name]))

        type_store.set_type_of(localization, declared_kwargs_var, kwargs_variable)
    else:
        if len(call_kwargs) > 0:
            found_error = True
            error_msg += "'{0}' got unexpected keyword arguments: {1}; ".format(fname, str(call_kwargs))

    if found_error:
        return ErrorType(localization, error_msg)

    return None


def create_call_to_type_inference_code(func, localization, keywords=list(), kwargs=None, starargs=None, line=0, column=0):
    call = ast.Call()

    if type(func) is tuple:
        tuple_node = ast.Tuple()
        tuple_node.elts = list(func)
        func = tuple_node

    ti_args = [localization, func]

    call.func = core_language.create_Name('invoke_member')
    call.lineno = line
    call.col_offset = column

    if starargs is None:
        call.starargs = data_structures.create_list([])
    else:
        call.starargs = data_structures.create_list(starargs)

    if kwargs is None:
        call.kwargs = data_structures.create_keyword_dict(None)
    else:
        call.kwargs = kwargs

    call.keywords = []
    call.args = ti_args  # core_language.create_list(ti_args)

    return call


def is_suitable_condition(localization, condition_type):
    if isinstance(condition_type, ErrorType):
        ErrorType(localization, "The type of this condition is erroneous")
        return False
    if not condition_type is bool:
        TypeWarning.instance(localization, "The type of this condition is not boolean. Is that what you really intend?")

    return True


def is_suitable_for_loop_condition(localization, condition_type):
    if isinstance(condition_type, ErrorType):
        ErrorType(localization, "The type of this for loop condition is erroneous")
        return False

    if not (isinstance(condition_type, PythonIndexableDataStructure)
            or (condition_type is str)
            or isinstance(condition_type, collections.Iterable)):
        ErrorType(localization, "The type of this for loop condition is erroneous")
        return False

    return True


def get_type_of_for_loop_variable(localization, condition_type):
    if isinstance(condition_type, PythonIndexableDataStructure):
        return condition_type.get_element_type()

    if (condition_type is str):
        return str

    if isinstance(condition_type, collections.Iterable):
        return condition_type.__iter__(localization)
    return ErrorType(localization, "STYPY FATAL ERROR: Unexpected type for a loop target")
