#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ast
import types

import stypy
import stypy_interface
from stypy.errors.type_error import StypyTypeError, ConstructorParameterError
from stypy.errors.type_warning import TypeWarning
from stypy.invokation.type_rules.type_groups.type_group_generator import Str, IterableObject
from stypy.reporting.localization import Localization
from stypy.reporting.print_utils import format_function_name
from stypy.stypy_parameters import ENABLE_CODING_ADVICES
from stypy.types import type_intercession
from stypy.types import union_type
from stypy.types import undefined_type
from stypy.types.standard_wrapper import wrap_contained_type
from stypy.types.type_containers import get_contained_elements_type, can_store_elements, \
    set_contained_elements_type_for_key, can_store_keypairs, get_key_types
from stypy.visitor.type_inference.visitor_utils import core_language, data_structures
from stypy.invokation.handlers import call_utilities

import numpy

"""
This file holds functions that are invoked by the generated source code of type inference programs. This code
usually need functions to perform common tasks that can be directly generated in Python source code, but this
will turn the source code of the type inference program into something much less manageable. Therefore, we
identified common tasks within this source code and encapsulated it into functions to be called, in order to
make the type inference programs smaller and more clear.

This file also contains functions to manage type idioms
"""


# ########################### FUNCTIONS TO PROCESS FUNCTION CALLS ############################


def __assign_arguments(localization, type_store, declared_arguments_list, type_of_args):
    """
    Auxiliary function to assign the declared argument names and types to a type store. The purpose of this function
    is to insert into the current type store and function context the name and value of the parameters so the code
    can use them.
    :param localization: Caller information
    :param type_store: Current type store (it is assumed that the generated source code has created a new function
    context for the called function
    :param declared_arguments_list: Argument names
    :param type_of_args: Type of arguments (order of argument names is used to establish a correspondence. If len of
    declared_argument_list and type_of_args is not the same, the lower length is used)
    :return:
    """

    # Calculate which list is sorter previous to iteration
    if len(declared_arguments_list) < len(type_of_args):
        min_len = len(declared_arguments_list)
    else:
        min_len = len(type_of_args)

    # Assign arguments one by one to the type store
    for i in xrange(min_len):
        type_store.set_type_of(localization, declared_arguments_list[i], type_of_args[i])


def __process_call_type_or_args(function_name, localization, declared_argument_name_list, call_varargs,
                                call_kwargs, defaults):
    """
    As type inference programs converts any function call to the func(*args, **kwargs) form, this function checks both
     parameters to assign its elements to declared named parameters, see if the call have been done with enough ones,
     if the call has an excess or parameters and, in the end, calculate the real variable argument list and keywords
     dictionary

    :param function_name: Called function name
    :param localization: Caller information
    :param declared_argument_name_list: Declared argument list names (example: ['a', 'b'] in def f(a, b))
    :param call_varargs: First parameter of the type inference function call
    :param call_kwargs: Second parameter of the type inference function call
    :param defaults: Declared defaults (values for those parameters that have a default value (are optional) in the
    function declaration (example: [3,4] in def f(a, b=3, c=4))
    :return:
    """

    # First: Calculate the type of the passed named parameters. They can be calculated by extracting them from
    # the parameter list or the keyword list. Values in both lists are not allowed for a parameter, and an error is
    # reported in that case
    call_type_of_args = []  # Per-call passed args
    cont = 0
    error_msg = ""
    found_error = False
    arg_count = 1

    # call_varargs = list(call_varargs)

    # Named parameters are extracted in declaration order
    for name in declared_argument_name_list:
        # Exist an argument on that position in the passed args?
        if len(call_varargs) > cont:
            call_type_of_args.append(call_varargs[cont])
            # If there is a keyword with the same name of the argument, a value is found for it in the arg list,
            # and it is not a default, report an error
            if name in call_kwargs and name not in defaults:
                found_error = True
                msg = "{0} got multiple values for keyword argument '{1}'; ".format(format_function_name(function_name),
                                                                                    name)
                error_msg += msg
                call_type_of_args.append(StypyTypeError(localization, msg, prints_msg=False))

            # One the argument is processed, we delete it to not to consider it as an extra vararg
            del call_varargs[cont]
        else:
            # If no argument is passed in the vararg list, search for a compatible keyword argument
            if name in call_kwargs:
                call_type_of_args.append(call_kwargs[name])
                # Remove the argument from the kwargs dict as we don't want it to appear as an extra kwarg
                del call_kwargs[name]
            else:
                # No value for this named argument, report the error
                found_error = True
                msg = "Insufficient number of arguments for {0}: Cannot find a value for argument" \
                      " number {1} ('{2}'); ".format(format_function_name(function_name), arg_count, name)
                error_msg += msg

                call_type_of_args.append(StypyTypeError(localization, msg, prints_msg=False))

        arg_count += 1

    # Errors found: Return the type of named args (with errors among them, the error messages, and an error found flag
    if found_error:
        return call_type_of_args, error_msg, True

    # No errors found: Return the type of named args, no error message and a no error found flag
    return call_type_of_args, "", False


def process_argument_values(localization, type_of_self, type_store, function_name,
                            declared_argument_name_list,
                            declared_varargs_var,
                            declared_kwargs_var,
                            declared_defaults,
                            call_varargs=list(),  # List of arguments to unpack (if present)
                            call_kwargs={},
                            allow_argument_keywords=True):  # Dictionary of keyword arguments to unpack (if present)
    """
    This long function is the responsible of checking all the parameters passed to a function call and make sure that
    the call is valid and possible. Argument passing in Python is a complex task, because there are several argument
    types and combinations, and the mechanism is rather flexible, so care was taken to try to identify those
    combinations, identify misuses of the call mechanism and assign the correct values to arguments. The function is
    long and complex, so documentation was placed to try to clarify the behaviour of each part.

    :param localization: Caller information
    :param type_of_self: Type of the owner of the function/method. Currently unused, may be used for reporting errors.
    :param type_store: Current type store (a function context for the current function have to be already set up)
    :param function_name: Name of the function/method/lambda function that is being invoked
    :param declared_argument_name_list: List of named arguments declared in the source code of the function
    (example ['a', 'n'] in def f(a, n))
    :param declared_varargs_var: Name of the parameter that holds the variable argument list (if any)
    (example: "args" in def f(*args))
    :param declared_kwargs_var: Name of the parameter that holds the keyword argument dictionary (if any).
    (example: "kwargs" in def f(**kwargs))
    :param declared_defaults: Declared default values for arguments (if present).
    (example: [3, 4] in def f(a=3, n=4)
    :param call_varargs: Calls to functions/methods in type inference programs only have two parameters: args
    (variable argument list) and kwargs (keyword argument dictionary). This is done in order to simplify call
     handling, as any function call can be expressed this way.
    Example: f(*args, **kwargs)
    Values for the declared arguments are extracted from the varargs list in order (so the rest of the arguments are
    the real variable list of arguments of the original function).
    :param call_kwargs: This dictionary holds pairs of (name, type). If name is in the declared argument list, the
     corresponding type is assigned to this named parameter. If it is not, it is left inside the kwargs dictionary of
     the function. In the end, the declared_kwargs_var values are those that will not be assigned to named parameters.
    :param allow_argument_keywords: Python API functions do not allow the usage of named keywords when calling them.
    This disallow the usage of this kind of calls and report errors if used with this kind of functions.
    :return:
    """

    # Error initialization
    found_error = False
    error_msg = ""

    # Special case: We are calling an Exception subclass with an init method that was autogenerated by the type inference
    # visitors. In that case, the exception construction methods are valid, and therefore we need to support calls using a
    # variable number of strings as parameters
    # if function_name.endswith("__init__") and len(declared_argument_name_list) == 0 and isinstance(type_of_self, Exception)
    #     and type(type_of_self) is not Exception

    # Call to an added method:
    if type(localization) is not Localization:
        if len(call_varargs) > 0 and type(call_varargs[0]) is Localization:
            localization = call_varargs[0]
        else:
            localization = Localization.get_current()

    # Is this function allowing argument keywords (f(a=3)? Then we must check if the call_kwargs parameter contains
    # a value. If it contains values, we must check that all of them belong to the declared defaults list. If one of
    # them it is not in this list, it is an error because the function do not admit initialized keyword parameters
    if not allow_argument_keywords and len(call_kwargs) > 0:
        for arg in call_kwargs:
            if arg not in declared_defaults:
                found_error = True
                error_msg += "{0} takes no keyword arguments; ".format(format_function_name(function_name))
                break

    # Store in the current context the declared function variable name information
    context = type_store
    context.declared_argument_name_list = declared_argument_name_list
    context.declared_varargs_var = declared_varargs_var
    context.declared_kwargs_var = declared_kwargs_var
    context.declared_defaults = declared_defaults

    # Defaults can be provided in the form of a list or tuple (values only, no name - value pairing) or in the form of
    # a dict (name-value pairing already done). In the first case we build a dictionary first to homogenize the
    # processing of these data
    if type(declared_defaults) is list or type(declared_defaults) is tuple:
        defaults_dict = {}
        # Defaults values are assigned beginning from the last parameter
        declared_argument_name_list.reverse()
        declared_defaults.reverse()
        cont = 0
        for value in declared_defaults:
            defaults_dict[declared_argument_name_list[cont]] = value
            cont += 1

        declared_argument_name_list.reverse()
        declared_defaults.reverse()
    else:
        defaults_dict = declared_defaults

    # Make varargs modifiable
    call_varargs = list(call_varargs)

    # Assign defaults to those values not present in the passed parameters
    for elem in defaults_dict:
        if elem not in call_kwargs:
            call_kwargs[elem] = defaults_dict[elem]

    # Check named parameters, variable argument list and argument keywords, assigning a value to the named arguments
    # (returned in call_type_of_args). call_vargargs and call_kwargs can lose elements if they are assigned as named
    # arguments types.
    call_type_of_args, error, found_error_on_call_args = __process_call_type_or_args(function_name,
                                                                                     localization,
                                                                                     declared_argument_name_list,
                                                                                     call_varargs,
                                                                                     call_kwargs,
                                                                                     defaults_dict)

    if found_error_on_call_args:  # Arg. arity error, return it
        error_msg += error

    found_error |= found_error_on_call_args

    # Assign arguments values
    __assign_arguments(localization, type_store, declared_argument_name_list, call_type_of_args)

    # Delete left defaults to not to consider them extra parameters (if there is any left it means that the
    # function has extra parameters, some of them have default values and a different value for them have not been
    # passed)
    left_kwargs = call_kwargs.keys()
    for name in left_kwargs:
        if name in defaults_dict:
            del call_kwargs[name]

    # var (star) args are composed by excess of args in the call once named arguments are processed
    if declared_varargs_var is not None:
        # Var args is a tuple of all the rest of the passed types. This tuple is created here and is assigned to
        # the current type store with the declared_varargs_var name
        excess_arguments = stypy_interface.get_builtin_python_type_instance(localization, "tuple")
        varargs = None
        for arg in call_varargs:
            varargs = union_type.UnionType.add(varargs, arg)
        stypy_interface.set_contained_elements_type(localization, excess_arguments, varargs)

        type_store.set_type_of(localization, declared_varargs_var, excess_arguments)
    else:
        # Arguments left in call_varargs and no variable list of arguments is declared? Somebody call us with too
        # many arguments.
        if len(call_varargs) > 0:
            found_error = True
            error_msg += "{0} got {1} more arguments than expected; ".format(format_function_name(function_name),
                                                                             str(len(call_varargs)))

    # keyword args are composed by the contents of the keywords list left from the argument processing. We create
    # a dictionary with these values and its associated var name.
    if declared_kwargs_var is not None:
        kwargs_variable = stypy_interface.get_builtin_python_type_instance(localization, "dict")

        # Create the kwargs dictionary
        for name in call_kwargs:
            str_ = stypy_interface.get_builtin_python_type_instance(localization, "str", value=name)
            set_contained_elements_type_for_key(kwargs_variable, str_, call_kwargs[name])

        type_store.set_type_of(localization, declared_kwargs_var, kwargs_variable)
    else:
        # Arguments left in call_kwargs and no keyword arguments variable declared? Somebody call us with keyword
        # parameters without accepting them or with wrong names.
        if len(call_kwargs) > 0:
            found_error = True
            error_msg += "{0} got unexpected keyword arguments: {1}; ".format(format_function_name(function_name),
                                                                              str(call_kwargs))

    # Create an error with the accumulated error messages of all the argument processing steps
    if found_error:
        # Special error for constructors, as they cannot return anything
        if "__init__" in function_name:
            return ConstructorParameterError(localization, error_msg)
        return StypyTypeError(localization, error_msg)

    return call_type_of_args, call_varargs, call_kwargs


def create_call_to_type_inference_code(func, localization, keywords=list(), kwargs=None, starargs=None, line=0,
                                       column=0):
    """
    Create the necessary Python code to call a function that performs the type inference of an existing function.
     Basically it calls the invoke method of the TypeInferenceProxy that represent the callable code, creating
     the *args and **kwargs call parameters we mentioned before.
    :param func: Function name to call
    :param localization: Caller information
    :param keywords: Unused. May be removed
    :param kwargs: keyword dictionary
    :param starargs: variable argument list
    :param line: Source line when this call is produced
    :param column: Source column when this call is produced
    :return:
    """
    call = ast.Call()

    # Initialize the arguments of the call. localization always goes first
    ti_args = [localization, func]

    # Call to type_inference_proxy_of_the_func.invoke
    call.func = core_language.create_Name('invoke')
    call.lineno = line
    call.col_offset = column

    # Create and assign starargs
    if starargs is None:
        call.starargs = data_structures.create_list([])
    else:
        call.starargs = data_structures.create_list(starargs)

    # Create and assign kwargs
    if kwargs is None:
        call.kwargs = data_structures.create_keyword_dict(None)
    else:
        call.kwargs = kwargs

    call.keywords = []

    # Assign named args (only localization)
    call.args = ti_args

    # Return call AST node
    return call


# ########################### VARIABLES FOR CONDITIONS TYPE CHECKING ############################


def evaluates_to_none(localization, condition_type):
    return condition_type is types.NoneType


def is_suitable_condition(localization, condition_type):
    """
    Checks if the type of a condition is suitable. Only checks if the type of a condition is an error, except if
    coding advices is enabled. In that case a warning is issued if the condition is not bool.
    :param localization: Caller information
    :param condition_type: Type of the condition
    :return:
    """
    if is_error_type(condition_type):
        return condition_type

    if ENABLE_CODING_ADVICES:
        if not type(condition_type) is bool:
            TypeWarning.instance(localization,
                                 "The type of this condition is not boolean ({0}). Is that what you really intend?".
                                 format(condition_type))

    return True


def is_error_type(type_):
    """
    Tells if the passed type represent some kind of error
    :param type_: Passed type
    :return: bool value
    """
    return isinstance(type_, StypyTypeError)


def can_represent_type(type_to_represent, type_to_compare):
    if type_to_represent == type(type_to_compare):
        return True

    if type(type_to_compare) is union_type.UnionType:
        for t in type_to_compare.types:
            if type_to_represent == type(t):
                return True

    return False


def is_suitable_for_loop_condition(localization, condition_type):
    """
    A loop must iterate an iterable object or data structure or an string. This function checks this
    :param localization: Caller information
    :param condition_type: Type of the condition
    :return:
    """
    if is_error_type(condition_type):
        return False

    if type(condition_type) is file:
        return True

    if not (can_store_elements(condition_type) or can_represent_type(Str, condition_type) or (
            can_represent_type(IterableObject, condition_type)) or call_utilities.is_iterable(condition_type)):
        StypyTypeError(localization, "The type of this for loop condition is erroneous")
        return False

    return True


def will_iterate_loop(localization, condition_type):
    """
    A loop must iterate an iterable object or data structure or an string. This function checks if the iterable object
    is empty (its contents are of the type UndefinedType). In that case, it does not iterate
    :param localization: Caller information
    :param condition_type: Type of the condition
    :return:
    """
    if is_error_type(condition_type):
        return False

    if type(condition_type) is file:
        return True

    if (can_store_elements(condition_type) or (
            can_represent_type(IterableObject, condition_type)) or call_utilities.is_iterable(condition_type)):
        try:
            t = get_contained_elements_type(condition_type)
            if type(t) is undefined_type.UndefinedType or t is undefined_type.UndefinedType:
                return False
        except:
            pass

        return True

    return True


def get_type_of_for_loop_variable(localization, condition_type):
    """
    A loop must iterate an iterable object or data structure or an string. This function returns the contents of
    whatever the loop is iterating
    :param localization: Caller information
    :param condition_type: Type of the condition
    :return:
    """

    if type(condition_type) is StypyTypeError:
        return condition_type

    if type(condition_type) is types.FileType:
        return str()

    if can_store_keypairs(condition_type):
        return wrap_contained_type(get_key_types(condition_type))

    # If the type of the condition can store elements, return the type of stored elements
    if can_store_elements(condition_type):
        return wrap_contained_type(get_contained_elements_type(condition_type))

    # If the type of the condition is some kind of string, return the type of string
    if can_represent_type(Str, condition_type):
        return wrap_contained_type(condition_type)

    # If the type of the condition is something iterable, return the result of calling its __iter__ method
    if can_represent_type(IterableObject, condition_type):
        iter_method = condition_type.get_type_of_member(localization, "__iter__")
        return wrap_contained_type(stypy_interface.invoke(localization, iter_method))

    if call_utilities.is_numpy_array(condition_type):
        return condition_type.get_contained_type()
    if call_utilities.is_ndenumerate(condition_type):
        contained = None
        for typ in condition_type.iter.coords:
            contained = union_type.UnionType.add(contained, typ)

        t = wrap_contained_type((contained,))
        t.set_contained_type(contained)
        return union_type.UnionType.add(t, numpy.int32)

    if call_utilities.is_ndindex(condition_type):
        t = wrap_contained_type((numpy.int32(),))
        t.set_contained_type(numpy.int32())
        return t

    return StypyTypeError(localization, "Invalid iterable type for a loop target ({0})".format(str(condition_type)))


# ################################## TYPE IDIOMS FUNCTIONS ###################################

# Is none idiom functions

def may_be_none(actual_type, expected_type):
    """
    Determines if the passed type may be NoneType
    :param actual_type:
    :param expected_type:
    :return:
    """
    if type(actual_type) is types.NoneType:
        return True, 0
    if actual_type is types.NoneType:
        return True, 0
    if stypy.types.type_inspection.is_union_type(actual_type):
        type_is_in_union = __type_is_in_union(actual_type.types, expected_type)
        if not type_is_in_union:
            return False, 0
        return True, len(actual_type.types) - 1
    return False, 0


def may_not_be_none(actual_type, expected_type):
    """
    Opposite function to the previous one
    :param actual_type:
    :param expected_type:
    :return:
    """
    if stypy.types.type_inspection.is_union_type(actual_type):
        # Type is not found, so it may not be type. Also type is found, but there are more types,
        # so it may not also be the type
        found = __type_is_in_union(actual_type.types, expected_type)
        type_is_not_in_union = not found or (found and len(actual_type.types) > 1)

        if type_is_not_in_union:
            return True, len(actual_type.types) - 1
        # All types found it is impossible that it may not be type
        return False, 0

    if type(actual_type) is types.NoneType:
        return False, 0
    if actual_type is types.NoneType:
        return False, 0

    return True, 0


# Type is idiom functions

def __type_is_in_union(type_list, expected_type):
    """
    Determines if a certain type is present in a type list
    :param type_list:
    :param expected_type:
    :return:
    """
    for typ in type_list:
        if type(typ) == expected_type:
            return True
        if expected_type is types.NoneType:
            if typ is types.NoneType:
                return True
            if type(typ) is types.NoneType:
                return True

    return False


def may_be_type(actual_type, expected_type):
    """
    Returns:
     1) if the actual type is the expected one, including the semantics of union types (int\/str may be int).
     2) It the number of types in the union type, if we suppress the actual type
     """
    if type(actual_type) == expected_type:
        return True, 0
    if stypy.types.type_inspection.is_union_type(actual_type):
        type_is_in_union = __type_is_in_union(actual_type.types, expected_type)
        if not type_is_in_union:
            return False, 0
        return True, len(actual_type.types) - 1
    return False, 0


def may_not_be_type(actual_type, expected_type):
    """
    Returns:
     1) if the actual type is not the expected one, including the semantics of union types (int\/str may not be float).
     2) It the number of types in the union type, if we suppress the actual type
     """
    if stypy.types.type_inspection.is_union_type(actual_type):
        # Type is not found, so it may not be type. Also type is found, but there are more types,
        # so it may not also be the type
        found = __type_is_in_union(actual_type.types, expected_type)
        type_is_not_in_union = not found or (found and len(actual_type.types) > 1)

        if type_is_not_in_union:
            return True, len(actual_type.types) - 1
        # All types found it is impossible that it may not be type
        return False, 0

    if not type(actual_type) == expected_type:
        return True, 0

    return False, 0


def remove_type_from_union(union_type_obj, type_to_remove):
    """
    Removes the specified type from the passed union type
    :param union_type_obj: Union type to remove from
    :param type_to_remove: Type to remove
    :return:
    """
    if not stypy.types.type_inspection.is_union_type(union_type_obj):
        return union_type_obj
    result = None
    for type_ in union_type_obj.types:
        if type_to_remove is types.NoneType:
            if type_ is not types.NoneType:
                result = union_type_obj.add(result, type_)
            if type(type_) is not types.NoneType:
                result = union_type_obj.add(result, type_)
        else:
            if not type(type_) == type_to_remove:
                result = union_type_obj.add(result, type_)

    return result


# isinstance idiom functions


def __subtype_is_in_union(type_list, expected_type):
    """
    Determines if a type list has a type that may be considered a subtype of the expected one.
    :param type_list:
    :param expected_type:
    :return:
    """
    for typ in type_list:
        if isinstance(typ, expected_type):
            return True
        if expected_type is types.NoneType:
            if typ is types.NoneType:
                return True
            if type(typ) is types.NoneType:
                return True

    return False


def may_be_subtype(expected_type, actual_type):
    """
    Returns:
     1) if the actual type is the expected one, including the semantics of union types (int\/str may be int).
     2) It the number of types in the union type, if we suppress the actual type
     """
    if isinstance(actual_type, expected_type):
        return True, 0
    if stypy.types.type_inspection.is_union_type(actual_type):
        subtype_is_in_union = __subtype_is_in_union(actual_type.types, expected_type)
        if not subtype_is_in_union:
            return False, 0
        return True, len(actual_type.types) - 1
    return False, 0


def may_not_be_subtype(expected_type, actual_type):
    """
    Returns:
     1) if the actual type is not the expected one, including the semantics of union types (int\/str may not be float).
     2) It the number of types in the union type, if we suppress the actual type
     """
    if stypy.types.type_inspection.is_union_type(actual_type):
        # Type is not found, so it may not be type. Also type is found, but there are more types,
        # so it may not also be the type
        found = __subtype_is_in_union(actual_type.types, expected_type)
        subtype_is_not_in_union = not found or (found and len(actual_type.types) > 1)

        if subtype_is_not_in_union:
            return True, len(actual_type.types) - 1
        # All types found it is impossible that it may not be type
        return False, 0

    if not isinstance(actual_type, expected_type):
        return True, 0

    return False, 0


def remove_subtype_from_union(union_type_obj, type_to_remove):
    """
    Removes the specified type from the passed union type
    :param union_type_obj: Union type to remove from
    :param type_to_remove: Type to remove
    :return:
    """
    if not stypy.types.type_inspection.is_union_type(union_type_obj):
        return union_type_obj
    result = None
    for type_ in union_type_obj.types:
        if not isinstance(type_, type_to_remove):
            result = union_type_obj.add(result, type_)
    return result


def remove_not_subtype_from_union(union_type_obj, type_to_remove):
    """
    Removes the specified type from the passed union type
    :param union_type_obj: Union type to remove from
    :param type_to_remove: Type to remove
    :return:
    """
    if not stypy.types.type_inspection.is_union_type(union_type_obj):
        return union_type_obj
    result = None
    for type_ in union_type_obj.types:
        if isinstance(type_, type_to_remove):
            result = union_type_obj.add(result, type_)
    return result


# hasattr idiom functions


def __member_is_in_union(type_list, member):
    """
    Determines if any of the types of a type list has a concrete member declared
    :param type_list:
    :param member:
    :return:
    """
    for typ in type_list:
        if type_intercession.has_attr(typ, member):
            return True

    return False


def may_provide_member(member, actual_type):
    """
    Returns:
     1) if the actual type is the expected one, including the semantics of union types (int\/str may be int).
     2) It the number of types in the union type, if we suppress the actual type
     """
    if stypy.types.type_inspection.is_union_type(actual_type):
        member_is_in_union = __member_is_in_union(actual_type.types, member)
        if not member_is_in_union:
            return False, 0
        return True, len(actual_type.types) - 1
    if type_intercession.has_attr(actual_type, member):
        return True, 0
    return False, 0


def may_not_provide_member(member, actual_type):
    """
    Returns:
     1) if the actual type is not the expected one, including the semantics of union types (int\/str may not be float).
     2) It the number of types in the union type, if we suppress the actual type
     """
    if stypy.types.type_inspection.is_union_type(actual_type):
        # Type is not found, so it may not be type. Also type is found, but there are more types,
        # so it may not also be the type
        found = __member_is_in_union(actual_type.types, member)
        member_is_not_in_union = not found or (found and len(actual_type.types) > 1)

        if member_is_not_in_union:
            return True, len(actual_type.types) - 1
        # All types found it is impossible that it may not be type
        return False, 0

    if not type_intercession.has_attr(actual_type, member):
        return True, 0

    return False, 0


def remove_member_provider_from_union(union_type_obj, member):
    """
    Removes the specified type from the passed union type
    :param union_type_obj: Union type to remove from
    :param member: Member to remove
    :return:
    """
    if not stypy.types.type_inspection.is_union_type(union_type_obj):
        return union_type_obj
    result = None
    for type_ in union_type_obj.types:
        if not type_intercession.has_attr(type_, member):
            result = union_type_obj.add(result, type_)
    return result


def remove_not_member_provider_from_union(union_type_obj, member):
    """
    Removes the specified type from the passed union type
    :param union_type_obj: Union type to remove from
    :param member: Member to remove
    :return:
    """
    if not stypy.types.type_inspection.is_union_type(union_type_obj):
        return union_type_obj
    result = None
    for type_ in union_type_obj.types:
        if type_intercession.has_attr(type_, member):
            result = union_type_obj.add(result, type_)
    return result
