#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
import sys
import types

import aux_functions
from python_operators import *
from stypy.errors.type_error import StypyTypeError
from stypy.errors.type_warning import TypeWarning
from stypy.invokation.handlers import handler_selection
from stypy.invokation.type_rules.type_groups import type_group_generator
from stypy.invokation.type_rules.type_groups import type_groups
from stypy.module_imports import python_imports
from stypy.reporting.localization import Localization
from stypy.types import type_containers
from stypy.types import type_inspection
from stypy.types import type_intercession
from stypy.types import undefined_type
from stypy.types import union_type
from stypy.types.known_python_types import get_sample_instance_for_type, needs_unique_instance, get_unique_instance
from stypy.types.standard_wrapper import wrap_contained_type, StandardWrapper
from stypy.invokation import handlers

builtins_module = sys.modules['__builtin__']

"""
This file is the interface that type inference files have to communicate with the stypy features. Only this functions
are accessible from type inference programs.
"""


# ############################################### TYPE HANDLING ##############################################


def wrap_type(type_):
    """
    Wrapping a type (putting a type into a wrapper class) is needed to handle certain types that need more information
    to store in order to properly deal with its type. This includes containers (list, dict...) and those types that
    has no __hash__ implementation, as Stypy uses the type hash to store them in SSA scenarios.
    Containers are wrapped to track their stored types properly. The rest of the types are not wrapped.
    :param type_:
    :return:
    """
    return wrap_contained_type(type_)


def get_new_type_instance(localization, type_instance):
    """
    Obtains an instance of the passed type instance. Stypy distinguish between a type instance and the type literal
     when calculating types, as opposite to other similar tools.
    :param localization:
    :param type_instance:
    :return:
    """
    try:
        return get_builtin_python_type_instance(localization, type(type_instance).__name__)
    except:
        return type_instance


def is_builtin_python_type(localization, type_name):
    """
    Determines if a type is a Python builtin type
    :param localization:
    :param type_name:
    :return:
    """
    # The name "None" is treated on a different way
    if type_name is "None":
        return True

    return hasattr(builtins_module, type_name)


def get_builtin_python_type(localization, type_name):
    """
    Gets a certain Python builtin type from its name
    :param localization:
    :param type_name:
    :return:
    """
    # The name "None" is treated on a different way
    if type_name is "None":
        return types.NoneType

    if hasattr(builtins_module, type_name):
        return getattr(builtins_module, type_name)

    return StypyTypeError.unknown_python_builtin_type_error(localization, type_name)


def get_builtin_python_type_instance(localization, type_name, value=None):
    """
    Gets a certain Python builtin type instance from its type name. Stypy distinguish between a type literal and a
    type instance.
    :param localization:
    :param type_name:
    :param value: If a value is provided, no instance is created an the value is returned as the type instance
    :return:
    """
    # The name "None" is treated on a different way
    if type_name is "None":
        return None

    if type_name == 'function':
        if value is not None:
            return value
        else:
            return types.FunctionType

    if type_name == 'builtin_function_or_method':
        if value is not None:
            return value
        else:
            return types.BuiltinFunctionType

    if hasattr(builtins_module, type_name):
        builtin = getattr(builtins_module, type_name)
        if value is not None:
            if builtin in [int, long, float]:
                return builtin()
            return wrap_type(builtin(value))

        try:
            if needs_unique_instance(builtin):
                return wrap_type(get_unique_instance(builtin))

            if inspect.isclass(builtin):
                return wrap_type(builtin())
        except:
            builtin = get_sample_instance_for_type(type_name)

        return wrap_type(builtin)

    try:
        return wrap_type(get_sample_instance_for_type(type_name))
    except:
        pass

    # If there was no way to create an instance of this type an a value is provided, use the value as a instance
    if value is not None:
        return value

    return StypyTypeError.unknown_python_builtin_type_error(localization, type_name)


# ############################################### TYPE CONTAINERS ##############################################

def add_contained_elements_type(localization, container, elements):
    """
    Adds an element to the passed container
    :param localization:
    :param container:
    :param elements:
    :return:
    """
    if is_error_type(container):
        return container
    existing = get_contained_elements_type(localization, container)
    if existing is undefined_type.UndefinedType:
        set_contained_elements_type(localization, container, elements)
    else:
        new = union_type.UnionType.add(existing, elements)
        set_contained_elements_type(localization, container, new)


def get_contained_elements_type(localization, container, multi_assign_arity=-1, multi_assign_index=-1):
    """
    Gets the type stored in a certain container
    :param localization:
    :param container:
    :return:
    """
    if is_error_type(container):
        return container

    return type_containers.get_contained_elements_type(container, multi_assign_arity, multi_assign_index)


def set_contained_elements_type(localization, container, elements):
    """
    Modifies the types stored by a container, dealing with union type indexes
    :param localization:
    :param container:
    :param elements:
    :return:
    """
    if type(elements) is tuple:
        if type_inspection.is_error(elements[0]):
            return elements[0]
        if type_inspection.is_union_type(elements[0]):
            errors = []
            # For each type of the union, set elements
            for t in elements[0].types:
                # Special case for dictionaties
                if len(elements) > 1:
                    result = __set_contained_elements_type(localization, container, (t, elements[1]))
                    if type_inspection.is_error(result):
                        errors.append(result)
                else:
                    result = __set_contained_elements_type(localization, container, t)
                    if type_inspection.is_error(result):
                        errors.append(result)

            # Everything is an error
            if len(errors) == len(elements[0].types):
                # Delete errors an produce a single one
                for e in errors:
                    StypyTypeError.remove_error_msg(e)
                return StypyTypeError(localization,
                                      "Indexes of indexable containers must be Integers or instances that "
                                      "implement the __index__ method")
            else:
                for e in errors:
                    e.turn_to_warning()
            return

    return __set_contained_elements_type(localization, container, elements)


def __set_contained_elements_type(localization, container, elements):
    """
    Modifies the types stored by a container, not dealing with union type indexes
    :param localization:
    :param container:
    :param elements:
    :return:
    """
    Localization.set_current(localization)
    if is_error_type(container):
        return container
    try:
        if type(elements) is tuple:
            # Key, value data structures
            if type_containers.can_store_keypairs(container):
                return type_containers.set_contained_elements_type_for_key(container, elements[0], elements[1])

            # Indexable data structures
            if type_containers.can_store_elements(container):
                if not type_group_generator.Integer == type(elements[0]):
                    if not hasattr(container, '__index__'):
                        # Ellipsis
                        if not (type_inspection.compare_type(elements[0], slice) and (type_inspection.compare_type(
                                elements[1], slice) or type_inspection.compare_type(
                            elements[1], list))):
                            return StypyTypeError(localization,
                                                  "Indexes of indexable containers must be Integers or instances that "
                                                  "implement the __index__ method")
                    else:
                        index_getter = getattr(container, '__index__')
                        if index_getter is not None:
                            if hasattr(index_getter,
                                       '__objclass__') and not index_getter.__objclass__.__name__ == "ndarray":
                                # res = invoke(localization, container, '__index__')
                                res = invoke(localization, index_getter)
                                if not type_group_generator.Integer == type(res):
                                    return StypyTypeError(localization,
                                                          "Indexes of indexable containers must be Integers or instances that "
                                                          "implement the __index__ method")

            else:
                # Other classes that define __setitem__
                m_set = type_containers.get_setitem_method(container)
                if m_set is not None:
                    try:
                        return m_set(localization, *elements)
                    except TypeError:
                        return m_set(*elements)

            if type_containers.is_slice(elements[0]):
                if type_containers.can_store_elements(elements[1]):
                    return type_containers.set_contained_elements_type(container,
                                                                       type_containers.get_contained_elements_type(
                                                                           elements[1]))
                else:
                    return type_containers.set_contained_elements_type(container, elements[1])
            else:
                return type_containers.set_contained_elements_type(container, elements[1])

        type_containers.set_contained_elements_type(container, elements)
    except Exception as ex:
        return StypyTypeError(localization, "Cannot store elements '{1}' into an object of type '{0}': {2}".format(
            type(container), str(elements), str(ex))
                              )


def del_contained_elements_type(localization, container, element):
    """
    Removes a type from a container
    :param localization:
    :param container:
    :param element:
    :return:
    """
    if is_error_type(container):
        return container
    return type_containers.del_contained_elements_type(container, element)


# ############################################### OPERATOR INVOKATION ##############################################

# This is a clone of the "operator" module that is used when invoking builtin operators. This is used to separate
# the "operator" Python module from the Python language operators implementation, because although its implementation
# is initially the same, builtin operators are not modifiable (as opposed to the ones offered by the operator module).
# This variable follows a Singleton pattern.
builtin_operators_module = None


def insert_dummy_func(module, new_name):
    """
    Auxiliar function to insert handlers to the "or" and "and" keyword operators, that have different behavior from
    the '|' and '&' operators when dealing with certain operand types.
    :param module:
    :param new_name:
    :return:
    """
    if new_name is "or_keyword":
        def f(x, y):
            return x or y
    else:
        def f(x, y):
            return x and y

    ret_func = types.FunctionType(f.func_code, f.func_globals, name=new_name,
                                  argdefs=f.func_defaults,
                                  closure=f.func_closure,
                                  )
    ret_func.__module__ = module
    return ret_func


def load_builtin_operators_module():
    """
    Loads the builtin Python operators logic that model the Python operator behavior, as a clone of the "operator"
    Python library module, that initially holds the same behavior for each operator. Once initially loaded, this logic
    cannot be altered (in Python we cannot overload the '+' operator behavior for builtin types, but we can modify the
    behavior of the equivalent operator.add function).
    :return: The behavior of the Python operators
    """
    global builtin_operators_module

    # First time calling an operator? Load operator logic
    if builtin_operators_module is None:
        operator_module = python_imports.import_python_library_module(None, 'operator')
        builtin_operators_module = operator_module

        # Special functions to the 'or' and 'and' keyword operators
        builtin_operators_module.or_keyword = insert_dummy_func("operator", "or_keyword")
        builtin_operators_module.and_keyword = insert_dummy_func("operator", "and_keyword")

    return builtin_operators_module


"""
Comparison operators
"""
comparison_operators = {
    # 'eq': lambda x, y: x == y,
    'ge': lambda x, y: x >= y,
    'gt': lambda x, y: x > y,
    'le': lambda x, y: x <= y,
    'lt': lambda x, y: x < y,
    'ne': lambda x, y: x != y,
}

"""
Operator that forces bool results
"""
forced_operator_result_checks = [
    (['lt', 'gt', 'lte', 'gte', 'le', 'ge'], type_group_generator.Integer),
]


def python_operator(localization, operator_symbol, *arguments):
    """
    Handles all the invocations to Python operators of the type inference program.
    :param localization: Caller information
    :param operator_symbol: Operator symbol ('+', '-',...). Symbols instead of operator names ('add', 'sub', ...)
    are used in the generated type inference program to improve readability
    :param arguments: Variable list of arguments of the operator
    :return: Return type of the operator call
    """
    global builtin_operators_module

    import types as python_types

    load_builtin_operators_module()

    try:
        # Test that this is a valid operator
        operator_name = operator_symbol_to_name(operator_symbol)
        # Obtain the operator call from the operator module
        operator_call = getattr(builtin_operators_module, operator_name)
    except:
        # If not a valid operator, return a type error
        return StypyTypeError(localization, "Unrecognized operator: {0}".format(operator_symbol))

    # PATCH: This specific operator reverses the argument order
    if operator_name == 'contains':
        arguments = tuple(reversed(arguments))

    # Invoke the operator and return its result type
    result = invoke(localization, operator_call, *arguments)

    # Check for comparison operator special semantics
    if type(arguments[0]) == python_types.ClassType and operator_name == 'eq':
        if type_intercession.has_member(localization, arguments[0], 'stypy__eq__'):
            if type(arguments[0].stypy__eq__) == python_types.UnboundMethodType:
                StypyTypeError.remove_error_msg(result)
                return arguments[0] == arguments[1]
        if type_intercession.has_member(localization, arguments[0], 'stypy__cmp__'):
            if type(arguments[0].stypy__cmp__) == python_types.UnboundMethodType:
                StypyTypeError.remove_error_msg(result)
                return arguments[0] == arguments[1]

    if isinstance(result, StypyTypeError) and operator_name in comparison_operators:
        try:
            # If a parameter is an error, we do not run this special case
            if len(filter(lambda param: type(param) is StypyTypeError, arguments)) > 0:
                return result

            ret = comparison_operators[operator_name](arguments[0], arguments[1])

            # The first error is more concrete than the second
            if type(ret) is StypyTypeError and type(result) is StypyTypeError:
                StypyTypeError.errors.remove(ret)
                return result

            import types
            # If the first argument is not an instance of a class that can overload operators, we return the inferred
            # error
            if not type(arguments[0]) == types.InstanceType:
                if not type(arguments[0]) == types.ClassType:
                    return result

            StypyTypeError.remove_errors_condition(
                lambda err: localization.file_name.replace("pyc", "py").split('/')[-1] in err.error_msg and
                            err.localization.line == localization.line and
                            err.localization.column == localization.column and operator_name in err.error_msg)
            return ret
        except:
            return result

    # for check_tuple in forced_operator_result_checks:
    #     if operator_name in check_tuple[0]:
    #         if isinstance(result, union_type.UnionType):
    #             types = result.get_types()
    #             found = len(filter(lambda t: check_tuple[1] == type(t), types)) > 0
    #             if not found:
    #                 return StypyTypeError(localization,
    #                                       "Operator {0} did not return an {1}".format(operator_name, check_tuple[1]))
    #             else:
    #                 return result
    #         else:
    #             if check_tuple[1] == type(result):
    #                 return result
    #             else:
    #                 return StypyTypeError(localization,
    #                                       "Operator {0} did not return an {1}".format(operator_name, check_tuple[1]))
    return result


# ############################################### CALLABLE INVOCATION ##############################################

def invoke(localization, callable_, *arguments, **keyword_arguments):
    """
    Calls a callable object
    :param localization:
    :param callable_:
    :param arguments:
    :param keyword_arguments:
    :return:
    """
    if is_error_type(callable_):
        return callable_

    return handler_selection.invoke(localization, callable_, *arguments, **keyword_arguments)


def process_argument_values(localization, type_of_self, type_store, function_name,
                            declared_argument_name_list,
                            declared_varargs_var,
                            declared_kwargs_var,
                            declared_defaults,
                            call_varargs=list(),  # List of arguments to unpack (if present)
                            call_kwargs={},
                            allow_argument_keywords=True):
    """
    Process the arguments of a function checking its type and placing them in the correct places to perform type
    inference operations within the function body
    :param localization:
    :param type_of_self:
    :param type_store:
    :param function_name:
    :param declared_argument_name_list:
    :param declared_varargs_var:
    :param declared_kwargs_var:
    :param declared_defaults:
    :param call_varargs:
    :param call_kwargs:
    :param allow_argument_keywords:
    :return:
    """
    return aux_functions.process_argument_values(localization, type_of_self, type_store, function_name,
                                                 declared_argument_name_list,
                                                 declared_varargs_var,
                                                 declared_kwargs_var,
                                                 declared_defaults,
                                                 call_varargs,
                                                 call_kwargs,
                                                 allow_argument_keywords)


def init_call_information(module_type_store, function_name, localization, argument_names, arguments):
    """
    Function body call initializer

    :param module_type_store:
    :param function_name:
    :param localization:
    :param argument_names:
    :param arguments:
    :return:
    """
    # Call to a compile-time declared method
    if type(localization) is Localization:
        # module_type_store.set_type_of(localization, 'self', type_of_self)
        localization.set_stack_trace(function_name, argument_names, arguments)
        if function_name == "__new__":
            module_type_store.set_type_of(localization, 'cls', module_type_store.get_type_of(localization, 'self'))
    else:
        # Call to a runtime added method, when the self parameter is among the method standard parameters instead of
        # in a particular position
        type_of_self = localization
        localization = arguments[0][0]
        arguments = arguments[1:]
        argument_names = argument_names[1:]

        module_type_store.set_type_of(localization, 'self', type_of_self)
        if function_name == "__new__":
            module_type_store.set_type_of(localization, 'cls', type_of_self)

        localization.set_stack_trace(function_name, argument_names, arguments)


def teardown_call_information(localization, arguments):
    """
    Function body call teardown
    :param localization:
    :param arguments:
    :return:
    """
    # Call to a compile-time declared method
    if type(localization) is Localization:
        localization.unset_stack_trace()
    else:
        # Call to a runtime added method, when the self parameter is among the method standard parameters instead of
        # in a particular position
        localization = arguments[0][0]
        localization.unset_stack_trace()


# ############################################### MODULE IMPORTS ##############################################

def generate_type_inference_code_for_module(localization, imported_module_name):
    """
    Call stypy to generate type inference code for a certain module if needed.
    :param localization:
    :param imported_module_name:
    :return:
    """
    return python_imports.generate_type_inference_code_for_module(localization, imported_module_name)


def import_module(localization, imported_module_name, origin_type_store, dest_type_store):
    """
    This function imports a module into a type store
    :param localization: Caller information
    :param imported_module_name: Name of the module
    :param origin_type_store: Type store of the imported module
    :param dest_type_store: Type store to add the module elements
    :return: None or a TypeError if the requested type do not exist
    """
    return python_imports.import_module(localization, imported_module_name, origin_type_store,
                                        dest_type_store)


def import_from_module(localization, imported_module_name, origin_type_store, dest_type_store,
                       *elements):
    """
    This function imports all the declared public members of a user-defined or Python library module into the specified
    type store
    It modules the from <module> import <element1>, <element2>, ... or * sentences and also the import <module> sentence
    :param localization: Caller information
    :param imported_module_name: Name of the module
    :param origin_type_store: Type store of the imported module
    :param dest_type_store: Type store to add the module elements
    :param elements: A variable list of arguments with the elements to import. The value '*' means all elements. No
    value models the "import <module>" sentence
    :return: None or a TypeError if the requested type do not exist
    """
    return python_imports.import_from_module(localization, imported_module_name, origin_type_store,
                                             dest_type_store,
                                             *elements)


def nest_module(localization, parent_module, child_module, child_module_type_store, parent_module_type_store):
    """
    Puts a module inside another module, to create module hierarchies
    :param localization:
    :param parent_module:
    :param child_module:
    :param child_module_type_store:
    :param parent_module_type_store:
    :return:
    """
    return python_imports.nest_module(localization, parent_module, child_module, child_module_type_store,
                                      parent_module_type_store)


def update_path_to_current_file_folder(current_file):
    """
    Updates the system path to the provided file folder
    :param current_file:
    :return:
    """
    python_imports.update_path_to_current_file_folder(current_file)


def remove_current_file_folder_from_path(current_file):
    """
    Removes from the system path the provided file folder
    :param current_file:
    :return:
    """
    python_imports.remove_current_file_folder_from_path(current_file)


# ############################################### TYPE INSPECTION ##############################################

def is_error_type(type_):
    """
    Check if a type is a type error
    :param type_:
    :return:
    """
    return type_inspection.is_error(type_)


def ensure_var_of_types(localization, var, var_description, *type_names):
    """
    This function is used to be sure that an specific var is of one of the specified types. This function is used
    by type inference programs when a variable must be of a collection of specific types for the program to be
    correct, which can happen in certain situations such as if conditions or loop tests.
    :param localization: Caller information
    :param var: Variable to test (TypeInferenceProxy)
    :param var_description: Description of the purpose of the tested variable, to show in a potential TypeError
    :param type_names: Accepted type names
    :return: None or a TypeError if the variable do not have a suitable type
    """
    if isinstance(var, type_groups.DynamicType) or type(var) is type_groups.DynamicType:
        return
    if isinstance(var, type_groups.UndefinedType) or type(var) is type_groups.UndefinedType:
        return

    python_type = type(var)

    # It is already a exception type (this is a special case for the raise statement)
    if python_type is types.TypeType and (type(var) is BaseException or issubclass(var, BaseException)):
        python_type = var

    for type_name in type_names:
        if isinstance(type_name, str):
            type_obj = eval("types." + type_name)
        else:
            type_obj = type_name

        if python_type is type_obj or issubclass(python_type, type_obj):
            return  # Suitable type found, end.

    return StypyTypeError(localization, var_description + " must be of one of the following types: " + str(type_names))


def ensure_var_has_members(localization, var, var_description, *member_names):
    """
    This function is used to make sure that a certain variable has an specific set of members, which may be needed
    when generating some type inference code that needs an specific structure o a certain object
    :param localization: Caller information
    :param var: Variable to test (TypeInferenceProxy)
    :param var_description: Description of the purpose of the tested variable, to show in a potential TypeError
    :param member_names: List of members that the type of the variable must have to be valid.
    :return: None or a TypeError if the variable do not have all passed members
    """
    python_type = var
    for type_name in member_names:
        if not type_intercession.has_member(localization, python_type, type_name):
            StypyTypeError(localization, var_description + " must have all of these members: " + str(member_names))
            return False

    return True


def __slice_bounds_checking(bound):
    """
    Checks if the bounds of an slice object have correct types
    :param bound:
    :return:
    """
    if bound is None or bound is types.NoneType:
        return [None], []

    if isinstance(bound, union_type.UnionType):
        types_to_check = bound.types
    else:
        types_to_check = [bound]

    right_types = []
    wrong_types = []
    for type_ in types_to_check:
        if type_group_generator.Integer == type(
                type_) or type_groups.CastsToIndex == type_ or handlers.call_utilities.is_numpy_array(type_):
            right_types.append(type_)
        else:
            wrong_types.append(type_)

    return right_types, wrong_types


def ensure_slice_bounds(localization, lower, upper, step):
    """
    Check the parameters of a created slice to make sure that the slice have correct bounds. If not, it returns a
    silent TypeError, as the specific problem (invalid lower, upper or step parameter is reported separately)
    :param localization: Caller information
    :param lower: Lower bound of the slice or None
    :param upper: Upper bound of the slice or None
    :param step: Step of the slice or None
    :return: A slice object or a TypeError is its parameters are invalid
    """
    error = False
    r, w = __slice_bounds_checking(lower)

    if len(w) > 0 and len(r) > 0:
        TypeWarning(localization, "Some of the possible types of the lower bound of the slice ({0}) are invalid".
                    format(lower))
    if len(w) > 0 and len(r) == 0:
        StypyTypeError(localization, "The type of the lower bound of the slice ({0}) is invalid".format(lower))
        error = True

    r, w = __slice_bounds_checking(upper)
    if len(w) > 0 and len(r) > 0:
        TypeWarning(localization, "Some of the possible types of the upper bound of the slice ({0}) are invalid".
                    format(upper))
    if len(w) > 0 and len(r) == 0:
        StypyTypeError(localization, "The type of the upper bound of the slice ({0}) is invalid".format(upper))
        error = True

    r, w = __slice_bounds_checking(step)
    if len(w) > 0 and len(r) > 0:
        TypeWarning(localization, "Some of the possible types of the step of the slice ({0}) are invalid".
                    format(step))
    if len(w) > 0 and len(r) == 0:
        StypyTypeError(localization, "The type of the step of the slice ({0}) is invalid".format(step))
        error = True

    if not error:
        return get_builtin_python_type_instance(localization, 'slice')
    else:
        return StypyTypeError(localization, "Type error when specifying slice bounds", prints_msg=False)


def enable_usage_of_dynamic_types_warning(localization, fname=""):
    """
    Enable the warning that informs the user that dynamic types are used and therefore type inference results may not
    be accurate
    :param localization:
    :return:
    """
    TypeWarning.enable_usage_of_dynamic_types_warning(localization, fname)


def is_suitable_condition(localization, cond_type):
    """
    Checks if a conditional condition has a suitable type
    :param localization:
    :param cond_type:
    :return:
    """
    return aux_functions.is_suitable_condition(localization, cond_type)


def evaluates_to_none(localization, cond_type):
    """
    Checks if a conditional condition is a NoneType
    :param localization:
    :param cond_type:
    :return:
    """
    return aux_functions.evaluates_to_none(localization, cond_type)


def is_suitable_for_loop_condition(localization, cond_type):
    """
    Checks if a loop condition has a suitable type
    :param localization:
    :param cond_type:
    :return:
    """
    return aux_functions.is_suitable_for_loop_condition(localization, cond_type)


def will_iterate_loop(localization, cond_type):
    """
    Checks if a loop condition is going to iterate. Empty lists or tuples do not iterate
    :param localization:
    :param cond_type:
    :return:
    """
    return aux_functions.will_iterate_loop(localization, cond_type)


def get_type_of_for_loop_variable(localization, cond_type):
    """
    Get the type of a loop variable
    :param localization:
    :param cond_type:
    :return:
    """
    return aux_functions.get_type_of_for_loop_variable(localization, cond_type)


# ############################################### TYPE IDIOMS ##############################################

def may_be_none(actual_type, expected_type):
    """
    Checks if a type may be none
    :param actual_type:
    :param expected_type:
    :return:
    """
    return aux_functions.may_be_none(actual_type, expected_type)


def may_not_be_none(actual_type, expected_type):
    """
    Checks if a type may be not none
    :param actual_type:
    :param expected_type:
    :return:
    """
    return aux_functions.may_not_be_none(actual_type, expected_type)


# Type is idiom

def may_be_type(actual_type, expected_type):
    """
    Checks if a type may be another type
    :param actual_type:
    :param expected_type:
    :return:
    """
    return aux_functions.may_be_type(actual_type, expected_type)


def may_not_be_type(actual_type, expected_type):
    """
    Checks if a type may not be another type
    :param actual_type:
    :param expected_type:
    :return:
    """
    return aux_functions.may_not_be_type(actual_type, expected_type)


def remove_type_from_union(union_type_obj, type_to_remove):
    """
    Removes a type from a union type
    :param union_type_obj:
    :param type_to_remove:
    :return:
    """
    return aux_functions.remove_type_from_union(union_type_obj, type_to_remove)


# isinstance idiom

def may_be_subtype(actual_type, expected_type):
    """
    Checks if a type may be a subtype of another type
    :param actual_type:
    :param expected_type:
    :return:
    """
    return aux_functions.may_be_subtype(actual_type, expected_type)


def may_not_be_subtype(actual_type, expected_type):
    """
    Checks if a type may not be a subtype of another type
    :param actual_type:
    :param expected_type:
    :return:
    """
    return aux_functions.may_not_be_subtype(actual_type, expected_type)


def remove_subtype_from_union(union_type_obj, type_to_remove):
    """
    Removes all members that may be a subtype of the passed type from a union type
    :param union_type_obj:
    :param type_to_remove:
    :return:
    """
    return aux_functions.remove_subtype_from_union(union_type_obj, type_to_remove)


def remove_not_subtype_from_union(union_type_obj, type_to_remove):
    """
    Removes all members that may not be a subtype of the passed type from a union type
    :param union_type_obj:
    :param type_to_remove:
    :return:
    """
    return aux_functions.remove_not_subtype_from_union(union_type_obj, type_to_remove)


# hasattr idiom

def may_provide_member(actual_type, member):
    """
    Checks if a type may provide the passed member
    :param actual_type:
    :param member:
    :return:
    """
    return aux_functions.may_provide_member(actual_type, member)


def may_not_provide_member(actual_type, member):
    """
    Checks if a type may not provide the passed member
    :param actual_type:
    :param member:
    :return:
    """
    return aux_functions.may_not_provide_member(actual_type, member)


def remove_member_provider_from_union(union_type_obj, member):
    """
    Removes all types from a union type that provide the passed member
    :param union_type_obj:
    :param member:
    :return:
    """
    return aux_functions.remove_member_provider_from_union(union_type_obj, member)


def remove_not_member_provider_from_union(union_type_obj, member):
    """
    Removes all types from a union type that don't provide the passed member
    :param union_type_obj:
    :param member:
    :return:
    """
    return aux_functions.remove_not_member_provider_from_union(union_type_obj, member)


def __istuple(tuple_):
    return isinstance(tuple_, StandardWrapper) and isinstance(tuple_.wrapped_type, tuple)


def __islist(obj):
    return isinstance(obj, StandardWrapper) and isinstance(obj.wrapped_type, list)


def __stypy_get_value_from_tuple(tuple_, total_length, position):
    try:
        if hasattr(tuple_.contained_types, 'types'):
            if len(tuple_.contained_types.types) == total_length:
                return tuple_.contained_types.types[position]
            else:
                if len(tuple_.contained_types.types) == 1:
                    return tuple_.contained_types.types[0]
                else:
                    return tuple_.contained_types
        else:
            if hasattr(tuple_.contained_types, 'wrapped_type'):
                if isinstance(tuple_.contained_types.wrapped_type, list):
                    return tuple_.contained_types
            else:
                #if type_group_generator.Number == type(tuple_.contained_types):
                return tuple_.contained_types
    except:
        return tuple_
    return tuple_


def __stypy_get_value_from_list(list_, total_length, position):
    try:
        t = list_.get_contained_type()
        if isinstance(t, union_type.UnionType):
            return list_
        else:
            return list_.wrapped_type[0]
    except:
        return list_


def stypy_get_value_from_tuple(tuple_, total_length, position):
    try:
        if __istuple(tuple_):
            return __stypy_get_value_from_tuple(tuple_, total_length, position)
        if __islist(tuple_):
            return __stypy_get_value_from_list(tuple_, total_length, position)
        if isinstance(tuple_, union_type.UnionType):
            typs = tuple_.types
            u = None
            # if len(typs) == total_length:
            #     return typs[position]
            for t in typs:
                if not __istuple(t):
                    return tuple_
                else:
                    u = union_type.UnionType.add(u, __stypy_get_value_from_tuple(t, total_length, position))

            return u
    except:
        return tuple_
    return tuple_
