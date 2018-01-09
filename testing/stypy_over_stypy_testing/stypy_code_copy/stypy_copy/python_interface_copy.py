import sys

from stypy_copy.errors_copy.type_error_copy import TypeError
from stypy_copy.errors_copy.type_warning_copy import TypeWarning
from stypy_copy.errors_copy.unsupported_features_copy import create_unsupported_python_feature_message
from stypy_copy.code_generation_copy.type_inference_programs_copy.python_operators_copy import *
from stypy_copy.python_lib_copy.module_imports_copy import python_imports_copy
from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType
from stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import ExtraTypeDefinitions
from stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy import type_group_generator_copy, type_groups_copy
from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy

"""
This file contains the stypy API that can be called inside the type inference generated programs source code.
These functions will be used to interact with stypy, extract type information and other necessary operations when
generating type inference code.
"""

"""
An object containing the Python __builtins__ module, containing the type inference functions for each Python builtin
"""
builtin_module = python_imports_copy.get_module_from_sys_cache('__builtin__')


def get_builtin_type(localization, type_name, value=UndefinedType):
    """
    Obtains a Python builtin type instance to represent the type of an object in Python. Optionally, a value for
    this object can be specified. Values for objects are not much used within the current version of stypy, but
    they are stored for future enhancements. Currently, values, if present, are taken into account for the hasattr,
    setattr and getattr builtin functions.

    :param localization: Caller information
    :param type_name: Name of the Python type to be created ("int", "float"...)
    :param value: Optional value for this type. Value must be of the speficied type. The function does not check this.
    :return: A TypeInferenceProxy representing the specified type or a TypeError if the specified type do not exist
    """

    # Type "NoneType" has an special treatment
    if "NoneType" in type_name:
        return python_imports_copy.import_from(localization, "None")

    # Python uses more builtin types than those defined in the types package. We created an special object to hold
    # an instance of each one of them. This ensures that this instance is returned.
    if hasattr(ExtraTypeDefinitions, type_name):
        ret_type = getattr(ExtraTypeDefinitions, type_name)
    else:
        # Type from the Python __builtins__ module
        ret_type = builtin_module.get_type_of_member(localization, type_name)

    # By default, types represent instances of these types (not type names)
    ret_type.set_type_instance(True)

    # Assign value if present
    if value is not UndefinedType:
        ret_type.set_value(value)

    return ret_type


def get_python_api_type(localization, module_name, type_name):
    """
    This function can obtain any type name for any Python module that have it declared. This way we can access
    non-builtin types such as those declared on the time module and so on, provided they exist within the specified
    module
    :param localization: Caller information
    :param module_name: Module name
    :param type_name: Type name within the module
    :return: A TypeInferenceProxy for the specified type or a TypeError if the requested type do not exist
    """
    # Import the module
    module = python_imports_copy.import_python_module(localization, module_name)
    if isinstance(module, TypeError):
        return module
    # Return the type declared as a member of the module
    return module.get_type_of_member(localization, type_name)


def import_elements_from_external_module(localization, imported_module_name, dest_type_store,
                                         *elements):
    """
    This function imports all the declared public members of a user-defined or Python library module into the specified
    type store
    It modules the from <module> import <element1>, <element2>, ... or * sentences and also the import <module> sentence
    :param localization: Caller information
    :param main_module_path: Path of the module to import, i. e. path of the .py file of the module
    :param imported_module_name: Name of the module
    :param dest_type_store: Type store to add the module elements
    :param elements: A variable list of arguments with the elements to import. The value '*' means all elements. No
    value models the "import <module>" sentence
    :return: None or a TypeError if the requested type do not exist
    """
    return python_imports_copy.import_elements_from_external_module(localization, imported_module_name,
                                                               dest_type_store, sys.path,
                                                               *elements)


def import_from(localization, member_name, module_name="__builtin__"):
    """
    Imports a single member from a module. If no module is specified, the builtin module is used instead. Models the
    "from <module> import <member>" sentence, being a sort version of the import_elements_from_external_module function
    but only for Python library modules
    :param localization: Caller information
    :param member_name: Member to import
    :param module_name: Python library module that contains the member or nothing to use the __builtins__ module
    :return: A TypeInferenceProxy for the specified member or a TypeError if the requested element do not exist
    """
    return python_imports_copy.import_from(localization, member_name, module_name)


def import_module(localization, module_name="__builtin__"):
    """
    Import a full Python library module (models the "import <module>" sentence for Python library modules
    :param localization: Caller information
    :param module_name: Module to import
    :return: A TypeInferenceProxy for the specified module or a TypeError if the requested module do not exist
    """
    return python_imports_copy.import_python_module(localization, module_name)


# This is a clone of the "operator" module that is used when invoking builtin operators. This is used to separate
# the "operator" Python module from the Python language operators implementation, because although its implementation
# is initially the same, builtin operators are not modifiable (as opposed to the ones offered by the operator module).
# This variable follows a Singleton pattern.
builtin_operators_module = None


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
        operator_module = python_imports_copy.import_python_module(None, 'operator')
        builtin_operators_module = operator_module.clone()
        builtin_operators_module.name = "builtin_operators"

    return builtin_operators_module


forced_operator_result_checks = [
    (['lt', 'gt', 'lte', 'gte', 'le', 'ge'], type_group_generator_copy.Integer),
]


def operator(localization, operator_symbol, *arguments):
    """
    Handles all the invokations to Python operators of the type inference program.
    :param localization: Caller information
    :param operator_symbol: Operator symbol ('+', '-',...). Symbols instead of operator names ('add', 'sub', ...)
    are used in the generated type inference program to improve readability
    :param arguments: Variable list of arguments of the operator
    :return: Return type of the operator call
    """
    global builtin_operators_module

    load_builtin_operators_module()

    try:
        # Test that this is a valid operator
        operator_name = operator_symbol_to_name(operator_symbol)
    except:
        # If not a valid operator, return a type error
        return TypeError(localization, "Unrecognized operator: {0}".format(operator_symbol))

    # Obtain the operator call from the operator module
    operator_call = builtin_operators_module.get_type_of_member(localization, operator_name)

    # PATCH: This specific operator reverses the argument order
    if operator_name == 'contains':
        arguments = tuple(reversed(arguments))

    # Invoke the operator and return its result type
    result = operator_call.invoke(localization, *arguments)
    for check_tuple in forced_operator_result_checks:
        if operator_name in check_tuple[0]:
            if check_tuple[1] == result:
                return result
            else:
                return TypeError(localization,
                                 "Operator {0} did not return an {1}".format(operator_name, check_tuple[1]))
    return result


def unsupported_python_feature(localization, feature, description=""):
    """
    This is called when the type inference program uses not yet supported by stypy Python feature
    :param localization: Caller information
    :param feature: Feature name
    :param description: Message to give to the user
    :return: A specific TypeError for this kind of problem
    """
    create_unsupported_python_feature_message(localization, feature, description)


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
    # TODO: What happens when a var has the DynamicType or UndefinedType type?
    python_type = var.get_python_type()
    for type_name in type_names:
        if type_name is str:
            type_obj = eval("types." + type_name)
        else:
            type_obj = type_name

        if python_type is type_obj:
            return  # Suitable type found, end.

    return TypeError(localization, var_description + " must be of one of the following types: " + str(type_names))


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
    python_type = var.get_python_entity()
    for type_name in member_names:
        if not hasattr(python_type, type_name):
            TypeError(localization, var_description + " must have all of these members: " + str(member_names))
            return False

    return True


def __slice_bounds_checking(bound):
    if bound is None:
        return [None], []

    if isinstance(bound, union_type_copy.UnionType):
        types_to_check = bound.types
    else:
        types_to_check = [bound]

    right_types = []
    wrong_types = []
    for type_ in types_to_check:
        if type_group_generator_copy.Integer == type_ or type_groups_copy.CastsToIndex == type_:
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
        TypeError(localization, "The type of the lower bound of the slice ({0}) is invalid".format(lower))
        error = True

    r, w = __slice_bounds_checking(upper)
    if len(w) > 0 and len(r) > 0:
        TypeWarning(localization, "Some of the possible types of the upper bound of the slice ({0}) are invalid".
                    format(upper))
    if len(w) > 0 and len(r) == 0:
        TypeError(localization, "The type of the upper bound of the slice ({0}) is invalid".format(upper))
        error = True

    r, w = __slice_bounds_checking(step)
    if len(w) > 0 and len(r) > 0:
        TypeWarning(localization, "Some of the possible types of the step of the slice ({0}) are invalid".
                    format(step))
    if len(w) > 0 and len(r) == 0:
        TypeError(localization, "The type of the step of the slice ({0}) is invalid".format(step))
        error = True

    if not error:
        return get_builtin_type(localization, 'slice')
    else:
        return TypeError(localization, "Type error when specifying slice bounds", prints_msg=False)
