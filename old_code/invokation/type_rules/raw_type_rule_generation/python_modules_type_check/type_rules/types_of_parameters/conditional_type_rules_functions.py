import types

from stypy.errors.type_error import TypeError
from stypy.errors.type_warning import TypeWarning
from stypy.python_lib.python_types.instantiation.known_python_types_management import get_type_name, get_type_from_name

"""
This file contains functions that can be called when dealing with type rules. These enhance the type rule
checking beyond the simple checking of a <tuple with param types>:<return type> type dictionary, as they
can check additional parameter properties such as if a parameter has a certain member.
"""


def __instance_type_has_the_attr(param_obj, name, return_type):
    """
    Utility function that looks for the first parameter whose type is an instance, test if has an attribute
    called 'name' and returns its associated return type or an ErrorType if the condition is not satisfied
    :param param_obj: Tuple of parameters passed on the call
    :param name: Name of the called function
    :param return_type: Expected return type if the condition is satisfied.
    :return:
    """
    obj = None
    localization = param_obj[-1]
    param_obj = param_obj[0:-1]

    for param in param_obj:
        if param is types.InstanceType:
            obj = param
            break

    if obj == None:
        return TypeError(localization, "The object of type {0} has no member called '{1}'"
                         .format(get_type_name(type(obj)), name))

    return __has_the_attr(localization, obj, name, return_type,
                          "The object of type {0} has no member called '{1}'".format(get_type_name(type(obj)), name))


def __nth_param_has_the_attr(param_obj, pos, name, return_type):
    localization = param_obj[-1]
    param_obj = param_obj[0:-1]

    obj = param_obj[pos]

    return __has_the_attr(localization, obj, name, return_type,
                          "The object of type {0} has no member called '{1}'".format(get_type_name(type(obj)), name))


def __nth_param_is_a_subtype_of(param_obj, pos, name, return_type):
    localization = param_obj[-1]
    param_obj = param_obj[0:-1]

    obj = param_obj[pos]
    parent = get_type_from_name(name)
    # if isinstance(obj, parent):
    if issubclass(obj, parent):
        return return_type
    else:
        return TypeError(localization,
                         "The object of type {0} is not a subtype of type '{1}'".format(get_type_name(type(obj)), name))
        # TypeWarning(localization, "The object of type {0} has to be a subtype of type '{1}'".format(get_type_name(type(obj)), name))
        # return return_type


def __has_the_attr(localization, param_obj, name, return_type, msg):
    # Instance types represent objects of any class and we have not access to its members, only to the members of the
    # InstanceType class. The same applies to ClassType
    if param_obj is types.InstanceType:
        TypeWarning.instance(localization, "The passed instance must implement the member {0} to allow this call".format(name))
        return return_type

    if param_obj is types.ClassType:
        TypeWarning.instance(localization, "The passed class must implement the member {0} to allow this call".format(name))
        return return_type

    if hasattr(param_obj, name):
        return return_type
    else:
        return TypeError(localization, msg)
        # TypeWarning(localization, "The passed object must implement the member {0} to allow this call".format(name))
        # return return_type


########################### CONDITIONAL TYPE HANDLING FUNCTIONS ############################


def first_instance_type_param_has_member(member_name, return_type):
    return lambda params: __instance_type_has_the_attr(params, member_name, return_type)


def nth_param_has_member(pos, member_name, return_type):
    return lambda params: __nth_param_has_the_attr(params, pos, member_name, return_type)


def first_param_has_member(member_name, return_type):
    return nth_param_has_member(1, member_name, return_type)


def nth_param_is_a_subtype_of(pos, member_name, return_type):
    return lambda params: __nth_param_is_a_subtype_of(params, pos, member_name, return_type)


def first_param_is_a_subtype_of(member_name, return_type):
    return nth_param_is_a_subtype_of(1, member_name, return_type)
