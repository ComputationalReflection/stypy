from stypy.python_lib.type_rules.raw_type_rule_generation.python_modules_type_check.type_rules.number_of_parameters.number_of_parameters import \
    get_num_of_parameters

import types_of_parameters_member_calls
import members_of_objects
from stypy.python_lib.python_types.instantiation.known_python_types_management import *
import recoverable_errors
from known_type_of_parameters_errors import is_known_type_error


# #####################  PRIVATE INTERFACE   ######################

type_constructors = ['bool', 'bytearray', 'bytes', 'classmethod', 'complex', 'dict', 'enumerate', 'file', 'float',
                     'frozenset', 'int', 'list', 'long', 'object', 'property', 'reversed', 'set', 'slice',
                     'staticmethod', 'str', 'super', 'tuple', 'type', 'unicode', 'xrange']


def __callable_filter(target, m):
    try:
        return callable(getattr(target, m)) and not inspect.isclass(getattr(target, m))
    except AttributeError:
        return False


def __non_callable_filter(target, m):
    try:
        return not callable(getattr(target, m)) or inspect.isclass(getattr(target, m))
    except AttributeError:
        return False


def __filter_callable_members(target, members):
    """
    Utility function to tell callable from non-callable members apart
    :param target: python element
    :param members: member names of the python element to be considered
    :return:
    """
    callable_members = filter(lambda m: __callable_filter(target, m), members)
    if hasattr(target, "__name__"):
        if target.__name__ == '__builtin__':
            callable_members += type_constructors

    non_callable_members = filter(lambda m: __non_callable_filter(target, m), members)

    return non_callable_members, callable_members


def __separate_callable_members(python_element):
    """
    Separates callable from non-callable members of a python element (module, class...)
    :param python_element: Module, class...
    :return: callable, non callable member lists
    """
    if inspect.isclass(python_element) and not python_element is types.InstanceType:
        try:
            value = get_type_sample_value(python_element)
            return __filter_callable_members(value, members_of_objects.get_members_of_object(python_element))
        except KeyError:
            pass

    return __filter_callable_members(python_element, members_of_objects.get_members_of_object(python_element))


def __type_member_filter(target, m):
    try:
        return inspect.isclass(getattr(target, m))
    except AttributeError:  # Some attributes are unobtainable
        return False


def __attribute_member_filter(target, m):
    try:
        return not inspect.isclass(getattr(target, m)) and not callable(getattr(target, m))
    except AttributeError:  # Some attributes are unobtainable
        return False


def __filter_types_and_attribute_members(target, members):
    """
    Utility function to tell types from attribute members apart
    :param target: python element
    :param members: member names of the python element to be considered
    :return:
    """
    type_members = filter(lambda m: __type_member_filter(target, m), members)
    attribute_members = filter(lambda m: __attribute_member_filter(target, m), members)

    return type_members, attribute_members


def __separate_types_and_attribute_members(python_element):
    """
    Separates attributes from types defined inside a python element (module, class...)
    :param python_element: Module, class...
    :return: type members, attribute member lists
    """
    if inspect.isclass(python_element) and not python_element is types.InstanceType:
        try:
            value = get_type_sample_value(python_element)
            return __filter_types_and_attribute_members(value, members_of_objects.get_members_of_object(python_element))
        except KeyError as k:
            print k

    return __filter_types_and_attribute_members(python_element,
                                                members_of_objects.get_members_of_object(python_element))


def __increase_arity(arities):
    return_list = []
    for num in arities:
        return_list.append(num + 1)

    return return_list


# #####################  PUBLIC INTERFACE   ######################


def get_attribute_members(type_):
    """
    Gets the attributes defined inside a module or class
    :param type_: Module or class
    :return: Attribute list
    """
    return __separate_types_and_attribute_members(type_)[1]


def get_type_members(type_):
    """
    Gets the types defined inside a module or class
    :param type_: Module or class
    :return: Type list
    """
    return __separate_types_and_attribute_members(type_)[0]


def get_callable_members(type_):
    """
    Gets the callable members inside a module or class
    :param type_: Module or class
    :return: Callable list
    """
    return __separate_callable_members(type_)[1]


def get_callable_members_parameter_arity(python_element, custom_test_parameters=None, maximum_arity=3,
                                         excluded_members=[]):
    """
    Gets a tuple of (callable member name, [admitted parameter numbers]
    :param python_element: Module, class...
    :param custom_test_parameters: Custom parameter list to perform "type polling" tests.
    :return: A tuple of (callable member name, [admitted parameter numbers]
    """
    non_callable_m, callable_members = __separate_callable_members(python_element)

    for member in excluded_members:
        if member in callable_members:
            callable_members.remove(member)

    try:
        value = get_type_sample_value(python_element)
        return map(lambda member: (member, get_num_of_parameters(getattr(value, member),
                                                                 custom_test_parameters=custom_test_parameters,
                                                                 max_parameters_to_consider=maximum_arity)),
                   callable_members)
    except (KeyError, AttributeError):
        return map(lambda member: (member, get_num_of_parameters(getattr(python_element, member),
                                                                 custom_test_parameters=custom_test_parameters,
                                                                 max_parameters_to_consider=maximum_arity)),
                   callable_members)


def get_type_rules_of_attribute_members(type_):
    """
    Gets the type rules of attribute members inside a module or class
    :param type_: Module, class...
    :return: dictionary of {name: type}
    """
    rules_dict = {}

    # Attribute members
    attribute_members = get_attribute_members(type_)
    for member in attribute_members:
        rules = types_of_parameters_member_calls.get_attribute_member_type_rules(type_, member)
        # Add rules to a general rules dictionary for members
        rules_dict[member] = rules

    return rules_dict


def get_type_rules_of_type_members(type_holder):
    type_members = get_type_members(type_holder)
    result = {}
    for type_ in type_members:
        # Builtins do not wear a parent module name
        if "__builtin" in get_type_name(type_holder):
            parent_name = ""
        else:
            parent_name = get_type_name(type_holder) + "."

        full_type = parent_name + type_
        # print full_type, is_known_type(full_type)
        if not is_known_type(full_type):
            full_type_name = "'" + full_type + "'"
            type_value = "type_instantiation.get_type_sample_value(" + full_type + ")"
            result[full_type] = (full_type_name, type_value)
        else:
            full_type_name = "known_python_types_handling.get_type_name(" + full_type + ")"
            type_value = "known_python_types_handling.get_type_sample_value(" + full_type + ")"
            result[full_type] = (full_type_name, type_value)
    return result


def get_type_rules_of_callable_members(type_, custom_test_parameters=None, maximum_arity=3, has_self=False,
                                       excluded_members=None, only_for_members=None):
    """
    Gets the type rules of callable members inside a module or class
    :param type_: Module, class...
    :return: dictionary of {name: {<parameter tuple>: <call return type>}}
    """
    rules_dict = {}
    errors_list = []

    use_excluded_members = not excluded_members == None
    use_specific_members = not only_for_members == None

    # Callable members
    arities = get_callable_members_parameter_arity(type_, custom_test_parameters, maximum_arity=maximum_arity,
                                                   excluded_members=excluded_members)
    cont = 1
    for (member, arity) in arities:
        if use_excluded_members:
            if member in excluded_members:
                continue

        if use_specific_members:
            if not member in only_for_members:
                continue

        if has_self:
            arity = __increase_arity(arity)
        print "Generating rules for callable member '{0}' ({1}) ({2}/{3})".format(member, arity, cont, len(arities))
        rules, errors = types_of_parameters_member_calls.get_callable_member_type_rules(type_, member, arity,
                                                                                        maximum_arity)

        # Add rules to a general rules dictionary for members
        rules_dict[member] = rules
        errors_list += errors
        cont += 1

    print "Filtering error list to remove known errors ({0} elements)".format(len(errors_list))

    # Due fo performance reasons, rules with more than 3 parameters are not processed to see if they are recovered.
    errors_list = filter(lambda type_rule: type_rule.get_number_of_params() <= 3, errors_list)

    # Filter error list to discard those that are well known type errors (no recovery or post-processing is possible)
    errors_list = filter(lambda type_rule: not is_known_type_error(type_rule.exception_msg), errors_list)

    once_filtered_errors = len(errors_list)

    print "Processing filtered error list to recover from some error messages ({0} elements)".format(
        once_filtered_errors)

    recoverable_errors.recover_error_rules(rules_dict, errors_list)

    print "Recovered {0} errors. End of rule generation.".format(once_filtered_errors - len(errors_list))
    return rules_dict, errors_list


def __get_parameters_numbers_from(rule_list):
    """
    Creates a list indicating the different amount of parameter arities present into a list of tuples (parameter
    combinations).
    :param tuple_list: List of tuples (parameter combinations from type rules)
    :return: A list of parameter arities [int]
    """
    param_list = []

    for type_rule in rule_list:
        num_params = type_rule.get_number_of_params()
        if not num_params in param_list:
            param_list.append(num_params)

    return param_list


def get_callable_members_parameter_arity_from_type_rules(type_rules):
    """
    Callable arities may not be precise once type rules are applied, as some parameter numbers may not be real parameter
    numbers once type rules are applied (the type number algorithm is optimistic). Therefore, once type rules are calculated
    parameter arities can be more precisely obtained counting minimum and maximum arities on type rule dictionaries.
    :param type_rules: Calculated type rules
    :return: A tuple of (callable member name, [admitted parameter numbers]
    """
    return_list = []
    for member in type_rules:
        return_list.append((member, __get_parameters_numbers_from(type_rules[member])))

    return return_list
