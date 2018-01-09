from stypy.python_lib.type_rules.raw_type_rule_generation.python_modules_type_check.type_rules.types_of_parameters.type_rule import TypeRule
import types
from stypy.python_lib.python_types.type_inference.undefined_type import UndefinedType

"""
Type rule postprocessing

The functions in this source file are used to recover some of the type rule errors reported in a "type polling" run over
callable code in order to accept additional parameter type combinations that may lead to a concrete error type.
For that reason, some error messages are identified, analyzed and processed so they are converted to a correct TypeRule,
ocassionaly requiring additional runtime data of its parameters (by using functions from the conditional_type_rules_
functions.py file).
"""

__predefined_return_types = {
    '__format__': ("str", ""),
    '__getattribute__': ("UndefinedType", UndefinedType()),
    '__delattr__': ("types.NoneType", None),
}


def __lookup_first_suitable_return_type(type_rules):
    if len(type_rules) > 0:
        if type_rules[0].member_name in __predefined_return_types:
            return __predefined_return_types[type_rules[0].member_name]

    for type_rule in type_rules:

        if not type(type_rule.return_obj) is types.NotImplementedType and not type_rule.is_error and \
                not type_rule.is_conditional_rule:
            return type_rule.return_type_name, type_rule.return_obj

    return (None, None)


def __build_type_check_function_call(function_name, params, return_type):
    params_str = ""
    for param in params:
        params_str += "'" + param + "', "

    params_str = params_str[0:-2]

    return "{0}({1}, {2})".format(function_name, params_str, return_type)


def __recover_from_member_error(type_rules, error_rule, check_function):
    """
    Converts an error rule to a TypeRule when an error like '...instance has no attribute <attribute_name>' is
    found. Which means that the call will be valid is an InstanceType parameter has that <attribute_name>
    :param type_rules: Correct type rules
    :param error_rule: Error type rules
    :return:
    """
    try:
        existing_rules = type_rules[error_rule.member_name]
        if len(existing_rules) == 0:
            return_type_name = "UndefinedType"  # No proper return type to put. Static rules should be used instead.

        return_type_name = __lookup_first_suitable_return_type(existing_rules)[0]
        if return_type_name is None:
            return_type_name = "UndefinedType"  # No proper return type to put. Static rules should be used instead.

        if "__" in error_rule.exception_msg:
            member_name = "__" + error_rule.exception_msg.split("__")[1] + "__"
        else:
            if "'" in error_rule.exception_msg:
                member_name = error_rule.exception_msg.split("'")[1]
            else:
                return

        new_rule = TypeRule(error_rule.owner_obj, error_rule.member_name, error_rule.param_types, None)

        new_rule.set_function_return_type(__build_type_check_function_call(check_function,
                                                                           [member_name],
                                                                           return_type_name)
        )

        type_rules[error_rule.member_name].append(new_rule)
    except Exception:
        pass


def __recover_from_first_instance_member_error(type_rules, error_rule):
    return __recover_from_member_error(type_rules, error_rule, "first_instance_type_param_has_member")


def __recover_from_first_param_member_error(type_rules, error_rule):
    return __recover_from_member_error(type_rules, error_rule, "first_param_has_member")


def __recover_from_error(type_rules, error_rule):
    """
    Converts an error rule to a TypeRule when an error like 'does not match format' is
    found. Which means that the call is valid but a concrete format of a string parameter is needed.
    :param type_rules: Correct type rules
    :param error_rule: Error type rules
    :return:
    """
    try:
        existing_rules = type_rules[error_rule.member_name]
        if len(existing_rules) == 0:
            return_type_obj = UndefinedType
        else:
            return_type_obj = __lookup_first_suitable_return_type(existing_rules)[1]

        type_rules[error_rule.member_name].append(
            TypeRule(error_rule.owner_obj, error_rule.member_name, error_rule.param_types, return_type_obj)
        )
    except KeyError:
        pass


def __recover_from_subtype_error(type_rules, error_rule):
    """
    Converts an error rule to a TypeRule when an error like '...instance has no attribute <attribute_name>' is
    found. Which means that the call will be valid is an InstanceType parameter has that <attribute_name>
    :param type_rules: Correct type rules
    :param error_rule: Error type rules
    :return:
    """
    try:
        existing_rules = type_rules[error_rule.member_name]
        if len(existing_rules) == 0:
            return_type_name = "UndefinedType"  # No proper return type to put. Static rules should be used instead.

        return_type_name = __lookup_first_suitable_return_type(existing_rules)[0]
        if return_type_name is None:
            return_type_name = "UndefinedType"  # No proper return type to put. Static rules should be used instead.

        type_name = error_rule.exception_msg.split(" ")[-1]
        new_rule = TypeRule(error_rule.owner_obj, error_rule.member_name, error_rule.param_types, None)

        new_rule.set_function_return_type(__build_type_check_function_call("first_param_is_a_subtype_of",
                                                                           [type_name],
                                                                           return_type_name)
        )

        type_rules[error_rule.member_name].append(new_rule)
    except Exception:
        pass


"""
Recoverable error message checkers
"""
recoverable_msgs = [
    # Error messages produced by the value of the polling parameters rather than a concrete type error
    (lambda msg: "does not match format" in msg, __recover_from_error),
    (lambda msg: "not in sequence" in msg, __recover_from_error),
    (lambda msg: "attempt to assign sequence of size" in msg, __recover_from_error),
    (lambda msg: "is not in list" in msg, __recover_from_error),
    (lambda msg: "index out of range" in msg, __recover_from_error),
    (lambda msg: "Invalid conversion specification" in msg, __recover_from_error),
    (lambda msg: "has no attribute 'foo'" in msg, __recover_from_error),

    (lambda msg: "exceptions.StopIteration" in msg, __recover_from_error),
    (lambda msg: "exceptions.KeyError" in msg, __recover_from_error),
    (lambda msg: "dictionary is empty" in msg, __recover_from_error),

    #Object has a concrete member related errors
    (lambda msg: "instance has no attribute" in msg, __recover_from_first_instance_member_error),
    (lambda msg: "object has no attribute" in msg, __recover_from_first_instance_member_error),
    (lambda msg: "has no attribute" in msg, __recover_from_first_param_member_error),
    (lambda msg: "has no" in msg and "method" in msg, __recover_from_first_param_member_error),

    #Subtyping related errors
    (lambda msg: "is not a subtype of" in msg, __recover_from_subtype_error),
]


def recover_error_rules(type_rules, error_rules):
    """
    Run through the error rules recovering those that are suitable to be recovered. If a recovery is possible, the
    new rule is added to the correct section of the type rules and the error rule is deleted.
    :param type_rules: Correct type rules
    :param error_rule: Error type rules
    :return:
    """
    recovered_rules = []
    for error_rule in error_rules:
        for msg_pair in recoverable_msgs:
            if msg_pair[0](error_rule.exception_msg):
                msg_pair[1](type_rules, error_rule)
                recovered_rules.append(error_rule)
                break

    for rule in recovered_rules:
        error_rules.remove(rule)


