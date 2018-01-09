from stypy.python_lib.python_types.instantiation.known_python_types_management import get_known_types_and_values
from stypy.python_lib.type_rules.raw_type_rule_generation.python_modules_type_check.type_rules.types_of_parameters.type_rule import TypeRule
from types_of_parameters_static_definitions import get_predefined_type_rules
from types_of_parameters_static_additional_definitions import get_additional_type_rules


def __invoke_0(type_, member_name):
    """
    Invokes a 0-parameter function or method, returning all possible valid and invalid calls using the
     known Python types defined in the known_python_types_handling.py file. If the call is valid, its return type
     is also calculated. Type information is stored in TypeRule instances.
    :param type_: Type that holds the function or method
    :param member_name: Name of the function or method
    :return: List of valid calls type rules, list of invalid calls type rules
    """
    code_to_invoke = getattr(type_, member_name)
    valid_calls = []
    invalid_calls = []
    try:
        valid_calls.append(
            TypeRule(type_, member_name, (), code_to_invoke()))
    except Exception as exc:
        invalid_calls.append(
            TypeRule(type_, member_name, (), exc, True)
        )
    return valid_calls, invalid_calls


def __invoke_1(type_, member_name):
    """
    Invokes a 1-parameter function or method, returning all possible valid and invalid calls using the
     known Python types defined in the known_python_types_handling.py file. If the call is valid, its return type
     is also calculated. Type information is stored in TypeRule instances.
    :param type_: Type that holds the function or method
    :param member_name: Name of the function or method
    :return: List of valid calls type rules, list of invalid calls type rules
    """
    known_types = get_known_types_and_values()
    code_to_invoke = getattr(type_, member_name)
    valid_calls = []
    invalid_calls = []

    for param1, param1_value in known_types:
        try:
            valid_calls.append(
                TypeRule(type_, member_name, (param1, ), code_to_invoke(param1_value))
            )
        except Exception as exc:
            invalid_calls.append(
                TypeRule(type_, member_name, (param1, ), exc, True)
            )
        except AttributeError as exc:
            invalid_calls.append(
                TypeRule(type_, member_name, (param1, ), exc, True)
            )

    return valid_calls, invalid_calls


def __invoke_2(type_, member_name):
    """
    Invokes a 2-parameter function or method, returning all possible valid and invalid calls using the
     known Python types defined in the known_python_types_handling.py file. If the call is valid, its return type
     is also calculated. Type information is stored in TypeRule instances.
    :param type_: Type that holds the function or method
    :param member_name: Name of the function or method
    :return: List of valid calls type rules, list of invalid calls type rules
    """
    known_types = get_known_types_and_values()
    code_to_invoke = getattr(type_, member_name)
    valid_calls = []
    invalid_calls = []

    for param1, param1_value in known_types:
        for param2, param2_value in known_types:
            try:
                valid_calls.append(
                    TypeRule(type_, member_name, (param1, param2), code_to_invoke(param1_value, param2_value))
                )
            except Exception as exc:
                invalid_calls.append(
                    TypeRule(type_, member_name, (param1, param2), exc, True)
                )

    return valid_calls, invalid_calls


def __invoke_3(type_, member_name):
    """
    Invokes a 3-parameter function or method, returning all possible valid and invalid calls using the
     known Python types defined in the known_python_types_handling.py file. If the call is valid, its return type
     is also calculated. Type information is stored in TypeRule instances.
    :param type_: Type that holds the function or method
    :param member_name: Name of the function or method
    :return: List of valid calls type rules, list of invalid calls type rules
    """
    known_types = get_known_types_and_values()
    code_to_invoke = getattr(type_, member_name)
    valid_calls = []
    invalid_calls = []

    for param1, param1_value in known_types:
        for param2, param2_value in known_types:
            for param3, param3_value in known_types:
                try:
                    valid_calls.append(
                        TypeRule(type_, member_name, (param1, param2, param3),
                                 code_to_invoke(param1_value, param2_value, param3_value))
                    )
                except Exception as exc:
                    invalid_calls.append(
                        TypeRule(type_, member_name, (param1, param2, param3), exc, True)
                    )

    return valid_calls, invalid_calls


def __invoke_4(type_, member_name):
    """
    Invokes a 4-parameter function or method, returning all possible valid and invalid calls using the
     known Python types defined in the known_python_types_handling.py file. If the call is valid, its return type
     is also calculated. Type information is stored in TypeRule instances.
    :param type_: Type that holds the function or method
    :param member_name: Name of the function or method
    :return: List of valid calls type rules, list of invalid calls type rules
    """
    known_types = get_known_types_and_values()
    code_to_invoke = getattr(type_, member_name)
    valid_calls = []
    invalid_calls = []

    for param1, param1_value in known_types:
        for param2, param2_value in known_types:
            for param3, param3_value in known_types:
                for param4, param4_value in known_types:
                    try:
                        valid_calls.append(
                            TypeRule(type_, member_name, (param1, param2, param3, param4),
                                     code_to_invoke(param1_value, param2_value, param3_value, param4_value))
                        )
                    except Exception as exc:
                        invalid_calls.append(
                            TypeRule(type_, member_name, (param1, param2, param3, param4), exc, True)
                        )

    return valid_calls, invalid_calls


def __invoke_5(type_, member_name):
    """
    Invokes a 4-parameter function or method, returning all possible valid and invalid calls using the
     known Python types defined in the known_python_types_handling.py file. If the call is valid, its return type
     is also calculated. Type information is stored in TypeRule instances.
    :param type_: Type that holds the function or method
    :param member_name: Name of the function or method
    :return: List of valid calls type rules, list of invalid calls type rules
    """
    known_types = get_known_types_and_values()
    code_to_invoke = getattr(type_, member_name)
    valid_calls = []
    invalid_calls = []

    for param1, param1_value in known_types:
        for param2, param2_value in known_types:
            for param3, param3_value in known_types:
                for param4, param4_value in known_types:
                    for param5, param5_value in known_types:
                        try:
                            valid_calls.append(
                                TypeRule(type_, member_name, (param1, param2, param3, param4, param5),
                                         code_to_invoke(param1_value, param2_value, param3_value, param4_value,
                                                        param5_value))
                            )
                        except Exception as exc:
                            invalid_calls.append(
                                TypeRule(type_, member_name, (param1, param2, param3, param4, param5), exc, True)
                            )

    return valid_calls, invalid_calls


"""
Assign a number of parameters with its type polling function.
"""
__callable_invokers = {
    0: __invoke_0,
    1: __invoke_1,
    2: __invoke_2,
    3: __invoke_3,
    4: __invoke_4,
    5: __invoke_5,
}


def get_attribute_member_type_rules(type_, member_name):
    """
    Creates a TypeRule instance for the attribute member <member_name> of type <type_>
    :param type_: Type
    :param member_name: attribute name
    :return: A TypeRule instance
    """
    member = getattr(type_, member_name)
    return TypeRule(type_, member_name, None, member)


def get_callable_member_type_rules(type_, member_name, arities, maximum_arity=3):
    """
    Creates a TypeRule instance for the callable member <member_name> of type <type_>
    :param type_: Type
    :param member_name: attribute name
    :param maximum_arity Limits the invocation to certain type polling functions to reduce consumed time.
    :return: A TypeRule instance
    """
    type_rules = []
    type_errors = []

    # Test if rules for this member are already predefined statically. If not, execute inference
    predefined_rules = get_predefined_type_rules(type_, member_name)
    if not predefined_rules is None:
        type_rules += predefined_rules
    else:
        for num in arities:
            try:
                if num > maximum_arity:
                    continue

                valid, invalid = __callable_invokers[num](type_, member_name)

                type_rules += valid
                type_errors += invalid
            except KeyError:
                raise Exception("No invoker for {0} parameters".format(num))

    #Add additional rules to those that have been predefined or inferred
    additional_rules = get_additional_type_rules(type_, member_name)
    if not additional_rules is None:
        for rule in additional_rules:
            if not rule in type_rules:
                type_rules.append(rule)

    return type_rules, type_errors

