#!/usr/bin/env python
# -*- coding: utf-8 -*-

from stypy.errors.type_error import StypyTypeError
from stypy.errors.type_warning import TypeWarning
from stypy.reporting.localization import Localization
from stypy.type_inference_programs.stypy_interface import invoke
from stypy.types.standard_wrapper import StandardWrapper
from stypy.types.type_containers import get_contained_elements_type, get_contained_elements_type_for_key
from stypy.types.union_type import UnionType
from stypy.visitor.type_inference.visitor_utils import stypy_functions


def assertTrue(cond, msg=""):
    assert cond, msg


def compare_types(type1, type2):
    if isinstance(type1, UnionType):
        if isinstance(type2, UnionType):
            type2 = type2.types
        if len(type1.types) != len(type2):
            assert False, "Different len of types ({0} != {1})".format(str(type1.types), str(type2))

        for t in type1.types:
            if t not in type2:
                assert False, "Different types ({0} != {1})".format(str(type1.types), str(type2))
    else:
        if isinstance(type1, StandardWrapper):
            type1 = type(type1.wrapped_type)
        if isinstance(type2, StandardWrapper):
            type2 = type(type2.wrapped_type)

        assert type1 == type2, "Different types ({0} != {1})".format(str(type1), str(type2))


def assert_if_not_error(obj):
    assert isinstance(obj, StypyTypeError), "{0} should be an error".format(obj)


def assert_equal_type_name(obj1, type_name):
    assert obj1.__name__ == type_name, "Different type names ({0} != {1})".format(obj1.__name__, type_name)


def get_elements_type(obj):
    return get_contained_elements_type(obj)


def get_values_from_key(obj, key):
    return get_contained_elements_type_for_key(obj, key)


def add_key_and_value_type(container, key, type_):
    try:
        existing = container[key]
    except KeyError:
        existing = None

    to_add = UnionType.add(existing, type_)
    container[key] = to_add


# ##################################################    #################################################

def create_union_type(types_):
    ret = None

    for elem in types_:
        ret = UnionType.add(ret, elem)

    return ret


def generic_0parameter_test(type_store, var_name, method_name, correct_return_type):
    obj = type_store.get_type_of(Localization(__file__), var_name)
    method = type_store.get_type_of_member(Localization(__file__), obj, method_name)

    result = invoke(Localization(__file__), method)

    compare_types(result, correct_return_type)


def generic_1parameter_test(type_store, var_name, method_name, correct_types, correct_return_types, incorrect_types,
                            expected_num_of_warnings,
                            expected_num_of_warnings_in_correct_calls=0):
    obj = type_store.get_type_of(Localization(__file__), var_name)
    method = type_store.get_type_of_member(Localization(__file__), obj, method_name)

    # Correct calls
    for i in range(len(correct_types)):
        result = invoke(Localization(__file__), method, correct_types[i])
        type_store.set_type_of(Localization(__file__), "result_i", result)
        compare_types(type_store.get_type_of(Localization(__file__), "result_i"),
                      correct_return_types[i])

    # Incorrect calls
    for i in range(len(incorrect_types)):
        result = invoke(Localization(__file__), method, incorrect_types[i])
        type_store.set_type_of(Localization(__file__), "result_i", result)
        assert_if_not_error(type_store.get_type_of(Localization(__file__), "result_i"))

    # Incorrect calls (arity)
    for i in range(len(correct_types)):
        result = invoke(Localization(__file__), method, correct_types[i], None)
        type_store.set_type_of(Localization(__file__), "result_i", result)
        assert_if_not_error(type_store.get_type_of(Localization(__file__), "result_i"))

    # Union type call (all correct)
    union = create_union_type(correct_types)
    result = invoke(Localization(__file__), method, union)

    if result is StypyTypeError:
        compare_types(result, correct_return_types[0])
    else:
        expected_result = create_union_type(stypy_functions.flatten_lists(*correct_return_types))
        compare_types(result, expected_result)

    assertTrue(len(TypeWarning.get_warning_msgs()) == expected_num_of_warnings_in_correct_calls,
               "got {0} warings, expected {1}".format(len(TypeWarning.get_warning_msgs()),
                                                      expected_num_of_warnings_in_correct_calls))

    # Union type call (mix correct/incorrect)
    union = create_union_type(correct_types + incorrect_types)
    result = invoke(Localization(__file__), method, union)
    if result is StypyTypeError:
        compare_types(result, correct_return_types[0])
    else:
        expected_result = create_union_type(stypy_functions.flatten_lists(*correct_return_types))
        compare_types(result, expected_result)

    assertTrue(len(TypeWarning.get_warning_msgs()) == expected_num_of_warnings,
               "got {0} warings, expected {1}".format(len(TypeWarning.get_warning_msgs()),
                                                      expected_num_of_warnings))

    TypeWarning.reset_warning_msgs()

    # Union type call (all incorrect)
    if len(incorrect_types) > 0:
        union = create_union_type(incorrect_types)
        result = invoke(Localization(__file__), method, union)
        assert_if_not_error(result)

        assertTrue(len(TypeWarning.warnings) == 0)  # All bad, no warnings


def generic_2parameter_test(type_store, var_name, method_name, correct_types, correct_return_types,
                            incorrect_types, expected_num_of_warnings,
                            expected_num_of_warnings_in_correct_calls=0):
    obj = type_store.get_type_of(Localization(__file__), var_name)
    method = type_store.get_type_of_member(Localization(__file__), obj, method_name)

    # Correct call
    for i in range(len(correct_types)):
        result = invoke(Localization(__file__), method, correct_types[i][0],
                        correct_types[i][1])
        compare_types(result, correct_return_types[i])

    # Incorrect call
    for i in range(len(incorrect_types)):
        result = invoke(Localization(__file__), method, incorrect_types[i][0],
                        incorrect_types[i][1])
        assert_if_not_error(result)

    # Incorrect arity call
    result = invoke(Localization(__file__), method, correct_types[0][0],
                    correct_types[0][1], None)
    assert_if_not_error(result)

    # Union type call (all correct)
    union = create_union_type(stypy_functions.flatten_lists(*correct_types))
    result = invoke(Localization(__file__), method, union, union)

    if result is StypyTypeError:
        compare_types(result, correct_return_types[0])
    else:
        expected_result = create_union_type(correct_return_types)
        compare_types(result, expected_result)

    assertTrue(len(TypeWarning.get_warning_msgs()) == expected_num_of_warnings_in_correct_calls,
               "got {0} warnings, expected {1}".format(len(TypeWarning.get_warning_msgs()),
                                                       expected_num_of_warnings_in_correct_calls))

    # Union type call (mix correct/incorrect)
    union = create_union_type(stypy_functions.flatten_lists(*correct_types) +
                              stypy_functions.flatten_lists(*incorrect_types))

    result = invoke(Localization(__file__), method, union, union)
    if result is TypeError:
        compare_types(result, correct_return_types[0])
    else:
        expected_result = create_union_type(correct_return_types)
        compare_types(result, expected_result)

    assertTrue(len(TypeWarning.get_warning_msgs()) == expected_num_of_warnings,
               "got {0} warnings, expected {1}".format(len(TypeWarning.get_warning_msgs()),
                                                       expected_num_of_warnings))

    TypeWarning.reset_warning_msgs()

    # Union type call (all incorrect)
    if len(incorrect_types) > 0:
        union = create_union_type(stypy_functions.flatten_lists(*incorrect_types))

        result = invoke(Localization(__file__), method, union, union)
        assert_if_not_error(result)
        assertTrue(len(TypeWarning.get_warning_msgs()) == 0)  # All bad, no warnings
