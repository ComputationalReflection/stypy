#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.types import union_type


# ############################# UNFOLDING OF POSSIBLE UNION TYPES  ######################################


def __has_union_types(type_list):
    """
    Determines if a list of types has union types inside it
    :param type_list: List of types
    :return: bool
    """
    return len(filter(lambda elem: isinstance(elem, union_type.UnionType), type_list)) > 0


def __is_union_type(obj):
    """
    Determines if an object is a union type
    :param obj: Any Python object
    :return: bool
    """
    return isinstance(obj, union_type.UnionType)


def clone_list(list_):
    """
    Shallow copy of a list.
    :param list_:
    :return:
    """
    result = []
    for elem in list_:
        result.append(elem)

    return result


def clone_dict(dict_):
    """
    Shallow copy of a dict
    :param dict_:
    :return:
    """
    result = {}
    for elem in dict_:
        result[elem] = dict_[elem]

    return result


def __unfold_union_types_from_args(argument_list, possible_argument_combinations_list):
    """
    Helper for the following function
    :param argument_list:
    :param possible_argument_combinations_list:
    :return:
    """
    if not __has_union_types(argument_list):
        if argument_list not in possible_argument_combinations_list:
            possible_argument_combinations_list.append(argument_list)
        return
    for cont in xrange(len(argument_list)):
        arg = argument_list[cont]
        # For each union type, make type checks using each of their contained types
        if __is_union_type(arg):
            for t in arg.types:
                clone = clone_list(argument_list)
                clone[cont] = t
                __unfold_union_types_from_args(clone, possible_argument_combinations_list)


def unfold_union_types_from_args(argument_list):
    """
    Turns [(int \/ long \/ str), str] into:
    [
        (int, str),
        (long, str),
        (str, str),
    ]
    Note that if multiple union types are present, all are used to create combinations. This function is recursive.
    :param argument_list:
    :return:
    """
    list_of_possible_args = []
    if __has_union_types(argument_list):
        __unfold_union_types_from_args(argument_list, list_of_possible_args)
        return list_of_possible_args
    else:
        return [argument_list]


def __unfold_union_types_from_kwargs(keyword_arguments_dict, possible_argument_combinations_list):
    """
    Helper for the following function
    :param keyword_arguments_dict:
    :param possible_argument_combinations_list:
    :return:
    """
    if not __has_union_types(keyword_arguments_dict.values()):
        if keyword_arguments_dict not in possible_argument_combinations_list:
            possible_argument_combinations_list.append(keyword_arguments_dict)
        return
    for elem in keyword_arguments_dict:
        arg = keyword_arguments_dict[elem]
        # For each union type, make type checks using each of their contained types
        if __is_union_type(arg):
            for t in arg.types:
                clone = clone_dict(keyword_arguments_dict)
                clone[elem] = t
                __unfold_union_types_from_kwargs(clone, possible_argument_combinations_list)


def unfold_union_types_from_kwargs(keyword_argument_dict):
    """
    Recursive function that does the same as its args-dealing equivalent, but with keyword arguments
    :param keyword_argument_dict:
    :return:
    """
    list_of_possible_kwargs = []
    if __has_union_types(keyword_argument_dict.values()):
        __unfold_union_types_from_kwargs(keyword_argument_dict, list_of_possible_kwargs)
        return list_of_possible_kwargs
    else:
        return [keyword_argument_dict]


def unfold_arguments(*args, **kwargs):
    """
    Turns parameter lists with union types into a a list of tuples. Each tuple contains a single type of every
     union type present in the original parameter list. Each tuple contains a different type of some of its union types
      from the other ones, so in the end all the possible combinations are generated and
     no union types are present in the result list. This is also done with keyword arguments. Note that if multiple
     union types with lots of contained types are present in the original parameter list, the result of this function
     may be very big. As later on every list returned by this function will be checked by a call handler, the
     performance of the type inference checking may suffer. However, we cannot check the types of Python library
     functions using other approaches, as union types cannot be properly expressed in type rules nor converted to a
     single Python value.
    :param args: Call arguments
    :param kwargs: Call keyword arguments
    :return:
    """

    # Decompose union types among arguments
    unfolded_arguments = unfold_union_types_from_args(args)
    # Decompose union types among keyword arguments
    unfolded_keyword_arguments = unfold_union_types_from_kwargs(kwargs)
    result_arg_kwarg_tuples = []

    # Only keyword arguments are passed? return and empty list with each dictionary
    if len(unfolded_arguments) == 0:
        if len(unfolded_keyword_arguments) > 0:
            for kwarg in unfolded_keyword_arguments:
                result_arg_kwarg_tuples.append(([], kwarg))
        else:
            # 0-argument call
            result_arg_kwarg_tuples.append(([], {}))
    else:
        # Combine each argument list returned with each keyword arguments dictionary returned, so we obtain all the
        # possible args, kwargs combinations.
        for arg in unfolded_arguments:
            if len(unfolded_keyword_arguments) > 0:
                for kwarg in unfolded_keyword_arguments:
                    result_arg_kwarg_tuples.append((arg, kwarg))
            else:
                result_arg_kwarg_tuples.append((arg, {}))

    return result_arg_kwarg_tuples
