#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.types import known_python_types
from stypy.types import type_containers
from stypy.types.known_python_types import types_without_value
from stypy.types.standard_wrapper import wrap_contained_type, StandardWrapper
from stypy.types.union_type import UnionType


def turn_to_type(obj):
    """
    As in our type analysis process we may have to deal with code whose sources are not available, calls to this code
    must be performed so we can obtain return values that can be used in our program analysis. This code is responsible
     of turning the values obtained from these calls to its types so stypy is able to use its return types to analyze
     the program.
    :param obj:
    :return:
    """
    # Already wrapped: it is already a type
    if type(obj) is StandardWrapper:
        return obj
    if type(obj) in types_without_value:
        obj = type(obj)()

    wrapped_obj = wrap_contained_type(obj)

    # Special handling for dicts and similar
    if type_containers.can_store_keypairs(wrapped_obj):
        collected_items = dict()
        keys = obj.keys()
        for key in keys:
            values_for_key = type_containers.get_contained_elements_type_for_key(obj, key)
            key_type = turn_to_type(key)
            value_types = turn_to_type(values_for_key)
            try:
                existing_values = collected_items[key_type]
            except:
                existing_values = None

            collected_items[key_type] = UnionType.add(value_types, existing_values)

        for key in keys:
            del obj[key]

        for key, value in collected_items.items():
            type_containers.set_contained_elements_type_for_key(wrapped_obj, key, value)

        return wrapped_obj

    # Special handling for containers
    if type_containers.can_store_elements(wrapped_obj):
        union_contained = None
        for elem in obj:
            elem = turn_to_type(elem)
            if elem is type:
                union_contained = UnionType.add(union_contained, elem)
            else:
                try:
                    union_contained = UnionType.add(union_contained, known_python_types.get_sample_instance_for_type(
                        type(elem).__name__))
                except Exception as exc:
                    union_contained = UnionType.add(union_contained, elem)

        wrapped_obj.set_contained_type(union_contained)
        return wrapped_obj

    return obj

# if __name__ == '__main__':
#     import numpy
#
#     t = turn_to_type(numpy.core.numerictypes.typeinfo)
#     pass
#     #     # t =  turn_to_type(l)
#     #     # print t
#     #
#     #     t =  turn_to_type(d)
#     #     print t
