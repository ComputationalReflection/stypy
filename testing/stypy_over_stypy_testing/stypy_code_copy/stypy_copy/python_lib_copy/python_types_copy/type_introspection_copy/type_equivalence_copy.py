from stypy_copy.python_lib_copy.python_types_copy.type_copy import Type

"""
This file implements an algorithm to compare types by its structure
"""

# TODO: Remove?
# --------------------
# Type Equivalence
# --------------------


# def equivalent_types(type1, type2):
#     """Type equivalence is much more complex; we start with similar identity"""
#     if runtime_type_inspection.is_union_type(type1):
#         return __are_equivalent_union_types(type1, type2)
#     if runtime_type_inspection.is_union_type(type2):
#         return __are_equivalent_union_types(type1, type2)
#
#     # #Two dictionaries are equal is their element and index type are the same and their type mappings
#     # if isinstance(type1, python_data_structures.PythonDictionary):
#     #
#     # if isinstance(type1, python_data_structures.PythonIndexableDataStructure):
#
#     return type1 == type2


# def __are_equivalent_union_types(type1, type2):
#     if not (runtime_type_inspection.is_union_type(type1)):
#         return False
#     if not (runtime_type_inspection.is_union_type(type2)):
#         return False
#     if len(type1.types) != len(type2.types):
#         return False
#     types2 = list(type2.types)
#     for t1 in type1.types:
#         t2_index = find_equivalent_type(t1, types2)
#         if t2_index == -1:
#             return False
#         del types2[t2_index]
#     return True


# def find_equivalent_type(type, type_list):
#     for i in range(len(type_list)):
#         if equivalent_types(type, type_list[i]):
#             return i
#     return -1

# non_comparable_members = ['__call__', '__delattr__', '__format__']

def structural_equivalence(type1, type2, exclude_special_properties=True):
    """
    Test if two types are structurally equal, optionally excluding special properties that should have been compared
    previously. This method is used by the TypeInferenceProxy __eq__ method.

    :param type1: Type to compare its structure
    :param type2: Type to compare its structure
    :param exclude_special_properties: Do not compare the value of certain special hardcoded properties, that have
    been processed previously in the TypeInferenceProxy __eq__ method.
    :return: bool
    """
    type1_members = dir(type1)
    type2_members = dir(type2)

    # We do not consider values if only one of the compared types has one
    value_in_type1 = 'value' in type1_members
    value_in_type2 = 'value' in type1_members

    if value_in_type1 and not value_in_type2:
        type1_members.remove('value')

    if value_in_type2 and not value_in_type1:
        type1_members.remove('value')

    same_structure = type1_members == type2_members
    if not same_structure:
        return False

    for member in type1_members:
        if exclude_special_properties:
            if member in Type.special_properties_for_equality:
                continue

        # try:
        member1 = getattr(type1, member)
        member2 = getattr(type2, member)

        # If both are wrapper types, we compare it
        if isinstance(member1, Type):  # and isinstance(member2, Type):
            if not member1.get_python_type() == member2.get_python_type():
                return False
        else:
            # Else we compare its types
            if not type(member1) == type(member2):
                return False
                # except:
                #     return False

                # for member in type1_members:
                #     if exclude_special_properties:
                #         if member in Type.special_properties_for_equality:
                #             continue
                #     member1 = getattr(type1, member)
                #     member2 = getattr(type2, member)
                #     try:
                #         # If both are wrapper types, we compare it
                #         if not member1.get_python_type() == member2.get_python_type():
                #             return False
                #     except:
                #         # Else we compare its types
                #         if not type(member1) == type(member2):
                #             return False

                # try:
                #     member1 = getattr(type1, member)
                #     member2 = getattr(type2, member)
                #
                #     #If both are wrapper types, we compare it
                #     if isinstance(member1, Type) and isinstance(member2, Type):
                #         if not member1.get_python_type() == member2.get_python_type():
                #             return False
                #     else:
                #         #Else we compare its types
                #         if not type(member1) == type(member2):
                #             return False
                # except:
                #     return False

    # for member in type2_members:
    #     # if member in non_comparable_members:
    #     #     continue
    #     try:
    #         member1 = getattr(type1, member)
    #         member2 = getattr(type2, member)
    #
    #         #If both are types, we compare it
    #         if isinstance(member1, Type) and isinstance(member2, Type):
    #             if not member1.get_python_type() == member2.get_python_type():
    #                 return False
    #         else:
    #             #Else we compare its types
    #             if not type(member1) == type(member2):
    #                 return False
    #         # if not getattr(type1, member) == getattr(type2, member):
    #         #     return False
    #     except:
    #         return False

    return True
