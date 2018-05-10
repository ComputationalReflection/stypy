import inspect
## AQUI: INCLUIR UNION_TYPE_COPY CUELGA POR RECURSION PROBLEM
from stypy_copy.errors_copy.type_error_copy import TypeError
from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy import RecursionType
import type_equivalence_copy
import stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.union_type_copy
from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType


"""
Several functions to identify what kind of type is a Python object
"""


# -------------
# Runtime type inspection
# -------------

# TODO: Remove?
# def is_type_store(obj_type):
#     return isinstance(obj_type, typestore.TypeStore)

#
# def is_object(obj_type):
#     if is_union_type(obj_type) or is_error_type(obj_type) or is_type_store(obj_type):
#         return False
#     return inspect.isclass(type(obj_type))


def is_class(class_type):
    """
    Determines if class_type is a Python class

    :param class_type: Type to test
    :return: bool
    """
    if is_union_type(class_type) or is_error_type(class_type):
        return False
    return inspect.isclass(class_type)


def is_union_type(the_type):
    """
    Determines if the_type is a UnionType
    :param the_type: Type to test
    :return: bool
    """
    return isinstance(the_type, stypy_copy.python_lib.python_types.type_inference.union_type.UnionType)


def is_undefined_type(the_type):
    """
    Determines if the_type is an UndefinedType
    :param the_type: Type to test
    :return: bool
    """
    return the_type == UndefinedType


def is_error_type(the_type):
    """
    Determines if the_type is an ErrorType
    :param the_type: Type to test
    :return: bool
    """
    return isinstance(the_type, TypeError)


def is_recursion_type(the_type):
    """
    Determines if the_type is a RecursionType
    :param the_type: Type to test
    :return: bool
    """
    return isinstance(the_type, RecursionType)


def is_property(the_type):
    """
    Determines if the_type is a Python property
    :param the_type: Type to test
    :return: bool
    """
    return isinstance(the_type, property)

# TODO: Remove?
# def __get_member_value(localization, member, type_of_obj, field_name):
#     # member = type_of_obj.__dict__[field_name]
#     if is_property(member):
#         return member.fget(localization, type_of_obj)
#     return member
#
#
# def get_type_of_member(localization, type_of_obj, field_name):
#     field_name = get_member_name(field_name)
#     return_type = __get_type_of_member(localization, type_of_obj, field_name)
#     if return_type is None:
#         return TypeError(localization, "The object does not provide a field named '%s'" % field_name)
#     else:
#         return __get_member_value(localization, return_type, type_of_obj, field_name)
#
#
# def __get_type_of_member_class_hierarchy(localization, type_of_obj, field_name):
#     if field_name in type_of_obj.__dict__:
#         # return __get_member_value (line, column, type_of_obj, field_name)
#         return type_of_obj.__dict__[field_name]
#     else:
#         for class_ in type_of_obj.__bases__:
#             return __get_type_of_member_class_hierarchy(localization, class_, field_name)
#         return None
#
#
# def __get_type_of_member(localization, type_of_obj, field_name):
#     if is_error_type(type_of_obj):
#         return type_of_obj
#
#     if inspect.ismodule(type_of_obj):
#         if field_name == "__class__":
#             return type_of_obj
#         if field_name in type_of_obj.__dict__:
#             return type_of_obj.__dict__[field_name]
#
#     if is_class(type_of_obj):
#         return __get_type_of_member_class_hierarchy(localization, type_of_obj, field_name)
#
#     if is_object(type_of_obj):
#         if field_name in type_of_obj.__dict__:
#             return type_of_obj.__dict__[field_name]
#         if field_name == "__class__":
#             return type_of_obj.__class__
#
#         return __get_type_of_member_class_hierarchy(localization, type_of_obj.__class__, field_name)
#
#     if is_union_type(type_of_obj):
#         inferred_types = []
#         error_types = []
#         for t in type_of_obj.types:
#             inferred_type = __get_type_of_member(localization, t, field_name)
#             #print "inferred_type = ", inferred_type
#             if not inferred_type is None:
#                 inferred_types.append(inferred_type)
#             else:
#                 error_types.append(t)
#         if len(inferred_types) == 0:
#             return None  # compiler error (no object provides the field)
#         if len(error_types) > 0:
#             Warning(localization, "The object may not provide a field named '%s'" % field_name)
#         inferred_type = None
#         for t in inferred_types:
#             inferred_type = stypy.python_lib.python_types.type_inference.union_type.UnionType.add(inferred_type, t)
#         return inferred_type
#
#     if is_type_store(type_of_obj):  # For modules
#         if field_name == "__class__":
#             return type_of_obj
#         return type_of_obj.get_type_of(localization, field_name)
#
#     return None  # compiler error
#
#
# def set_type_of_member(localization, type_of_obj, field_name, type_of_field):
#     field_name = get_member_name(field_name)
#
#     if is_class(type_of_obj):
#         member = __get_type_of_member(localization, type_of_obj, field_name)
#         if is_property(member):
#             member.fset(localization, type_of_obj, type_of_field)
#             return
#
#         type_of_obj.__dict__[field_name] = type_of_field
#         return
#
#     if is_object(type_of_obj):
#         member = __get_type_of_member(localization, type_of_obj, field_name)
#
#         if is_property(member):
#             member.fset(localization, type_of_obj, type_of_field)
#             return
#
#         type_of_obj.__dict__[field_name] = type_of_field
#         return
#
#     if is_type_store(type_of_obj):  # For modules
#         type_of_obj.set_type_of(localization, field_name, type_of_field)
#
#
# def invoke_member(localization, member, *args, **kwargs):
#     owner = None
#     if len(args) > 0:
#         owner = args[0]
#
#     if isinstance(owner, typestore.TypeStore) or inspect.ismodule(owner):
#         return member(localization, *args[1:], **kwargs)
#     else:
#         if inspect.isfunction(member) or inspect.isclass(member):
#             return member(localization, *args, **kwargs)
#         if type(owner) is types.InstanceType or type(owner) is types.ClassType:
#             return member(localization, owner, *args, **kwargs)


# # --------------------
# # Subtyping
# # --------------------
#
# def is_subtype(type1, type2):
#     if type_equivalence.equivalent_types(type1, type2):
#         return True
#     if type_equivalence.equivalent_types(type1, int) and type_equivalence.equivalent_types(type2, float):
#         return True
#     if is_union_type(type1):
#         for each_type in type1.types:
#             if not is_subtype(each_type, type2):
#                 return False
#         return True
#     if is_union_type(type2):
#         for each_type in type2.types:
#             if not is_subtype(type1, each_type):
#                 return False
#         return True
