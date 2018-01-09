#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect
import sys

import type_containers
from stypy import contexts
from stypy import invokation
from stypy import type_inference_programs
from stypy import types
from stypy.errors.type_error import StypyTypeError
from stypy.errors.type_warning import TypeWarning
from stypy.reporting.localization import Localization
from stypy.types.type_wrapper import TypeWrapper
from type_containers import get_contained_elements_type
from type_inspection import is_error, is_old_style_class

"""
This file contains functions that provide intercession-related capabilities
"""


def get_superclasses(clazz):
    """
    Get the superclasses of a certain class
    :param clazz:
    :return:
    """
    if hasattr(clazz, "__mro__"):
        return clazz.__mro__

    if hasattr(clazz, "__bases__"):
        return clazz.__bases__

    return ()


def __get_module_file(obj):
    """
    Gets the file of the module that declare the passed object
    :param obj:
    :return:
    """
    try:
        return sys.modules[obj.__module__].__file__
    except:
        return None


def __private_variable_name(name):
    """
    Determines if a certain variable name can be considered private
    :param name:
    :return:
    """
    if not name.startswith("__"):
        return False
    if name.startswith("__") and name.endswith("__"):
        return False

    return True


def has_attr(obj, name):
    """
    Determines if the provided obj has the attribute name
    :param obj:
    :param name:
    :return:
    """
    if not __private_variable_name(name):
        return hasattr(obj, name)

    import types as python_types
    mangled_name = None
    if hasattr(obj, '__class__') and not inspect.ismodule(obj):
        mangled_name = "_" + obj.__class__.__name__ + name

    if type(obj) is python_types.ClassType and hasattr(obj, '__name__'):
        mangled_name = "_" + obj.__name__ + name

    if mangled_name is not None:
        if isinstance(obj, types.union_type.UnionType):
            return not isinstance(obj.get_type_of_member(name), StypyTypeError)
        return hasattr(obj, mangled_name)
    return hasattr(obj, name)


def get_attr(obj, name):
    """
    Get the attribute name from the provided obj
    :param obj:
    :param name:
    :return:
    """
    if not __private_variable_name(name):
        return getattr(obj, name)

    if hasattr(obj, name):
        return getattr(obj, name)
    else:
        import types as python_types
        mangled_name = None
        if hasattr(obj, '__class__') and not inspect.ismodule(obj):
            mangled_name = "_" + obj.__class__.__name__ + name

        if type(obj) is python_types.ClassType and hasattr(obj, '__name__'):
            mangled_name = "_" + obj.__name__ + name

        if mangled_name is not None and hasattr(obj, mangled_name):
            return getattr(obj, mangled_name)

    return getattr(obj, name)


def set_attr(obj, name, value):
    """
    Sets the attribute name from the provided obj to value
    :param obj:
    :param name:
    :param value:
    :return:
    """
    if not __private_variable_name(name):
        # Type changing an object with a UnionType as value requires special handling
        if name == '__class__' and isinstance(value, types.union_type.UnionType) and hasattr(obj, '__class__'):
            new_base = types.union_type.create_union_type_class(value, is_old_style_class(obj.__class__))
            return setattr(obj, name, new_base)

        # Changing the inheritance tree of a class with a UnionType as value requires special handling
        if name == '__bases__':
            if isinstance(value, types.union_type.UnionType):
                bases_union_type = None
                for type_ in value.types:
                    bases_union_type = types.union_type.UnionType.add(bases_union_type,
                                                                      get_contained_elements_type(type_))

                new_base = types.union_type.create_union_type_class(bases_union_type, True)
                return setattr(obj, name, (new_base,))
            if isinstance(value, TypeWrapper):
                wrapped = value.get_wrapped_type()
                if isinstance(wrapped, tuple):
                    contained = get_contained_elements_type(value)
                    if not isinstance(contained, list):
                        contained = [contained]

                    return setattr(obj, name, tuple(contained))

        if inspect.ismethod(obj):
            # Method properties have to be written over its associated im_func
            return setattr(obj.im_func, name, value)
        else:
            return setattr(obj, name, value)

    if hasattr(obj, name):
        if inspect.ismethod(obj):
            # Method properties have to be written over its associated im_func
            return setattr(obj.im_func, name, value)
        else:
            return setattr(obj, name, value)
    else:
        import types as python_types
        mangled_name = None
        if hasattr(obj, '__class__') and not inspect.ismodule(obj):
            mangled_name = "_" + obj.__class__.__name__ + name

        if type(obj) is python_types.ClassType and hasattr(obj, '__name__'):
            mangled_name = "_" + obj.__name__ + name

        if mangled_name is not None and hasattr(obj, mangled_name):
            return setattr(obj, mangled_name, value)

    return setattr(obj, name, value)


def has_member(localization, obj, member):
    """
    Checks if the provided obj has the passed member
    :param localization:
    :param obj:
    :param member:
    :return:
    """
    module_file = __get_module_file(obj)
    if module_file is None:
        return hasattr(obj, member)
    else:
        if contexts.context.Context.exist_context_for_module(module_file):
            defining_context = contexts.context.Context.get_current_active_context_for_module(module_file)
            return defining_context.has_member(localization, obj, member)
        else:
            return hasattr(obj, member)


def del_attr(obj, name):
    """
    Del the attribute name from the passed object
    :param obj:
    :param name:
    :return:
    """
    if not __private_variable_name(name):
        return delattr(obj, name)

    if hasattr(obj, name):
        return delattr(obj, name)
    else:
        import types as python_types
        mangled_name = None
        if hasattr(obj, '__class__') and not inspect.ismodule(obj):
            mangled_name = "_" + obj.__class__.__name__ + name

        if type(obj) is python_types.ClassType and hasattr(obj, '__name__'):
            mangled_name = "_" + obj.__name__ + name

        if mangled_name is not None and hasattr(obj, mangled_name):
            return delattr(obj, mangled_name)

    return delattr(obj, name)


def get_member_from_object(localization, obj, name):
    """
    Get the member name from the object obj
    :param localization:
    :param obj:
    :param name:
    :return:
    """
    if is_error(obj):
        return obj

    Localization.set_current(localization)
    if isinstance(obj, contexts.context.Context):
        return obj.get_type_of(localization, name)

    if isinstance(obj, TypeWrapper) and obj.is_declared_member(name):
        try:
            return obj.get_type_of_member(name)
        except:
            return StypyTypeError.member_not_defined_error(localization, obj, name)

    if has_attr(obj, name):
        ret_type = type_inference_programs.stypy_interface.wrap_type(get_attr(obj, name))
        if name == '__bases__':
            base_type = None
            bases = obj.__bases__
            if not isinstance(bases, tuple):
                bases = (bases,)
            for type_ in bases:
                base_type = types.union_type.UnionType.add(base_type, type_)
            type_containers.set_contained_elements_type(ret_type, base_type)

        return ret_type

    if isinstance(obj, invokation.type_rules.type_groups.type_groups.DynamicType) or \
            (type(obj) is invokation.type_rules.type_groups.type_groups.DynamicType):
        return invokation.type_rules.type_groups.type_groups.DynamicType()

    # Variable not found
    return StypyTypeError.member_not_defined_error(localization, obj, name)


def get_member(localization, obj, member):
    """
    Version of the previous method that take into consideration current contexts
    :param localization:
    :param obj:
    :param member:
    :return:
    """
    module_file = __get_module_file(obj)
    if module_file is None:
        return get_member_from_object(localization, obj, member)
    else:
        if contexts.context.Context.exist_context_for_module(module_file):
            defining_context = contexts.context.Context.get_current_active_context_for_module(module_file)
            return defining_context.get_type_of_member(localization, obj, member)
        else:
            return get_member_from_object(localization, obj, member)


def set_member_to_object(localization, obj, name, value):
    """
    Sets the member name from the object obj to value
    :param localization:
    :param obj:
    :param name:
    :param value:
    :return:
    """
    if is_error(obj):
        return obj
    try:
        Localization.set_current(localization)
        if isinstance(obj, contexts.context.Context):
            return obj.set_type_of(localization, name, value)

        if isinstance(obj, TypeWrapper) and obj.is_declared_member(name):
            return obj.set_type_of_member(name, value)

        set_attr(obj, name, value)
    except Exception as exc:
        return StypyTypeError.member_cannot_be_set_error(localization, obj, name, value, str(exc))


def set_member(localization, obj, member, value):
    """
    Version of the previous method that take into consideration current contexts
    :param localization:
    :param obj:
    :param member:
    :param value:
    :return:
    """
    module_file = __get_module_file(obj)
    if module_file is None:
        return set_member_to_object(localization, obj, member, value)
    else:
        if contexts.context.Context.exist_context_for_module(module_file):
            defining_context = contexts.context.Context.get_current_active_context_for_module(module_file)
            return defining_context.set_type_of_member(localization, obj, member, value)
        else:
            return set_member_to_object(localization, obj, member, value)


def del_member_from_object(localization, obj, name):
    """
    Removes the member name from the object obj
    :param localization:
    :param obj:
    :param name:
    :return:
    """
    if is_error(obj):
        return obj
    try:
        Localization.set_current(localization)
        if isinstance(obj, TypeWrapper):
            if has_attr(obj, '__delitem__'):
                return type_inference_programs.stypy_interface.invoke(localization, obj.__delitem__, [name])
            if obj.is_declared_member(name):
                return obj.del_member(name)

        member_type = get_member(localization, obj, name)

        if isinstance(member_type, types.union_type.UnionType):
            if types.undefined_type.UndefinedType in member_type:
                TypeWarning(localization,
                            "Attempt to delete the potentially unexisting member '{0}' from an object".format(name))
        if isinstance(member_type, StypyTypeError):
            return member_type

        if isinstance(obj, contexts.context.Context):
            obj.del_type_of(localization, name)
            return

        del_attr(obj, name)
    except Exception as exc:
        if isinstance(obj, invokation.type_rules.type_groups.type_groups.DynamicType):
            return invokation.type_rules.type_groups.type_groups.DynamicType()
        return StypyTypeError.member_cannot_be_deleted_error(localization, obj, name, str(exc))


def del_member(localization, obj, member):
    """
    Version of the previous method that take into consideration current contexts
    :param localization:
    :param obj:
    :param member:
    :return:
    """
    module_file = __get_module_file(obj)
    if module_file is None:
        return del_member_from_object(localization, obj, member)
    else:
        if contexts.context.Context.exist_context_for_module(module_file):
            defining_context = contexts.context.Context.get_current_active_context_for_module(module_file)
            return defining_context.del_member(localization, obj, member)
        else:
            return del_member_from_object(localization, obj, member, )


def change_type(localization, obj, new_type):
    """
    Change the type of obj to new_type
    :param localization:
    :param obj:
    :param new_type:
    :return:
    """
    try:
        return set_member(localization, obj, '__class__', new_type)
    except Exception as exc:
        return StypyTypeError(localization,
                              "Cannot change the type of '{0}' to '{1}: {2}".format(str(obj), str(new_type), str(exc)))


def change_base_types(localization, obj, new_type):
    """
    Chenge the base type of obj to new_type
    :param localization:
    :param obj:
    :param new_type:
    :return:
    """
    try:
        return set_member(localization, obj, '__bases__', new_type)
    except Exception as exc:
        return StypyTypeError(localization,
                              "Cannot change the type of '{0}' to '{1}: {2}".format(str(obj), str(new_type), str(exc)))


def supports_intercession(obj):
    """
    Determines if an object supports structural reflection. An object supports it if it has a __dict__ property and its
    type is dict (instead of the read-only dictproxy)

    :param obj: Any Python object
    :return: bool
    """
    if not hasattr(obj, '__dict__'):
        return False

    if type(obj.__dict__) is dict:
        return True
    else:
        try:
            setattr(obj, "__stypy_probe", None)
            delattr(obj, "__stypy_probe")
            return True
        except:
            return False
