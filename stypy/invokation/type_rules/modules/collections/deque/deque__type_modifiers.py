#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import types

from stypy.errors.type_error import StypyTypeError
from stypy.type_inference_programs.stypy_interface import get_builtin_python_type_instance, wrap_contained_type
from stypy.types import undefined_type
from stypy.types import union_type
from stypy.types.standard_wrapper import StandardWrapper
from stypy.types.type_containers import get_contained_elements_type, \
    set_contained_elements_type, can_store_elements
from stypy.types.type_inspection import is_method, is_function, is_class, get_self, is_str, compare_type


class TypeModifiers:
    # ###################### LIST TYPE MODIFIERS #######################

    @staticmethod  # Constructor  (__init__)
    def deque(localization, proxy_obj, arguments):
        ret_type = wrap_contained_type(collections.deque())
        if len(arguments) > 0:
            params = arguments[0]
            if is_str(type(params)):
                set_contained_elements_type(ret_type,
                                            get_builtin_python_type_instance(localization,
                                                                             type(params).__name__))
            else:
                existing_type = get_contained_elements_type(params)
                if existing_type is not None:
                    set_contained_elements_type(ret_type, existing_type)

        return ret_type

    # @staticmethod
    # def __iadd__(localization, proxy_obj, arguments):
    #     ret_type = get_builtin_python_type_instance(localization, 'list')
    #
    #     existing_type = get_contained_elements_type(get_self(proxy_obj))
    #     params = arguments[0]
    #     if can_store_elements(params):
    #         if existing_type is not None:
    #             new_type = existing_type
    #         else:
    #             new_type = None
    #         other_type = get_contained_elements_type(params)
    #         if not isinstance(other_type, union_type.UnionType):
    #             other_type = [other_type]
    #         else:
    #             other_type = other_type.types
    #
    #         for par in other_type:
    #             new_type = union_type.UnionType.add(new_type, par)
    #     else:
    #         new_type = union_type.UnionType.add(existing_type, arguments[0])
    #
    #     set_contained_elements_type(ret_type, new_type)
    #
    #     return ret_type
    #
    # @staticmethod
    # def __add__(localization, proxy_obj, arguments):
    #     ret_type = get_builtin_python_type_instance(localization, 'list')
    #
    #     existing_type = get_contained_elements_type(get_self(proxy_obj))
    #     params = arguments[0]
    #     if can_store_elements(params):
    #         if existing_type is not None:
    #             if not isinstance(existing_type, union_type.UnionType):
    #                 new_type = existing_type
    #             else:
    #                 new_type = existing_type.duplicate()
    #         else:
    #             new_type = None
    #         other_type = get_contained_elements_type(params)
    #         if not isinstance(other_type, union_type.UnionType):
    #             other_type = [other_type]
    #         else:
    #             other_type = other_type.types
    #
    #         for par in other_type:
    #             new_type = union_type.UnionType.add(new_type, par)
    #     else:
    #         new_type = union_type.UnionType.add(existing_type, arguments[0])
    #
    #     set_contained_elements_type(ret_type, new_type)
    #
    #     return ret_type
    #
    # @staticmethod
    # def __iter__(localization, proxy_obj, arguments):
    #     self_object = get_self(proxy_obj)
    #     listiterator = iter(self_object)
    #     wrap = StandardWrapper(listiterator)
    #     set_contained_elements_type(wrap,
    #                                 get_contained_elements_type(get_self(proxy_obj)))
    #
    #     return wrap
    #
    # ####### TYPE MODIFIERS #######
    #
    # @staticmethod
    # def __getitem__(localization, callable_, arguments):
    #     self_obj = callable_.__self__
    #     get_item = getattr(self_obj, "__getitem__")
    #
    #     def invoke_getitem():
    #         try:
    #             return get_item(0)
    #         except IndexError as ie:
    #             return undefined_type.UndefinedType
    #
    #     if compare_type(arguments[0], slice):
    #         ret_type = get_builtin_python_type_instance(localization, 'list')
    #
    #         set_contained_elements_type(ret_type, invoke_getitem())
    #
    #         return ret_type
    #
    #     return invoke_getitem()
    #
    # @staticmethod
    # def __getslice__(localization, proxy_obj, arguments):
    #     ret_type = get_builtin_python_type_instance(localization, 'list')
    #
    #     set_contained_elements_type(ret_type,
    #                                 get_contained_elements_type(get_self(proxy_obj)))
    #
    #     return ret_type
    #
    # @staticmethod
    # def extend(localization, proxy_obj, arguments):
    #     existing_type = get_contained_elements_type(get_self(proxy_obj))
    #     params = arguments[0]
    #     if can_store_elements(params):
    #         new_type = existing_type
    #         other_type = get_contained_elements_type(params)
    #         if not isinstance(other_type, collections.Iterable):
    #             other_type = [other_type]
    #         for par in other_type:
    #             new_type = union_type.UnionType.add(new_type, par)
    #     else:
    #         return StypyTypeError.wrong_parameter_type_error(localization, "iterable", type(params).__name__)
    #
    #     set_contained_elements_type(get_self(proxy_obj), new_type)
    #
    #     return types.NoneType
    #
    # @staticmethod
    # def insert(localization, proxy_obj, arguments):
    #     existing_type = get_contained_elements_type(get_self(proxy_obj))
    #     params = arguments[0]
    #     if can_store_elements(params):
    #         new_type = existing_type
    #         other_type = params.get_elements_type()
    #         if not isinstance(other_type, collections.Iterable):
    #             other_type = [other_type]
    #         for par in other_type:
    #             new_type = union_type.UnionType.add(new_type, par)
    #     else:
    #         new_type = union_type.UnionType.add(existing_type, arguments[1])
    #
    #     set_contained_elements_type(get_self(proxy_obj), new_type)
    #
    #     return types.NoneType

    @staticmethod
    def append(localization, callable_, arguments):
        self_instance = StandardWrapper.get_wrapper_of(callable_.__self__)
        existing_type = get_contained_elements_type(self_instance)
        if existing_type is undefined_type.UndefinedType:
            new_type = arguments[0]
        else:
            new_type = union_type.UnionType.add(existing_type, arguments[0])
        set_contained_elements_type(self_instance, new_type)
        return types.NoneType

    @staticmethod
    def pop(localization, callable_, arguments):
        self_instance = StandardWrapper.get_wrapper_of(callable_.__self__)
        return get_contained_elements_type(self_instance)

    @staticmethod
    def popleft(localization, callable_, arguments):
        self_instance = StandardWrapper.get_wrapper_of(callable_.__self__)
        return get_contained_elements_type(self_instance)
    #
    # @staticmethod
    # def __setitem__(localization, callable_, arguments):
    #     return TypeModifiers.append(localization, callable_, [arguments[-1]])
    #
    # @staticmethod
    # def __setslice__(localization, proxy_obj, arguments):
    #     existing_type = get_contained_elements_type(get_self(proxy_obj))
    #     params = arguments[2]
    #     if can_store_elements(params):
    #         new_type = existing_type
    #         other_type = get_contained_elements_type(params)
    #         if not isinstance(other_type, collections.Iterable):
    #             other_type = [other_type]
    #         for par in other_type:
    #             new_type = union_type.UnionType.add(new_type, par)
    #     else:
    #         new_type = union_type.UnionType.add(existing_type, arguments[2])
    #
    #     set_contained_elements_type(get_self(proxy_obj), new_type)
    #
    #     return types.NoneType
    #
    # @staticmethod
    # def __mul__(localization, proxy_obj, arguments):
    #     ret_type = get_builtin_python_type_instance(localization, 'list')
    #
    #     existing_type = get_contained_elements_type(get_self(proxy_obj))
    #     new_type = existing_type.clone()
    #     set_contained_elements_type(ret_type, new_type)
    #
    #     return ret_type
    #
    # @staticmethod
    # def sort(localization, proxy_obj, arguments):
    #     if len(arguments) == 0:
    #         return types.NoneType
    #
    #     if type(arguments[0]) is dict:
    #         return types.NoneType
    #
    #     for arg in arguments:
    #         if not (is_method(arg) or is_function(arg) or is_class(arg)):
    #             return StypyTypeError.wrong_parameter_type_error(localization,
    #                                                              "callable function or method (or entities that define a __call__ method)",
    #                                                              type(arg).__name__)
    #
    #     return types.NoneType
    #
    # @staticmethod
    # def pop(localization, proxy_obj, arguments):
    #     return get_contained_elements_type(get_self(proxy_obj))
