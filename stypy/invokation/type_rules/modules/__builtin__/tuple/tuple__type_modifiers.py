#!/usr/bin/env python
# -*- coding: utf-8 -*-

from stypy.type_inference_programs.stypy_interface import get_builtin_python_type_instance
from stypy.types import union_type
from stypy.types.standard_wrapper import StandardWrapper, TypeWrapper
from stypy.types.type_containers import get_contained_elements_type, \
    set_contained_elements_type, can_store_elements
from stypy.types.type_inspection import get_self, is_str, compare_type


class TypeModifiers:
    # ###################### TUPLE TYPE MODIFIERS #######################

    @staticmethod  # Constructor  (__init__)
    def tuple(localization, proxy_obj, arguments):
        ret_type = get_builtin_python_type_instance(localization, 'tuple')
        if len(arguments) > 0:
            params = arguments[0]
            if is_str(type(params)):
                set_contained_elements_type(ret_type,
                                            get_builtin_python_type_instance(localization, type(params).__name__))
                return ret_type

            existing_type = get_contained_elements_type(params)
            if isinstance(existing_type, union_type.UnionType):
                types_ = existing_type.types
                ordered = None
                for type_ in types_:
                    ordered = union_type.UnionType.add(ordered, type_)

                set_contained_elements_type(ret_type, ordered)
            else:
                set_contained_elements_type(ret_type, existing_type)

        # ret_type.known_elements = True
        return ret_type

    @staticmethod
    def __add__(localization, proxy_obj, arguments):
        ret_type = get_builtin_python_type_instance(localization, 'tuple')

        existing_type = get_contained_elements_type(TypeWrapper.get_wrapper_of(get_self(proxy_obj)))
        params = arguments[0]
        if can_store_elements(params):
            if existing_type is not None:
                new_type = existing_type.duplicate()
            else:
                new_type = None
            other_type = get_contained_elements_type(params)
            if not isinstance(other_type, union_type.UnionType):
                other_type = [other_type]
            else:
                other_type = other_type.types

            for par in other_type:
                new_type = union_type.UnionType.add(new_type, par)
        else:
            new_type = union_type.UnionType.add(existing_type, arguments[0])

        set_contained_elements_type(ret_type, new_type)

        return ret_type

    @staticmethod
    def __iter__(localization, proxy_obj, arguments):
        iter = StandardWrapper(get_self(proxy_obj).__iter__())
        set_contained_elements_type(iter,
                                    get_contained_elements_type(TypeWrapper.get_wrapper_of(get_self(proxy_obj))))

        return iter

    @staticmethod
    def __getitem__(localization, proxy_obj, arguments):
        if compare_type(arguments[0], slice):
            ret_type = get_builtin_python_type_instance(localization, "tuple")

            set_contained_elements_type(ret_type,
                                        get_contained_elements_type(TypeWrapper.get_wrapper_of(get_self(proxy_obj))))

            return ret_type

        return get_contained_elements_type(TypeWrapper.get_wrapper_of(get_self(proxy_obj)))

    @staticmethod
    def __getslice__(localization, proxy_obj, arguments):
        ret_type = get_builtin_python_type_instance(localization, "tuple")
        set_contained_elements_type(ret_type,
                                    get_contained_elements_type(TypeWrapper.get_wrapper_of(get_self(proxy_obj))))

        return ret_type

    @staticmethod
    def __mul__(localization, proxy_obj, arguments):
        ret_type = get_builtin_python_type_instance(localization, "tuple")
        existing_type = get_contained_elements_type(TypeWrapper.get_wrapper_of(get_self(proxy_obj)))
        new_type = existing_type
        set_contained_elements_type(ret_type, new_type)

        return ret_type

    @staticmethod
    def __getnewargs__(localization, proxy_obj, arguments):
        ret_type = get_builtin_python_type_instance(localization, "tuple")
        # existing_type = get_contained_elements_type(get_self(proxy_obj))
        set_contained_elements_type(ret_type, TypeWrapper.get_wrapper_of(get_self(proxy_obj)))

        return ret_type
