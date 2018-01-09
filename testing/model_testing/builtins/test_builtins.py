#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print path
if not path in sys.path:
    sys.path.append(path)

import unittest

from stypy.contexts.context import Context
from stypy.invokation.type_rules.type_groups.type_groups import DynamicType
from stypy.type_inference_programs.stypy_interface import *
from stypy.types import undefined_type
from stypy.types.known_python_types import ExtraTypeDefinitions
from stypy.types.type_intercession import get_member
from testing.model_testing.model_testing_common import *
from stypy.ssa.ssa_context import SSAContext


class TestBuiltins(unittest.TestCase):
    def __obtain_list_item(self, list_name, return_var_name):
        __temp_1 = self.type_store.get_type_of(Localization(__file__), list_name)
        __temp_2 = __temp_1.get_type_of_member(Localization(__file__), "__getitem__")
        __temp_3 = __temp_2.invoke(Localization(__file__), int())
        self.type_store.set_type_of(Localization(__file__), return_var_name, __temp_3)

    def __create_list(self, type_store, name, types):
        list_ = get_builtin_python_type_instance(None, "list")
        type_store.set_type_of(self.loc, name, list_)

        append_method = type_store.get_type_of_member(self.loc, list_, "append")
        for type in types:
            invoke(None, append_method, type)

    # ################################################## TESTING FUNCTIONS #################################################

    def setUp(self):
        # Create a type store
        self.type_store = Context(None, __file__)
        self.loc = Localization(__file__)

        # Create a list
        # l = [1, 2, "a"]
        self.__create_list(self.type_store, "mixed_type_list", [int(), str()])
        self.__create_list(self.type_store, "mixed_type_list2", [complex(), str()])
        self.__create_list(self.type_store, "int_list", [int()])
        self.str_obj = get_builtin_python_type_instance(self.loc, "str", "x")
        self.int_obj = get_builtin_python_type_instance(self.loc, "int", 3)
        self.float_obj = get_builtin_python_type_instance(self.loc, "float", 3.14)
        StypyTypeError.reset_error_msgs()
        TypeWarning.reset_warning_msgs()

    def test_bytearray(self):
        obj = get_builtin_python_type(self.loc, "bytearray")
        compare_types(obj, bytearray)
        obj_instance = invoke(self.loc, obj)
        compare_types(type(obj_instance), bytearray)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(type(obj_instance), bytearray)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        compare_types(type(obj_instance), bytearray)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(type(obj_instance), bytearray)

        compare_types(type(get_elements_type(obj_instance)), int)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(type(obj_instance), bytearray)

        compare_types(type(get_elements_type(obj_instance)), int)
        self.assertTrue(len(TypeWarning.get_warning_msgs()) == 1)

    def test_all(self):
        obj = get_builtin_python_type_instance(self.loc, "all")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        compare_types(type(obj_instance), bool)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(type(obj_instance), bool)

        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(type(obj_instance), bool)

    def test_set(self):
        obj = get_builtin_python_type(self.loc, "set")
        compare_types(obj, set)
        obj_instance = invoke(self.loc, obj)
        compare_types(obj_instance, set)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        compare_types(obj_instance, set)
        compare_types(get_elements_type(obj_instance), str())

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(get_elements_type(obj_instance), [complex(), str()])

        obj_instance = invoke(self.loc, obj, self.str_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(obj_instance, set)
        compare_types(get_elements_type(obj_instance), int())

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(obj_instance, set)
        compare_types(get_elements_type(obj_instance), [int(), str()])

    def test_help(self):
        obj = get_builtin_python_type_instance(self.loc, "help")
        assert_equal_type_name(type(obj), "_Helper")
        obj_instance = invoke(self.loc, obj)
        compare_types(type(obj_instance), str)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(type(obj_instance), str)

    def test_vars(self):
        t = self.type_store.open_function_context("test_func")
        t.set_type_of(self.loc, "local_1", get_builtin_python_type_instance(self.loc, "int"))
        t.set_type_of(self.loc, "local_2", get_builtin_python_type_instance(self.loc, "str"))
        obj = get_builtin_python_type_instance(self.loc, "vars")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        compare_types(obj_instance, dict)
        compare_types(type(get_values_from_key(obj_instance, "local_1")), int)
        compare_types(type(get_values_from_key(obj_instance, "local_2")), str)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, type(self.int_obj))
        compare_types(obj_instance, dict)
        self.assertTrue(len(obj_instance) == 56)

        t.close_function_context()

    def test_bool(self):
        obj = get_builtin_python_type_instance(self.loc, "bool")
        compare_types(type(obj), bool)
        obj = bool
        obj_instance = invoke(self.loc, obj)
        compare_types(type(obj_instance), bool)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(type(obj_instance), bool)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        compare_types(type(obj_instance), bool)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(type(obj_instance), bool)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(type(obj_instance), bool)

    def test_float(self):
        class Foo:
            def __float__(self, localization):
                return get_builtin_python_type_instance(localization, 'float')

        obj = get_builtin_python_type(self.loc, "float")
        compare_types(obj, float)
        obj_instance = invoke(self.loc, obj)
        compare_types(type(obj_instance), float)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(type(obj_instance), float)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        assert_if_not_error(obj_instance)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        param = Foo()
        obj_instance = invoke(self.loc, obj, param)
        compare_types(type(obj_instance), float)

    def test_import(self):
        obj = get_builtin_python_type_instance(self.loc, "__import__")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj, "math")
        compare_types((obj_instance), types.ModuleType)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, "math", self.int_obj)
        compare_types((obj_instance), types.ModuleType)

        obj_instance = invoke(self.loc, obj, "math", self.int_obj, self.int_obj)
        compare_types((obj_instance), types.ModuleType)

        obj_instance = invoke(self.loc, obj, "math", self.int_obj, self.int_obj, self.int_obj)
        compare_types((obj_instance), types.ModuleType)

        obj_instance = invoke(self.loc, obj, "math", self.int_obj, self.int_obj, self.int_obj, self.str_obj)
        assert_if_not_error(obj_instance)

    def test_unicode(self):
        class WrongFoo:
            def __str__(self, localization):
                return get_builtin_python_type_instance(localization, 'float')

        class Foo:
            def __str__(self, localization):
                return get_builtin_python_type_instance(localization, 'str')

        obj = get_builtin_python_type(self.loc, "unicode")
        compare_types(obj, unicode)
        obj_instance = invoke(self.loc, obj)
        compare_types(type(obj_instance), unicode)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(type(obj_instance), unicode)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        compare_types(type(obj_instance), unicode)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(type(obj_instance), unicode)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        param = Foo()
        obj_instance = invoke(self.loc, obj, param)
        compare_types(type(obj_instance), unicode)

        param = WrongFoo()
        obj_instance = invoke(self.loc, obj, param)
        assert_if_not_error(obj_instance)

    def test_enumerate(self):
        class Foo:
            def __iter__(self, localization):
                t = get_builtin_python_type_instance(localization, "listiterator")
                set_contained_elements_type(localization, t,
                                            get_builtin_python_type_instance(localization, 'float', 3))

                return t

        obj = get_builtin_python_type(self.loc, "enumerate")
        compare_types(obj, enumerate)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        compare_types(obj_instance, enumerate)
        compare_types(get_elements_type(obj_instance), tuple)
        compare_types(get_elements_type(get_elements_type(obj_instance)), [int(), str()])

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(obj_instance, enumerate)
        compare_types(get_elements_type(obj_instance), tuple)
        compare_types(get_elements_type(get_elements_type(obj_instance)), [int(), complex(), str()])

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(obj_instance, enumerate)
        compare_types(get_elements_type(obj_instance), tuple)
        compare_types(get_elements_type(get_elements_type(obj_instance)), [int(), str()])

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        param = Foo()
        obj_instance = invoke(self.loc, obj, param)
        compare_types(obj_instance, enumerate)
        compare_types(get_elements_type(obj_instance), tuple)
        compare_types(get_elements_type(get_elements_type(obj_instance)), [int(), float(3)])

    def test_reduce(self):
        obj = get_builtin_python_type_instance(self.loc, "reduce")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        def my_func1(localization, *args, **kwargs):
            result = python_operator(localization, '+', args[0], args[1])
            return result

        func = my_func1
        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, func, list_)
        compare_types(type(obj_instance), int)

        def my_func2(localization, *args, **kwargs):
            param1 = get_builtin_python_type(localization, "str")
            arg1 = invoke(localization, param1, args[0])
            param1 = get_builtin_python_type(localization, "str")
            arg2 = invoke(localization, param1, args[1])

            result = python_operator(localization, '+', arg1, arg2)
            return result

        func2 = my_func2
        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, func2, list_)
        compare_types(type(obj_instance), str)

        def invalid_func(localization, *args, **kwargs):
            raise Exception("This always fails")

        func3 = invalid_func
        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, func3, list_)
        assert_if_not_error(obj_instance)

        initial = get_builtin_python_type_instance(self.loc, "int")
        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, func, list_, initial)
        compare_types(type(obj_instance), int)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, func2, list_, initial)
        compare_types(type(obj_instance), str)

        initial2 = get_builtin_python_type_instance(self.loc, "str")
        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, func2, list_, initial2)
        compare_types(type(obj_instance), str)

    def test_list(self):
        obj = get_builtin_python_type(self.loc, "list")
        compare_types(obj, list)
        obj_instance = invoke(self.loc, obj)
        compare_types(obj_instance, list)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        compare_types(obj_instance, list)
        compare_types(type(get_elements_type(obj_instance)), str)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types((get_elements_type(obj_instance)), [complex(), str()])

        obj_instance = invoke(self.loc, obj, self.str_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types((obj_instance), list)
        compare_types(type(get_elements_type(obj_instance)), int)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types((obj_instance), list)
        compare_types((get_elements_type(obj_instance)), [int(), str()])

    def test_coerce(self):
        obj = get_builtin_python_type_instance(self.loc, "coerce")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        assert_if_not_error(obj_instance)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_)
        assert_if_not_error(obj_instance)

        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, list_)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        compare_types(obj_instance, tuple)
        compare_types(type(get_elements_type(obj_instance)), int)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.float_obj)
        compare_types(obj_instance, tuple)
        compare_types(get_elements_type(obj_instance), [self.int_obj, self.float_obj])

        obj_instance = invoke(self.loc, obj, self.int_obj, self.str_obj)
        assert_if_not_error(obj_instance)

    def test_intern(self):
        obj = get_builtin_python_type_instance(self.loc, "intern")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        compare_types(type(obj_instance), str)

    def test_globals(self):
        t = Context(None, __file__)
        t.set_type_of(self.loc, "global_1", get_builtin_python_type_instance(self.loc, "int"))
        t.set_type_of(self.loc, "global_2", get_builtin_python_type_instance(self.loc, "str"))
        obj = get_builtin_python_type_instance(self.loc, "globals")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        compare_types(obj_instance, dict)
        #compare_types(obj_instance.wrapped_type['__builtins__'], dict)
        compare_types(obj_instance.wrapped_type['__doc__'], None)
        compare_types(obj_instance.wrapped_type['__name__'], str())
        compare_types(obj_instance.wrapped_type['__package__'], None)
        compare_types(obj_instance.wrapped_type['global_1'], int())
        compare_types(obj_instance.wrapped_type['global_2'], str())

    def test_issubclass(self):
        obj = get_builtin_python_type_instance(self.loc, "issubclass")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        type1 = int
        type2 = complex

        obj_instance = invoke(self.loc, obj, type1, type2)
        compare_types(type(obj_instance), bool)

        instance1 = get_builtin_python_type_instance(self.loc, "int")
        instance2 = get_builtin_python_type_instance(self.loc, "complex")

        obj_instance = invoke(self.loc, obj, instance1, instance2)
        assert_if_not_error(obj_instance)

    def test_divmod(self):
        obj = get_builtin_python_type_instance(self.loc, "divmod")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        compare_types(obj_instance, tuple)
        compare_types(type(get_elements_type(obj_instance)), int)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.float_obj)
        compare_types(obj_instance, tuple)
        compare_types(type(get_elements_type(obj_instance)), float)

    def test_file(self):
        obj = file #get_builtin_python_type_instance(self.loc, "file")
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        compare_types(type(obj_instance), file)

        obj_instance = invoke(self.loc, obj, self.str_obj, self.str_obj)
        compare_types(type(obj_instance), file)

        obj_instance = invoke(self.loc, obj, self.str_obj, self.str_obj, self.int_obj)
        compare_types(type(obj_instance), file)

        class Foo:
            def __trunc__(self, localization):
                return get_builtin_python_type_instance(localization, "int")

        trunc_instance = Foo()
        obj_instance = invoke(self.loc, obj, self.str_obj, self.str_obj, trunc_instance)
        compare_types(type(obj_instance), file)

    def test_unichr(self):
        obj = get_builtin_python_type_instance(self.loc, "unichr")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(type(obj_instance), unicode)

        class Foo:
            def __trunc__(self, localization):
                return get_builtin_python_type_instance(localization, "int")

        trunc_instance = Foo()
        obj_instance = invoke(self.loc, obj, trunc_instance)
        compare_types(type(obj_instance), unicode)

    def test_apply(self):
        class Foo1:
            def __call__(self, localization):
                return get_builtin_python_type_instance(None, "int")

        instance1 = Foo1()

        class Foo2:
            def __call__(self, localization, param):
                return get_builtin_python_type_instance(None, type(param).__name__)

        class Foo2b:
            def __call__(self, localization, param, param2):
                return get_builtin_python_type_instance(None, type(param2).__name__)

        instance2 = Foo2()
        instance2b = Foo2b()

        class Foo3:
            def __call__(self, localization, *param):
                return param[-1]

        instance3 = Foo3()

        apply_func = get_builtin_python_type_instance(self.loc, "apply")
        compare_types(type(apply_func), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, apply_func)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, apply_func, instance1)
        compare_types(type(obj_instance), int)

        tuple_ = get_builtin_python_type(self.loc, "tuple")
        types_ = get_builtin_python_type_instance(self.loc, "list")
        set_contained_elements_type(self.loc, types_, self.float_obj)
        tuple_ = invoke(self.loc, tuple_, types_)
        obj_instance = invoke(self.loc, apply_func, instance2, tuple_)
        compare_types(type(obj_instance), float)

        obj_instance = invoke(self.loc, apply_func, instance2b, tuple_)
        compare_types(type(obj_instance), float)

        dict_ = get_builtin_python_type_instance(self.loc, "dict")
        add_key_and_value_type(dict_, self.str_obj, self.int_obj)
        obj_instance = invoke(self.loc, apply_func, instance3, tuple_, dict_)
        compare_types(obj_instance, dict)

        builtins = load_builtin_operators_module()
        add = self.type_store.get_type_of_member(self.loc, builtins, "add")
        obj_instance = invoke(self.loc, apply_func, add, tuple_)
        compare_types(type(obj_instance), float)

    def test_isinstance(self):
        obj = get_builtin_python_type_instance(self.loc, "isinstance")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, int)
        compare_types(type(obj_instance), bool)

        obj_instance = invoke(self.loc, obj, self.str_obj, str)
        compare_types(type(obj_instance), bool)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_, list)
        compare_types(type(obj_instance), bool)

        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, list_, tuple)
        compare_types(type(obj_instance), bool)

    def test_next(self):
        obj = get_builtin_python_type_instance(self.loc, "next")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        assert_if_not_error(obj_instance)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        iter_list_call = self.type_store.get_type_of_member(self.loc, list_, "__iter__")
        iter_list = invoke(self.loc, iter_list_call)
        compare_types(iter_list, ExtraTypeDefinitions.listiterator)

        obj_instance = invoke(self.loc, obj, list_)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, iter_list)
        compare_types(obj_instance, [complex(), str()])

        obj_instance = invoke(self.loc, obj, iter_list, self.float_obj)
        compare_types(obj_instance, [complex(), str(), float(3.14)])

        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        iter_list_call = self.type_store.get_type_of_member(self.loc, list_, "__iter__")
        iter_list = invoke(self.loc, iter_list_call)
        compare_types(iter_list, ExtraTypeDefinitions.listiterator)

        obj_instance = invoke(self.loc, obj, list_)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, iter_list)
        compare_types(type(obj_instance), int)

        obj_instance = invoke(self.loc, obj, iter_list, self.float_obj)
        compare_types(obj_instance, [int(), float(3.14)])

    def test_any(self):
        obj = get_builtin_python_type_instance(self.loc, "any")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        assert_if_not_error(obj_instance)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(type(obj_instance), bool)

    def test_locals(self):
        context = self.type_store.open_function_context("test_func2")
        context.set_type_of(self.loc, "local_1", get_builtin_python_type_instance(self.loc, "int"))
        context.set_type_of(self.loc, "local_2", get_builtin_python_type_instance(self.loc, "str"))
        obj = get_builtin_python_type_instance(self.loc, "locals")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        compare_types(obj_instance, dict)
        compare_types(type(obj_instance.wrapped_type['local_1']), int)
        compare_types(type(obj_instance.wrapped_type['local_2']), str)

        self.type_store.close_function_context()

    def test_filter(self):
        obj = get_builtin_python_type_instance(self.loc, "filter")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        assert_if_not_error(obj_instance)

        def my_func1(localization, *args, **kwargs):
            return False

        func1 = my_func1

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, func1, list_)
        compare_types(obj_instance, list)
        compare_types(get_elements_type(obj_instance), [complex(), str()])

        obj_instance = invoke(self.loc, obj, func1, self.str_obj)
        compare_types(obj_instance, list)
        compare_types(get_elements_type(obj_instance), str())

    def test_slice(self):
        obj = get_builtin_python_type(self.loc, "slice")
        compare_types(obj, slice)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(get_elements_type(obj_instance), [int(), types.NoneType])

        obj_instance = invoke(self.loc, obj, self.str_obj)
        compare_types(get_elements_type(obj_instance), [str(), types.NoneType])

        def my_func1(localization, *args, **kwargs):
            return False

        func1 = my_func1

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, func1, list_)
        compare_types(obj_instance, slice)
        compare_types(get_elements_type(obj_instance), [list_, False])

        obj_instance = invoke(self.loc, obj, func1, self.str_obj)
        compare_types(obj_instance, slice)
        compare_types(get_elements_type(obj_instance), [str(), False])

    def test_copyright(self):
        obj = get_builtin_python_type(self.loc, "copyright")
        assert_equal_type_name(type(obj), "_Printer")
        obj_instance = invoke(self.loc, obj)
        compare_types(type(obj_instance), types.NoneType)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

    def test_min(self):
        obj = get_builtin_python_type_instance(self.loc, "min")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        compare_types(type(obj_instance), str)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.float_obj)
        compare_types(obj_instance, [self.int_obj, self.float_obj])

        obj_instance = invoke(self.loc, obj, self.int_obj, self.float_obj, self.int_obj, self.float_obj, self.int_obj,
                              self.float_obj, self.int_obj, self.float_obj)
        compare_types(obj_instance, [int(), float()])

        tuple_ = get_builtin_python_type_instance(self.loc, "tuple")
        set_contained_elements_type(self.loc, tuple_, self.float_obj)
        obj_instance = invoke(self.loc, obj, tuple_)
        compare_types(type(obj_instance), float)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(obj_instance, [complex(), str()])

        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(type(obj_instance), int)

        def func(localization, *args, **kwargs):
            return get_builtin_python_type_instance(localization, "int")

        dict = {'key': func}
        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_, **dict)
        compare_types(obj_instance, [complex(), str()])

    def test_open(self):
        obj = get_builtin_python_type_instance(self.loc, "open")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        compare_types(type(obj_instance), file)

        obj_instance = invoke(self.loc, obj, self.str_obj, self.str_obj)
        compare_types(type(obj_instance), file)

        obj_instance = invoke(self.loc, obj, self.str_obj, self.str_obj, self.int_obj)
        compare_types(type(obj_instance), file)

        class Foo:
            def __trunc__(self, localization):
                return get_builtin_python_type_instance(localization, "int")

        trunc_instance = Foo()
        obj_instance = invoke(self.loc, obj, self.str_obj, self.str_obj, trunc_instance)
        compare_types(type(obj_instance), file)

    def test_sum(self):
        obj = get_builtin_python_type_instance(self.loc, "sum")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        class Foo:
            def __add__(self, localization, other):
                return get_builtin_python_type_instance(localization, "int")

        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(type(obj_instance), int)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_)
        assert_if_not_error(obj_instance)

        instance = Foo()
        list_ = get_builtin_python_type_instance(self.loc, "list")
        set_contained_elements_type(self.loc, list_, instance)
        obj_instance = invoke(self.loc, obj, list_)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, list_, self.int_obj)
        assert_if_not_error(obj_instance)

    def test_chr(self):
        obj = get_builtin_python_type_instance(self.loc, "chr")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(type(obj_instance), str)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.str_obj)
        assert_if_not_error(obj_instance)

        class Foo:
            def __trunc__(self, localization):
                return get_builtin_python_type_instance(localization, "int")

        trunc_instance = Foo()
        obj_instance = invoke(self.loc, obj, trunc_instance)
        compare_types(type(obj_instance), str)

    def test_hex(self):
        obj = get_builtin_python_type_instance(self.loc, "hex")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(type(obj_instance), str)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.str_obj)
        assert_if_not_error(obj_instance)

        class Foo:
            def __hex__(self, localization):
                return get_builtin_python_type_instance(localization, "int")

        class Foo2:
            def __hex__(self, localization):
                return get_builtin_python_type_instance(localization, "str")

        hex_instance = Foo()
        obj_instance = invoke(self.loc, obj, hex_instance)
        assert_if_not_error(obj_instance)

        hex_instance = Foo2()
        obj_instance = invoke(self.loc, obj, hex_instance)
        compare_types(type(obj_instance), str)

    def test_execfile(self):
        obj = get_builtin_python_type_instance(self.loc, "execfile")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        compare_types(type(obj_instance), DynamicType)

        dict_ = get_builtin_python_type_instance(self.loc, "dict")
        obj_instance = invoke(self.loc, obj, self.str_obj, dict_)
        compare_types(type(obj_instance), DynamicType)

        obj_instance = invoke(self.loc, obj, self.str_obj, dict_, dict_)
        compare_types(type(obj_instance), DynamicType)

    def test_long(self):
        obj = get_builtin_python_type(self.loc, "long")
        compare_types(obj, types.LongType)
        obj_instance = invoke(self.loc, obj)
        compare_types(type(obj_instance), long)

        obj_instance = invoke(self.loc, obj, self.float_obj)
        compare_types(type(obj_instance), long)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        compare_types(type(obj_instance), long)

        obj_instance = invoke(self.loc, obj, self.float_obj, self.int_obj)
        compare_types(type(obj_instance), long)

        obj_instance = invoke(self.loc, obj, self.float_obj, self.float_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.str_obj)
        assert_if_not_error(obj_instance)

        class Foo:
            def __trunc__(self, localization):
                return get_builtin_python_type_instance(localization, "int")

        trunc_instance = Foo()
        obj_instance = invoke(self.loc, obj, trunc_instance)
        compare_types(type(obj_instance), long)

        obj_instance = invoke(self.loc, obj, self.int_obj, trunc_instance)
        compare_types(type(obj_instance), long)

        obj_instance = invoke(self.loc, obj, trunc_instance, trunc_instance)
        compare_types(type(obj_instance), long)

    def test_id(self):
        obj = get_builtin_python_type_instance(self.loc, "id")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        type1 = int
        type2 = complex

        obj_instance = invoke(self.loc, obj, type1)
        compare_types(type(obj_instance), long)

        obj_instance = invoke(self.loc, obj, type2)
        compare_types(type(obj_instance), long)

        instance1 = get_builtin_python_type_instance(self.loc, "int")
        instance2 = get_builtin_python_type_instance(self.loc, "complex")

        obj_instance = invoke(self.loc, obj, instance1)
        compare_types(type(obj_instance), long)

        obj_instance = invoke(self.loc, obj, instance2)
        compare_types(type(obj_instance), long)

    def test_xrange(self):
        obj = get_builtin_python_type_instance(self.loc, "xrange")
        compare_types(obj, types.XRangeType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.float_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        compare_types(obj_instance, xrange)
        compare_types(type(get_elements_type(obj_instance)), int)

        obj_instance = invoke(self.loc, obj, self.float_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.float_obj, self.float_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.str_obj)
        assert_if_not_error(obj_instance)

        class Foo:
            def __trunc__(self, localization):
                return get_builtin_python_type_instance(localization, "int")

        trunc_instance = Foo()
        obj_instance = invoke(self.loc, obj, trunc_instance)
        compare_types(obj_instance, xrange)
        compare_types(type(get_elements_type(obj_instance)), int)

        obj_instance = invoke(self.loc, obj, self.int_obj, trunc_instance)
        compare_types(obj_instance, xrange)
        compare_types(type(get_elements_type(obj_instance)), int)

        obj_instance = invoke(self.loc, obj, trunc_instance, trunc_instance)
        compare_types(obj_instance, xrange)
        compare_types(type(get_elements_type(obj_instance)), int)

    def test_int(self):
        obj = get_builtin_python_type(self.loc, "int")
        compare_types((obj), types.IntType)
        obj_instance = invoke(self.loc, obj)
        compare_types(type(obj_instance), int)

        obj_instance = invoke(self.loc, obj, self.float_obj)
        compare_types(type(obj_instance), int)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        compare_types(type(obj_instance), int)

        obj_instance = invoke(self.loc, obj, self.float_obj, self.int_obj)
        compare_types(type(obj_instance), int)

        obj_instance = invoke(self.loc, obj, self.float_obj, self.float_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.str_obj)
        assert_if_not_error(obj_instance)

        class Foo:
            def __trunc__(self, localization):
                return get_builtin_python_type_instance(localization, "int")

        trunc_instance = Foo()
        obj_instance = invoke(self.loc, obj, trunc_instance)
        compare_types(type(obj_instance), int)

        obj_instance = invoke(self.loc, obj, self.int_obj, trunc_instance)
        compare_types(type(obj_instance), int)

        obj_instance = invoke(self.loc, obj, trunc_instance, trunc_instance)
        compare_types(type(obj_instance), int)

    def test_getattr(self):
        obj = get_builtin_python_type_instance(self.loc, "getattr")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        class Foo:
            def __add__(self, localization, other):
                return get_builtin_python_type_instance(localization, "int")

        Foo.attclass = int
        foo_inst = Foo()
        foo_inst.attinst = str

        class_ = Foo
        instance = foo_inst
        val1 = get_builtin_python_type_instance(self.loc, "str", "attclass")
        val2 = get_builtin_python_type_instance(self.loc, "str", "attinst")
        val3 = get_builtin_python_type_instance(self.loc, "str", "not_exist")
        val4 = get_builtin_python_type_instance(self.loc, "str")

        obj_instance = invoke(self.loc, obj, class_, val1)
        compare_types(obj_instance, int)

        obj_instance = invoke(self.loc, obj, class_, val2)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, class_, val3)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, class_, val4)
        compare_types(obj_instance, [Foo.__add__, str(), int])

        obj_instance = invoke(self.loc, obj, instance, val1)
        compare_types(obj_instance, int)

        obj_instance = invoke(self.loc, obj, instance, val2)
        compare_types(obj_instance, str)

        obj_instance = invoke(self.loc, obj, instance, val3)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, instance, val4)
        compare_types(obj_instance, [foo_inst.__add__, str, str(), int])

    def test_abs(self):
        class Foo:
            def __abs__(self, localization):
                return get_builtin_python_type_instance(localization, 'int')

        obj = get_builtin_python_type_instance(self.loc, "abs")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(type(obj_instance), int)

        complex_inst = get_builtin_python_type_instance(self.loc, "complex")
        obj_instance = invoke(self.loc, obj, complex_inst)
        compare_types(type(obj_instance), float)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        param = Foo()
        obj_instance = invoke(self.loc, obj, param)
        compare_types(type(obj_instance), int)

    def test_exit(self):
        obj = get_builtin_python_type(self.loc, "exit")
        assert_equal_type_name(type(obj), "Quitter")
        obj_instance = invoke(self.loc, obj)
        compare_types(obj_instance, undefined_type.UndefinedType)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(obj_instance, undefined_type.UndefinedType)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(obj_instance, undefined_type.UndefinedType)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

    def test_pow(self):
        obj = get_builtin_python_type_instance(self.loc, "pow")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        class Foo:
            def __pow__(self, localization, other, another):
                return get_builtin_python_type_instance(localization, "int")

        complex_inst = get_builtin_python_type_instance(self.loc, "complex")
        obj_instance = invoke(self.loc, obj, complex_inst, self.int_obj)
        compare_types(type(obj_instance), types.ComplexType)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        compare_types(type(obj_instance), int)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_, list_)
        assert_if_not_error(obj_instance)

        instance = instance = Foo()
        obj_instance = invoke(self.loc, obj, instance, self.int_obj, self.int_obj)
        compare_types(type(obj_instance), int)

        bool_inst = get_builtin_python_type_instance(self.loc, "bool")
        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj, bool_inst)
        compare_types(type(obj_instance), int)

        long_inst = get_builtin_python_type_instance(self.loc, "long")
        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj, long_inst)
        compare_types(type(obj_instance), long)

    def test_input(self):
        obj = get_builtin_python_type_instance(self.loc, "input")
        assert_equal_type_name(obj, "input")
        obj_instance = invoke(self.loc, obj)
        compare_types((obj_instance), DynamicType)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types((obj_instance), DynamicType)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types((obj_instance), DynamicType)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

    def test_type(self):
        obj = get_builtin_python_type_instance(self.loc, "type")
        assert_equal_type_name(obj, "type")
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(obj_instance, int)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        compare_types(obj_instance, str)

        class Foo:
            def __pow__(self, localization, other, another):
                return get_builtin_python_type_instance(localization, "int")

        complex_inst = get_builtin_python_type_instance(self.loc, "complex")
        obj_instance = invoke(self.loc, obj, complex_inst)
        compare_types(obj_instance, types.ComplexType)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(obj_instance, int)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_, list_)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, list_)
        compare_types(obj_instance, list)

        instance = Foo()
        obj_instance = invoke(self.loc, obj, instance)
        compare_types(obj_instance, type(Foo()))

        bool_inst = get_builtin_python_type_instance(self.loc, "bool")
        obj_instance = invoke(self.loc, obj, bool_inst)
        compare_types(obj_instance, bool)

        long_inst = get_builtin_python_type_instance(self.loc, "long")
        obj_instance = invoke(self.loc, obj, long_inst)
        compare_types(obj_instance, long)

        type_inst = get_builtin_python_type_instance(self.loc, "type")
        obj_instance = invoke(self.loc, obj, type_inst)
        compare_types(obj_instance, types.TypeType)

    def test_oct(self):
        obj = get_builtin_python_type_instance(self.loc, "oct")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(type(obj_instance), str)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.str_obj)
        assert_if_not_error(obj_instance)

        class Foo:
            def __oct__(self, localization):
                return get_builtin_python_type_instance(localization, "int")

        class Foo2:
            def __oct__(self, localization):
                return get_builtin_python_type_instance(localization, "str")

        hex_instance = Foo()
        obj_instance = invoke(self.loc, obj, hex_instance)
        assert_if_not_error(obj_instance)

        hex_instance = Foo2()
        obj_instance = invoke(self.loc, obj, hex_instance)
        compare_types(type(obj_instance), str)

    def test_bin(self):
        obj = get_builtin_python_type_instance(self.loc, "bin")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        bool_inst = get_builtin_python_type_instance(self.loc, "bool")
        obj_instance = invoke(self.loc, obj, bool_inst)
        compare_types(type(obj_instance), str)

        long_inst = get_builtin_python_type_instance(self.loc, "long")
        obj_instance = invoke(self.loc, obj, long_inst)
        compare_types(type(obj_instance), str)

    def test_map(self):
        obj = get_builtin_python_type(self.loc, "map")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, self.int_obj, list_)
        assert_if_not_error(obj_instance)

        func = get_builtin_python_type(self.loc, 'str')
        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, func, list_)
        compare_types(obj_instance, list)
        compare_types(type(get_elements_type(obj_instance)), str)

        def my_func1(localization, *args, **kwargs):
            param1 = get_builtin_python_type(localization, "str")
            result = invoke(localization, param1, args[0])

            return result

        func = my_func1
        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, func, list_)
        compare_types(obj_instance, list)
        compare_types(type(get_elements_type(obj_instance)), str)

        def my_func2(localization, *args, **kwargs):
            param2 = get_builtin_python_type_instance(localization, "int")
            result = python_operator(localization, '/', args[0], param2)

            return result

        func2 = my_func2
        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, func2, list_)
        compare_types(obj_instance, list)
        compare_types(type(get_elements_type(obj_instance)), complex)

        def invalid_func(localization, *args, **kwargs):
            return StypyTypeError(localization, "This always fails")

        func3 = invalid_func
        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, func3, list_)
        assert_if_not_error(obj_instance)

        def my_func3(localization, *args, **kwargs):
            if len(args) > 2:
                return StypyTypeError(localization, "Too many arguments")

            result = python_operator(localization, '+', args[0], args[1])

            return result

        func4 = my_func3
        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, func4, list_, list_)
        compare_types(obj_instance, list)
        compare_types(type(get_elements_type(obj_instance)), int)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, func4, list_, list_, list_)
        assert_if_not_error(obj_instance)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, func2, list_, list_)
        compare_types(obj_instance, list)
        compare_types(type(get_elements_type(obj_instance)), complex)

        def my_func4(localization, *args, **kwargs):
            str_var = get_builtin_python_type_instance(localization, "str")
            result = python_operator(localization, '+', args[0], str_var)

            return result

        func5 = my_func4
        str_var = get_builtin_python_type_instance(self.loc, "str")

        obj_instance = invoke(self.loc, obj, func5, str_var)
        compare_types(obj_instance, list)
        compare_types(type(get_elements_type(obj_instance)), str)

        obj_instance = invoke(self.loc, obj, func2, str_var)
        assert_if_not_error(obj_instance)

    def test_zip(self):
        obj = get_builtin_python_type_instance(self.loc, "zip")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(obj_instance, list)
        compare_types(get_elements_type(obj_instance), tuple)
        compare_types(type(get_elements_type(get_elements_type(obj_instance))), int)

        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, list_, list_)
        compare_types(obj_instance, list)
        compare_types(get_elements_type(obj_instance), tuple)
        compare_types(type(get_elements_type(get_elements_type(obj_instance))), int)

        list_2 = self.type_store.get_type_of(Localization(__file__), "mixed_type_list")
        obj_instance = invoke(self.loc, obj, list_, list_2)
        compare_types(obj_instance, list)
        compare_types(get_elements_type(obj_instance), tuple)
        compare_types(get_elements_type(get_elements_type(obj_instance)), [int(), str()])

        list_3 = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_, list_3)
        compare_types(obj_instance, list)
        compare_types(get_elements_type(obj_instance), tuple)
        compare_types(get_elements_type(get_elements_type(obj_instance)), [int(), str(), complex()])

        str_instance = get_builtin_python_type_instance(self.loc, "str")
        obj_instance = invoke(self.loc, obj, list_, str_instance)
        compare_types(obj_instance, list)
        compare_types(get_elements_type(obj_instance), tuple)
        compare_types(get_elements_type(get_elements_type(obj_instance)), [int(), str()])

        str_instance = get_builtin_python_type_instance(self.loc, "str")
        obj_instance = invoke(self.loc, obj, str_instance, str_instance)
        compare_types(obj_instance, list)
        compare_types(get_elements_type(obj_instance), tuple)
        compare_types(type(get_elements_type(get_elements_type(obj_instance))), str)

    def test_hash(self):
        obj = get_builtin_python_type_instance(self.loc, "hash")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(type(obj_instance), int)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        compare_types(type(obj_instance), int)

        class Foo:
            def __trunc__(self, localization):
                return get_builtin_python_type_instance(localization, "int")

        trunc_instance = Foo()
        obj_instance = invoke(self.loc, obj, trunc_instance)
        compare_types(type(obj_instance), int)

    def test_format(self):
        obj = get_builtin_python_type_instance(self.loc, "format")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(type(obj_instance), str)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        compare_types(type(obj_instance), str)

        obj_instance = invoke(self.loc, obj, self.str_obj, self.str_obj)
        compare_types(type(obj_instance), str)

        class Foo:
            def __trunc__(self, localization):
                return get_builtin_python_type_instance(localization, "int")

        trunc_instance = Foo()
        obj_instance = invoke(self.loc, obj, trunc_instance)
        compare_types(type(obj_instance), str)

    def test_max(self):
        obj = get_builtin_python_type_instance(self.loc, "max")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        compare_types(type(obj_instance), str)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.float_obj)
        compare_types(obj_instance, [self.int_obj, self.float_obj])

        obj_instance = invoke(self.loc, obj, self.int_obj, self.float_obj, self.int_obj, self.float_obj, self.int_obj,
                              self.float_obj, self.int_obj, self.float_obj)
        compare_types(obj_instance, [self.int_obj, self.float_obj])

        tuple_ = get_builtin_python_type_instance(self.loc, "tuple")
        set_contained_elements_type(self.loc, tuple_, self.float_obj)
        obj_instance = invoke(self.loc, obj, tuple_)
        compare_types(type(obj_instance), float)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(obj_instance, [complex(), str()])

        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(type(obj_instance), int)

        def func(localization, *args, **kwargs):
            return get_builtin_python_type_instance(localization, "int")

        dict = {'key': func}
        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_, **dict)
        compare_types(obj_instance, [complex(), str()])

    def test_reversed(self):
        obj = get_builtin_python_type_instance(self.loc, "reversed")
        assert_equal_type_name(obj, "reversed")
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(obj_instance, reversed)
        compare_types(type(get_elements_type(obj_instance)), int)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(obj_instance, reversed)
        compare_types(get_elements_type(obj_instance), [complex(), str()])

        str_ = get_builtin_python_type_instance(self.loc, "str")
        obj_instance = invoke(self.loc, obj, str_)
        compare_types(obj_instance, reversed)
        compare_types(get_elements_type(obj_instance), str())

    def test_object(self):
        obj = get_builtin_python_type_instance(self.loc, "object")
        assert_equal_type_name(type(obj), "object")
        obj = object
        obj_instance = invoke(self.loc, obj)
        compare_types(type(obj_instance), object)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

    def test_quit(self):
        obj = get_builtin_python_type_instance(self.loc, "quit")
        assert_equal_type_name(type(obj), "Quitter")
        obj_instance = invoke(self.loc, obj)
        compare_types(obj_instance, undefined_type.UndefinedType)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(obj_instance, undefined_type.UndefinedType)

    def test_len(self):
        obj = get_builtin_python_type_instance(self.loc, "len")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        class Foo:
            def __len__(self, localization):
                return get_builtin_python_type_instance(localization, "int")

        class Foo2:
            def __len__(self, localization):
                return get_builtin_python_type_instance(localization, "str")

        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(type(obj_instance), int)

        instance = Foo()
        obj_instance = invoke(self.loc, obj, instance)
        compare_types(type(obj_instance), int)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        compare_types(type(obj_instance), int)

        instance = Foo2()
        obj_instance = invoke(self.loc, obj, instance)
        assert_if_not_error(obj_instance)

    def test_repr(self):
        obj = get_builtin_python_type_instance(self.loc, "repr")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(type(obj_instance), str)

        class Foo:
            def __repr__(self, localization):
                return get_builtin_python_type_instance(localization, "str")

        class Foo2:
            def __repr__(self, localization):
                return get_builtin_python_type_instance(localization, "int")

        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(type(obj_instance), str)

        instance = Foo()
        obj_instance = invoke(self.loc, obj, instance)
        compare_types(type(obj_instance), str)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        compare_types(type(obj_instance), str)

        instance = Foo2()
        obj_instance = invoke(self.loc, obj, instance)
        assert_if_not_error(obj_instance)

    def test_callable(self):
        obj = get_builtin_python_type_instance(self.loc, "callable")
        # self.assert_equal_type_name(obj, "__builtin__.quit")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(type(obj_instance), bool)

    def test_credits(self):
        obj = get_builtin_python_type(self.loc, "credits")
        assert_equal_type_name(type(obj), "_Printer")
        obj_instance = invoke(self.loc, obj)
        compare_types(type(obj_instance), types.NoneType)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

    def test_tuple(self):
        obj = get_builtin_python_type(self.loc, "tuple")
        compare_types(obj, tuple)
        obj_instance = invoke(self.loc, obj)
        compare_types(obj_instance, tuple)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        compare_types(obj_instance, tuple)
        compare_types(type(get_elements_type(obj_instance)), str)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(get_elements_type(obj_instance), [complex(), str()])

        obj_instance = invoke(self.loc, obj, self.str_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(obj_instance, tuple)
        compare_types(type(get_elements_type(obj_instance)), int)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(obj_instance, tuple)
        compare_types(get_elements_type(obj_instance), [int(), str()])

    def test_eval(self):
        obj = get_builtin_python_type_instance(self.loc, "eval")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        code = get_builtin_python_type_instance(self.loc, "CodeType")
        obj_instance = invoke(self.loc, obj, code)
        compare_types(type(obj_instance), DynamicType)

        dict_ = get_builtin_python_type_instance(self.loc, "dict")
        obj_instance = invoke(self.loc, obj, code, dict_)
        compare_types(type(obj_instance), DynamicType)

        obj_instance = invoke(self.loc, obj, code, dict_, dict_)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, code, dict_, dict_, dict_)
        assert_if_not_error(obj_instance)

        str_ = get_builtin_python_type_instance(self.loc, "str")
        obj_instance = invoke(self.loc, obj, str_, dict_, dict_)
        compare_types(type(obj_instance), DynamicType)

    def test_frozenset(self):
        obj = get_builtin_python_type(self.loc, "frozenset")
        compare_types(obj, frozenset)
        obj_instance = invoke(self.loc, obj)
        compare_types(obj_instance, frozenset)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        compare_types(obj_instance, frozenset)
        compare_types(get_elements_type(obj_instance), str())

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(get_elements_type(obj_instance), [complex(), str()])

        obj_instance = invoke(self.loc, obj, self.str_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(obj_instance, frozenset)
        compare_types(get_elements_type(obj_instance), int())

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(obj_instance, frozenset)
        compare_types(get_elements_type(obj_instance), [int(), str()])

    def test_sorted(self):
        obj = get_builtin_python_type_instance(self.loc, "sorted")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(obj_instance, list)
        compare_types(type(get_elements_type(obj_instance)), int)

        def my_func1(localization, *args, **kwargs):
            param1 = get_builtin_python_type_instance(localization, "str", args[0])

            return param1

        func = my_func1
        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list")
        obj_instance = invoke(self.loc, obj, list_, func)
        compare_types(obj_instance, list)
        compare_types(get_elements_type(obj_instance), [int(), str()])

        obj_instance = invoke(self.loc, obj, list_, func, func)
        compare_types(obj_instance, list)
        compare_types(get_elements_type(obj_instance), [int(), str()])

        bool_obj = get_builtin_python_type_instance(self.loc, "bool")
        obj_instance = invoke(self.loc, obj, list_, func, func, bool_obj)
        compare_types(obj_instance, list)
        compare_types(get_elements_type(obj_instance), [int(), str()])

        # str
        str_ = get_builtin_python_type_instance(self.loc, "str")
        obj_instance = invoke(self.loc, obj, str_)
        compare_types(obj_instance, list)
        compare_types(type(get_elements_type(obj_instance)), str)

        func = my_func1
        obj_instance = invoke(self.loc, obj, str_, func)
        compare_types(obj_instance, list)
        compare_types(type(get_elements_type(obj_instance)), str)

        obj_instance = invoke(self.loc, obj, str_, func, func)
        compare_types(obj_instance, list)
        compare_types(type(get_elements_type(obj_instance)), str)

        bool_obj = get_builtin_python_type_instance(self.loc, "bool")
        obj_instance = invoke(self.loc, obj, str_, func, func, bool_obj)
        compare_types(obj_instance, list)
        compare_types(type(get_elements_type(obj_instance)), str)

    def test_ord(self):
        obj = get_builtin_python_type_instance(self.loc, "ord")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, list_)
        assert_if_not_error(obj_instance)

        str_ = get_builtin_python_type_instance(self.loc, "str")
        obj_instance = invoke(self.loc, obj, str_)
        compare_types(type(obj_instance), int)

    def test_super(self):
        obj = get_builtin_python_type(self.loc, "super")
        compare_types(obj, super)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        int_class = get_builtin_python_type(self.loc, "int")

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, int_class)
        compare_types(type(obj_instance), super)
        compare_types(get_member(self.loc, obj_instance, "__thisclass__"), int)
        compare_types(get_member(self.loc, obj_instance, "__self_class__"), None)

        str_class = get_builtin_python_type(self.loc, "str")

        obj_instance = invoke(self.loc, obj, str_class)
        compare_types(type(obj_instance), super)
        compare_types(get_member(self.loc, obj_instance, "__thisclass__"), str)
        compare_types(get_member(self.loc, obj_instance, "__self_class__"), None)

        obj_instance = invoke(self.loc, obj, str_class, self.str_obj)
        compare_types(type(obj_instance), super)
        compare_types(get_member(self.loc, obj_instance, "__thisclass__"), str)
        compare_types(get_member(self.loc, obj_instance, "__self_class__"), str)

        obj_instance = invoke(self.loc, obj, self.str_obj, self.str_obj)
        assert_if_not_error(obj_instance)

        class Foo(object):
            att = int

            def func(self, localization):
                return get_builtin_python_type_instance(localization, "int")

        foo_class = Foo
        foo_instance = Foo()

        obj_instance = invoke(self.loc, obj, foo_class, foo_instance)
        compare_types(type(obj_instance), super)
        compare_types(get_member(self.loc, obj_instance, "__thisclass__"), foo_class)
        compare_types(get_member(self.loc, obj_instance, "__self_class__"), foo_class)

    def test_hasattr(self):
        obj = get_builtin_python_type_instance(self.loc, "hasattr")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj, self.str_obj)
        compare_types(type(obj_instance), bool)

        class Foo:
            att = int

            def func(self, localization):
                return get_builtin_python_type_instance(localization, "int")

        foo_class = Foo
        foo_instance = Foo()

        att_str = get_builtin_python_type_instance(self.loc, "str", value="att")
        func_str = get_builtin_python_type_instance(self.loc, "str", value="func")
        ne_str = get_builtin_python_type_instance(self.loc, "str", value="ne")

        obj_instance = invoke(self.loc, obj, foo_class, att_str)
        compare_types(type(obj_instance), bool)
        compare_types(obj_instance, True)

        obj_instance = invoke(self.loc, obj, foo_class, func_str)
        compare_types(type(obj_instance), bool)
        compare_types(obj_instance, True)

        obj_instance = invoke(self.loc, obj, foo_class, ne_str)
        compare_types(type(obj_instance), bool)
        compare_types(obj_instance, False)

        obj_instance = invoke(self.loc, obj, foo_instance, att_str)
        compare_types(type(obj_instance), bool)
        compare_types(obj_instance, True)

        obj_instance = invoke(self.loc, obj, foo_instance, func_str)
        compare_types(type(obj_instance), bool)
        compare_types(obj_instance, True)

        obj_instance = invoke(self.loc, obj, foo_instance, ne_str)
        compare_types(type(obj_instance), bool)
        compare_types(obj_instance, False)

    def test_delattr(self):
        obj = get_builtin_python_type_instance(self.loc, "delattr")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj, self.str_obj)
        assert_if_not_error(obj_instance)

        class Foo:
            att = int

            def func(self, localization):
                return get_builtin_python_type_instance(localization, "int")

        foo_class = Foo
        foo_instance = Foo()

        att_str = get_builtin_python_type_instance(self.loc, "str", value="att")
        func_str = get_builtin_python_type_instance(self.loc, "str", value="func")
        ne_str = get_builtin_python_type_instance(self.loc, "str", value="ne")

        obj_instance = invoke(self.loc, obj, foo_class, att_str)
        compare_types(type(obj_instance), types.NoneType)

        obj_instance = invoke(self.loc, obj, foo_class, func_str)
        compare_types(type(obj_instance), types.NoneType)

        obj_instance = invoke(self.loc, obj, foo_class, ne_str)
        assert_if_not_error(obj_instance)

        obj = get_builtin_python_type_instance(self.loc, "hasattr")

        obj_instance = invoke(self.loc, obj, foo_class, att_str)
        compare_types(type(obj_instance), bool)
        compare_types(obj_instance, False)

        obj_instance = invoke(self.loc, obj, foo_class, func_str)
        compare_types(type(obj_instance), bool)
        compare_types(obj_instance, False)

        obj_instance = invoke(self.loc, obj, foo_class, ne_str)
        compare_types(type(obj_instance), bool)
        compare_types(obj_instance, False)

        obj_instance = invoke(self.loc, obj, foo_instance, att_str)
        compare_types(type(obj_instance), bool)
        compare_types(obj_instance, False)

        obj_instance = invoke(self.loc, obj, foo_instance, func_str)
        compare_types(type(obj_instance), bool)
        compare_types(obj_instance, False)

        obj_instance = invoke(self.loc, obj, foo_instance, ne_str)
        compare_types(type(obj_instance), bool)
        compare_types(obj_instance, False)

    def test_dict(self):
        obj = get_builtin_python_type(self.loc, "dict")
        compare_types(obj, dict)
        obj_instance = invoke(self.loc, obj)
        compare_types(obj_instance, dict)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        assert_if_not_error(obj_instance)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        tuple_obj = get_builtin_python_type(self.loc, "tuple")
        tuple1 = invoke(self.loc, tuple_obj, list_)

        list_param1 = get_builtin_python_type_instance(self.loc, "list")
        self.type_store.set_type_of(self.loc, "list_param1", list_param1)

        append = self.type_store.get_type_of_member(self.loc, list_param1, "append")
        invoke(self.loc, append, tuple1)
        obj_instance = invoke(self.loc, obj, list_param1)

        keys = obj_instance.keys()
        values = obj_instance.values()
        self.assertTrue(
                get_builtin_python_type_instance(self.loc,
                                                 "complex") in keys and get_builtin_python_type_instance(
                        self.loc, "str") in keys)

        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")

        tuple_obj = get_builtin_python_type(self.loc, "tuple")
        tuple1 = invoke(self.loc, tuple_obj, list_)

        list_param2 = get_builtin_python_type_instance(self.loc, "list")
        self.type_store.set_type_of(self.loc, "list_param2", list_param2)

        append = self.type_store.get_type_of_member(self.loc, list_param2, "append")
        invoke(self.loc, append, tuple1)
        obj_instance = invoke(self.loc, obj, list_param2)

        keys = obj_instance.keys()
        values = obj_instance.values()
        self.assertTrue(isinstance(keys[0], int))
        self.assertTrue(isinstance(values[0], int))

        obj_instance2 = invoke(self.loc, obj, obj_instance)
        compare_types(obj_instance2, dict)
        keys = obj_instance2.keys()
        values = obj_instance2.values()

        self.assertTrue(isinstance(keys[0], int))
        self.assertTrue(isinstance(values[0], int))

        obj_instance = invoke(self.loc, obj, self.str_obj, self.int_obj)
        assert_if_not_error(obj_instance)

    def test_setattr(self):
        obj = get_builtin_python_type_instance(self.loc, "setattr")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj, self.str_obj)
        assert_if_not_error(obj_instance)

        class Foo:
            pass

        foo_class = Foo
        foo_instance = Foo()

        att_str = get_builtin_python_type_instance(self.loc, "str", value="att")
        func_str = get_builtin_python_type_instance(self.loc, "str", value="func")
        ne_str = get_builtin_python_type_instance(self.loc, "str", value="ne")

        val = get_builtin_python_type_instance(self.loc, "int")

        obj_instance = invoke(self.loc, obj, foo_class, att_str, val)
        compare_types(type(obj_instance), types.NoneType)

        obj_instance = invoke(self.loc, obj, foo_class, func_str, val)
        compare_types(type(obj_instance), types.NoneType)

        obj = get_builtin_python_type_instance(self.loc, "hasattr")

        obj_instance = invoke(self.loc, obj, foo_class, att_str)
        compare_types(type(obj_instance), bool)
        compare_types(obj_instance, True)

        obj_instance = invoke(self.loc, obj, foo_class, func_str)
        compare_types(type(obj_instance), bool)
        compare_types(obj_instance, True)

        obj_instance = invoke(self.loc, obj, foo_class, ne_str)
        compare_types(type(obj_instance), bool)
        compare_types(obj_instance, False)

        obj_instance = invoke(self.loc, obj, foo_instance, att_str)
        compare_types(type(obj_instance), bool)
        compare_types(obj_instance, True)

        obj_instance = invoke(self.loc, obj, foo_instance, func_str)
        compare_types(type(obj_instance), bool)
        compare_types(obj_instance, True)

        obj_instance = invoke(self.loc, obj, foo_instance, ne_str)
        compare_types(type(obj_instance), bool)
        compare_types(obj_instance, False)

    def test_license(self):
        obj = get_builtin_python_type_instance(self.loc, "license")
        assert_equal_type_name(type(obj), "_Printer")
        obj_instance = invoke(self.loc, obj)
        compare_types(type(obj_instance), types.NoneType)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

    def test_classmethod(self):
        obj = get_builtin_python_type(self.loc, "classmethod")
        compare_types(obj, classmethod)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(type(obj_instance), classmethod)

        class Foo:
            def __oct__(self, localization):
                return get_builtin_python_type_instance(localization, "int")

        hex_instance = Foo()
        obj_instance = invoke(self.loc, obj, hex_instance)
        compare_types(type(obj_instance), classmethod)

    def test_raw_input(self):
        obj = get_builtin_python_type_instance(self.loc, "input")
        assert_equal_type_name(obj, "input")
        obj_instance = invoke(self.loc, obj)
        compare_types(obj_instance, DynamicType)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(obj_instance, DynamicType)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(obj_instance, DynamicType)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

    def test_bytes(self):
        obj = get_builtin_python_type(self.loc, "bytes")
        compare_types(obj, bytes)
        obj_instance = invoke(self.loc, obj)
        compare_types(type(obj_instance), str)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(type(obj_instance), str)

        class Foo:
            def __str__(self, localization):
                return get_builtin_python_type_instance(localization, "int")

        oct_instance = Foo()
        obj_instance = invoke(self.loc, obj, oct_instance)
        assert_if_not_error(obj_instance)

    def test_iter(self):
        obj = get_builtin_python_type_instance(self.loc, "iter")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        compare_types(obj_instance, ExtraTypeDefinitions.iterator)
        compare_types(type(get_elements_type(obj_instance)), str)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list2")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(obj_instance, ExtraTypeDefinitions.listiterator)
        compare_types(get_elements_type(obj_instance), [complex(), str()])

        obj_instance = invoke(self.loc, obj, self.str_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(obj_instance, ExtraTypeDefinitions.listiterator)
        compare_types(type(get_elements_type(obj_instance)), int)

        list_ = self.type_store.get_type_of(Localization(__file__), "mixed_type_list")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(obj_instance, ExtraTypeDefinitions.listiterator)
        compare_types(get_elements_type(obj_instance), [int(), str()])

        class Foo:
            def __call__(self, localization, *args, **kwargs):
                return float()

        obj_instance = invoke(self.loc, obj, Foo(), int())
        compare_types(obj_instance, ExtraTypeDefinitions.callable_iterator)
        compare_types(type(get_elements_type(obj_instance)), float)

    def test_compile(self):
        obj = get_builtin_python_type_instance(self.loc, "compile")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj, self.str_obj, self.str_obj)
        compare_types(type(obj_instance), DynamicType)

        obj_instance = invoke(self.loc, obj, self.str_obj, self.str_obj, self.str_obj, self.int_obj, self.int_obj)
        compare_types(type(obj_instance), DynamicType)

    def test_reload(self):
        obj = get_builtin_python_type_instance(self.loc, "reload")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        assert_if_not_error(obj_instance)

        dest = Context(self.type_store, "func")
        import_from_module(self.loc, "time", None, dest)
        module_ = dest.get_type_of(self.loc, "time")

        obj_instance = invoke(self.loc, obj, module_)
        compare_types(obj_instance, module_)
        dest.close_function_context()

    def test_range(self):
        obj = get_builtin_python_type_instance(self.loc, "range")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.float_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(obj_instance, list)
        compare_types(type(get_elements_type(obj_instance)), int)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        compare_types(obj_instance, list)
        compare_types(type(get_elements_type(obj_instance)), int)

        obj_instance = invoke(self.loc, obj, self.float_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.str_obj)
        assert_if_not_error(obj_instance)

        class Foo:
            def __trunc__(self, localization):
                return get_builtin_python_type_instance(localization, "int")

        class Foo2:
            def __trunc__(self, localization):
                return get_builtin_python_type_instance(localization, "list")

        trunc_instance = Foo()
        wrong_instance = Foo2()

        obj_instance = invoke(self.loc, obj, trunc_instance)
        compare_types(obj_instance, list)
        compare_types(type(get_elements_type(obj_instance)), int)

        obj_instance = invoke(self.loc, obj, trunc_instance, self.int_obj)
        compare_types(obj_instance, list)
        compare_types(type(get_elements_type(obj_instance)), int)

        obj_instance = invoke(self.loc, obj, trunc_instance, trunc_instance)
        compare_types(obj_instance, list)
        compare_types(type(get_elements_type(obj_instance)), int)

        obj_instance = invoke(self.loc, obj, wrong_instance)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, wrong_instance, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, wrong_instance, trunc_instance)
        assert_if_not_error(obj_instance)

    def test_staticmethod(self):
        obj = get_builtin_python_type(self.loc, "staticmethod")
        compare_types(obj, staticmethod)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(type(obj_instance), staticmethod)

        class Foo:
            def __oct__(self, localization):
                return get_builtin_python_type_instance(localization, "int")

        hex_instance = Foo()
        obj_instance = invoke(self.loc, obj, hex_instance)
        compare_types(type(obj_instance), staticmethod)

    def test_str(self):
        obj = get_builtin_python_type(self.loc, "str")
        compare_types(obj, str)
        obj_instance = invoke(self.loc, obj)
        compare_types(type(obj_instance), str)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(type(obj_instance), str)

        class Foo:
            def __str__(self, localization):
                return get_builtin_python_type_instance(localization, "str")

        class Foo2:
            def __str__(self, localization):
                return get_builtin_python_type_instance(localization, "int")

        list_ = self.type_store.get_type_of(Localization(__file__), "int_list")
        obj_instance = invoke(self.loc, obj, list_)
        compare_types(type(obj_instance), str)

        instance = Foo()
        obj_instance = invoke(self.loc, obj, instance)
        compare_types(type(obj_instance), str)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        compare_types(type(obj_instance), str)

        instance = Foo2()
        obj_instance = invoke(self.loc, obj, instance)
        assert_if_not_error(obj_instance)

    def test_complex(self):
        obj = get_builtin_python_type(self.loc, "complex")
        compare_types(obj, types.ComplexType)
        obj_instance = invoke(self.loc, obj)
        compare_types(type(obj_instance), complex)

        obj_instance = invoke(self.loc, obj, self.float_obj)
        compare_types(type(obj_instance), complex)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        compare_types(type(obj_instance), complex)

        obj_instance = invoke(self.loc, obj, self.float_obj, self.int_obj)
        compare_types(type(obj_instance), complex)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.str_obj)
        assert_if_not_error(obj_instance)

        class Foo:
            def __float__(self, localization):
                return get_builtin_python_type_instance(localization, "float")

        class Foo2:
            def __float__(self, localization):
                return get_builtin_python_type_instance(localization, "list")

        float_instance = Foo()
        wrong_instance = Foo2()

        obj_instance = invoke(self.loc, obj, float_instance)
        compare_types(type(obj_instance), complex)

        obj_instance = invoke(self.loc, obj, self.int_obj, float_instance)
        compare_types(type(obj_instance), complex)

        obj_instance = invoke(self.loc, obj, float_instance, float_instance)
        compare_types(type(obj_instance), complex)

        obj_instance = invoke(self.loc, obj, wrong_instance)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, wrong_instance)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, wrong_instance, wrong_instance)
        assert_if_not_error(obj_instance)

    def test_property(self):
        obj = get_builtin_python_type_instance(self.loc, "property")
        compare_types(type(obj), property)
        obj = property
        obj_instance = invoke(self.loc, obj)
        compare_types(type(obj_instance), property)

        def get(self, localization):
            return get_builtin_python_type_instance(localization, "int")

        def set(self, localization, *args):
            return get_builtin_python_type_instance(localization, "int")

        def del_(self, localization):
            return get_builtin_python_type_instance(localization, "int")

        fget = get
        fset = set
        fdel = del_
        fdoc = get_builtin_python_type_instance(self.loc, "str")

        obj_instance = invoke(self.loc, obj, fget)
        compare_types(type(obj_instance), property)
        compare_types(type(get_member(self.loc, obj_instance, "fget")), types.FunctionType)

        obj_instance = invoke(self.loc, obj, fget, fset)
        compare_types(type(obj_instance), property)
        compare_types(type(get_member(self.loc, obj_instance, "fget")), types.FunctionType)
        compare_types(type(get_member(self.loc, obj_instance, "fset")), types.FunctionType)

        obj_instance = invoke(self.loc, obj, fget, fset, fdel)
        compare_types(type(obj_instance), property)
        compare_types(type(get_member(self.loc, obj_instance, "fget")), types.FunctionType)
        compare_types(type(get_member(self.loc, obj_instance, "fset")), types.FunctionType)
        compare_types(type(get_member(self.loc, obj_instance, "fdel")), types.FunctionType)

        obj_instance = invoke(self.loc, obj, fget, fset, fdel, fdoc)
        compare_types(type(obj_instance), property)
        compare_types(type(get_member(self.loc, obj_instance, "fget")), types.FunctionType)
        compare_types(type(get_member(self.loc, obj_instance, "fset")), types.FunctionType)
        compare_types(type(get_member(self.loc, obj_instance, "fdel")), types.FunctionType)
        compare_types(type(get_member(self.loc, obj_instance, "__doc__")), str)

    def test_round(self):
        obj = get_builtin_python_type_instance(self.loc, "round")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.float_obj)
        compare_types(type(obj_instance), float)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        compare_types(type(obj_instance), float)

        obj_instance = invoke(self.loc, obj, self.float_obj, self.int_obj)
        compare_types(type(obj_instance), float)

        obj_instance = invoke(self.loc, obj, self.str_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.str_obj)
        assert_if_not_error(obj_instance)

        class Foo:
            def __float__(self, localization):
                return get_builtin_python_type_instance(localization, "float")

        class Foo2:
            def __float__(self, localization):
                return get_builtin_python_type_instance(localization, "list")

        class Foo3:
            def __index__(self, localization):
                return get_builtin_python_type_instance(localization, "int")

        float_instance = Foo()
        wrong_instance = Foo2()
        index_instance = Foo3()

        obj_instance = invoke(self.loc, obj, float_instance)
        compare_types(type(obj_instance), float)

        obj_instance = invoke(self.loc, obj, float_instance, self.int_obj)
        compare_types(type(obj_instance), float)

        obj_instance = invoke(self.loc, obj, float_instance, index_instance)
        compare_types(type(obj_instance), float)

        obj_instance = invoke(self.loc, obj, wrong_instance)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, wrong_instance, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, wrong_instance, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

    def test_dir(self):
        obj = get_builtin_python_type_instance(self.loc, "dir")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        compare_types(obj_instance, list)
        compare_types(type(get_elements_type(obj_instance)), str)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        compare_types(obj_instance, list)
        compare_types(type(get_elements_type(obj_instance)), str)

        class Foo:
            def __oct__(self, localization):
                return get_builtin_python_type_instance(localization, "int")

        hex_instance = Foo()
        obj_instance = invoke(self.loc, obj, hex_instance)
        compare_types(obj_instance, list)
        compare_types(type(get_elements_type(obj_instance)), str)

    def test_cmp(self):
        obj = get_builtin_python_type_instance(self.loc, "cmp")
        compare_types(type(obj), types.BuiltinFunctionType)
        obj_instance = invoke(self.loc, obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj)
        assert_if_not_error(obj_instance)

        obj_instance = invoke(self.loc, obj, self.int_obj, self.str_obj)
        compare_types(type(obj_instance), int)
