#!/usr/bin/env python
# -*- coding: utf-8 -*-
import types
import unittest

from stypy.contexts.context import Context
from stypy.type_inference_programs import stypy_interface
from stypy.types import known_python_types
from stypy.types.type_intercession import supports_intercession
from testing.model_testing.model_testing_common import *


class TestPythonTypeStandard(unittest.TestCase):
    def setUp(self):
        self.loc = Localization(__file__)
        self.builtins_ = stypy_interface.builtins_module
        self.type_store = Context(None, __file__)

    # ########################################## TESTS ###########################################

    def test_load_python_module(self):
        import math
        stypy_interface.import_module(self.loc, "math", math, self.type_store)
        math_module = self.type_store.get_type_of(self.loc, "math")
        compare_types(type(math_module), types.ModuleType)
        assert_if_not_error(invoke(self.loc, math_module))

    def test_load_builtin_type(self):
        int_ = self.type_store.get_type_of_member(self.loc, self.builtins_, 'int')
        self.assertFalse(supports_intercession(int_))
        compare_types(int_, types.IntType)
        temp = invoke(self.loc, int_)
        compare_types(type(temp), types.IntType)

        list_ = self.type_store.get_type_of_member(self.loc, self.builtins_, 'list')
        self.assertFalse(supports_intercession(list_))
        compare_types(list_, types.ListType)
        temp = invoke(self.loc, list_)
        compare_types(temp, types.ListType)

    def test_load_builtin_function(self):
        range_ = self.type_store.get_type_of_member(self.loc, self.builtins_, 'range')
        self.assertFalse(supports_intercession(range_))
        compare_types(type(range_), types.BuiltinFunctionType)
        temp = invoke(self.loc, range_)
        assert_if_not_error(temp)
        temp = invoke(self.loc, range_, int())

        compare_types(temp, types.ListType)
        temp = invoke(self.loc, range_, int(), int())
        compare_types(temp, types.ListType)
        temp = invoke(self.loc, range_, int(), int(), int())

        compare_types(temp, types.ListType)
        temp = invoke(self.loc, range_, int(), int(), int(), int())
        assert_if_not_error(temp)

    def test_load_builtin_attribute(self):
        doc = self.type_store.get_type_of_member(self.loc, self.builtins_, '__doc__')
        self.assertFalse(supports_intercession(doc))
        compare_types(type(doc), types.StringType)
        assert_if_not_error(invoke(self.loc, doc))

    def test_load_nested_members(self):
        doc = self.type_store.get_type_of_member(self.loc, self.builtins_, '__doc__')
        self.assertFalse(supports_intercession(doc))
        compare_types(type(doc), types.StringType)
        assert_if_not_error(invoke(self.loc, doc))

        title = self.type_store.get_type_of_member(self.loc, doc, 'title')
        self.assertFalse(supports_intercession(title))
        assert_equal_type_name(type(title), "builtin_function_or_method")
        compare_types(type(invoke(self.loc, title)), types.StringType)

        str_ = self.type_store.get_type_of_member(self.loc, title, '__str__')
        self.assertFalse(supports_intercession(str_))
        assert_equal_type_name(type(str_), "method-wrapper")
        compare_types(type(invoke(self.loc, str_)), types.StringType)

    def test_load_builtin_class(self):
        a_e = self.type_store.get_type_of_member(self.loc, self.builtins_, 'ArithmeticError')
        self.assertFalse(supports_intercession(a_e))
        compare_types(a_e, ArithmeticError)
        temp = invoke(self.loc, a_e)
        compare_types(type(temp), ArithmeticError)
        self.assertTrue(supports_intercession(temp))

    def test_load_user_defined_class(self):
        u_m = known_python_types
        self.assertTrue(supports_intercession(u_m))
        compare_types(type(u_m), types.ModuleType)
        u_d = self.type_store.get_type_of_member(self.loc, u_m, 'ExtraTypeDefinitions')
        self.assertTrue(supports_intercession(u_d))
        compare_types(u_d, known_python_types.ExtraTypeDefinitions)

    def test_builtin_class_members(self):
        a_e = self.type_store.get_type_of_member(self.loc, self.builtins_, 'ArithmeticError')
        self.assertFalse(supports_intercession(a_e))
        compare_types(a_e, ArithmeticError)
        temp = invoke(None, a_e)
        compare_types(type(temp), ArithmeticError)
        self.assertTrue(supports_intercession(temp))

        unic = self.type_store.get_type_of_member(self.loc, a_e, '__unicode__')
        assert_equal_type_name(type(unic), "method_descriptor")
        compare_types(type(invoke(self.loc, unic, temp)), types.UnicodeType)

        doc = self.type_store.get_type_of_member(self.loc, a_e, '__doc__')
        assert_if_not_error(invoke(None, doc))

    def test_user_defined_class_class_members(self):
        u_m = known_python_types
        self.assertTrue(supports_intercession(u_m))
        compare_types(type(u_m), types.ModuleType)
        u_d = self.type_store.get_type_of_member(self.loc, u_m, 'ExtraTypeDefinitions')
        self.assertTrue(supports_intercession(u_d))
        compare_types(u_d, known_python_types.ExtraTypeDefinitions)

        setit = self.type_store.get_type_of_member(self.loc, u_d, 'setiterator')
        assert_equal_type_name(setit, "setiterator")
        assert_if_not_error(invoke(self.loc, setit))

        doc = self.type_store.get_type_of_member(self.loc, u_d, '__doc__')
        assert_if_not_error(invoke(self.loc, doc))
