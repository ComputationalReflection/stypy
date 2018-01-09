#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from stypy.contexts.context import Context
from stypy.errors.type_error import StypyTypeError
from stypy.errors.type_warning import TypeWarning
from stypy.reporting.localization import Localization
from stypy.type_inference_programs.stypy_interface import invoke
from stypy.types.union_type import UnionType
from testing.model_testing.model_testing_common import compare_types


class TestPythonUnionTypeInvoke(unittest.TestCase):
    def setUp(self):
        self.loc = Localization(__file__)
        StypyTypeError.reset_error_msgs()
        TypeWarning.reset_warning_msgs()
        self.type_store = Context(None, __file__)

    def test_simple_union_invoke(self):
        union = UnionType.add(int(), str())

        islower = self.type_store.get_type_of_member(self.loc, union, "islower")
        res = invoke(self.loc, islower)

        self.assertTrue(len(TypeWarning.get_warning_msgs()) == 1)
        compare_types(res, False)

    def test_union_invoke_return_types(self):
        class Foo:
            def method(self, localization):
                return True

            def method_2(self, localization):
                return str()

        class Foo2:
            def method(self, localization):
                return False

            def method_2(self, localization):
                return int()

        # Access a member that can be provided only by some of the types in the union
        union = UnionType.add(Foo(),
                              Foo2())

        method = self.type_store.get_type_of_member(self.loc, union, "method_2")
        res = invoke(self.loc, method)
        self.assertTrue(len(TypeWarning.get_warning_msgs()) == 0)

        compare_types(res, [str(), int()])

    def test_call_simple_union_type_param(self):
        union_param = UnionType.create_from_type_list([int(),
                                                             str(),
                                                             float()])

        import math
        math_module = math
        sqrt_func = self.type_store.get_type_of_member(self.loc, math_module, "sqrt")
        ret = invoke(self.loc, sqrt_func, union_param)

        self.assertTrue(len(TypeWarning.get_warning_msgs()) == 1)

        compare_types(ret, float())
