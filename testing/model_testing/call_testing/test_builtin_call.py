#!/usr/bin/env python
# -*- coding: utf-8 -*-
import types
import unittest

from stypy.contexts.context import Context
from stypy.errors.type_error import StypyTypeError
from stypy.errors.type_warning import TypeWarning
from stypy.reporting.localization import Localization
from stypy.ssa.ssa_context import SSAContext
from stypy.type_inference_programs.stypy_interface import invoke, get_builtin_python_type_instance
from testing.model_testing.model_testing_common import compare_types
from stypy.types.undefined_type import UndefinedType
from stypy.types.union_type import UnionType
from stypy.invokation.type_rules.type_groups.type_groups import DynamicType
from stypy.type_inference_programs.stypy_interface import import_module


class TestCallBuiltins(unittest.TestCase):
    def setUp(self):
        StypyTypeError.reset_error_msgs()
        parent_context = Context(None, __file__)
        self.context = SSAContext(parent_context, "func")
        self.localization = Localization(__file__, 1, 1)
        Localization.set_current(self.localization)

    def test_right_call_builtin_module_function_simple_types(self):
        func = self.context.get_type_of(self.localization, "pow")
        compare_types(type(func), types.BuiltinFunctionType)

        ret = invoke(self.localization, func, int(), int())
        compare_types(type(ret), int)

    def test_wrong_call_builtin_module_function_simple_types(self):
        func = self.context.get_type_of(self.localization, "abs")
        compare_types(type(func), types.BuiltinFunctionType)

        # Wrong param type
        ret = invoke(self.localization, func, [list()])
        compare_types(type(ret), StypyTypeError)

        # Wrong param number
        ret = invoke(self.localization, func, [int(), int()])
        compare_types(type(ret), StypyTypeError)

    def test_right_call_builtin_module_class(self):
        type_ = self.context.get_type_of(self.localization, "list")
        compare_types(type_, list)

        ret = invoke(self.localization, type_, [])
        compare_types(ret, list)

    def test_wrong_call_builtin_module_class(self):
        type_ = self.context.get_type_of(self.localization, "list")
        compare_types(type_, list)

        # Wrong type
        ret = invoke(self.localization, type_, int())
        compare_types(type(ret), StypyTypeError)

        # Wrong number of parameters
        ret = invoke(self.localization, type_, str(), int())
        compare_types(type(ret), StypyTypeError)

    def test_right_call_builtin_module_class_method(self):
        type_ = get_builtin_python_type_instance(self.localization, "list")
        compare_types(type_, list)

        method = self.context.get_type_of_member(self.localization, type_, "append")
        compare_types(type(method), types.BuiltinFunctionType)

        ret = invoke(self.localization, method, int())
        compare_types(type(ret), types.NoneType)

        method = self.context.get_type_of_member(self.localization, type_, "__getitem__")
        ret = invoke(self.localization, method, int())
        compare_types(type(ret), int)

    def test_wrong_call_builtin_module_class_method(self):
        type_ = self.context.get_type_of(self.localization, "list")
        compare_types(type_, list)

        method = self.context.get_type_of_member(self.localization, type_, "__getitem__")
        ret = invoke(self.localization, method, [list()])
        compare_types(type(ret), StypyTypeError)

        method = self.context.get_type_of_member(self.localization, type_, "__getitem__")
        ret = invoke(self.localization, method, [int(), int()])
        compare_types(type(ret), StypyTypeError)

    def test_call_varargs(self):
        func = self.context.get_type_of(self.localization, "min")
        compare_types(type(func), types.BuiltinFunctionType)

        ret = invoke(self.localization, func, int(), int(), float(), list())
        compare_types(ret, [int(), float(), list()])

    def test_call_undefined_args(self):
        obj = self.context.get_type_of(self.localization, "file")
        compare_types(obj, types.FileType)

        ret = invoke(self.localization, obj, str(), UndefinedType)
        compare_types(type(ret), StypyTypeError)

        # Wrong param
        ret = invoke(self.localization, obj, int(), UndefinedType)
        compare_types(type(ret), StypyTypeError)

        # Wrong arity
        ret = invoke(self.localization, obj, int(), UndefinedType, int(), int(), int())
        compare_types(type(ret), StypyTypeError)

    def test_call_dynamic_args(self):
        obj = self.context.get_type_of(self.localization, "file")
        compare_types(obj, types.FileType)

        ret = invoke(self.localization, obj, str(), DynamicType())
        compare_types(type(ret), DynamicType)

        # Wrong param
        ret = invoke(self.localization, obj, int(), DynamicType())
        compare_types(type(ret), StypyTypeError)

        # Wrong arity
        ret = invoke(self.localization, obj, int(), DynamicType(), int(), int(), int())
        compare_types(type(ret), StypyTypeError)

    def test_call_dependent_types(self):
        class Foo:
            def __trunc__(self, localization):
                return int()

        obj = self.context.get_type_of(self.localization, "xrange")
        compare_types(obj, xrange)

        ret = invoke(self.localization, obj, Foo(), Foo())
        compare_types(ret, xrange)

    def test_call_union_type(self):
        abs_func = self.context.get_type_of(self.localization, 'abs')
        union_valid = UnionType.create_from_type_list([True, complex()])
        union_mixed = UnionType.create_from_type_list([int(), list()])
        union_wrong = UnionType.create_from_type_list([dict(), list()])

        # Two valid types
        ret = invoke(self.localization, abs_func, union_valid)
        compare_types(ret, [int(), float()])

        # Two types: one valid, one invalid
        ret = invoke(self.localization, abs_func, union_mixed)
        compare_types(type(ret), int)
        assert len(TypeWarning.get_warning_msgs()) == 1

        # Two invalid types
        ret = invoke(self.localization, abs_func, union_wrong)
        compare_types(type(ret), StypyTypeError)
        assert len(StypyTypeError.get_error_msgs()) == 1

    def test_call_type_objects(self):
        # Test that calls that admit type instances (3, "hi"...) do not admit passing type objects directly (int, str)
        abs_func = self.context.get_type_of(self.localization, 'abs')
        ret = invoke(self.localization, abs_func, int)
        compare_types(type(ret), StypyTypeError)

    def test_call_default_handler(self):
        import zlib
        import_module(self.localization, 'zlib', zlib, self.context)
        zlib_module = self.context.get_type_of(self.localization, 'zlib')
        compress_func = self.context.get_type_of_member(self.localization, zlib_module, 'compress')
        ret = invoke(self.localization, compress_func, str())
        compare_types(type(ret), str)
