#!/usr/bin/env python
# -*- coding: utf-8 -*-
import types
import unittest

from stypy.contexts.context import Context
from stypy.errors.advice import Advice
from stypy.errors.type_error import StypyTypeError
from stypy.errors.type_warning import TypeWarning
from stypy.reporting.localization import Localization
from testing.model_testing.model_testing_common import compare_types, assert_if_not_error


class TestPythonTypeStoreContexts(unittest.TestCase):
    def setUp(self):
        self.loc = Localization(__file__)
        self.type_store = Context(None, __file__)
        StypyTypeError.reset_error_msgs()
        TypeWarning.reset_warning_msgs()
        Advice.reset_advice_msgs()

    # Use builtins
    def test_load_builtin(self):
        res = self.type_store.get_type_of(self.loc, "list")
        compare_types(res, types.ListType)

        res = self.type_store.get_type_of(self.loc, "bool")
        compare_types(res, types.BooleanType)

        res = self.type_store.get_type_of(self.loc, "None")
        compare_types(res, types.NoneType)

        res = self.type_store.get_type_of(self.loc, "dict")
        compare_types(res, types.DictType)

        res = self.type_store.get_type_of(self.loc, "ArithmeticError")
        compare_types(res, ArithmeticError)

        res = self.type_store.get_type_of(self.loc, "range")
        compare_types(type(res), types.BuiltinFunctionType)

    # Use same-context defined variables (attributes, functions, classes, objects, modules)
    def test_load_var_same_context(self):
        self.type_store.set_type_of(self.loc, "var", int())
        compare_types(type(self.type_store.get_type_of(self.loc, "var")), types.IntType)

        def fun():
            pass

        self.type_store.set_type_of(self.loc, "func", fun)
        compare_types(type(self.type_store.get_type_of(self.loc, "func")), types.FunctionType)

        class Foo:
            pass

        self.type_store.set_type_of(self.loc, "Foo", Foo)
        compare_types(type(self.type_store.get_type_of(self.loc, "Foo")), types.ClassType)

        self.type_store.set_type_of(self.loc, "Foo_instance", Foo())
        compare_types(type(self.type_store.get_type_of(self.loc, "Foo_instance")), types.InstanceType)

        import math

        self.type_store.set_type_of(self.loc, "math", math)
        compare_types(type(self.type_store.get_type_of(self.loc, "math")), types.ModuleType)

    # Use parent context defined variables
    def test_load_var_parent_context(self):
        self.type_store.set_type_of(self.loc, "var", int())

        def fun():
            pass

        self.type_store.set_type_of(self.loc, "func", fun)

        class Foo:
            pass

        self.type_store.set_type_of(self.loc, "Foo", Foo)

        self.type_store.set_type_of(self.loc, "Foo_instance", Foo())

        import math

        self.type_store.set_type_of(self.loc, "math", math)

        self.type_store = self.type_store.open_function_context("new_func")

        compare_types(type(self.type_store.get_type_of(self.loc, "var")), types.IntType)
        compare_types(type(self.type_store.get_type_of(self.loc, "func")), types.FunctionType)
        compare_types(type(self.type_store.get_type_of(self.loc, "Foo")), types.ClassType)
        compare_types(type(self.type_store.get_type_of(self.loc, "Foo_instance")), types.InstanceType)
        compare_types(type(self.type_store.get_type_of(self.loc, "math")), types.ModuleType)

        self.type_store.set_type_of(self.loc, "other_var", float())

        self.type_store = self.type_store.open_function_context("nested_func")

        compare_types(type(self.type_store.get_type_of(self.loc, "var")), types.IntType)
        compare_types(type(self.type_store.get_type_of(self.loc, "func")), types.FunctionType)
        compare_types(type(self.type_store.get_type_of(self.loc, "Foo")), types.ClassType)
        compare_types(type(self.type_store.get_type_of(self.loc, "Foo_instance")), types.InstanceType)
        compare_types(type(self.type_store.get_type_of(self.loc, "math")), types.ModuleType)

        self.type_store.set_type_of(self.loc, "even_other_var", list())
        compare_types(type(self.type_store.get_type_of(self.loc, "other_var")), types.FloatType)

        self.type_store = self.type_store.open_function_context("other_nested_func")

        compare_types(type(self.type_store.get_type_of(self.loc, "var")), types.IntType)
        compare_types(type(self.type_store.get_type_of(self.loc, "func")), types.FunctionType)
        compare_types(type(self.type_store.get_type_of(self.loc, "Foo")), types.ClassType)
        compare_types(type(self.type_store.get_type_of(self.loc, "Foo_instance")), types.InstanceType)
        compare_types(type(self.type_store.get_type_of(self.loc, "math")), types.ModuleType)

        compare_types(type(self.type_store.get_type_of(self.loc, "even_other_var")), types.ListType)

        self.type_store = self.type_store.close_function_context()  # End other_nested_func
        self.type_store = self.type_store.close_function_context()  # End nested_func

        res = self.type_store.get_type_of(self.loc, "even_other_var")
        assert_if_not_error(res)

        self.type_store = self.type_store.close_function_context()  # End new_func

        res = self.type_store.get_type_of(self.loc, "other_var")
        assert_if_not_error(res)

    def test_use_global_in_global_context(self):
        num_warnings_before = len(TypeWarning.get_warning_msgs())
        self.type_store.declare_global(Localization(__file__, 1, 1), "global_var")
        self.type_store.declare_global(Localization(__file__, 2, 2), "global_func")

        self.type_store.set_type_of(Localization(__file__, 3, 3), "global_var", int)
        self.type_store.set_type_of(Localization(__file__, 4, 4), "global_func", types.FunctionType)

        compare_types(self.type_store.get_type_of(Localization(__file__, 5, 5), "global_var"), types.IntType)
        compare_types(self.type_store.get_type_of(Localization(__file__, 6, 6), "global_func"), types.FunctionType)
        num_warnings_after = len(TypeWarning.get_warning_msgs())

        assert len(Advice.get_advice_msgs()) == 2

        assert num_warnings_after == num_warnings_before

    def test_use_global_in_global_context_after_usage(self):
        self.type_store.set_type_of(self.loc, "global_var", int)
        self.type_store.set_type_of(self.loc, "global_func", types.FunctionType)
        self.type_store.declare_global(Localization(__file__, 1, 1), "global_var")
        self.type_store.declare_global(Localization(__file__, 2, 2), "global_func")

        compare_types(self.type_store.get_type_of(Localization(__file__, 3, 3), "global_var"), types.IntType)
        compare_types(self.type_store.get_type_of(Localization(__file__, 4, 4), "global_func"), types.FunctionType)

        assert len(Advice.get_advice_msgs()) == 2

    # Use globals
    def test_existing_global_var_usage(self):
        self.type_store.set_type_of(self.loc, "global_var", int)
        self.type_store.set_type_of(self.loc, "global_func", types.FunctionType)

        self.type_store = self.type_store.open_function_context("nested_func")
        self.type_store.declare_global(self.loc, "global_var")
        self.type_store.declare_global(self.loc, "global_func")

        num_warnings_before = len(TypeWarning.get_warning_msgs())
        compare_types(self.type_store.get_type_of(self.loc, "global_var"), types.IntType)
        compare_types(self.type_store.get_type_of(self.loc, "global_func"), types.FunctionType)
        num_warnings_after = len(TypeWarning.get_warning_msgs())

        # TypeWarning.print_warning_msgs()
        assert num_warnings_after == num_warnings_before

        res = self.type_store.set_type_of(self.loc, "global_var", str)
        res = self.type_store.set_type_of(self.loc, "global_func", types.NoneType)
        assert res is None
        self.type_store = self.type_store.close_function_context()
        compare_types(self.type_store.get_type_of(self.loc, "global_var"), types.StringType)
        compare_types(self.type_store.get_type_of(self.loc, "global_func"), types.NoneType)

    # Use a global variable without marking it as global (python error)
    def test_global_var_wrong_usage(self):
        self.type_store.set_type_of(self.loc, "global_var", int)
        self.type_store.set_type_of(self.loc, "global_func", types.FunctionType)

        self.type_store = self.type_store.open_function_context("nested_func")
        num_warnings_before = len(TypeWarning.get_warning_msgs())
        compare_types(self.type_store.get_type_of(self.loc, "global_var"), types.IntType)
        compare_types(self.type_store.get_type_of(self.loc, "global_func"), types.FunctionType)
        num_warnings_after = len(TypeWarning.get_warning_msgs())

        assert num_warnings_after == num_warnings_before + 2

        self.type_store.set_type_of(self.loc, "global_var", float)
        assert len(StypyTypeError.get_error_msgs()) == 1
        self.type_store.set_type_of(self.loc, "global_func", types.NoneType)
        assert len(StypyTypeError.get_error_msgs()) == 2

        self.type_store = self.type_store.close_function_context()

        num_warnings_after = len(TypeWarning.get_warning_msgs())
        assert num_warnings_after == num_warnings_before

    def test_wrong_global_keyword_in_func(self):
        self.type_store.declare_global(Localization(__file__, 1, 1), "global_var")
        self.type_store.declare_global(Localization(__file__, 2, 2), "global_func")

        self.type_store.set_type_of(self.loc, "global_var", int)
        self.type_store.set_type_of(self.loc, "global_func", types.FunctionType)

        self.type_store = self.type_store.open_function_context("nested_func")
        num_warnings_before = len(TypeWarning.get_warning_msgs())
        compare_types(self.type_store.get_type_of(self.loc, "global_var"), types.IntType)
        compare_types(self.type_store.get_type_of(self.loc, "global_func"), types.FunctionType)
        num_warnings_after = len(TypeWarning.get_warning_msgs())

        # TypeWarning.print_warning_msgs()
        assert num_warnings_after == num_warnings_before + 2

        self.type_store.declare_global(Localization(__file__, 3, 3), "global_var")
        assert len(Advice.get_advice_msgs()) == 3

        self.type_store.declare_global(Localization(__file__, 4, 4), "global_func")
        assert len(Advice.get_advice_msgs()) == 4

        self.type_store = self.type_store.close_function_context()

    def test_read_global_variable_in_func_no_global_keyword(self):
        self.type_store.declare_global(self.loc, "global_var")
        self.type_store.declare_global(self.loc, "global_func")

        self.type_store.set_type_of(self.loc, "global_var", int)
        self.type_store.set_type_of(self.loc, "global_func", types.FunctionType)

        self.type_store = self.type_store.open_function_context("nested_func")
        num_warnings_before = len(TypeWarning.get_warning_msgs())
        compare_types(self.type_store.get_type_of(self.loc, "global_var"), types.IntType)
        compare_types(self.type_store.get_type_of(self.loc, "global_func"), types.FunctionType)
        num_warnings_after = len(TypeWarning.get_warning_msgs())

        assert num_warnings_after == num_warnings_before + 2

        self.type_store = self.type_store.close_function_context()

    def test_unexisting_global_variable_in_func(self):
        self.type_store = self.type_store.open_function_context("nested_func")

        self.type_store.declare_global(self.loc, "non_existing")
        assert len(Advice.get_advice_msgs()) == 1

        res = self.type_store.get_type_of(self.loc, "non_existing")
        assert_if_not_error(res)

        self.type_store = self.type_store.close_function_context()

    def test_local_variable_turned_global_after(self):
        self.type_store.declare_global(Localization(__file__, 1, 1), "global_var")
        self.type_store.declare_global(Localization(__file__, 2, 2), "global_func")
        assert len(Advice.get_advice_msgs()) == 2

        self.type_store.set_type_of(self.loc, "global_var", int)
        self.type_store.set_type_of(self.loc, "global_func", types.FunctionType)

        self.type_store = self.type_store.open_function_context("nested_func")
        num_warnings_before = len(TypeWarning.get_warning_msgs())
        compare_types(self.type_store.get_type_of(self.loc, "global_var"), types.IntType)
        compare_types(self.type_store.get_type_of(self.loc, "global_func"), types.FunctionType)
        num_warnings_after = len(TypeWarning.get_warning_msgs())

        assert num_warnings_after == num_warnings_before + 2

        self.type_store.declare_global(Localization(__file__, 3, 3), "global_var")
        assert len(Advice.get_advice_msgs()) == 3

        self.type_store.declare_global(Localization(__file__, 4, 4), "global_func")
        assert len(Advice.get_advice_msgs()) == 4

        self.type_store.declare_global(Localization(__file__, 5, 5), "non_existing")
        assert len(Advice.get_advice_msgs()) == 5

        assert_if_not_error(self.type_store.get_type_of(self.loc, "non_existing"))
        self.type_store.set_type_of(Localization(__file__, 6, 6), "foo", int)
        self.type_store.declare_global(Localization(__file__, 7, 7), "foo")
        assert len(Advice.get_advice_msgs()) == 6

        self.type_store.set_type_of(Localization(__file__, 8, 8), "foo_func", types.FunctionType)
        self.type_store.declare_global(Localization(__file__, 8, 8), "foo_func")
        assert len(Advice.get_advice_msgs()) == 7

        compare_types(self.type_store.get_type_of(self.loc, "foo"), types.IntType)

        self.type_store = self.type_store.close_function_context()
        compare_types(self.type_store.get_type_of(self.loc, "foo"), types.IntType)
        compare_types(self.type_store.get_type_of(self.loc, "foo_func"), types.FunctionType)

    def test_local_variable_turned_global_before(self):
        self.type_store.declare_global(Localization(__file__, 1, 1), "global_var")
        self.type_store.set_type_of(Localization(__file__, 2, 2), "global_var", int)

        self.type_store = self.type_store.open_function_context("nested_func")
        num_warnings_before = len(TypeWarning.get_warning_msgs())
        compare_types(self.type_store.get_type_of(Localization(__file__, 3, 3), "global_var"), types.IntType)
        num_warnings_after = len(TypeWarning.get_warning_msgs())

        assert num_warnings_after == num_warnings_before + 1

        self.type_store.declare_global(Localization(__file__, 4, 4), "global_var")
        assert len(Advice.get_advice_msgs()) == 2

        self.type_store.declare_global(self.loc, "non_existing")

        assert_if_not_error(self.type_store.get_type_of(Localization(__file__, 5, 5), "non_existing"))

        self.type_store.declare_global(Localization(__file__, 6, 6), "foo")
        self.type_store.set_type_of(Localization(__file__, 7, 7), "foo", int)
        assert len(Advice.get_advice_msgs()) == 4

        self.type_store.declare_global(Localization(__file__, 8, 8), "foo_func")
        self.type_store.set_type_of(Localization(__file__, 9, 9), "foo_func", types.FunctionType)
        assert len(Advice.get_advice_msgs()) == 5

        compare_types(self.type_store.get_type_of(self.loc, "foo"), types.IntType)
        compare_types(self.type_store.get_type_of(self.loc, "foo_func"), types.FunctionType)
        self.type_store = self.type_store.close_function_context()

        compare_types(self.type_store.get_type_of(self.loc, "foo"), types.IntType)
        compare_types(self.type_store.get_type_of(self.loc, "foo_func"), types.FunctionType)

    def test_local_variable_turned_global_nested_context(self):
        self.type_store.declare_global(self.loc, "global_var")
        self.type_store.set_type_of(self.loc, "global_var", int)
        assert len(Advice.get_advice_msgs()) == 1

        self.type_store = self.type_store.open_function_context("nested_func")
        num_warnings_before = len(TypeWarning.get_warning_msgs())
        compare_types(self.type_store.get_type_of(Localization(__file__, 1, 1), "global_var"), types.IntType)
        num_warnings_after = len(TypeWarning.get_warning_msgs())

        assert num_warnings_after == num_warnings_before + 1

        self.type_store.declare_global(Localization(__file__, 2, 2), "global_var")
        assert len(Advice.get_advice_msgs()) == 2

        self.type_store.declare_global(Localization(__file__, 3, 3), "non_existing")
        assert len(Advice.get_advice_msgs()) == 3

        assert_if_not_error(self.type_store.get_type_of(self.loc, "non_existing"))

        self.type_store = self.type_store.open_function_context("other_func")

        self.type_store.set_type_of(Localization(__file__, 4, 4), "foo", int)
        self.type_store.declare_global(Localization(__file__, 5, 5), "foo")

        assert len(Advice.get_advice_msgs()) == 4

        self.type_store.declare_global(Localization(__file__, 6, 6), "foo_func")
        self.type_store.set_type_of(Localization(__file__, 6, 6), "foo_func", types.FunctionType)

        assert len(Advice.get_advice_msgs()) == 5

        self.type_store = self.type_store.close_function_context()

        compare_types(self.type_store.get_type_of(Localization(__file__, 1, 1), "foo"), types.IntType)
        self.type_store.set_type_of(Localization(__file__, 1, 1), "foo", types.FloatType)

        assert len(StypyTypeError.get_error_msgs()) == 2

        self.type_store = self.type_store.close_function_context()

        compare_types(self.type_store.get_type_of(self.loc, "foo"), types.IntType)
        compare_types(self.type_store.get_type_of(self.loc, "foo_func"), types.FunctionType)

    def test_local_variable_turned_global_nested_context_local_name_collision(self):
        self.type_store.declare_global(Localization(__file__, 1, 1), "global_var")
        self.type_store.set_type_of(Localization(__file__, 2, 2), "global_var", int)
        assert len(Advice.get_advice_msgs()) == 1

        self.type_store = self.type_store.open_function_context("nested_func")
        num_warnings_before = len(TypeWarning.get_warning_msgs())
        compare_types(self.type_store.get_type_of(Localization(__file__, 3, 3), "global_var"), types.IntType)
        num_warnings_after = len(TypeWarning.get_warning_msgs())

        assert num_warnings_after == num_warnings_before + 1

        self.type_store.declare_global(Localization(__file__, 4, 4), "global_var")
        assert len(Advice.get_advice_msgs()) == 2

        self.type_store.declare_global(Localization(__file__, 5, 5), "non_existing")
        assert len(Advice.get_advice_msgs()) == 3

        assert_if_not_error(self.type_store.get_type_of(Localization(__file__, 6, 6), "non_existing"))

        self.type_store = self.type_store.open_function_context("other_func")
        self.type_store.set_type_of(Localization(__file__, 7, 7), "foo", int)
        self.type_store.declare_global(Localization(__file__, 8, 8), "foo")
        assert len(Advice.get_advice_msgs()) == 4

        self.type_store.declare_global(Localization(__file__, 9, 9), "foo_func")
        self.type_store.set_type_of(Localization(__file__, 10, 10), "foo_func", types.FunctionType)
        assert len(Advice.get_advice_msgs()) == 5

        self.type_store = self.type_store.close_function_context()

        self.type_store.set_type_of(self.loc, "foo", types.FloatType)

        compare_types(self.type_store.get_type_of(self.loc, "foo"), types.FloatType)

        self.type_store = self.type_store.close_function_context()

        compare_types(self.type_store.get_type_of(self.loc, "foo"), types.IntType)
        compare_types(self.type_store.get_type_of(self.loc, "foo_func"), types.FunctionType)
