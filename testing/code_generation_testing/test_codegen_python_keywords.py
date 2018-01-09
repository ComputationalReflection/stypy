#!/usr/bin/env python
# -*- coding: utf-8 -*-
from codegen_testing_common import TestCommon
from stypy.errors.type_error import StypyTypeError

class TestCodeGenPythonKeywords(TestCommon):
    def test_lambda_map(self):
        file_path = self.file_path + "/without_classes/higher_order/lambda_map.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_print_variables(self):
        file_path = self.file_path + "/without_classes/print/print_variables.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_slice_creation(self):
        file_path = self.file_path + "/without_classes/slice/slice_creation.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_subscripts(self):
        file_path = self.file_path + "/without_classes/subscripts/basic_subscripts.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_delete(self):
        import math
        old_sin = math.sin
        file_path = self.file_path + "/without_classes/delete/test_delete_simple.py"
        result = self.run_stypy_with_program(file_path)

        math.sin = old_sin
        self.assertEqual(result, 0)

    def test_delete_data_structures(self):
        file_path = self.file_path + "/without_classes/delete/test_delete_data_structures.py"
        result = self.run_stypy_with_program(file_path)

        # for err in StypyTypeError.get_error_msgs():
        #       print err

        self.assertEqual(result, 0)
        self.assertEqual(len(StypyTypeError.get_error_msgs()), 7)

    def test_delete_builtins(self):
        file_path = self.file_path + "/without_classes/delete/test_delete_builtins.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_delete_attribute(self):
        file_path = self.file_path + "/without_classes/delete/test_delete_attribute.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_delete_builtin_vars(self):
        file_path = self.file_path + "/without_classes/delete/test_delete_builtin_vars.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_assert_usage(self):
        file_path = self.file_path + "/without_classes/assert/test_assert_usage.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_yield_usage(self):
        file_path = self.file_path + "/without_classes/yield/yield_simple_usage.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_yield_none(self):
        file_path = self.file_path + "/without_classes/yield/yield_none.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_with_usage(self):
        file_path = self.file_path + "/without_classes/with/simple_with_usage.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_with_usage_no_var(self):
        file_path = self.file_path + "/without_classes/with/simple_with_usage_no_var.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_with_usage_invalid_context(self):
        file_path = self.file_path + "/without_classes/with/simple_with_usage_invalid_context.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_repr_usage(self):
        file_path = self.file_path + "/without_classes/repr/repr_simple.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_alias_usage(self):
        file_path = self.file_path + "/without_classes/alias/alias_usage.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)
