#!/usr/bin/env python
# -*- coding: utf-8 -*-
from codegen_testing_common import TestCommon


class TestNoClassCodeGenCalls(TestCommon):
    def test_basic_function_calls(self):
        file_path = self.file_path + "/without_classes/functions/basic_function_call.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_basic_function_calls_multiple_calls(self):
        file_path = self.file_path + "/without_classes/functions/basic_function_call_multiple_call.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_function_in_function_call(self):
        file_path = self.file_path + "/without_classes/functions/function_in_function_call.py"
        result = self.run_stypy_with_program(file_path)
        self.assertEqual(result, 0)

    def test_nested_function_calls(self):
        file_path = self.file_path + "/without_classes/functions/nested_function_call.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_multiple_return_function_call(self):
        file_path = self.file_path + "/without_classes/functions/multiple_return_function_call.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_all_type_of_params_function_calls(self):
        file_path = self.file_path + "/without_classes/functions/all_type_of_params_function_call.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_builtin_function_calls(self):
        file_path = self.file_path + "/without_classes/python_library_calls/builtin_function_calls.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_builtin_types_calls(self):
        file_path = self.file_path + "/without_classes/python_library_calls/builtin_types_calls.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_function_call_args_subscript(self):
        file_path = self.file_path + "/without_classes/functions/function_call_args_subscript.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_function_call_with_lambdas(self):
        file_path = self.file_path + "/without_classes/functions/function_call_with_lambdas.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_function_call_with_or_params(self):
        file_path = self.file_path + "/without_classes/functions/function_call_with_or_params.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_function_call_lambda_closures(self):
        file_path = self.file_path + "/without_classes/functions/function_call_lambda_closures.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)