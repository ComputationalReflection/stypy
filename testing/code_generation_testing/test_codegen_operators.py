#!/usr/bin/env python
# -*- coding: utf-8 -*-
from codegen_testing_common import TestCommon


class TestNoClassCodeGenOperators(TestCommon):
    def test_basic_arithmetic(self):
        file_path = self.file_path + "/without_classes/operators/basic_arithmetic_operators.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_basic_bool(self):
        file_path = self.file_path + "/without_classes/operators/basic_bool_operators.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_simple_comparations(self):
        file_path = self.file_path + "/without_classes/comparations/simple_comparations.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_multiple_comparations(self):
        file_path = self.file_path + "/without_classes/comparations/multiple_comparations.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_special_methods_operators(self):
        file_path = self.file_path + "/without_classes/operators/error_special_methods_operators.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)