#!/usr/bin/env python
# -*- coding: utf-8 -*-
from codegen_testing_common import TestCommon


class TestNoClassCodeGenLoops(TestCommon):
    def test_while_outside_and_while(self):
        file_path = self.file_path + "/without_classes/while_statements/variable_outside_and_while.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_for_range(self):
        file_path = self.file_path + "/without_classes/for_statements/for_range.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_for_string(self):
        file_path = self.file_path + "/without_classes/for_statements/for_string.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_for_tuple_var(self):
        file_path = self.file_path + "/without_classes/for_statements/for_tuple_var.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_for_multiple_tuple_var(self):
        file_path = self.file_path + "/without_classes/for_statements/for_multiple_tuple_var.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_for_union_types_condition(self):
        file_path = self.file_path + "/without_classes/for_statements/for_union_types_condition.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_for_error_condition(self):
        file_path = self.file_path + "/without_classes/for_statements/error_for_condition.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)