#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.errors import type_warning
from stypy.errors import type_error
from codegen_testing_common import TestCommon


class TestNoClassCodeGenSSA(TestCommon):
    def test_ssa_function_body_error(self):
        file_path = self.file_path + "/without_classes/ssa/ssa_function_body_error.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_ssa_function_body_error_condition(self):
        file_path = self.file_path + "/without_classes/ssa/ssa_function_body_error_condition.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_ssa_function_body_error_nested(self):
        file_path = self.file_path + "/without_classes/ssa/ssa_function_body_error_nested.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_ssa_function_body_error_noparam(self):
        file_path = self.file_path + "/without_classes/ssa/ssa_function_body_error_noparam.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)
