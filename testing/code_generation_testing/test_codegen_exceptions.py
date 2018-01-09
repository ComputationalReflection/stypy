#!/usr/bin/env python
# -*- coding: utf-8 -*-
from codegen_testing_common import TestCommon


class TestCodeGenExceptions(TestCommon):
    def test_try_except(self):
        file_path = self.file_path + "/without_classes/exceptions/try_except.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_try_multiple_except(self):
        file_path = self.file_path + "/without_classes/exceptions/try_multiple_except.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_try_except_else(self):
        file_path = self.file_path + "/without_classes/exceptions/try_except_else.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_try_except_else_finally(self):
        file_path = self.file_path + "/without_classes/exceptions/try_except_else_finally.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_try_except_raise(self):
        file_path = self.file_path + "/without_classes/exceptions/try_except_raise.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_try_empty_raise(self):
        file_path = self.file_path + "/without_classes/exceptions/try_empty_raise.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_raise_exceptions(self):
        file_path = self.file_path + "/without_classes/exceptions/raise_exception.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_raise_assertion_error(self):
        file_path = self.file_path + "/without_classes/exceptions/raise_assertion_error.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)