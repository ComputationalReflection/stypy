#!/usr/bin/env python
# -*- coding: utf-8 -*-
from codegen_testing_common import TestCommon


class TestOther(TestCommon):
    def test_main_usage(self):
        file_path = self.file_path + "/without_classes/main/main_usage.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_syntax_errors(self):
        file_path = self.file_path + "/without_classes/syntax_errors/syntax_error.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, -2)

    def test_get_doc_attributes(self):
        file_path = self.file_path + "/without_classes/structural/get_doc_attributes.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)