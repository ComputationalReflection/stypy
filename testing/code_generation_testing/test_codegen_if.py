#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.errors import type_warning
from stypy.errors import type_error
from codegen_testing_common import TestCommon


class TestNoClassCodeGenIf(TestCommon):
    def test_if_outside_and_if(self):
        file_path = self.file_path + "/without_classes/if_statements/variable_outside_and_if.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_variable_both_if_branches(self):
        file_path = self.file_path + "/without_classes/if_statements/variable_both_if_branches.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)
        self.assertEquals(len(type_warning.TypeWarning.get_warning_msgs()), 1)

    def test_flow_sensitive_function_definition(self):
        file_path = self.file_path + "/without_classes/if_statements/flow_sensitive_function_definition.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)
        self.assertEquals(len(type_error.StypyTypeError.get_error_msgs()), 1)
        self.assertEquals(len(type_warning.TypeWarning.get_warning_msgs()), 1)

    def test_variable_only_if_branch(self):
        file_path = self.file_path + "/without_classes/if_statements/variable_only_if_branch.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)
        self.assertEquals(len(type_warning.TypeWarning.get_warning_msgs()), 1)

    def test_error_if_condition(self):
        file_path = self.file_path + "/without_classes/if_statements/error_if_condition.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)