#!/usr/bin/env python
# -*- coding: utf-8 -*-
from testing.code_generation_testing.codegen_testing_common import TestCommon


class TestNoClassCodeGenAssignments(TestCommon):
    def test_basic_assignments(self):
        file_path = self.file_path + "/without_classes/assignments/basic_assignments.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_nonbasic_assignments(self):
        file_path = self.file_path + "/without_classes/assignments/nonbasic_assignments.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_augment_assigments(self):
        file_path = self.file_path + "/without_classes/assignments/augment_assignments.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_array_assigments(self):
        file_path = self.file_path + "/without_classes/assignments/array_assignments.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_tuple_assigments(self):
        file_path = self.file_path + "/without_classes/assignments/tuple_assignments.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_multiple_assigments(self):
        file_path = self.file_path + "/without_classes/assignments/multiple_assignments.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_multi_tuple_assigments(self):
        file_path = self.file_path + "/without_classes/assignments/multi_tuple_assignments.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_tuple_assigments_2(self):
        file_path = self.file_path + "/without_classes/assignments/tuple_assignments_2.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_multiple_assigments2(self):
        file_path = self.file_path + "/without_classes/assignments/multiple_assignments2.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.assertEqual(result, 0)