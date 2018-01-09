#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print path
if not path in sys.path:
    sys.path.append(path)

from testing.code_generation_testing.codegen_testing_common import TestCommon


class TestBasicNumpyArrays(TestCommon):
    def test_numpy_arrays(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_arrays.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_array_any(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_array_any.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_array_argmax(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_array_argmax.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_array_argsort(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_array_argsort.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_array_arithmetic(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_array_arithmetic.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_array_broadcasting(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_array_broadcasting.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_array_broadcasting_2(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_array_broadcasting_2.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_array_broadcasting_3(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_array_broadcasting_3.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_array_broadcasting_4(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_array_broadcasting_4.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_array_copy(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_array_copy.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_array_creation(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_array_creation.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_array_creation_0_dimension(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_array_creation_0_dimension.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_array_creation_1_dimension(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_array_creation_1_dimension.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_array_creation_2_dimension(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_array_creation_2_dimension.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_array_datatypes(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_array_datatypes.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_array_flat(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_array_flat.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_array_indexing(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_array_indexing.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_array_indexing_slices(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_array_indexing_slices.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_array_math(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_array_math.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_array_mathematical(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_array_mathematical.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_array_mean(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_array_mean.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_array_product(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_array_product.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_array_ravel(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_array_ravel.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    # Stypy do not support ndarray child types
    # def test_numpy_array_view(self):
    #     file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_array_view.py"
    #     result = self.run_stypy_with_program(file_path, output_results=False)
    #
    #     self.assertEqual(result, 0)

    def test_numpy_boolean_array_indexing(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_boolean_array_indexing.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_integer_array_indexing(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_integer_array_indexing.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_integer_array_indexing_2(self):
        file_path = self.file_path + "/numpy/basic_numpy/arrays/numpy_integer_array_indexing_2.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)
