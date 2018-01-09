#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print path
if not path in sys.path:
    sys.path.append(path)

from stypy.errors import type_error, type_warning
from testing.code_generation_testing.codegen_testing_common import TestCommon


class TestBasicNumpy(TestCommon):
    def test_main_usage(self):
        file_path = self.file_path + "/numpy/basic_numpy/numpy_usage.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        assert (len(type_error.StypyTypeError.errors) == 2 or len(type_error.StypyTypeError.errors) == 1)
        assert (len(type_warning.TypeWarning.warnings) == 7 or len(type_warning.TypeWarning.warnings) == 0)
        self.assertEqual(result, 0)

    def test_import_umath_pyd(self):
        file_path = self.file_path + "/numpy/basic_numpy/import_umath_pyd.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_import_numerictypes(self):
        file_path = self.file_path + "/numpy/basic_numpy/import_numeric_types.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        # assert (len(type_error.StypyTypeError.errors) == 1 or len(type_error.StypyTypeError.errors) == 0)
        # assert (len(type_warning.TypeWarning.warnings) == 7 or len(type_warning.TypeWarning.warnings) == 0)
        self.assertEqual(result, 0)

    def test_import_pyd(self):
        file_path = self.file_path + "/numpy/basic_numpy/import_pyd.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        # assert (len(type_error.StypyTypeError.errors) == 1 or len(type_error.StypyTypeError.errors) == 0)
        # assert (len(type_warning.TypeWarning.warnings) == 7 or len(type_warning.TypeWarning.warnings) == 0)

        self.assertEqual(result, 0)

    def test_import_numpy(self):
        file_path = self.file_path + "/numpy/basic_numpy/import_numpy.py"
        result = self.run_stypy_with_program(file_path, output_results=False)
        #
        # assert (len(type_error.StypyTypeError.errors) == 1 or len(type_error.StypyTypeError.errors) == 0)
        # assert (len(type_warning.TypeWarning.warnings) == 7 or len(type_warning.TypeWarning.warnings) == 0)

        self.assertEqual(result, 0)

    def test_import_numpy_numeric_types(self):
        file_path = self.file_path + "/numpy/basic_numpy/import_numpy_numeric_types.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        # assert (len(type_error.StypyTypeError.errors) == 1 or len(type_error.StypyTypeError.errors) == 0)
        # assert (len(type_warning.TypeWarning.warnings) == 7 or len(type_warning.TypeWarning.warnings) == 0)

        self.assertEqual(result, 0)

    def test_import_numpy_core(self):
        file_path = self.file_path + "/numpy/basic_numpy/import_numpy_core.py"
        result = self.run_stypy_with_program(file_path, output_results=False)
        #
        # assert (len(type_error.StypyTypeError.errors) == 1 or len(type_error.StypyTypeError.errors) == 0)
        # assert (len(type_warning.TypeWarning.warnings) == 7 or len(type_warning.TypeWarning.warnings) == 0)

        self.assertEqual(result, 0)

    def test_import_numpy_type_check(self):
        file_path = self.file_path + "/numpy/basic_numpy/import_numpy_type_check.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        # assert (len(type_error.StypyTypeError.errors) == 1 or len(type_error.StypyTypeError.errors) == 0)
        # assert (len(type_warning.TypeWarning.warnings) == 7 or len(type_warning.TypeWarning.warnings) == 0)

        self.assertEqual(result, 0)

    def test_numpy_mathematical_functions_arithmetical(self):
        file_path = self.file_path + "/numpy/basic_numpy/numpy_mathematical_functions_arithmetical.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_mathematical_functions_complex(self):
        file_path = self.file_path + "/numpy/basic_numpy/numpy_mathematical_functions_complex.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_mathematical_functions_exponents(self):
        file_path = self.file_path + "/numpy/basic_numpy/numpy_mathematical_functions_exponents.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_mathematical_functions_fp(self):
        file_path = self.file_path + "/numpy/basic_numpy/numpy_mathematical_functions_fp.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_mathematical_functions_hyperbolic(self):
        file_path = self.file_path + "/numpy/basic_numpy/numpy_mathematical_functions_hyperbolic.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    # Hangs unit tests
    def test_numpy_mathematical_functions_misc(self):
        file_path = self.file_path + "/numpy/basic_numpy/numpy_mathematical_functions_misc.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    # Hangs unit tests
    def test_numpy_mathematical_functions_other(self):
        file_path = self.file_path + "/numpy/basic_numpy/numpy_mathematical_functions_other.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_mathematical_functions_rounding(self):
        file_path = self.file_path + "/numpy/basic_numpy/numpy_mathematical_functions_rounding.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    # Hangs unit tests
    def test_numpy_mathematical_functions_sums(self):
        file_path = self.file_path + "/numpy/basic_numpy/numpy_mathematical_functions_sums.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_mathematical_functions_trigonometrical(self):
        file_path = self.file_path + "/numpy/basic_numpy/numpy_mathematical_functions_trigonometrical.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_number_comparison(self):
        file_path = self.file_path + "/numpy/basic_numpy/numpy_number_comparison.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_version_and_config(self):
        file_path = self.file_path + "/numpy/basic_numpy/numpy_version_and_config.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)
