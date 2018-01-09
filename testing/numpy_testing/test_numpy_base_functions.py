#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print path
if not path in sys.path:
    sys.path.append(path)

from testing.code_generation_testing.codegen_testing_common import TestCommon


class TestBasicNumpyFunctions(TestCommon):
    def test_numpy_abs(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_abs.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_add(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_add.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_all(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_all.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_arange(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_arange.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_arange_2(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_arange_2.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_argpartition(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_argpartition.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_array_readonly(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_array_readonly.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_array_sqrt_arctan2(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_array_sqrt_arctan2.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_ascontiguousarray(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_ascontiguousarray.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_astype(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_astype.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_atleast_2d(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_atleast_2d.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_bincount(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_bincount.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_copy(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_copy.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_core_fromarrays(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_core_fromarrays.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_cumsum(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_cumsum.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_diag(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_diag.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_dtype(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_dtype.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_einsum(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_einsum.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_eye(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_eye.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_fft(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_fft.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_floor_ceil_trunc(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_floor_ceil_trunc.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_fromiter(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_fromiter.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_genfromtxt(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_genfromtxt.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_hstack_vstack(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_hstack_vstack.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_identity(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_identity.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_iinfo_finfo(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_iinfo_finfo.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_indices(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_indices.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_linspace(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_linspace.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_logical_not_negative(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_logical_not_negative.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_lookfor(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_lookfor.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_masked_array(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_masked_array.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_maximum_minimum(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_maximum_minimum.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_meshgrid(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_meshgrid.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_ndenumerate_ndindex(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_ndenumerate_ndindex.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_nonzero(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_nonzero.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_ogrid_mgrid(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_ogrid_mgrid.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_ones(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_ones.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_poly1d(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_poly1d.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_put(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_put.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_random(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_random.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_random_randint(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_random_randint.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_random_uniform(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_random_uniform.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_repeat(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_repeat.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_reshape(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_reshape.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_roll_sort(self): # Problematica
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_roll_sort.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_setprintoptions(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_setprintoptions.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_shape(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_shape.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_stride_tricks(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_stride_tricks.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_subtract_linalg(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_subtract_linalg.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_tensordot(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_tensordot.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_tile(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_tile.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_triu(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_triu.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_unique(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_unique.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_unpackbits(self): # Problematica
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_unpackbits.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_unravel_index(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_unravel_index.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_zeros(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_zeros.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_numpy_zeros_like_cos_sin(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/numpy_zeros_like_cos_sin.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)
