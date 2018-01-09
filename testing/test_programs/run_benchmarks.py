def run_bench_app():
    try:
        from numpy_benchmarks import bench_app

        c1 = bench_app.LaplaceInplace()
        c1.setup('normal')
        # Type error
        c1.setup(0)
        c1.time_it(None)
    except Exception as ex:
        print "1: " + str(ex)

    try:
        from numpy_benchmarks import bench_app

        c1 = bench_app.LaplaceInplace()
        # Type error
        c1.setup(0)
        c1.time_it(None)
    except Exception as ex:
        print "2: " + str(ex)

    try:
        from numpy_benchmarks import bench_app

        c2 = bench_app.MaxesOfDots()
        c2.setup()
        c2.time_it()
    except Exception as ex:
        print "3: " + str(ex)


def run_bench_core():
    try:
        from numpy_benchmarks import bench_core

        c1 = bench_core.Core()
        c1.setup()
        c1.time_arange_100()
        c1.time_array_1()
        c1.time_array_empty()
        c1.time_array_l()
        c1.time_array_l1()
        c1.time_array_l100()
        c1.time_diagflat_l50_l50()
        c1.time_diagflat_l100()
        c1.time_dstack_l()
        c1.time_empty_100()
        c1.time_eye_100()
        c1.time_eye_3000()
        c1.time_hstack_l()
        c1.time_identity_100()
        c1.time_identity_3000()
        c1.time_ones_100()
        c1.time_tril_l10x10()
        c1.time_triu_l10x10()
        c1.time_vstack_l()
        c1.time_zeros_100()
    except Exception as ex:
        print "4: " + str(ex)

    try:
        from numpy_benchmarks import bench_core

        c1 = bench_core.Temporaries()
        c1.setup()
        c1.time_large()
        c1.time_large2()
        c1.time_mid()
        c1.time_mid2()
    except Exception as ex:
        print "5: " + str(ex)

    try:
        from numpy_benchmarks import bench_core

        c1 = bench_core.CorrConv()
        c1.setup(10, 50, 'full')
        c1.time_convolve(10, 50, 'full')
        c1.time_correlate(10, 50, 'full')
    except Exception as ex:
        print "6: " + str(ex)

    try:
        from numpy_benchmarks import bench_core

        c1 = bench_core.CountNonzero()
        c1.setup(3, 1000000, str)
        c1.time_count_nonzero(3, 50, str)
        c1.time_count_nonzero_axis(3, 50, str)
        c1.time_count_nonzero_multi_axis(3, 50, str)

    except Exception as ex:
        print "6: " + str(ex)

    try:
        from numpy_benchmarks import bench_core

        c1 = bench_core.PackBits()
        c1.setup(bool)
        c1.time_packbits(bool)
        c1.time_packbits_axis0(bool)
        c1.time_packbits_axis1(bool)
    except Exception as ex:
        print "7: " + str(ex)

    try:
        from numpy_benchmarks import bench_core

        c1 = bench_core.UnpackBits()
        c1.setup()
        c1.time_unpackbits()
        c1.time_unpackbits_axis0()
        c1.time_unpackbits_axis1()
    except Exception as ex:
        print "7: " + str(ex)

    try:
        from numpy_benchmarks import bench_core

        c1 = bench_core.Indices()
        c1.time_indices()
    except Exception as ex:
        print "7: " + str(ex)


def run_bench_function_base():
    try:
        from numpy_benchmarks import bench_function_base

        c1 = bench_function_base.Bincount()
        c1.setup()
        c1.time_bincount()
        c1.time_weights()
    except Exception as ex:
        print "8: " + str(ex)

    try:
        from numpy_benchmarks import bench_function_base

        c1 = bench_function_base.Median()
        c1.setup()
        c1.time_even()
        c1.time_odd()
        c1.time_even_inplace()
        c1.time_odd_inplace()
        c1.time_even_small()
        c1.time_odd_small()

    except Exception as ex:
        print "9: " + str(ex)

    try:
        from numpy_benchmarks import bench_function_base

        c1 = bench_function_base.Percentile()
        c1.setup()
        c1.time_percentile()
        c1.time_quartile()

    except Exception as ex:
        print "10: " + str(ex)

    try:
        from numpy_benchmarks import bench_function_base

        c1 = bench_function_base.Select()
        c1.setup()
        c1.time_select()
        c1.time_select_larger()

    except Exception as ex:
        print "11: " + str(ex)

    try:
        from numpy_benchmarks import bench_function_base

        c1 = bench_function_base.Sort()
        c1.setup()
        c1.time_sort()
        c1.time_sort_random()
        c1.time_sort_inplace()
        c1.time_sort_equal()
        c1.time_sort_many_equal()
        c1.time_sort_worst()
        c1.time_argsort()
        c1.time_argsort_random()

    except Exception as ex:
        print "12: " + str(ex)

    try:
        from numpy_benchmarks import bench_function_base

        c1 = bench_function_base.Where()
        c1.setup()
        c1.time_1()
        c1.time_2()
        c1.time_2_broadcast()

    except Exception as ex:
        print "11: " + str(ex)


def run_bench_indexing():
    try:
        from numpy_benchmarks import bench_indexing

        c1 = bench_indexing.Indexing()
        c1.setup("indexes_", "I", '')
        c1.time_op("indexes_", "I", '')
    except Exception as ex:
        print "12: " + str(ex)

    try:
        from numpy_benchmarks import bench_indexing

        c1 = bench_indexing.IndexingSeparate()
        c1.setup()
        c1.teardown()
        c1.time_mmap_slicing()
        c1.time_mmap_fancy_indexing()
    except Exception as ex:
        print "13: " + str(ex)

    try:
        from numpy_benchmarks import bench_indexing

        c1 = bench_indexing.IndexingStructured0D()
        c1.setup()
        c1.time_array_slice()
        c1.time_array_all()
        c1.time_scalar_slice()
        c1.time_scalar_all()
    except Exception as ex:
        print "14: " + str(ex)


def run_bench_io():
    try:
        from numpy_benchmarks import bench_io

        c1 = bench_io.Copy()
        c1.setup("float64")
        c1.time_memcpy("float64")
        c1.time_cont_assign("float64")
        c1.time_strided_copy("float64")
        c1.time_strided_assign("float64")
    except Exception as ex:
        print "15: " + str(ex)

    try:
        from numpy_benchmarks import bench_io

        c1 = bench_io.CopyTo()
        c1.setup()
        c1.time_copyto()
        c1.time_copyto_sparse()
        c1.time_copyto_dense()
        c1.time_copyto_8_sparse()
        c1.time_copyto_8_dense()
    except Exception as ex:
        print "16: " + str(ex)

    try:
        from numpy_benchmarks import bench_io

        c1 = bench_io.Savez()
        c1.setup()
        c1.time_vb_savez_squares()
    except Exception as ex:
        print "17: " + str(ex)


def run_bench_linalg():
    try:
        from numpy_benchmarks import bench_linalg

        c1 = bench_linalg.Eindot()
        c1.setup()
        c1.time_dot_a_b()
        c1.time_dot_d_dot_b_c()
        c1.time_dot_trans_a_at()
        c1.time_dot_trans_a_atc()
        c1.time_dot_trans_at_a()
        c1.time_dot_trans_atc_a()
        c1.time_einsum_i_ij_j()
        c1.time_einsum_ij_jk_a_b()
        c1.time_einsum_ijk_jil_kl()
        c1.time_inner_trans_a_ac()
        c1.time_matmul_a_b()
        c1.time_matmul_d_matmul_b_c()
        c1.time_matmul_trans_a_at()
        c1.time_matmul_trans_a_atc()
        c1.time_matmul_trans_at_a()
        c1.time_matmul_trans_atc_a()
        c1.time_tensordot_a_b_axes_1_0_0_1()

    except Exception as ex:
        print "18: " + str(ex)

    try:
        from numpy_benchmarks import bench_linalg

        c1 = bench_linalg.Linalg()
        c1.setup("svd", "float64")
        c1.time_op("svd", "float64")
    except Exception as ex:
        print "19: " + str(ex)

    try:
        from numpy_benchmarks import bench_linalg

        c1 = bench_linalg.Lstsq()
        c1.setup()
        c1.time_numpy_linalg_lstsq_a__b_float64()
    except Exception as ex:
        print "20: " + str(ex)


def run_bench_ma():
    try:
        from numpy_benchmarks import bench_ma

        c1 = bench_ma.MA()
        c1.setup()
        c1.time_masked_array()
        c1.time_masked_array_l100()
        c1.time_masked_array_l100_t100()

    except Exception as ex:
        print "21: " + str(ex)

    try:
        from numpy_benchmarks import bench_ma

        c1 = bench_ma.Indexing()
        c1.setup(False, 1, 100)
        c1.time_scalar(False, 1, 100)
        c1.time_0d(False, 1, 100)
        c1.time_1d(False, 1, 100)

    except Exception as ex:
        print "22: " + str(ex)

    try:
        from numpy_benchmarks import bench_ma

        c1 = bench_ma.MA()
        c1.setup()
        c1.time_masked_array()
        c1.time_masked_array_l100()
        c1.time_masked_array_l100_t100()

    except Exception as ex:
        print "23: " + str(ex)

    try:
        from numpy_benchmarks import bench_ma

        c1 = bench_ma.UFunc()
        c1.setup(False, False, 100)
        c1.time_scalar(False, False, 100)
        c1.time_scalar_1d(False, False, 100)
        c1.time_1d(False, False, 100)
        c1.time_2d(False, False, 100)
    except Exception as ex:
        print "24: " + str(ex)

    try:
        from numpy_benchmarks import bench_ma

        c1 = bench_ma.Concatenate()
        c1.setup('ndarray', 100)
        c1.time_it('ndarray', 100)
    except Exception as ex:
        print "25: " + str(ex)


def run_bench_random():
    try:
        from numpy_benchmarks import bench_random

        c1 = bench_random.Random()
        c1.setup('normal')
        c1.time_rng('normal')
    except Exception as ex:
        print "26: " + str(ex)

    try:
        from numpy_benchmarks import bench_random

        c1 = bench_random.Shuffle()
        c1.setup()
        c1.time_100000()
    except Exception as ex:
        print "27: " + str(ex)

    try:
        from numpy_benchmarks import bench_random

        c1 = bench_random.Randint()

        c1.time_randint_fast()
        c1.time_randint_slow()
    except Exception as ex:
        print "28: " + str(ex)

    try:
        from numpy_benchmarks import bench_random

        c1 = bench_random.Randint_dtype()

        c1.setup("uint64")
        c1.time_randint_fast("uint64")
        c1.time_randint_slow("uint64")
    except Exception as ex:
        print "29: " + str(ex)


def run_bench_reduce():
    try:
        from numpy_benchmarks import bench_reduce

        c1 = bench_reduce.AddReduce()
        c1.setup()
        c1.time_axis_0()
        c1.time_axis_1()
    except Exception as ex:
        print "30: " + str(ex)

    try:
        from numpy_benchmarks import bench_reduce

        c1 = bench_reduce.AddReduceSeparate()
        c1.setup(0, 'float64')
        c1.time_reduce(0, 'float64')
    except Exception as ex:
        print "31: " + str(ex)

    try:
        from numpy_benchmarks import bench_reduce

        c1 = bench_reduce.AnyAll()
        c1.setup()
        c1.time_all_fast()
        c1.time_all_slow()
        c1.time_any_fast()
        c1.time_any_slow()
    except Exception as ex:
        print "32: " + str(ex)

    try:
        from numpy_benchmarks import bench_reduce

        c1 = bench_reduce.MinMax()
        c1.setup('float64')
        c1.time_min('float64')
        c1.time_max('float64')
    except Exception as ex:
        print "33: " + str(ex)

    try:
        from numpy_benchmarks import bench_reduce

        c1 = bench_reduce.SmallReduction()
        c1.setup()
        c1.time_small()
    except Exception as ex:
        print "34: " + str(ex)


def run_bench_shape_base():
    try:
        from numpy_benchmarks import bench_shape_base

        c1 = bench_shape_base.Block()
        c1.setup(100)
        c1.time_block_simple_row_wise(100)
        c1.time_block_simple_column_wise(100)
        c1.time_block_complicated(100)
        c1.time_nested(100)
        c1.time_3d(100)
        c1.time_no_lists(100)
    except Exception as ex:
        print "35: " + str(ex)


def run_bench_ufunc():
    try:
        from numpy_benchmarks import bench_ufunc

        c1 = bench_ufunc.Broadcast()
        c1.setup()
        c1.time_broadcast()

    except Exception as ex:
        print "36: " + str(ex)

    try:
        from numpy_benchmarks import bench_ufunc

        c1 = bench_ufunc.UFunc()
        c1.setup('abs')
        c1.time_ufunc_types('abs')

    except Exception as ex:
        print "37: " + str(ex)

    try:
        from numpy_benchmarks import bench_ufunc

        c1 = bench_ufunc.Custom()
        c1.setup()
        c1.time_nonzero()
        c1.time_not_bool()
        c1.time_and_bool()
        c1.time_or_bool()

    except Exception as ex:
        print "38: " + str(ex)

    try:
        from numpy_benchmarks import bench_ufunc

        c1 = bench_ufunc.CustomInplace()
        c1.setup()
        c1.time_char_or()
        c1.time_char_or_temp()
        c1.time_int_or()
        c1.time_int_or_temp()
        c1.time_float_add()
        c1.time_float_add_temp()
        c1.time_double_add()
        c1.time_double_add_temp()
    except Exception as ex:
        print "39: " + str(ex)

    try:
        from numpy_benchmarks import bench_ufunc

        c1 = bench_ufunc.CustomScalar()
        c1.setup('float64')
        c1.time_add_scalar2('float64')
        c1.time_divide_scalar2('float64')
        c1.time_divide_scalar2_inplace('float64')
        c1.time_less_than_scalar2('float64')

    except Exception as ex:
        print "40: " + str(ex)

    try:
        from numpy_benchmarks import bench_ufunc

        c1 = bench_ufunc.Scalar()
        c1.setup()
        c1.time_add_scalar()
        c1.time_add_scalar_conv()
        c1.time_add_scalar_conv_complex()

    except Exception as ex:
        print "41: " + str(ex)


if __name__ == '__main__':
    print "Running run_bench_app\n"
    # run_bench_app()
    print "Running run_bench_core\n"
    # run_bench_core()
    print "Running run_bench_function_base\n"
    # run_bench_function_base()
    print "Running run_bench_indexing\n"
    # run_bench_indexing()
    print "Running run_bench_io\n"
    # run_bench_io()
    print "Running run_bench_linalg\n"
    # run_bench_linalg()
    print "Running run_bench_ma\n"
    # run_bench_ma()
    print "Running run_bench_random\n"
    # run_bench_random()
    print "Running run_bench_reduce\n"
    # run_bench_reduce()
    print "Running run_bench_shape_base\n"
    # run_bench_shape_base()
    print "Running run_bench_ufunc\n"
    run_bench_ufunc()
