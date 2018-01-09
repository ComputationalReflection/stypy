
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import assert_array_almost_equal, assert_
5: from scipy.sparse import csr_matrix
6: 
7: 
8: def _check_csr_rowslice(i, sl, X, Xcsr):
9:     np_slice = X[i, sl]
10:     csr_slice = Xcsr[i, sl]
11:     assert_array_almost_equal(np_slice, csr_slice.toarray()[0])
12:     assert_(type(csr_slice) is csr_matrix)
13: 
14: 
15: def test_csr_rowslice():
16:     N = 10
17:     np.random.seed(0)
18:     X = np.random.random((N, N))
19:     X[X > 0.7] = 0
20:     Xcsr = csr_matrix(X)
21: 
22:     slices = [slice(None, None, None),
23:               slice(None, None, -1),
24:               slice(1, -2, 2),
25:               slice(-2, 1, -2)]
26: 
27:     for i in range(N):
28:         for sl in slices:
29:             _check_csr_rowslice(i, sl, X, Xcsr)
30: 
31: 
32: def test_csr_getrow():
33:     N = 10
34:     np.random.seed(0)
35:     X = np.random.random((N, N))
36:     X[X > 0.7] = 0
37:     Xcsr = csr_matrix(X)
38: 
39:     for i in range(N):
40:         arr_row = X[i:i + 1, :]
41:         csr_row = Xcsr.getrow(i)
42: 
43:         assert_array_almost_equal(arr_row, csr_row.toarray())
44:         assert_(type(csr_row) is csr_matrix)
45: 
46: 
47: def test_csr_getcol():
48:     N = 10
49:     np.random.seed(0)
50:     X = np.random.random((N, N))
51:     X[X > 0.7] = 0
52:     Xcsr = csr_matrix(X)
53: 
54:     for i in range(N):
55:         arr_col = X[:, i:i + 1]
56:         csr_col = Xcsr.getcol(i)
57: 
58:         assert_array_almost_equal(arr_col, csr_col.toarray())
59:         assert_(type(csr_col) is csr_matrix)
60: 
61: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_459657 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_459657) is not StypyTypeError):

    if (import_459657 != 'pyd_module'):
        __import__(import_459657)
        sys_modules_459658 = sys.modules[import_459657]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_459658.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_459657)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_array_almost_equal, assert_' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_459659 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_459659) is not StypyTypeError):

    if (import_459659 != 'pyd_module'):
        __import__(import_459659)
        sys_modules_459660 = sys.modules[import_459659]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_459660.module_type_store, module_type_store, ['assert_array_almost_equal', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_459660, sys_modules_459660.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_array_almost_equal, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_array_almost_equal', 'assert_'], [assert_array_almost_equal, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_459659)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.sparse import csr_matrix' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_459661 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse')

if (type(import_459661) is not StypyTypeError):

    if (import_459661 != 'pyd_module'):
        __import__(import_459661)
        sys_modules_459662 = sys.modules[import_459661]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse', sys_modules_459662.module_type_store, module_type_store, ['csr_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_459662, sys_modules_459662.module_type_store, module_type_store)
    else:
        from scipy.sparse import csr_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse', None, module_type_store, ['csr_matrix'], [csr_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse', import_459661)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')


@norecursion
def _check_csr_rowslice(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_check_csr_rowslice'
    module_type_store = module_type_store.open_function_context('_check_csr_rowslice', 8, 0, False)
    
    # Passed parameters checking function
    _check_csr_rowslice.stypy_localization = localization
    _check_csr_rowslice.stypy_type_of_self = None
    _check_csr_rowslice.stypy_type_store = module_type_store
    _check_csr_rowslice.stypy_function_name = '_check_csr_rowslice'
    _check_csr_rowslice.stypy_param_names_list = ['i', 'sl', 'X', 'Xcsr']
    _check_csr_rowslice.stypy_varargs_param_name = None
    _check_csr_rowslice.stypy_kwargs_param_name = None
    _check_csr_rowslice.stypy_call_defaults = defaults
    _check_csr_rowslice.stypy_call_varargs = varargs
    _check_csr_rowslice.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_check_csr_rowslice', ['i', 'sl', 'X', 'Xcsr'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_check_csr_rowslice', localization, ['i', 'sl', 'X', 'Xcsr'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_check_csr_rowslice(...)' code ##################

    
    # Assigning a Subscript to a Name (line 9):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 9)
    tuple_459663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 9)
    # Adding element type (line 9)
    # Getting the type of 'i' (line 9)
    i_459664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 17), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 17), tuple_459663, i_459664)
    # Adding element type (line 9)
    # Getting the type of 'sl' (line 9)
    sl_459665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 20), 'sl')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 17), tuple_459663, sl_459665)
    
    # Getting the type of 'X' (line 9)
    X_459666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 15), 'X')
    # Obtaining the member '__getitem__' of a type (line 9)
    getitem___459667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 15), X_459666, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 9)
    subscript_call_result_459668 = invoke(stypy.reporting.localization.Localization(__file__, 9, 15), getitem___459667, tuple_459663)
    
    # Assigning a type to the variable 'np_slice' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'np_slice', subscript_call_result_459668)
    
    # Assigning a Subscript to a Name (line 10):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 10)
    tuple_459669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 10)
    # Adding element type (line 10)
    # Getting the type of 'i' (line 10)
    i_459670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 21), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 21), tuple_459669, i_459670)
    # Adding element type (line 10)
    # Getting the type of 'sl' (line 10)
    sl_459671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 24), 'sl')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 21), tuple_459669, sl_459671)
    
    # Getting the type of 'Xcsr' (line 10)
    Xcsr_459672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 16), 'Xcsr')
    # Obtaining the member '__getitem__' of a type (line 10)
    getitem___459673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 16), Xcsr_459672, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 10)
    subscript_call_result_459674 = invoke(stypy.reporting.localization.Localization(__file__, 10, 16), getitem___459673, tuple_459669)
    
    # Assigning a type to the variable 'csr_slice' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'csr_slice', subscript_call_result_459674)
    
    # Call to assert_array_almost_equal(...): (line 11)
    # Processing the call arguments (line 11)
    # Getting the type of 'np_slice' (line 11)
    np_slice_459676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 30), 'np_slice', False)
    
    # Obtaining the type of the subscript
    int_459677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 60), 'int')
    
    # Call to toarray(...): (line 11)
    # Processing the call keyword arguments (line 11)
    kwargs_459680 = {}
    # Getting the type of 'csr_slice' (line 11)
    csr_slice_459678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 40), 'csr_slice', False)
    # Obtaining the member 'toarray' of a type (line 11)
    toarray_459679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 40), csr_slice_459678, 'toarray')
    # Calling toarray(args, kwargs) (line 11)
    toarray_call_result_459681 = invoke(stypy.reporting.localization.Localization(__file__, 11, 40), toarray_459679, *[], **kwargs_459680)
    
    # Obtaining the member '__getitem__' of a type (line 11)
    getitem___459682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 40), toarray_call_result_459681, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 11)
    subscript_call_result_459683 = invoke(stypy.reporting.localization.Localization(__file__, 11, 40), getitem___459682, int_459677)
    
    # Processing the call keyword arguments (line 11)
    kwargs_459684 = {}
    # Getting the type of 'assert_array_almost_equal' (line 11)
    assert_array_almost_equal_459675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 11)
    assert_array_almost_equal_call_result_459685 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), assert_array_almost_equal_459675, *[np_slice_459676, subscript_call_result_459683], **kwargs_459684)
    
    
    # Call to assert_(...): (line 12)
    # Processing the call arguments (line 12)
    
    
    # Call to type(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'csr_slice' (line 12)
    csr_slice_459688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 17), 'csr_slice', False)
    # Processing the call keyword arguments (line 12)
    kwargs_459689 = {}
    # Getting the type of 'type' (line 12)
    type_459687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'type', False)
    # Calling type(args, kwargs) (line 12)
    type_call_result_459690 = invoke(stypy.reporting.localization.Localization(__file__, 12, 12), type_459687, *[csr_slice_459688], **kwargs_459689)
    
    # Getting the type of 'csr_matrix' (line 12)
    csr_matrix_459691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 31), 'csr_matrix', False)
    # Applying the binary operator 'is' (line 12)
    result_is__459692 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 12), 'is', type_call_result_459690, csr_matrix_459691)
    
    # Processing the call keyword arguments (line 12)
    kwargs_459693 = {}
    # Getting the type of 'assert_' (line 12)
    assert__459686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 12)
    assert__call_result_459694 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), assert__459686, *[result_is__459692], **kwargs_459693)
    
    
    # ################# End of '_check_csr_rowslice(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_check_csr_rowslice' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_459695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_459695)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_check_csr_rowslice'
    return stypy_return_type_459695

# Assigning a type to the variable '_check_csr_rowslice' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '_check_csr_rowslice', _check_csr_rowslice)

@norecursion
def test_csr_rowslice(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_csr_rowslice'
    module_type_store = module_type_store.open_function_context('test_csr_rowslice', 15, 0, False)
    
    # Passed parameters checking function
    test_csr_rowslice.stypy_localization = localization
    test_csr_rowslice.stypy_type_of_self = None
    test_csr_rowslice.stypy_type_store = module_type_store
    test_csr_rowslice.stypy_function_name = 'test_csr_rowslice'
    test_csr_rowslice.stypy_param_names_list = []
    test_csr_rowslice.stypy_varargs_param_name = None
    test_csr_rowslice.stypy_kwargs_param_name = None
    test_csr_rowslice.stypy_call_defaults = defaults
    test_csr_rowslice.stypy_call_varargs = varargs
    test_csr_rowslice.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_csr_rowslice', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_csr_rowslice', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_csr_rowslice(...)' code ##################

    
    # Assigning a Num to a Name (line 16):
    int_459696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 8), 'int')
    # Assigning a type to the variable 'N' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'N', int_459696)
    
    # Call to seed(...): (line 17)
    # Processing the call arguments (line 17)
    int_459700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 19), 'int')
    # Processing the call keyword arguments (line 17)
    kwargs_459701 = {}
    # Getting the type of 'np' (line 17)
    np_459697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 17)
    random_459698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 4), np_459697, 'random')
    # Obtaining the member 'seed' of a type (line 17)
    seed_459699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 4), random_459698, 'seed')
    # Calling seed(args, kwargs) (line 17)
    seed_call_result_459702 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), seed_459699, *[int_459700], **kwargs_459701)
    
    
    # Assigning a Call to a Name (line 18):
    
    # Call to random(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Obtaining an instance of the builtin type 'tuple' (line 18)
    tuple_459706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 18)
    # Adding element type (line 18)
    # Getting the type of 'N' (line 18)
    N_459707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 26), 'N', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 26), tuple_459706, N_459707)
    # Adding element type (line 18)
    # Getting the type of 'N' (line 18)
    N_459708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 29), 'N', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 26), tuple_459706, N_459708)
    
    # Processing the call keyword arguments (line 18)
    kwargs_459709 = {}
    # Getting the type of 'np' (line 18)
    np_459703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 18)
    random_459704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), np_459703, 'random')
    # Obtaining the member 'random' of a type (line 18)
    random_459705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), random_459704, 'random')
    # Calling random(args, kwargs) (line 18)
    random_call_result_459710 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), random_459705, *[tuple_459706], **kwargs_459709)
    
    # Assigning a type to the variable 'X' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'X', random_call_result_459710)
    
    # Assigning a Num to a Subscript (line 19):
    int_459711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 17), 'int')
    # Getting the type of 'X' (line 19)
    X_459712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'X')
    
    # Getting the type of 'X' (line 19)
    X_459713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 6), 'X')
    float_459714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 10), 'float')
    # Applying the binary operator '>' (line 19)
    result_gt_459715 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 6), '>', X_459713, float_459714)
    
    # Storing an element on a container (line 19)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 4), X_459712, (result_gt_459715, int_459711))
    
    # Assigning a Call to a Name (line 20):
    
    # Call to csr_matrix(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'X' (line 20)
    X_459717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 22), 'X', False)
    # Processing the call keyword arguments (line 20)
    kwargs_459718 = {}
    # Getting the type of 'csr_matrix' (line 20)
    csr_matrix_459716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 11), 'csr_matrix', False)
    # Calling csr_matrix(args, kwargs) (line 20)
    csr_matrix_call_result_459719 = invoke(stypy.reporting.localization.Localization(__file__, 20, 11), csr_matrix_459716, *[X_459717], **kwargs_459718)
    
    # Assigning a type to the variable 'Xcsr' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'Xcsr', csr_matrix_call_result_459719)
    
    # Assigning a List to a Name (line 22):
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_459720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    
    # Call to slice(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'None' (line 22)
    None_459722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 20), 'None', False)
    # Getting the type of 'None' (line 22)
    None_459723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 26), 'None', False)
    # Getting the type of 'None' (line 22)
    None_459724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 32), 'None', False)
    # Processing the call keyword arguments (line 22)
    kwargs_459725 = {}
    # Getting the type of 'slice' (line 22)
    slice_459721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 14), 'slice', False)
    # Calling slice(args, kwargs) (line 22)
    slice_call_result_459726 = invoke(stypy.reporting.localization.Localization(__file__, 22, 14), slice_459721, *[None_459722, None_459723, None_459724], **kwargs_459725)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 13), list_459720, slice_call_result_459726)
    # Adding element type (line 22)
    
    # Call to slice(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'None' (line 23)
    None_459728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 20), 'None', False)
    # Getting the type of 'None' (line 23)
    None_459729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 26), 'None', False)
    int_459730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 32), 'int')
    # Processing the call keyword arguments (line 23)
    kwargs_459731 = {}
    # Getting the type of 'slice' (line 23)
    slice_459727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 14), 'slice', False)
    # Calling slice(args, kwargs) (line 23)
    slice_call_result_459732 = invoke(stypy.reporting.localization.Localization(__file__, 23, 14), slice_459727, *[None_459728, None_459729, int_459730], **kwargs_459731)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 13), list_459720, slice_call_result_459732)
    # Adding element type (line 22)
    
    # Call to slice(...): (line 24)
    # Processing the call arguments (line 24)
    int_459734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 20), 'int')
    int_459735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 23), 'int')
    int_459736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 27), 'int')
    # Processing the call keyword arguments (line 24)
    kwargs_459737 = {}
    # Getting the type of 'slice' (line 24)
    slice_459733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 14), 'slice', False)
    # Calling slice(args, kwargs) (line 24)
    slice_call_result_459738 = invoke(stypy.reporting.localization.Localization(__file__, 24, 14), slice_459733, *[int_459734, int_459735, int_459736], **kwargs_459737)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 13), list_459720, slice_call_result_459738)
    # Adding element type (line 22)
    
    # Call to slice(...): (line 25)
    # Processing the call arguments (line 25)
    int_459740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 20), 'int')
    int_459741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 24), 'int')
    int_459742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 27), 'int')
    # Processing the call keyword arguments (line 25)
    kwargs_459743 = {}
    # Getting the type of 'slice' (line 25)
    slice_459739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 14), 'slice', False)
    # Calling slice(args, kwargs) (line 25)
    slice_call_result_459744 = invoke(stypy.reporting.localization.Localization(__file__, 25, 14), slice_459739, *[int_459740, int_459741, int_459742], **kwargs_459743)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 13), list_459720, slice_call_result_459744)
    
    # Assigning a type to the variable 'slices' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'slices', list_459720)
    
    
    # Call to range(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'N' (line 27)
    N_459746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 19), 'N', False)
    # Processing the call keyword arguments (line 27)
    kwargs_459747 = {}
    # Getting the type of 'range' (line 27)
    range_459745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 13), 'range', False)
    # Calling range(args, kwargs) (line 27)
    range_call_result_459748 = invoke(stypy.reporting.localization.Localization(__file__, 27, 13), range_459745, *[N_459746], **kwargs_459747)
    
    # Testing the type of a for loop iterable (line 27)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 27, 4), range_call_result_459748)
    # Getting the type of the for loop variable (line 27)
    for_loop_var_459749 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 27, 4), range_call_result_459748)
    # Assigning a type to the variable 'i' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'i', for_loop_var_459749)
    # SSA begins for a for statement (line 27)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'slices' (line 28)
    slices_459750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 18), 'slices')
    # Testing the type of a for loop iterable (line 28)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 28, 8), slices_459750)
    # Getting the type of the for loop variable (line 28)
    for_loop_var_459751 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 28, 8), slices_459750)
    # Assigning a type to the variable 'sl' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'sl', for_loop_var_459751)
    # SSA begins for a for statement (line 28)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to _check_csr_rowslice(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'i' (line 29)
    i_459753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 32), 'i', False)
    # Getting the type of 'sl' (line 29)
    sl_459754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 35), 'sl', False)
    # Getting the type of 'X' (line 29)
    X_459755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 39), 'X', False)
    # Getting the type of 'Xcsr' (line 29)
    Xcsr_459756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 42), 'Xcsr', False)
    # Processing the call keyword arguments (line 29)
    kwargs_459757 = {}
    # Getting the type of '_check_csr_rowslice' (line 29)
    _check_csr_rowslice_459752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), '_check_csr_rowslice', False)
    # Calling _check_csr_rowslice(args, kwargs) (line 29)
    _check_csr_rowslice_call_result_459758 = invoke(stypy.reporting.localization.Localization(__file__, 29, 12), _check_csr_rowslice_459752, *[i_459753, sl_459754, X_459755, Xcsr_459756], **kwargs_459757)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_csr_rowslice(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_csr_rowslice' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_459759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_459759)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_csr_rowslice'
    return stypy_return_type_459759

# Assigning a type to the variable 'test_csr_rowslice' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'test_csr_rowslice', test_csr_rowslice)

@norecursion
def test_csr_getrow(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_csr_getrow'
    module_type_store = module_type_store.open_function_context('test_csr_getrow', 32, 0, False)
    
    # Passed parameters checking function
    test_csr_getrow.stypy_localization = localization
    test_csr_getrow.stypy_type_of_self = None
    test_csr_getrow.stypy_type_store = module_type_store
    test_csr_getrow.stypy_function_name = 'test_csr_getrow'
    test_csr_getrow.stypy_param_names_list = []
    test_csr_getrow.stypy_varargs_param_name = None
    test_csr_getrow.stypy_kwargs_param_name = None
    test_csr_getrow.stypy_call_defaults = defaults
    test_csr_getrow.stypy_call_varargs = varargs
    test_csr_getrow.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_csr_getrow', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_csr_getrow', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_csr_getrow(...)' code ##################

    
    # Assigning a Num to a Name (line 33):
    int_459760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 8), 'int')
    # Assigning a type to the variable 'N' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'N', int_459760)
    
    # Call to seed(...): (line 34)
    # Processing the call arguments (line 34)
    int_459764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 19), 'int')
    # Processing the call keyword arguments (line 34)
    kwargs_459765 = {}
    # Getting the type of 'np' (line 34)
    np_459761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 34)
    random_459762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 4), np_459761, 'random')
    # Obtaining the member 'seed' of a type (line 34)
    seed_459763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 4), random_459762, 'seed')
    # Calling seed(args, kwargs) (line 34)
    seed_call_result_459766 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), seed_459763, *[int_459764], **kwargs_459765)
    
    
    # Assigning a Call to a Name (line 35):
    
    # Call to random(...): (line 35)
    # Processing the call arguments (line 35)
    
    # Obtaining an instance of the builtin type 'tuple' (line 35)
    tuple_459770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 35)
    # Adding element type (line 35)
    # Getting the type of 'N' (line 35)
    N_459771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 26), 'N', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 26), tuple_459770, N_459771)
    # Adding element type (line 35)
    # Getting the type of 'N' (line 35)
    N_459772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 29), 'N', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 26), tuple_459770, N_459772)
    
    # Processing the call keyword arguments (line 35)
    kwargs_459773 = {}
    # Getting the type of 'np' (line 35)
    np_459767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 35)
    random_459768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), np_459767, 'random')
    # Obtaining the member 'random' of a type (line 35)
    random_459769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), random_459768, 'random')
    # Calling random(args, kwargs) (line 35)
    random_call_result_459774 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), random_459769, *[tuple_459770], **kwargs_459773)
    
    # Assigning a type to the variable 'X' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'X', random_call_result_459774)
    
    # Assigning a Num to a Subscript (line 36):
    int_459775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 17), 'int')
    # Getting the type of 'X' (line 36)
    X_459776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'X')
    
    # Getting the type of 'X' (line 36)
    X_459777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 6), 'X')
    float_459778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 10), 'float')
    # Applying the binary operator '>' (line 36)
    result_gt_459779 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 6), '>', X_459777, float_459778)
    
    # Storing an element on a container (line 36)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 4), X_459776, (result_gt_459779, int_459775))
    
    # Assigning a Call to a Name (line 37):
    
    # Call to csr_matrix(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'X' (line 37)
    X_459781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 22), 'X', False)
    # Processing the call keyword arguments (line 37)
    kwargs_459782 = {}
    # Getting the type of 'csr_matrix' (line 37)
    csr_matrix_459780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 11), 'csr_matrix', False)
    # Calling csr_matrix(args, kwargs) (line 37)
    csr_matrix_call_result_459783 = invoke(stypy.reporting.localization.Localization(__file__, 37, 11), csr_matrix_459780, *[X_459781], **kwargs_459782)
    
    # Assigning a type to the variable 'Xcsr' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'Xcsr', csr_matrix_call_result_459783)
    
    
    # Call to range(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'N' (line 39)
    N_459785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 19), 'N', False)
    # Processing the call keyword arguments (line 39)
    kwargs_459786 = {}
    # Getting the type of 'range' (line 39)
    range_459784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 13), 'range', False)
    # Calling range(args, kwargs) (line 39)
    range_call_result_459787 = invoke(stypy.reporting.localization.Localization(__file__, 39, 13), range_459784, *[N_459785], **kwargs_459786)
    
    # Testing the type of a for loop iterable (line 39)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 39, 4), range_call_result_459787)
    # Getting the type of the for loop variable (line 39)
    for_loop_var_459788 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 39, 4), range_call_result_459787)
    # Assigning a type to the variable 'i' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'i', for_loop_var_459788)
    # SSA begins for a for statement (line 39)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 40):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 40)
    i_459789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 20), 'i')
    # Getting the type of 'i' (line 40)
    i_459790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 22), 'i')
    int_459791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 26), 'int')
    # Applying the binary operator '+' (line 40)
    result_add_459792 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 22), '+', i_459790, int_459791)
    
    slice_459793 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 40, 18), i_459789, result_add_459792, None)
    slice_459794 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 40, 18), None, None, None)
    # Getting the type of 'X' (line 40)
    X_459795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 18), 'X')
    # Obtaining the member '__getitem__' of a type (line 40)
    getitem___459796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 18), X_459795, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
    subscript_call_result_459797 = invoke(stypy.reporting.localization.Localization(__file__, 40, 18), getitem___459796, (slice_459793, slice_459794))
    
    # Assigning a type to the variable 'arr_row' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'arr_row', subscript_call_result_459797)
    
    # Assigning a Call to a Name (line 41):
    
    # Call to getrow(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'i' (line 41)
    i_459800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 30), 'i', False)
    # Processing the call keyword arguments (line 41)
    kwargs_459801 = {}
    # Getting the type of 'Xcsr' (line 41)
    Xcsr_459798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 18), 'Xcsr', False)
    # Obtaining the member 'getrow' of a type (line 41)
    getrow_459799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 18), Xcsr_459798, 'getrow')
    # Calling getrow(args, kwargs) (line 41)
    getrow_call_result_459802 = invoke(stypy.reporting.localization.Localization(__file__, 41, 18), getrow_459799, *[i_459800], **kwargs_459801)
    
    # Assigning a type to the variable 'csr_row' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'csr_row', getrow_call_result_459802)
    
    # Call to assert_array_almost_equal(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'arr_row' (line 43)
    arr_row_459804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 34), 'arr_row', False)
    
    # Call to toarray(...): (line 43)
    # Processing the call keyword arguments (line 43)
    kwargs_459807 = {}
    # Getting the type of 'csr_row' (line 43)
    csr_row_459805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 43), 'csr_row', False)
    # Obtaining the member 'toarray' of a type (line 43)
    toarray_459806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 43), csr_row_459805, 'toarray')
    # Calling toarray(args, kwargs) (line 43)
    toarray_call_result_459808 = invoke(stypy.reporting.localization.Localization(__file__, 43, 43), toarray_459806, *[], **kwargs_459807)
    
    # Processing the call keyword arguments (line 43)
    kwargs_459809 = {}
    # Getting the type of 'assert_array_almost_equal' (line 43)
    assert_array_almost_equal_459803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 43)
    assert_array_almost_equal_call_result_459810 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), assert_array_almost_equal_459803, *[arr_row_459804, toarray_call_result_459808], **kwargs_459809)
    
    
    # Call to assert_(...): (line 44)
    # Processing the call arguments (line 44)
    
    
    # Call to type(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'csr_row' (line 44)
    csr_row_459813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 21), 'csr_row', False)
    # Processing the call keyword arguments (line 44)
    kwargs_459814 = {}
    # Getting the type of 'type' (line 44)
    type_459812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'type', False)
    # Calling type(args, kwargs) (line 44)
    type_call_result_459815 = invoke(stypy.reporting.localization.Localization(__file__, 44, 16), type_459812, *[csr_row_459813], **kwargs_459814)
    
    # Getting the type of 'csr_matrix' (line 44)
    csr_matrix_459816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 33), 'csr_matrix', False)
    # Applying the binary operator 'is' (line 44)
    result_is__459817 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 16), 'is', type_call_result_459815, csr_matrix_459816)
    
    # Processing the call keyword arguments (line 44)
    kwargs_459818 = {}
    # Getting the type of 'assert_' (line 44)
    assert__459811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 44)
    assert__call_result_459819 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), assert__459811, *[result_is__459817], **kwargs_459818)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_csr_getrow(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_csr_getrow' in the type store
    # Getting the type of 'stypy_return_type' (line 32)
    stypy_return_type_459820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_459820)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_csr_getrow'
    return stypy_return_type_459820

# Assigning a type to the variable 'test_csr_getrow' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'test_csr_getrow', test_csr_getrow)

@norecursion
def test_csr_getcol(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_csr_getcol'
    module_type_store = module_type_store.open_function_context('test_csr_getcol', 47, 0, False)
    
    # Passed parameters checking function
    test_csr_getcol.stypy_localization = localization
    test_csr_getcol.stypy_type_of_self = None
    test_csr_getcol.stypy_type_store = module_type_store
    test_csr_getcol.stypy_function_name = 'test_csr_getcol'
    test_csr_getcol.stypy_param_names_list = []
    test_csr_getcol.stypy_varargs_param_name = None
    test_csr_getcol.stypy_kwargs_param_name = None
    test_csr_getcol.stypy_call_defaults = defaults
    test_csr_getcol.stypy_call_varargs = varargs
    test_csr_getcol.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_csr_getcol', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_csr_getcol', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_csr_getcol(...)' code ##################

    
    # Assigning a Num to a Name (line 48):
    int_459821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 8), 'int')
    # Assigning a type to the variable 'N' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'N', int_459821)
    
    # Call to seed(...): (line 49)
    # Processing the call arguments (line 49)
    int_459825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 19), 'int')
    # Processing the call keyword arguments (line 49)
    kwargs_459826 = {}
    # Getting the type of 'np' (line 49)
    np_459822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 49)
    random_459823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 4), np_459822, 'random')
    # Obtaining the member 'seed' of a type (line 49)
    seed_459824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 4), random_459823, 'seed')
    # Calling seed(args, kwargs) (line 49)
    seed_call_result_459827 = invoke(stypy.reporting.localization.Localization(__file__, 49, 4), seed_459824, *[int_459825], **kwargs_459826)
    
    
    # Assigning a Call to a Name (line 50):
    
    # Call to random(...): (line 50)
    # Processing the call arguments (line 50)
    
    # Obtaining an instance of the builtin type 'tuple' (line 50)
    tuple_459831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 50)
    # Adding element type (line 50)
    # Getting the type of 'N' (line 50)
    N_459832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 26), 'N', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 26), tuple_459831, N_459832)
    # Adding element type (line 50)
    # Getting the type of 'N' (line 50)
    N_459833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 29), 'N', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 26), tuple_459831, N_459833)
    
    # Processing the call keyword arguments (line 50)
    kwargs_459834 = {}
    # Getting the type of 'np' (line 50)
    np_459828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 50)
    random_459829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), np_459828, 'random')
    # Obtaining the member 'random' of a type (line 50)
    random_459830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), random_459829, 'random')
    # Calling random(args, kwargs) (line 50)
    random_call_result_459835 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), random_459830, *[tuple_459831], **kwargs_459834)
    
    # Assigning a type to the variable 'X' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'X', random_call_result_459835)
    
    # Assigning a Num to a Subscript (line 51):
    int_459836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 17), 'int')
    # Getting the type of 'X' (line 51)
    X_459837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'X')
    
    # Getting the type of 'X' (line 51)
    X_459838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 6), 'X')
    float_459839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 10), 'float')
    # Applying the binary operator '>' (line 51)
    result_gt_459840 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 6), '>', X_459838, float_459839)
    
    # Storing an element on a container (line 51)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 4), X_459837, (result_gt_459840, int_459836))
    
    # Assigning a Call to a Name (line 52):
    
    # Call to csr_matrix(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'X' (line 52)
    X_459842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 22), 'X', False)
    # Processing the call keyword arguments (line 52)
    kwargs_459843 = {}
    # Getting the type of 'csr_matrix' (line 52)
    csr_matrix_459841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'csr_matrix', False)
    # Calling csr_matrix(args, kwargs) (line 52)
    csr_matrix_call_result_459844 = invoke(stypy.reporting.localization.Localization(__file__, 52, 11), csr_matrix_459841, *[X_459842], **kwargs_459843)
    
    # Assigning a type to the variable 'Xcsr' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'Xcsr', csr_matrix_call_result_459844)
    
    
    # Call to range(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'N' (line 54)
    N_459846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 19), 'N', False)
    # Processing the call keyword arguments (line 54)
    kwargs_459847 = {}
    # Getting the type of 'range' (line 54)
    range_459845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 13), 'range', False)
    # Calling range(args, kwargs) (line 54)
    range_call_result_459848 = invoke(stypy.reporting.localization.Localization(__file__, 54, 13), range_459845, *[N_459846], **kwargs_459847)
    
    # Testing the type of a for loop iterable (line 54)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 54, 4), range_call_result_459848)
    # Getting the type of the for loop variable (line 54)
    for_loop_var_459849 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 54, 4), range_call_result_459848)
    # Assigning a type to the variable 'i' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'i', for_loop_var_459849)
    # SSA begins for a for statement (line 54)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 55):
    
    # Obtaining the type of the subscript
    slice_459850 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 55, 18), None, None, None)
    # Getting the type of 'i' (line 55)
    i_459851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 23), 'i')
    # Getting the type of 'i' (line 55)
    i_459852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 25), 'i')
    int_459853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 29), 'int')
    # Applying the binary operator '+' (line 55)
    result_add_459854 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 25), '+', i_459852, int_459853)
    
    slice_459855 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 55, 18), i_459851, result_add_459854, None)
    # Getting the type of 'X' (line 55)
    X_459856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 18), 'X')
    # Obtaining the member '__getitem__' of a type (line 55)
    getitem___459857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 18), X_459856, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 55)
    subscript_call_result_459858 = invoke(stypy.reporting.localization.Localization(__file__, 55, 18), getitem___459857, (slice_459850, slice_459855))
    
    # Assigning a type to the variable 'arr_col' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'arr_col', subscript_call_result_459858)
    
    # Assigning a Call to a Name (line 56):
    
    # Call to getcol(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'i' (line 56)
    i_459861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 30), 'i', False)
    # Processing the call keyword arguments (line 56)
    kwargs_459862 = {}
    # Getting the type of 'Xcsr' (line 56)
    Xcsr_459859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 18), 'Xcsr', False)
    # Obtaining the member 'getcol' of a type (line 56)
    getcol_459860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 18), Xcsr_459859, 'getcol')
    # Calling getcol(args, kwargs) (line 56)
    getcol_call_result_459863 = invoke(stypy.reporting.localization.Localization(__file__, 56, 18), getcol_459860, *[i_459861], **kwargs_459862)
    
    # Assigning a type to the variable 'csr_col' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'csr_col', getcol_call_result_459863)
    
    # Call to assert_array_almost_equal(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'arr_col' (line 58)
    arr_col_459865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 34), 'arr_col', False)
    
    # Call to toarray(...): (line 58)
    # Processing the call keyword arguments (line 58)
    kwargs_459868 = {}
    # Getting the type of 'csr_col' (line 58)
    csr_col_459866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 43), 'csr_col', False)
    # Obtaining the member 'toarray' of a type (line 58)
    toarray_459867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 43), csr_col_459866, 'toarray')
    # Calling toarray(args, kwargs) (line 58)
    toarray_call_result_459869 = invoke(stypy.reporting.localization.Localization(__file__, 58, 43), toarray_459867, *[], **kwargs_459868)
    
    # Processing the call keyword arguments (line 58)
    kwargs_459870 = {}
    # Getting the type of 'assert_array_almost_equal' (line 58)
    assert_array_almost_equal_459864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 58)
    assert_array_almost_equal_call_result_459871 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), assert_array_almost_equal_459864, *[arr_col_459865, toarray_call_result_459869], **kwargs_459870)
    
    
    # Call to assert_(...): (line 59)
    # Processing the call arguments (line 59)
    
    
    # Call to type(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'csr_col' (line 59)
    csr_col_459874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 21), 'csr_col', False)
    # Processing the call keyword arguments (line 59)
    kwargs_459875 = {}
    # Getting the type of 'type' (line 59)
    type_459873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'type', False)
    # Calling type(args, kwargs) (line 59)
    type_call_result_459876 = invoke(stypy.reporting.localization.Localization(__file__, 59, 16), type_459873, *[csr_col_459874], **kwargs_459875)
    
    # Getting the type of 'csr_matrix' (line 59)
    csr_matrix_459877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 33), 'csr_matrix', False)
    # Applying the binary operator 'is' (line 59)
    result_is__459878 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 16), 'is', type_call_result_459876, csr_matrix_459877)
    
    # Processing the call keyword arguments (line 59)
    kwargs_459879 = {}
    # Getting the type of 'assert_' (line 59)
    assert__459872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 59)
    assert__call_result_459880 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), assert__459872, *[result_is__459878], **kwargs_459879)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_csr_getcol(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_csr_getcol' in the type store
    # Getting the type of 'stypy_return_type' (line 47)
    stypy_return_type_459881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_459881)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_csr_getcol'
    return stypy_return_type_459881

# Assigning a type to the variable 'test_csr_getcol' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'test_csr_getcol', test_csr_getcol)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
