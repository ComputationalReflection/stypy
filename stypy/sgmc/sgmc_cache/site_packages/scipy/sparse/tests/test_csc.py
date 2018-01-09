
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import assert_array_almost_equal, assert_
5: from scipy.sparse import csr_matrix, csc_matrix
6: 
7: 
8: def test_csc_getrow():
9:     N = 10
10:     np.random.seed(0)
11:     X = np.random.random((N, N))
12:     X[X > 0.7] = 0
13:     Xcsc = csc_matrix(X)
14: 
15:     for i in range(N):
16:         arr_row = X[i:i + 1, :]
17:         csc_row = Xcsc.getrow(i)
18: 
19:         assert_array_almost_equal(arr_row, csc_row.toarray())
20:         assert_(type(csc_row) is csr_matrix)
21: 
22: 
23: def test_csc_getcol():
24:     N = 10
25:     np.random.seed(0)
26:     X = np.random.random((N, N))
27:     X[X > 0.7] = 0
28:     Xcsc = csc_matrix(X)
29: 
30:     for i in range(N):
31:         arr_col = X[:, i:i + 1]
32:         csc_col = Xcsc.getcol(i)
33: 
34:         assert_array_almost_equal(arr_col, csc_col.toarray())
35:         assert_(type(csc_col) is csc_matrix)
36: 
37: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_459529 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_459529) is not StypyTypeError):

    if (import_459529 != 'pyd_module'):
        __import__(import_459529)
        sys_modules_459530 = sys.modules[import_459529]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_459530.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_459529)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_array_almost_equal, assert_' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_459531 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_459531) is not StypyTypeError):

    if (import_459531 != 'pyd_module'):
        __import__(import_459531)
        sys_modules_459532 = sys.modules[import_459531]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_459532.module_type_store, module_type_store, ['assert_array_almost_equal', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_459532, sys_modules_459532.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_array_almost_equal, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_array_almost_equal', 'assert_'], [assert_array_almost_equal, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_459531)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.sparse import csr_matrix, csc_matrix' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_459533 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse')

if (type(import_459533) is not StypyTypeError):

    if (import_459533 != 'pyd_module'):
        __import__(import_459533)
        sys_modules_459534 = sys.modules[import_459533]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse', sys_modules_459534.module_type_store, module_type_store, ['csr_matrix', 'csc_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_459534, sys_modules_459534.module_type_store, module_type_store)
    else:
        from scipy.sparse import csr_matrix, csc_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse', None, module_type_store, ['csr_matrix', 'csc_matrix'], [csr_matrix, csc_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse', import_459533)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')


@norecursion
def test_csc_getrow(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_csc_getrow'
    module_type_store = module_type_store.open_function_context('test_csc_getrow', 8, 0, False)
    
    # Passed parameters checking function
    test_csc_getrow.stypy_localization = localization
    test_csc_getrow.stypy_type_of_self = None
    test_csc_getrow.stypy_type_store = module_type_store
    test_csc_getrow.stypy_function_name = 'test_csc_getrow'
    test_csc_getrow.stypy_param_names_list = []
    test_csc_getrow.stypy_varargs_param_name = None
    test_csc_getrow.stypy_kwargs_param_name = None
    test_csc_getrow.stypy_call_defaults = defaults
    test_csc_getrow.stypy_call_varargs = varargs
    test_csc_getrow.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_csc_getrow', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_csc_getrow', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_csc_getrow(...)' code ##################

    
    # Assigning a Num to a Name (line 9):
    int_459535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 8), 'int')
    # Assigning a type to the variable 'N' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'N', int_459535)
    
    # Call to seed(...): (line 10)
    # Processing the call arguments (line 10)
    int_459539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 19), 'int')
    # Processing the call keyword arguments (line 10)
    kwargs_459540 = {}
    # Getting the type of 'np' (line 10)
    np_459536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 10)
    random_459537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 4), np_459536, 'random')
    # Obtaining the member 'seed' of a type (line 10)
    seed_459538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 4), random_459537, 'seed')
    # Calling seed(args, kwargs) (line 10)
    seed_call_result_459541 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), seed_459538, *[int_459539], **kwargs_459540)
    
    
    # Assigning a Call to a Name (line 11):
    
    # Call to random(...): (line 11)
    # Processing the call arguments (line 11)
    
    # Obtaining an instance of the builtin type 'tuple' (line 11)
    tuple_459545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 11)
    # Adding element type (line 11)
    # Getting the type of 'N' (line 11)
    N_459546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 26), 'N', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 26), tuple_459545, N_459546)
    # Adding element type (line 11)
    # Getting the type of 'N' (line 11)
    N_459547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 29), 'N', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 26), tuple_459545, N_459547)
    
    # Processing the call keyword arguments (line 11)
    kwargs_459548 = {}
    # Getting the type of 'np' (line 11)
    np_459542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 11)
    random_459543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 8), np_459542, 'random')
    # Obtaining the member 'random' of a type (line 11)
    random_459544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 8), random_459543, 'random')
    # Calling random(args, kwargs) (line 11)
    random_call_result_459549 = invoke(stypy.reporting.localization.Localization(__file__, 11, 8), random_459544, *[tuple_459545], **kwargs_459548)
    
    # Assigning a type to the variable 'X' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'X', random_call_result_459549)
    
    # Assigning a Num to a Subscript (line 12):
    int_459550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 17), 'int')
    # Getting the type of 'X' (line 12)
    X_459551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'X')
    
    # Getting the type of 'X' (line 12)
    X_459552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 6), 'X')
    float_459553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 10), 'float')
    # Applying the binary operator '>' (line 12)
    result_gt_459554 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 6), '>', X_459552, float_459553)
    
    # Storing an element on a container (line 12)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 4), X_459551, (result_gt_459554, int_459550))
    
    # Assigning a Call to a Name (line 13):
    
    # Call to csc_matrix(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'X' (line 13)
    X_459556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 22), 'X', False)
    # Processing the call keyword arguments (line 13)
    kwargs_459557 = {}
    # Getting the type of 'csc_matrix' (line 13)
    csc_matrix_459555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 11), 'csc_matrix', False)
    # Calling csc_matrix(args, kwargs) (line 13)
    csc_matrix_call_result_459558 = invoke(stypy.reporting.localization.Localization(__file__, 13, 11), csc_matrix_459555, *[X_459556], **kwargs_459557)
    
    # Assigning a type to the variable 'Xcsc' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'Xcsc', csc_matrix_call_result_459558)
    
    
    # Call to range(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'N' (line 15)
    N_459560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 19), 'N', False)
    # Processing the call keyword arguments (line 15)
    kwargs_459561 = {}
    # Getting the type of 'range' (line 15)
    range_459559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 13), 'range', False)
    # Calling range(args, kwargs) (line 15)
    range_call_result_459562 = invoke(stypy.reporting.localization.Localization(__file__, 15, 13), range_459559, *[N_459560], **kwargs_459561)
    
    # Testing the type of a for loop iterable (line 15)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 15, 4), range_call_result_459562)
    # Getting the type of the for loop variable (line 15)
    for_loop_var_459563 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 15, 4), range_call_result_459562)
    # Assigning a type to the variable 'i' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'i', for_loop_var_459563)
    # SSA begins for a for statement (line 15)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 16):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 16)
    i_459564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 20), 'i')
    # Getting the type of 'i' (line 16)
    i_459565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 22), 'i')
    int_459566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 26), 'int')
    # Applying the binary operator '+' (line 16)
    result_add_459567 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 22), '+', i_459565, int_459566)
    
    slice_459568 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 16, 18), i_459564, result_add_459567, None)
    slice_459569 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 16, 18), None, None, None)
    # Getting the type of 'X' (line 16)
    X_459570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 18), 'X')
    # Obtaining the member '__getitem__' of a type (line 16)
    getitem___459571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 18), X_459570, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 16)
    subscript_call_result_459572 = invoke(stypy.reporting.localization.Localization(__file__, 16, 18), getitem___459571, (slice_459568, slice_459569))
    
    # Assigning a type to the variable 'arr_row' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'arr_row', subscript_call_result_459572)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to getrow(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 'i' (line 17)
    i_459575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 30), 'i', False)
    # Processing the call keyword arguments (line 17)
    kwargs_459576 = {}
    # Getting the type of 'Xcsc' (line 17)
    Xcsc_459573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 18), 'Xcsc', False)
    # Obtaining the member 'getrow' of a type (line 17)
    getrow_459574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 18), Xcsc_459573, 'getrow')
    # Calling getrow(args, kwargs) (line 17)
    getrow_call_result_459577 = invoke(stypy.reporting.localization.Localization(__file__, 17, 18), getrow_459574, *[i_459575], **kwargs_459576)
    
    # Assigning a type to the variable 'csc_row' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'csc_row', getrow_call_result_459577)
    
    # Call to assert_array_almost_equal(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'arr_row' (line 19)
    arr_row_459579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 34), 'arr_row', False)
    
    # Call to toarray(...): (line 19)
    # Processing the call keyword arguments (line 19)
    kwargs_459582 = {}
    # Getting the type of 'csc_row' (line 19)
    csc_row_459580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 43), 'csc_row', False)
    # Obtaining the member 'toarray' of a type (line 19)
    toarray_459581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 43), csc_row_459580, 'toarray')
    # Calling toarray(args, kwargs) (line 19)
    toarray_call_result_459583 = invoke(stypy.reporting.localization.Localization(__file__, 19, 43), toarray_459581, *[], **kwargs_459582)
    
    # Processing the call keyword arguments (line 19)
    kwargs_459584 = {}
    # Getting the type of 'assert_array_almost_equal' (line 19)
    assert_array_almost_equal_459578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 19)
    assert_array_almost_equal_call_result_459585 = invoke(stypy.reporting.localization.Localization(__file__, 19, 8), assert_array_almost_equal_459578, *[arr_row_459579, toarray_call_result_459583], **kwargs_459584)
    
    
    # Call to assert_(...): (line 20)
    # Processing the call arguments (line 20)
    
    
    # Call to type(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'csc_row' (line 20)
    csc_row_459588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 21), 'csc_row', False)
    # Processing the call keyword arguments (line 20)
    kwargs_459589 = {}
    # Getting the type of 'type' (line 20)
    type_459587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'type', False)
    # Calling type(args, kwargs) (line 20)
    type_call_result_459590 = invoke(stypy.reporting.localization.Localization(__file__, 20, 16), type_459587, *[csc_row_459588], **kwargs_459589)
    
    # Getting the type of 'csr_matrix' (line 20)
    csr_matrix_459591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 33), 'csr_matrix', False)
    # Applying the binary operator 'is' (line 20)
    result_is__459592 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 16), 'is', type_call_result_459590, csr_matrix_459591)
    
    # Processing the call keyword arguments (line 20)
    kwargs_459593 = {}
    # Getting the type of 'assert_' (line 20)
    assert__459586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 20)
    assert__call_result_459594 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), assert__459586, *[result_is__459592], **kwargs_459593)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_csc_getrow(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_csc_getrow' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_459595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_459595)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_csc_getrow'
    return stypy_return_type_459595

# Assigning a type to the variable 'test_csc_getrow' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'test_csc_getrow', test_csc_getrow)

@norecursion
def test_csc_getcol(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_csc_getcol'
    module_type_store = module_type_store.open_function_context('test_csc_getcol', 23, 0, False)
    
    # Passed parameters checking function
    test_csc_getcol.stypy_localization = localization
    test_csc_getcol.stypy_type_of_self = None
    test_csc_getcol.stypy_type_store = module_type_store
    test_csc_getcol.stypy_function_name = 'test_csc_getcol'
    test_csc_getcol.stypy_param_names_list = []
    test_csc_getcol.stypy_varargs_param_name = None
    test_csc_getcol.stypy_kwargs_param_name = None
    test_csc_getcol.stypy_call_defaults = defaults
    test_csc_getcol.stypy_call_varargs = varargs
    test_csc_getcol.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_csc_getcol', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_csc_getcol', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_csc_getcol(...)' code ##################

    
    # Assigning a Num to a Name (line 24):
    int_459596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 8), 'int')
    # Assigning a type to the variable 'N' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'N', int_459596)
    
    # Call to seed(...): (line 25)
    # Processing the call arguments (line 25)
    int_459600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 19), 'int')
    # Processing the call keyword arguments (line 25)
    kwargs_459601 = {}
    # Getting the type of 'np' (line 25)
    np_459597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 25)
    random_459598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 4), np_459597, 'random')
    # Obtaining the member 'seed' of a type (line 25)
    seed_459599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 4), random_459598, 'seed')
    # Calling seed(args, kwargs) (line 25)
    seed_call_result_459602 = invoke(stypy.reporting.localization.Localization(__file__, 25, 4), seed_459599, *[int_459600], **kwargs_459601)
    
    
    # Assigning a Call to a Name (line 26):
    
    # Call to random(...): (line 26)
    # Processing the call arguments (line 26)
    
    # Obtaining an instance of the builtin type 'tuple' (line 26)
    tuple_459606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 26)
    # Adding element type (line 26)
    # Getting the type of 'N' (line 26)
    N_459607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 26), 'N', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 26), tuple_459606, N_459607)
    # Adding element type (line 26)
    # Getting the type of 'N' (line 26)
    N_459608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 29), 'N', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 26), tuple_459606, N_459608)
    
    # Processing the call keyword arguments (line 26)
    kwargs_459609 = {}
    # Getting the type of 'np' (line 26)
    np_459603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 26)
    random_459604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), np_459603, 'random')
    # Obtaining the member 'random' of a type (line 26)
    random_459605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), random_459604, 'random')
    # Calling random(args, kwargs) (line 26)
    random_call_result_459610 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), random_459605, *[tuple_459606], **kwargs_459609)
    
    # Assigning a type to the variable 'X' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'X', random_call_result_459610)
    
    # Assigning a Num to a Subscript (line 27):
    int_459611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 17), 'int')
    # Getting the type of 'X' (line 27)
    X_459612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'X')
    
    # Getting the type of 'X' (line 27)
    X_459613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 6), 'X')
    float_459614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 10), 'float')
    # Applying the binary operator '>' (line 27)
    result_gt_459615 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 6), '>', X_459613, float_459614)
    
    # Storing an element on a container (line 27)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 4), X_459612, (result_gt_459615, int_459611))
    
    # Assigning a Call to a Name (line 28):
    
    # Call to csc_matrix(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'X' (line 28)
    X_459617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 22), 'X', False)
    # Processing the call keyword arguments (line 28)
    kwargs_459618 = {}
    # Getting the type of 'csc_matrix' (line 28)
    csc_matrix_459616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 11), 'csc_matrix', False)
    # Calling csc_matrix(args, kwargs) (line 28)
    csc_matrix_call_result_459619 = invoke(stypy.reporting.localization.Localization(__file__, 28, 11), csc_matrix_459616, *[X_459617], **kwargs_459618)
    
    # Assigning a type to the variable 'Xcsc' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'Xcsc', csc_matrix_call_result_459619)
    
    
    # Call to range(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'N' (line 30)
    N_459621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 19), 'N', False)
    # Processing the call keyword arguments (line 30)
    kwargs_459622 = {}
    # Getting the type of 'range' (line 30)
    range_459620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 13), 'range', False)
    # Calling range(args, kwargs) (line 30)
    range_call_result_459623 = invoke(stypy.reporting.localization.Localization(__file__, 30, 13), range_459620, *[N_459621], **kwargs_459622)
    
    # Testing the type of a for loop iterable (line 30)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 30, 4), range_call_result_459623)
    # Getting the type of the for loop variable (line 30)
    for_loop_var_459624 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 30, 4), range_call_result_459623)
    # Assigning a type to the variable 'i' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'i', for_loop_var_459624)
    # SSA begins for a for statement (line 30)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 31):
    
    # Obtaining the type of the subscript
    slice_459625 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 31, 18), None, None, None)
    # Getting the type of 'i' (line 31)
    i_459626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 23), 'i')
    # Getting the type of 'i' (line 31)
    i_459627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 25), 'i')
    int_459628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 29), 'int')
    # Applying the binary operator '+' (line 31)
    result_add_459629 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 25), '+', i_459627, int_459628)
    
    slice_459630 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 31, 18), i_459626, result_add_459629, None)
    # Getting the type of 'X' (line 31)
    X_459631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 18), 'X')
    # Obtaining the member '__getitem__' of a type (line 31)
    getitem___459632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 18), X_459631, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 31)
    subscript_call_result_459633 = invoke(stypy.reporting.localization.Localization(__file__, 31, 18), getitem___459632, (slice_459625, slice_459630))
    
    # Assigning a type to the variable 'arr_col' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'arr_col', subscript_call_result_459633)
    
    # Assigning a Call to a Name (line 32):
    
    # Call to getcol(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'i' (line 32)
    i_459636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 30), 'i', False)
    # Processing the call keyword arguments (line 32)
    kwargs_459637 = {}
    # Getting the type of 'Xcsc' (line 32)
    Xcsc_459634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 18), 'Xcsc', False)
    # Obtaining the member 'getcol' of a type (line 32)
    getcol_459635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 18), Xcsc_459634, 'getcol')
    # Calling getcol(args, kwargs) (line 32)
    getcol_call_result_459638 = invoke(stypy.reporting.localization.Localization(__file__, 32, 18), getcol_459635, *[i_459636], **kwargs_459637)
    
    # Assigning a type to the variable 'csc_col' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'csc_col', getcol_call_result_459638)
    
    # Call to assert_array_almost_equal(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'arr_col' (line 34)
    arr_col_459640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 34), 'arr_col', False)
    
    # Call to toarray(...): (line 34)
    # Processing the call keyword arguments (line 34)
    kwargs_459643 = {}
    # Getting the type of 'csc_col' (line 34)
    csc_col_459641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 43), 'csc_col', False)
    # Obtaining the member 'toarray' of a type (line 34)
    toarray_459642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 43), csc_col_459641, 'toarray')
    # Calling toarray(args, kwargs) (line 34)
    toarray_call_result_459644 = invoke(stypy.reporting.localization.Localization(__file__, 34, 43), toarray_459642, *[], **kwargs_459643)
    
    # Processing the call keyword arguments (line 34)
    kwargs_459645 = {}
    # Getting the type of 'assert_array_almost_equal' (line 34)
    assert_array_almost_equal_459639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 34)
    assert_array_almost_equal_call_result_459646 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), assert_array_almost_equal_459639, *[arr_col_459640, toarray_call_result_459644], **kwargs_459645)
    
    
    # Call to assert_(...): (line 35)
    # Processing the call arguments (line 35)
    
    
    # Call to type(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'csc_col' (line 35)
    csc_col_459649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 21), 'csc_col', False)
    # Processing the call keyword arguments (line 35)
    kwargs_459650 = {}
    # Getting the type of 'type' (line 35)
    type_459648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 16), 'type', False)
    # Calling type(args, kwargs) (line 35)
    type_call_result_459651 = invoke(stypy.reporting.localization.Localization(__file__, 35, 16), type_459648, *[csc_col_459649], **kwargs_459650)
    
    # Getting the type of 'csc_matrix' (line 35)
    csc_matrix_459652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 33), 'csc_matrix', False)
    # Applying the binary operator 'is' (line 35)
    result_is__459653 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 16), 'is', type_call_result_459651, csc_matrix_459652)
    
    # Processing the call keyword arguments (line 35)
    kwargs_459654 = {}
    # Getting the type of 'assert_' (line 35)
    assert__459647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 35)
    assert__call_result_459655 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), assert__459647, *[result_is__459653], **kwargs_459654)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_csc_getcol(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_csc_getcol' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_459656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_459656)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_csc_getcol'
    return stypy_return_type_459656

# Assigning a type to the variable 'test_csc_getcol' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'test_csc_getcol', test_csc_getcol)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
