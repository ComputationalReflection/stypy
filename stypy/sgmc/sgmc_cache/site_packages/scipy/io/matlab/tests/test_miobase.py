
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Testing miobase module
2: '''
3: 
4: import numpy as np
5: 
6: from numpy.testing import assert_equal
7: from pytest import raises as assert_raises
8: 
9: from scipy.io.matlab.miobase import matdims
10: 
11: 
12: def test_matdims():
13:     # Test matdims dimension finder
14:     assert_equal(matdims(np.array(1)), (1, 1))  # numpy scalar
15:     assert_equal(matdims(np.array([1])), (1, 1))  # 1d array, 1 element
16:     assert_equal(matdims(np.array([1,2])), (2, 1))  # 1d array, 2 elements
17:     assert_equal(matdims(np.array([[2],[3]])), (2, 1))  # 2d array, column vector
18:     assert_equal(matdims(np.array([[2,3]])), (1, 2))  # 2d array, row vector
19:     # 3d array, rowish vector
20:     assert_equal(matdims(np.array([[[2,3]]])), (1, 1, 2))
21:     assert_equal(matdims(np.array([])), (0, 0))  # empty 1d array
22:     assert_equal(matdims(np.array([[]])), (0, 0))  # empty 2d
23:     assert_equal(matdims(np.array([[[]]])), (0, 0, 0))  # empty 3d
24:     # Optional argument flips 1-D shape behavior.
25:     assert_equal(matdims(np.array([1,2]), 'row'), (1, 2))  # 1d array, 2 elements
26:     # The argument has to make sense though
27:     assert_raises(ValueError, matdims, np.array([1,2]), 'bizarre')
28:     # Check empty sparse matrices get their own shape
29:     from scipy.sparse import csr_matrix, csc_matrix
30:     assert_equal(matdims(csr_matrix(np.zeros((3, 3)))), (3, 3))
31:     assert_equal(matdims(csc_matrix(np.zeros((2, 2)))), (2, 2))
32: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_143987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', ' Testing miobase module\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_143988 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_143988) is not StypyTypeError):

    if (import_143988 != 'pyd_module'):
        __import__(import_143988)
        sys_modules_143989 = sys.modules[import_143988]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_143989.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_143988)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.testing import assert_equal' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_143990 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing')

if (type(import_143990) is not StypyTypeError):

    if (import_143990 != 'pyd_module'):
        __import__(import_143990)
        sys_modules_143991 = sys.modules[import_143990]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', sys_modules_143991.module_type_store, module_type_store, ['assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_143991, sys_modules_143991.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', None, module_type_store, ['assert_equal'], [assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', import_143990)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from pytest import assert_raises' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_143992 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest')

if (type(import_143992) is not StypyTypeError):

    if (import_143992 != 'pyd_module'):
        __import__(import_143992)
        sys_modules_143993 = sys.modules[import_143992]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', sys_modules_143993.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_143993, sys_modules_143993.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', import_143992)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.io.matlab.miobase import matdims' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_143994 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.io.matlab.miobase')

if (type(import_143994) is not StypyTypeError):

    if (import_143994 != 'pyd_module'):
        __import__(import_143994)
        sys_modules_143995 = sys.modules[import_143994]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.io.matlab.miobase', sys_modules_143995.module_type_store, module_type_store, ['matdims'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_143995, sys_modules_143995.module_type_store, module_type_store)
    else:
        from scipy.io.matlab.miobase import matdims

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.io.matlab.miobase', None, module_type_store, ['matdims'], [matdims])

else:
    # Assigning a type to the variable 'scipy.io.matlab.miobase' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.io.matlab.miobase', import_143994)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')


@norecursion
def test_matdims(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_matdims'
    module_type_store = module_type_store.open_function_context('test_matdims', 12, 0, False)
    
    # Passed parameters checking function
    test_matdims.stypy_localization = localization
    test_matdims.stypy_type_of_self = None
    test_matdims.stypy_type_store = module_type_store
    test_matdims.stypy_function_name = 'test_matdims'
    test_matdims.stypy_param_names_list = []
    test_matdims.stypy_varargs_param_name = None
    test_matdims.stypy_kwargs_param_name = None
    test_matdims.stypy_call_defaults = defaults
    test_matdims.stypy_call_varargs = varargs
    test_matdims.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_matdims', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_matdims', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_matdims(...)' code ##################

    
    # Call to assert_equal(...): (line 14)
    # Processing the call arguments (line 14)
    
    # Call to matdims(...): (line 14)
    # Processing the call arguments (line 14)
    
    # Call to array(...): (line 14)
    # Processing the call arguments (line 14)
    int_144000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 34), 'int')
    # Processing the call keyword arguments (line 14)
    kwargs_144001 = {}
    # Getting the type of 'np' (line 14)
    np_143998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 25), 'np', False)
    # Obtaining the member 'array' of a type (line 14)
    array_143999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 25), np_143998, 'array')
    # Calling array(args, kwargs) (line 14)
    array_call_result_144002 = invoke(stypy.reporting.localization.Localization(__file__, 14, 25), array_143999, *[int_144000], **kwargs_144001)
    
    # Processing the call keyword arguments (line 14)
    kwargs_144003 = {}
    # Getting the type of 'matdims' (line 14)
    matdims_143997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 17), 'matdims', False)
    # Calling matdims(args, kwargs) (line 14)
    matdims_call_result_144004 = invoke(stypy.reporting.localization.Localization(__file__, 14, 17), matdims_143997, *[array_call_result_144002], **kwargs_144003)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 14)
    tuple_144005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 14)
    # Adding element type (line 14)
    int_144006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 40), tuple_144005, int_144006)
    # Adding element type (line 14)
    int_144007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 40), tuple_144005, int_144007)
    
    # Processing the call keyword arguments (line 14)
    kwargs_144008 = {}
    # Getting the type of 'assert_equal' (line 14)
    assert_equal_143996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 14)
    assert_equal_call_result_144009 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), assert_equal_143996, *[matdims_call_result_144004, tuple_144005], **kwargs_144008)
    
    
    # Call to assert_equal(...): (line 15)
    # Processing the call arguments (line 15)
    
    # Call to matdims(...): (line 15)
    # Processing the call arguments (line 15)
    
    # Call to array(...): (line 15)
    # Processing the call arguments (line 15)
    
    # Obtaining an instance of the builtin type 'list' (line 15)
    list_144014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 15)
    # Adding element type (line 15)
    int_144015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 34), list_144014, int_144015)
    
    # Processing the call keyword arguments (line 15)
    kwargs_144016 = {}
    # Getting the type of 'np' (line 15)
    np_144012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 25), 'np', False)
    # Obtaining the member 'array' of a type (line 15)
    array_144013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 25), np_144012, 'array')
    # Calling array(args, kwargs) (line 15)
    array_call_result_144017 = invoke(stypy.reporting.localization.Localization(__file__, 15, 25), array_144013, *[list_144014], **kwargs_144016)
    
    # Processing the call keyword arguments (line 15)
    kwargs_144018 = {}
    # Getting the type of 'matdims' (line 15)
    matdims_144011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 17), 'matdims', False)
    # Calling matdims(args, kwargs) (line 15)
    matdims_call_result_144019 = invoke(stypy.reporting.localization.Localization(__file__, 15, 17), matdims_144011, *[array_call_result_144017], **kwargs_144018)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 15)
    tuple_144020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 15)
    # Adding element type (line 15)
    int_144021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 42), tuple_144020, int_144021)
    # Adding element type (line 15)
    int_144022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 42), tuple_144020, int_144022)
    
    # Processing the call keyword arguments (line 15)
    kwargs_144023 = {}
    # Getting the type of 'assert_equal' (line 15)
    assert_equal_144010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 15)
    assert_equal_call_result_144024 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), assert_equal_144010, *[matdims_call_result_144019, tuple_144020], **kwargs_144023)
    
    
    # Call to assert_equal(...): (line 16)
    # Processing the call arguments (line 16)
    
    # Call to matdims(...): (line 16)
    # Processing the call arguments (line 16)
    
    # Call to array(...): (line 16)
    # Processing the call arguments (line 16)
    
    # Obtaining an instance of the builtin type 'list' (line 16)
    list_144029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 16)
    # Adding element type (line 16)
    int_144030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 34), list_144029, int_144030)
    # Adding element type (line 16)
    int_144031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 34), list_144029, int_144031)
    
    # Processing the call keyword arguments (line 16)
    kwargs_144032 = {}
    # Getting the type of 'np' (line 16)
    np_144027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 25), 'np', False)
    # Obtaining the member 'array' of a type (line 16)
    array_144028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 25), np_144027, 'array')
    # Calling array(args, kwargs) (line 16)
    array_call_result_144033 = invoke(stypy.reporting.localization.Localization(__file__, 16, 25), array_144028, *[list_144029], **kwargs_144032)
    
    # Processing the call keyword arguments (line 16)
    kwargs_144034 = {}
    # Getting the type of 'matdims' (line 16)
    matdims_144026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 17), 'matdims', False)
    # Calling matdims(args, kwargs) (line 16)
    matdims_call_result_144035 = invoke(stypy.reporting.localization.Localization(__file__, 16, 17), matdims_144026, *[array_call_result_144033], **kwargs_144034)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 16)
    tuple_144036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 16)
    # Adding element type (line 16)
    int_144037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 44), tuple_144036, int_144037)
    # Adding element type (line 16)
    int_144038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 44), tuple_144036, int_144038)
    
    # Processing the call keyword arguments (line 16)
    kwargs_144039 = {}
    # Getting the type of 'assert_equal' (line 16)
    assert_equal_144025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 16)
    assert_equal_call_result_144040 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), assert_equal_144025, *[matdims_call_result_144035, tuple_144036], **kwargs_144039)
    
    
    # Call to assert_equal(...): (line 17)
    # Processing the call arguments (line 17)
    
    # Call to matdims(...): (line 17)
    # Processing the call arguments (line 17)
    
    # Call to array(...): (line 17)
    # Processing the call arguments (line 17)
    
    # Obtaining an instance of the builtin type 'list' (line 17)
    list_144045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 17)
    # Adding element type (line 17)
    
    # Obtaining an instance of the builtin type 'list' (line 17)
    list_144046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 17)
    # Adding element type (line 17)
    int_144047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 35), list_144046, int_144047)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 34), list_144045, list_144046)
    # Adding element type (line 17)
    
    # Obtaining an instance of the builtin type 'list' (line 17)
    list_144048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 17)
    # Adding element type (line 17)
    int_144049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 39), list_144048, int_144049)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 34), list_144045, list_144048)
    
    # Processing the call keyword arguments (line 17)
    kwargs_144050 = {}
    # Getting the type of 'np' (line 17)
    np_144043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 25), 'np', False)
    # Obtaining the member 'array' of a type (line 17)
    array_144044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 25), np_144043, 'array')
    # Calling array(args, kwargs) (line 17)
    array_call_result_144051 = invoke(stypy.reporting.localization.Localization(__file__, 17, 25), array_144044, *[list_144045], **kwargs_144050)
    
    # Processing the call keyword arguments (line 17)
    kwargs_144052 = {}
    # Getting the type of 'matdims' (line 17)
    matdims_144042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 17), 'matdims', False)
    # Calling matdims(args, kwargs) (line 17)
    matdims_call_result_144053 = invoke(stypy.reporting.localization.Localization(__file__, 17, 17), matdims_144042, *[array_call_result_144051], **kwargs_144052)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 17)
    tuple_144054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 17)
    # Adding element type (line 17)
    int_144055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 48), tuple_144054, int_144055)
    # Adding element type (line 17)
    int_144056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 48), tuple_144054, int_144056)
    
    # Processing the call keyword arguments (line 17)
    kwargs_144057 = {}
    # Getting the type of 'assert_equal' (line 17)
    assert_equal_144041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 17)
    assert_equal_call_result_144058 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), assert_equal_144041, *[matdims_call_result_144053, tuple_144054], **kwargs_144057)
    
    
    # Call to assert_equal(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Call to matdims(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Call to array(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Obtaining an instance of the builtin type 'list' (line 18)
    list_144063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 18)
    # Adding element type (line 18)
    
    # Obtaining an instance of the builtin type 'list' (line 18)
    list_144064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 18)
    # Adding element type (line 18)
    int_144065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 35), list_144064, int_144065)
    # Adding element type (line 18)
    int_144066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 35), list_144064, int_144066)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_144063, list_144064)
    
    # Processing the call keyword arguments (line 18)
    kwargs_144067 = {}
    # Getting the type of 'np' (line 18)
    np_144061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 25), 'np', False)
    # Obtaining the member 'array' of a type (line 18)
    array_144062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 25), np_144061, 'array')
    # Calling array(args, kwargs) (line 18)
    array_call_result_144068 = invoke(stypy.reporting.localization.Localization(__file__, 18, 25), array_144062, *[list_144063], **kwargs_144067)
    
    # Processing the call keyword arguments (line 18)
    kwargs_144069 = {}
    # Getting the type of 'matdims' (line 18)
    matdims_144060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 17), 'matdims', False)
    # Calling matdims(args, kwargs) (line 18)
    matdims_call_result_144070 = invoke(stypy.reporting.localization.Localization(__file__, 18, 17), matdims_144060, *[array_call_result_144068], **kwargs_144069)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 18)
    tuple_144071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 18)
    # Adding element type (line 18)
    int_144072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 46), tuple_144071, int_144072)
    # Adding element type (line 18)
    int_144073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 46), tuple_144071, int_144073)
    
    # Processing the call keyword arguments (line 18)
    kwargs_144074 = {}
    # Getting the type of 'assert_equal' (line 18)
    assert_equal_144059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 18)
    assert_equal_call_result_144075 = invoke(stypy.reporting.localization.Localization(__file__, 18, 4), assert_equal_144059, *[matdims_call_result_144070, tuple_144071], **kwargs_144074)
    
    
    # Call to assert_equal(...): (line 20)
    # Processing the call arguments (line 20)
    
    # Call to matdims(...): (line 20)
    # Processing the call arguments (line 20)
    
    # Call to array(...): (line 20)
    # Processing the call arguments (line 20)
    
    # Obtaining an instance of the builtin type 'list' (line 20)
    list_144080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 20)
    # Adding element type (line 20)
    
    # Obtaining an instance of the builtin type 'list' (line 20)
    list_144081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 20)
    # Adding element type (line 20)
    
    # Obtaining an instance of the builtin type 'list' (line 20)
    list_144082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 20)
    # Adding element type (line 20)
    int_144083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 36), list_144082, int_144083)
    # Adding element type (line 20)
    int_144084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 36), list_144082, int_144084)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 35), list_144081, list_144082)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 34), list_144080, list_144081)
    
    # Processing the call keyword arguments (line 20)
    kwargs_144085 = {}
    # Getting the type of 'np' (line 20)
    np_144078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 25), 'np', False)
    # Obtaining the member 'array' of a type (line 20)
    array_144079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 25), np_144078, 'array')
    # Calling array(args, kwargs) (line 20)
    array_call_result_144086 = invoke(stypy.reporting.localization.Localization(__file__, 20, 25), array_144079, *[list_144080], **kwargs_144085)
    
    # Processing the call keyword arguments (line 20)
    kwargs_144087 = {}
    # Getting the type of 'matdims' (line 20)
    matdims_144077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 17), 'matdims', False)
    # Calling matdims(args, kwargs) (line 20)
    matdims_call_result_144088 = invoke(stypy.reporting.localization.Localization(__file__, 20, 17), matdims_144077, *[array_call_result_144086], **kwargs_144087)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 20)
    tuple_144089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 20)
    # Adding element type (line 20)
    int_144090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 48), tuple_144089, int_144090)
    # Adding element type (line 20)
    int_144091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 48), tuple_144089, int_144091)
    # Adding element type (line 20)
    int_144092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 48), tuple_144089, int_144092)
    
    # Processing the call keyword arguments (line 20)
    kwargs_144093 = {}
    # Getting the type of 'assert_equal' (line 20)
    assert_equal_144076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 20)
    assert_equal_call_result_144094 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), assert_equal_144076, *[matdims_call_result_144088, tuple_144089], **kwargs_144093)
    
    
    # Call to assert_equal(...): (line 21)
    # Processing the call arguments (line 21)
    
    # Call to matdims(...): (line 21)
    # Processing the call arguments (line 21)
    
    # Call to array(...): (line 21)
    # Processing the call arguments (line 21)
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_144099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    
    # Processing the call keyword arguments (line 21)
    kwargs_144100 = {}
    # Getting the type of 'np' (line 21)
    np_144097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 25), 'np', False)
    # Obtaining the member 'array' of a type (line 21)
    array_144098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 25), np_144097, 'array')
    # Calling array(args, kwargs) (line 21)
    array_call_result_144101 = invoke(stypy.reporting.localization.Localization(__file__, 21, 25), array_144098, *[list_144099], **kwargs_144100)
    
    # Processing the call keyword arguments (line 21)
    kwargs_144102 = {}
    # Getting the type of 'matdims' (line 21)
    matdims_144096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 17), 'matdims', False)
    # Calling matdims(args, kwargs) (line 21)
    matdims_call_result_144103 = invoke(stypy.reporting.localization.Localization(__file__, 21, 17), matdims_144096, *[array_call_result_144101], **kwargs_144102)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 21)
    tuple_144104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 21)
    # Adding element type (line 21)
    int_144105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 41), tuple_144104, int_144105)
    # Adding element type (line 21)
    int_144106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 41), tuple_144104, int_144106)
    
    # Processing the call keyword arguments (line 21)
    kwargs_144107 = {}
    # Getting the type of 'assert_equal' (line 21)
    assert_equal_144095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 21)
    assert_equal_call_result_144108 = invoke(stypy.reporting.localization.Localization(__file__, 21, 4), assert_equal_144095, *[matdims_call_result_144103, tuple_144104], **kwargs_144107)
    
    
    # Call to assert_equal(...): (line 22)
    # Processing the call arguments (line 22)
    
    # Call to matdims(...): (line 22)
    # Processing the call arguments (line 22)
    
    # Call to array(...): (line 22)
    # Processing the call arguments (line 22)
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_144113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_144114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 34), list_144113, list_144114)
    
    # Processing the call keyword arguments (line 22)
    kwargs_144115 = {}
    # Getting the type of 'np' (line 22)
    np_144111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 25), 'np', False)
    # Obtaining the member 'array' of a type (line 22)
    array_144112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 25), np_144111, 'array')
    # Calling array(args, kwargs) (line 22)
    array_call_result_144116 = invoke(stypy.reporting.localization.Localization(__file__, 22, 25), array_144112, *[list_144113], **kwargs_144115)
    
    # Processing the call keyword arguments (line 22)
    kwargs_144117 = {}
    # Getting the type of 'matdims' (line 22)
    matdims_144110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'matdims', False)
    # Calling matdims(args, kwargs) (line 22)
    matdims_call_result_144118 = invoke(stypy.reporting.localization.Localization(__file__, 22, 17), matdims_144110, *[array_call_result_144116], **kwargs_144117)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 22)
    tuple_144119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 22)
    # Adding element type (line 22)
    int_144120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 43), tuple_144119, int_144120)
    # Adding element type (line 22)
    int_144121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 43), tuple_144119, int_144121)
    
    # Processing the call keyword arguments (line 22)
    kwargs_144122 = {}
    # Getting the type of 'assert_equal' (line 22)
    assert_equal_144109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 22)
    assert_equal_call_result_144123 = invoke(stypy.reporting.localization.Localization(__file__, 22, 4), assert_equal_144109, *[matdims_call_result_144118, tuple_144119], **kwargs_144122)
    
    
    # Call to assert_equal(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Call to matdims(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Call to array(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Obtaining an instance of the builtin type 'list' (line 23)
    list_144128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 23)
    # Adding element type (line 23)
    
    # Obtaining an instance of the builtin type 'list' (line 23)
    list_144129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 23)
    # Adding element type (line 23)
    
    # Obtaining an instance of the builtin type 'list' (line 23)
    list_144130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 23)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 35), list_144129, list_144130)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 34), list_144128, list_144129)
    
    # Processing the call keyword arguments (line 23)
    kwargs_144131 = {}
    # Getting the type of 'np' (line 23)
    np_144126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 25), 'np', False)
    # Obtaining the member 'array' of a type (line 23)
    array_144127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 25), np_144126, 'array')
    # Calling array(args, kwargs) (line 23)
    array_call_result_144132 = invoke(stypy.reporting.localization.Localization(__file__, 23, 25), array_144127, *[list_144128], **kwargs_144131)
    
    # Processing the call keyword arguments (line 23)
    kwargs_144133 = {}
    # Getting the type of 'matdims' (line 23)
    matdims_144125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 17), 'matdims', False)
    # Calling matdims(args, kwargs) (line 23)
    matdims_call_result_144134 = invoke(stypy.reporting.localization.Localization(__file__, 23, 17), matdims_144125, *[array_call_result_144132], **kwargs_144133)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 23)
    tuple_144135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 45), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 23)
    # Adding element type (line 23)
    int_144136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 45), tuple_144135, int_144136)
    # Adding element type (line 23)
    int_144137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 45), tuple_144135, int_144137)
    # Adding element type (line 23)
    int_144138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 45), tuple_144135, int_144138)
    
    # Processing the call keyword arguments (line 23)
    kwargs_144139 = {}
    # Getting the type of 'assert_equal' (line 23)
    assert_equal_144124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 23)
    assert_equal_call_result_144140 = invoke(stypy.reporting.localization.Localization(__file__, 23, 4), assert_equal_144124, *[matdims_call_result_144134, tuple_144135], **kwargs_144139)
    
    
    # Call to assert_equal(...): (line 25)
    # Processing the call arguments (line 25)
    
    # Call to matdims(...): (line 25)
    # Processing the call arguments (line 25)
    
    # Call to array(...): (line 25)
    # Processing the call arguments (line 25)
    
    # Obtaining an instance of the builtin type 'list' (line 25)
    list_144145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 25)
    # Adding element type (line 25)
    int_144146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 34), list_144145, int_144146)
    # Adding element type (line 25)
    int_144147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 34), list_144145, int_144147)
    
    # Processing the call keyword arguments (line 25)
    kwargs_144148 = {}
    # Getting the type of 'np' (line 25)
    np_144143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 25), 'np', False)
    # Obtaining the member 'array' of a type (line 25)
    array_144144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 25), np_144143, 'array')
    # Calling array(args, kwargs) (line 25)
    array_call_result_144149 = invoke(stypy.reporting.localization.Localization(__file__, 25, 25), array_144144, *[list_144145], **kwargs_144148)
    
    str_144150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 42), 'str', 'row')
    # Processing the call keyword arguments (line 25)
    kwargs_144151 = {}
    # Getting the type of 'matdims' (line 25)
    matdims_144142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 17), 'matdims', False)
    # Calling matdims(args, kwargs) (line 25)
    matdims_call_result_144152 = invoke(stypy.reporting.localization.Localization(__file__, 25, 17), matdims_144142, *[array_call_result_144149, str_144150], **kwargs_144151)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 25)
    tuple_144153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 51), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 25)
    # Adding element type (line 25)
    int_144154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 51), tuple_144153, int_144154)
    # Adding element type (line 25)
    int_144155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 51), tuple_144153, int_144155)
    
    # Processing the call keyword arguments (line 25)
    kwargs_144156 = {}
    # Getting the type of 'assert_equal' (line 25)
    assert_equal_144141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 25)
    assert_equal_call_result_144157 = invoke(stypy.reporting.localization.Localization(__file__, 25, 4), assert_equal_144141, *[matdims_call_result_144152, tuple_144153], **kwargs_144156)
    
    
    # Call to assert_raises(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'ValueError' (line 27)
    ValueError_144159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 18), 'ValueError', False)
    # Getting the type of 'matdims' (line 27)
    matdims_144160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 30), 'matdims', False)
    
    # Call to array(...): (line 27)
    # Processing the call arguments (line 27)
    
    # Obtaining an instance of the builtin type 'list' (line 27)
    list_144163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 48), 'list')
    # Adding type elements to the builtin type 'list' instance (line 27)
    # Adding element type (line 27)
    int_144164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 48), list_144163, int_144164)
    # Adding element type (line 27)
    int_144165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 48), list_144163, int_144165)
    
    # Processing the call keyword arguments (line 27)
    kwargs_144166 = {}
    # Getting the type of 'np' (line 27)
    np_144161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 39), 'np', False)
    # Obtaining the member 'array' of a type (line 27)
    array_144162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 39), np_144161, 'array')
    # Calling array(args, kwargs) (line 27)
    array_call_result_144167 = invoke(stypy.reporting.localization.Localization(__file__, 27, 39), array_144162, *[list_144163], **kwargs_144166)
    
    str_144168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 56), 'str', 'bizarre')
    # Processing the call keyword arguments (line 27)
    kwargs_144169 = {}
    # Getting the type of 'assert_raises' (line 27)
    assert_raises_144158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 27)
    assert_raises_call_result_144170 = invoke(stypy.reporting.localization.Localization(__file__, 27, 4), assert_raises_144158, *[ValueError_144159, matdims_144160, array_call_result_144167, str_144168], **kwargs_144169)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 4))
    
    # 'from scipy.sparse import csr_matrix, csc_matrix' statement (line 29)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
    import_144171 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 29, 4), 'scipy.sparse')

    if (type(import_144171) is not StypyTypeError):

        if (import_144171 != 'pyd_module'):
            __import__(import_144171)
            sys_modules_144172 = sys.modules[import_144171]
            import_from_module(stypy.reporting.localization.Localization(__file__, 29, 4), 'scipy.sparse', sys_modules_144172.module_type_store, module_type_store, ['csr_matrix', 'csc_matrix'])
            nest_module(stypy.reporting.localization.Localization(__file__, 29, 4), __file__, sys_modules_144172, sys_modules_144172.module_type_store, module_type_store)
        else:
            from scipy.sparse import csr_matrix, csc_matrix

            import_from_module(stypy.reporting.localization.Localization(__file__, 29, 4), 'scipy.sparse', None, module_type_store, ['csr_matrix', 'csc_matrix'], [csr_matrix, csc_matrix])

    else:
        # Assigning a type to the variable 'scipy.sparse' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'scipy.sparse', import_144171)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
    
    
    # Call to assert_equal(...): (line 30)
    # Processing the call arguments (line 30)
    
    # Call to matdims(...): (line 30)
    # Processing the call arguments (line 30)
    
    # Call to csr_matrix(...): (line 30)
    # Processing the call arguments (line 30)
    
    # Call to zeros(...): (line 30)
    # Processing the call arguments (line 30)
    
    # Obtaining an instance of the builtin type 'tuple' (line 30)
    tuple_144178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 30)
    # Adding element type (line 30)
    int_144179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 46), tuple_144178, int_144179)
    # Adding element type (line 30)
    int_144180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 46), tuple_144178, int_144180)
    
    # Processing the call keyword arguments (line 30)
    kwargs_144181 = {}
    # Getting the type of 'np' (line 30)
    np_144176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 36), 'np', False)
    # Obtaining the member 'zeros' of a type (line 30)
    zeros_144177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 36), np_144176, 'zeros')
    # Calling zeros(args, kwargs) (line 30)
    zeros_call_result_144182 = invoke(stypy.reporting.localization.Localization(__file__, 30, 36), zeros_144177, *[tuple_144178], **kwargs_144181)
    
    # Processing the call keyword arguments (line 30)
    kwargs_144183 = {}
    # Getting the type of 'csr_matrix' (line 30)
    csr_matrix_144175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 25), 'csr_matrix', False)
    # Calling csr_matrix(args, kwargs) (line 30)
    csr_matrix_call_result_144184 = invoke(stypy.reporting.localization.Localization(__file__, 30, 25), csr_matrix_144175, *[zeros_call_result_144182], **kwargs_144183)
    
    # Processing the call keyword arguments (line 30)
    kwargs_144185 = {}
    # Getting the type of 'matdims' (line 30)
    matdims_144174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 17), 'matdims', False)
    # Calling matdims(args, kwargs) (line 30)
    matdims_call_result_144186 = invoke(stypy.reporting.localization.Localization(__file__, 30, 17), matdims_144174, *[csr_matrix_call_result_144184], **kwargs_144185)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 30)
    tuple_144187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 57), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 30)
    # Adding element type (line 30)
    int_144188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 57), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 57), tuple_144187, int_144188)
    # Adding element type (line 30)
    int_144189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 60), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 57), tuple_144187, int_144189)
    
    # Processing the call keyword arguments (line 30)
    kwargs_144190 = {}
    # Getting the type of 'assert_equal' (line 30)
    assert_equal_144173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 30)
    assert_equal_call_result_144191 = invoke(stypy.reporting.localization.Localization(__file__, 30, 4), assert_equal_144173, *[matdims_call_result_144186, tuple_144187], **kwargs_144190)
    
    
    # Call to assert_equal(...): (line 31)
    # Processing the call arguments (line 31)
    
    # Call to matdims(...): (line 31)
    # Processing the call arguments (line 31)
    
    # Call to csc_matrix(...): (line 31)
    # Processing the call arguments (line 31)
    
    # Call to zeros(...): (line 31)
    # Processing the call arguments (line 31)
    
    # Obtaining an instance of the builtin type 'tuple' (line 31)
    tuple_144197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 31)
    # Adding element type (line 31)
    int_144198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 46), tuple_144197, int_144198)
    # Adding element type (line 31)
    int_144199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 46), tuple_144197, int_144199)
    
    # Processing the call keyword arguments (line 31)
    kwargs_144200 = {}
    # Getting the type of 'np' (line 31)
    np_144195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 36), 'np', False)
    # Obtaining the member 'zeros' of a type (line 31)
    zeros_144196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 36), np_144195, 'zeros')
    # Calling zeros(args, kwargs) (line 31)
    zeros_call_result_144201 = invoke(stypy.reporting.localization.Localization(__file__, 31, 36), zeros_144196, *[tuple_144197], **kwargs_144200)
    
    # Processing the call keyword arguments (line 31)
    kwargs_144202 = {}
    # Getting the type of 'csc_matrix' (line 31)
    csc_matrix_144194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 25), 'csc_matrix', False)
    # Calling csc_matrix(args, kwargs) (line 31)
    csc_matrix_call_result_144203 = invoke(stypy.reporting.localization.Localization(__file__, 31, 25), csc_matrix_144194, *[zeros_call_result_144201], **kwargs_144202)
    
    # Processing the call keyword arguments (line 31)
    kwargs_144204 = {}
    # Getting the type of 'matdims' (line 31)
    matdims_144193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 17), 'matdims', False)
    # Calling matdims(args, kwargs) (line 31)
    matdims_call_result_144205 = invoke(stypy.reporting.localization.Localization(__file__, 31, 17), matdims_144193, *[csc_matrix_call_result_144203], **kwargs_144204)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 31)
    tuple_144206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 57), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 31)
    # Adding element type (line 31)
    int_144207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 57), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 57), tuple_144206, int_144207)
    # Adding element type (line 31)
    int_144208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 60), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 57), tuple_144206, int_144208)
    
    # Processing the call keyword arguments (line 31)
    kwargs_144209 = {}
    # Getting the type of 'assert_equal' (line 31)
    assert_equal_144192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 31)
    assert_equal_call_result_144210 = invoke(stypy.reporting.localization.Localization(__file__, 31, 4), assert_equal_144192, *[matdims_call_result_144205, tuple_144206], **kwargs_144209)
    
    
    # ################# End of 'test_matdims(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_matdims' in the type store
    # Getting the type of 'stypy_return_type' (line 12)
    stypy_return_type_144211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_144211)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_matdims'
    return stypy_return_type_144211

# Assigning a type to the variable 'test_matdims' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'test_matdims', test_matdims)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
