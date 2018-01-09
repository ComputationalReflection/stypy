
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Functions that operate on sparse matrices
2: '''
3: 
4: from __future__ import division, print_function, absolute_import
5: 
6: __all__ = ['count_blocks','estimate_blocksize']
7: 
8: from .csr import isspmatrix_csr, csr_matrix
9: from .csc import isspmatrix_csc
10: from ._sparsetools import csr_count_blocks
11: 
12: 
13: def extract_diagonal(A):
14:     raise NotImplementedError('use .diagonal() instead')
15: 
16: #def extract_diagonal(A):
17: #    '''extract_diagonal(A) returns the main diagonal of A.'''
18: #    #TODO extract k-th diagonal
19: #    if isspmatrix_csr(A) or isspmatrix_csc(A):
20: #        fn = getattr(sparsetools, A.format + "_diagonal")
21: #        y = empty( min(A.shape), dtype=upcast(A.dtype) )
22: #        fn(A.shape[0],A.shape[1],A.indptr,A.indices,A.data,y)
23: #        return y
24: #    elif isspmatrix_bsr(A):
25: #        M,N = A.shape
26: #        R,C = A.blocksize
27: #        y = empty( min(M,N), dtype=upcast(A.dtype) )
28: #        fn = sparsetools.bsr_diagonal(M//R, N//C, R, C, \
29: #                A.indptr, A.indices, ravel(A.data), y)
30: #        return y
31: #    else:
32: #        return extract_diagonal(csr_matrix(A))
33: 
34: 
35: def estimate_blocksize(A,efficiency=0.7):
36:     '''Attempt to determine the blocksize of a sparse matrix
37: 
38:     Returns a blocksize=(r,c) such that
39:         - A.nnz / A.tobsr( (r,c) ).nnz > efficiency
40:     '''
41:     if not (isspmatrix_csr(A) or isspmatrix_csc(A)):
42:         A = csr_matrix(A)
43: 
44:     if A.nnz == 0:
45:         return (1,1)
46: 
47:     if not 0 < efficiency < 1.0:
48:         raise ValueError('efficiency must satisfy 0.0 < efficiency < 1.0')
49: 
50:     high_efficiency = (1.0 + efficiency) / 2.0
51:     nnz = float(A.nnz)
52:     M,N = A.shape
53: 
54:     if M % 2 == 0 and N % 2 == 0:
55:         e22 = nnz / (4 * count_blocks(A,(2,2)))
56:     else:
57:         e22 = 0.0
58: 
59:     if M % 3 == 0 and N % 3 == 0:
60:         e33 = nnz / (9 * count_blocks(A,(3,3)))
61:     else:
62:         e33 = 0.0
63: 
64:     if e22 > high_efficiency and e33 > high_efficiency:
65:         e66 = nnz / (36 * count_blocks(A,(6,6)))
66:         if e66 > efficiency:
67:             return (6,6)
68:         else:
69:             return (3,3)
70:     else:
71:         if M % 4 == 0 and N % 4 == 0:
72:             e44 = nnz / (16 * count_blocks(A,(4,4)))
73:         else:
74:             e44 = 0.0
75: 
76:         if e44 > efficiency:
77:             return (4,4)
78:         elif e33 > efficiency:
79:             return (3,3)
80:         elif e22 > efficiency:
81:             return (2,2)
82:         else:
83:             return (1,1)
84: 
85: 
86: def count_blocks(A,blocksize):
87:     '''For a given blocksize=(r,c) count the number of occupied
88:     blocks in a sparse matrix A
89:     '''
90:     r,c = blocksize
91:     if r < 1 or c < 1:
92:         raise ValueError('r and c must be positive')
93: 
94:     if isspmatrix_csr(A):
95:         M,N = A.shape
96:         return csr_count_blocks(M,N,r,c,A.indptr,A.indices)
97:     elif isspmatrix_csc(A):
98:         return count_blocks(A.T,(c,r))
99:     else:
100:         return count_blocks(csr_matrix(A),blocksize)
101: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_379391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', ' Functions that operate on sparse matrices\n')

# Assigning a List to a Name (line 6):

# Assigning a List to a Name (line 6):
__all__ = ['count_blocks', 'estimate_blocksize']
module_type_store.set_exportable_members(['count_blocks', 'estimate_blocksize'])

# Obtaining an instance of the builtin type 'list' (line 6)
list_379392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
str_379393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 11), 'str', 'count_blocks')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_379392, str_379393)
# Adding element type (line 6)
str_379394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 26), 'str', 'estimate_blocksize')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_379392, str_379394)

# Assigning a type to the variable '__all__' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '__all__', list_379392)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.sparse.csr import isspmatrix_csr, csr_matrix' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_379395 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse.csr')

if (type(import_379395) is not StypyTypeError):

    if (import_379395 != 'pyd_module'):
        __import__(import_379395)
        sys_modules_379396 = sys.modules[import_379395]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse.csr', sys_modules_379396.module_type_store, module_type_store, ['isspmatrix_csr', 'csr_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_379396, sys_modules_379396.module_type_store, module_type_store)
    else:
        from scipy.sparse.csr import isspmatrix_csr, csr_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse.csr', None, module_type_store, ['isspmatrix_csr', 'csr_matrix'], [isspmatrix_csr, csr_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse.csr' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse.csr', import_379395)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.sparse.csc import isspmatrix_csc' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_379397 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse.csc')

if (type(import_379397) is not StypyTypeError):

    if (import_379397 != 'pyd_module'):
        __import__(import_379397)
        sys_modules_379398 = sys.modules[import_379397]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse.csc', sys_modules_379398.module_type_store, module_type_store, ['isspmatrix_csc'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_379398, sys_modules_379398.module_type_store, module_type_store)
    else:
        from scipy.sparse.csc import isspmatrix_csc

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse.csc', None, module_type_store, ['isspmatrix_csc'], [isspmatrix_csc])

else:
    # Assigning a type to the variable 'scipy.sparse.csc' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse.csc', import_379397)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.sparse._sparsetools import csr_count_blocks' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_379399 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse._sparsetools')

if (type(import_379399) is not StypyTypeError):

    if (import_379399 != 'pyd_module'):
        __import__(import_379399)
        sys_modules_379400 = sys.modules[import_379399]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse._sparsetools', sys_modules_379400.module_type_store, module_type_store, ['csr_count_blocks'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_379400, sys_modules_379400.module_type_store, module_type_store)
    else:
        from scipy.sparse._sparsetools import csr_count_blocks

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse._sparsetools', None, module_type_store, ['csr_count_blocks'], [csr_count_blocks])

else:
    # Assigning a type to the variable 'scipy.sparse._sparsetools' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse._sparsetools', import_379399)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')


@norecursion
def extract_diagonal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'extract_diagonal'
    module_type_store = module_type_store.open_function_context('extract_diagonal', 13, 0, False)
    
    # Passed parameters checking function
    extract_diagonal.stypy_localization = localization
    extract_diagonal.stypy_type_of_self = None
    extract_diagonal.stypy_type_store = module_type_store
    extract_diagonal.stypy_function_name = 'extract_diagonal'
    extract_diagonal.stypy_param_names_list = ['A']
    extract_diagonal.stypy_varargs_param_name = None
    extract_diagonal.stypy_kwargs_param_name = None
    extract_diagonal.stypy_call_defaults = defaults
    extract_diagonal.stypy_call_varargs = varargs
    extract_diagonal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'extract_diagonal', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'extract_diagonal', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'extract_diagonal(...)' code ##################

    
    # Call to NotImplementedError(...): (line 14)
    # Processing the call arguments (line 14)
    str_379402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 30), 'str', 'use .diagonal() instead')
    # Processing the call keyword arguments (line 14)
    kwargs_379403 = {}
    # Getting the type of 'NotImplementedError' (line 14)
    NotImplementedError_379401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'NotImplementedError', False)
    # Calling NotImplementedError(args, kwargs) (line 14)
    NotImplementedError_call_result_379404 = invoke(stypy.reporting.localization.Localization(__file__, 14, 10), NotImplementedError_379401, *[str_379402], **kwargs_379403)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 14, 4), NotImplementedError_call_result_379404, 'raise parameter', BaseException)
    
    # ################# End of 'extract_diagonal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'extract_diagonal' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_379405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_379405)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'extract_diagonal'
    return stypy_return_type_379405

# Assigning a type to the variable 'extract_diagonal' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'extract_diagonal', extract_diagonal)

@norecursion
def estimate_blocksize(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_379406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 36), 'float')
    defaults = [float_379406]
    # Create a new context for function 'estimate_blocksize'
    module_type_store = module_type_store.open_function_context('estimate_blocksize', 35, 0, False)
    
    # Passed parameters checking function
    estimate_blocksize.stypy_localization = localization
    estimate_blocksize.stypy_type_of_self = None
    estimate_blocksize.stypy_type_store = module_type_store
    estimate_blocksize.stypy_function_name = 'estimate_blocksize'
    estimate_blocksize.stypy_param_names_list = ['A', 'efficiency']
    estimate_blocksize.stypy_varargs_param_name = None
    estimate_blocksize.stypy_kwargs_param_name = None
    estimate_blocksize.stypy_call_defaults = defaults
    estimate_blocksize.stypy_call_varargs = varargs
    estimate_blocksize.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'estimate_blocksize', ['A', 'efficiency'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'estimate_blocksize', localization, ['A', 'efficiency'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'estimate_blocksize(...)' code ##################

    str_379407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, (-1)), 'str', 'Attempt to determine the blocksize of a sparse matrix\n\n    Returns a blocksize=(r,c) such that\n        - A.nnz / A.tobsr( (r,c) ).nnz > efficiency\n    ')
    
    
    
    # Evaluating a boolean operation
    
    # Call to isspmatrix_csr(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'A' (line 41)
    A_379409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 27), 'A', False)
    # Processing the call keyword arguments (line 41)
    kwargs_379410 = {}
    # Getting the type of 'isspmatrix_csr' (line 41)
    isspmatrix_csr_379408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'isspmatrix_csr', False)
    # Calling isspmatrix_csr(args, kwargs) (line 41)
    isspmatrix_csr_call_result_379411 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), isspmatrix_csr_379408, *[A_379409], **kwargs_379410)
    
    
    # Call to isspmatrix_csc(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'A' (line 41)
    A_379413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 48), 'A', False)
    # Processing the call keyword arguments (line 41)
    kwargs_379414 = {}
    # Getting the type of 'isspmatrix_csc' (line 41)
    isspmatrix_csc_379412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 33), 'isspmatrix_csc', False)
    # Calling isspmatrix_csc(args, kwargs) (line 41)
    isspmatrix_csc_call_result_379415 = invoke(stypy.reporting.localization.Localization(__file__, 41, 33), isspmatrix_csc_379412, *[A_379413], **kwargs_379414)
    
    # Applying the binary operator 'or' (line 41)
    result_or_keyword_379416 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 12), 'or', isspmatrix_csr_call_result_379411, isspmatrix_csc_call_result_379415)
    
    # Applying the 'not' unary operator (line 41)
    result_not__379417 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 7), 'not', result_or_keyword_379416)
    
    # Testing the type of an if condition (line 41)
    if_condition_379418 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 4), result_not__379417)
    # Assigning a type to the variable 'if_condition_379418' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'if_condition_379418', if_condition_379418)
    # SSA begins for if statement (line 41)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 42):
    
    # Assigning a Call to a Name (line 42):
    
    # Call to csr_matrix(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'A' (line 42)
    A_379420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 23), 'A', False)
    # Processing the call keyword arguments (line 42)
    kwargs_379421 = {}
    # Getting the type of 'csr_matrix' (line 42)
    csr_matrix_379419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'csr_matrix', False)
    # Calling csr_matrix(args, kwargs) (line 42)
    csr_matrix_call_result_379422 = invoke(stypy.reporting.localization.Localization(__file__, 42, 12), csr_matrix_379419, *[A_379420], **kwargs_379421)
    
    # Assigning a type to the variable 'A' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'A', csr_matrix_call_result_379422)
    # SSA join for if statement (line 41)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'A' (line 44)
    A_379423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 7), 'A')
    # Obtaining the member 'nnz' of a type (line 44)
    nnz_379424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 7), A_379423, 'nnz')
    int_379425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 16), 'int')
    # Applying the binary operator '==' (line 44)
    result_eq_379426 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 7), '==', nnz_379424, int_379425)
    
    # Testing the type of an if condition (line 44)
    if_condition_379427 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 4), result_eq_379426)
    # Assigning a type to the variable 'if_condition_379427' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'if_condition_379427', if_condition_379427)
    # SSA begins for if statement (line 44)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 45)
    tuple_379428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 45)
    # Adding element type (line 45)
    int_379429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 16), tuple_379428, int_379429)
    # Adding element type (line 45)
    int_379430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 16), tuple_379428, int_379430)
    
    # Assigning a type to the variable 'stypy_return_type' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'stypy_return_type', tuple_379428)
    # SSA join for if statement (line 44)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    int_379431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 11), 'int')
    # Getting the type of 'efficiency' (line 47)
    efficiency_379432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'efficiency')
    # Applying the binary operator '<' (line 47)
    result_lt_379433 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 11), '<', int_379431, efficiency_379432)
    float_379434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 28), 'float')
    # Applying the binary operator '<' (line 47)
    result_lt_379435 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 11), '<', efficiency_379432, float_379434)
    # Applying the binary operator '&' (line 47)
    result_and__379436 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 11), '&', result_lt_379433, result_lt_379435)
    
    # Applying the 'not' unary operator (line 47)
    result_not__379437 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 7), 'not', result_and__379436)
    
    # Testing the type of an if condition (line 47)
    if_condition_379438 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 4), result_not__379437)
    # Assigning a type to the variable 'if_condition_379438' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'if_condition_379438', if_condition_379438)
    # SSA begins for if statement (line 47)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 48)
    # Processing the call arguments (line 48)
    str_379440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 25), 'str', 'efficiency must satisfy 0.0 < efficiency < 1.0')
    # Processing the call keyword arguments (line 48)
    kwargs_379441 = {}
    # Getting the type of 'ValueError' (line 48)
    ValueError_379439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 48)
    ValueError_call_result_379442 = invoke(stypy.reporting.localization.Localization(__file__, 48, 14), ValueError_379439, *[str_379440], **kwargs_379441)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 48, 8), ValueError_call_result_379442, 'raise parameter', BaseException)
    # SSA join for if statement (line 47)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 50):
    
    # Assigning a BinOp to a Name (line 50):
    float_379443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 23), 'float')
    # Getting the type of 'efficiency' (line 50)
    efficiency_379444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 29), 'efficiency')
    # Applying the binary operator '+' (line 50)
    result_add_379445 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 23), '+', float_379443, efficiency_379444)
    
    float_379446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 43), 'float')
    # Applying the binary operator 'div' (line 50)
    result_div_379447 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 22), 'div', result_add_379445, float_379446)
    
    # Assigning a type to the variable 'high_efficiency' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'high_efficiency', result_div_379447)
    
    # Assigning a Call to a Name (line 51):
    
    # Assigning a Call to a Name (line 51):
    
    # Call to float(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'A' (line 51)
    A_379449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'A', False)
    # Obtaining the member 'nnz' of a type (line 51)
    nnz_379450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 16), A_379449, 'nnz')
    # Processing the call keyword arguments (line 51)
    kwargs_379451 = {}
    # Getting the type of 'float' (line 51)
    float_379448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 10), 'float', False)
    # Calling float(args, kwargs) (line 51)
    float_call_result_379452 = invoke(stypy.reporting.localization.Localization(__file__, 51, 10), float_379448, *[nnz_379450], **kwargs_379451)
    
    # Assigning a type to the variable 'nnz' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'nnz', float_call_result_379452)
    
    # Assigning a Attribute to a Tuple (line 52):
    
    # Assigning a Subscript to a Name (line 52):
    
    # Obtaining the type of the subscript
    int_379453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 4), 'int')
    # Getting the type of 'A' (line 52)
    A_379454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 10), 'A')
    # Obtaining the member 'shape' of a type (line 52)
    shape_379455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 10), A_379454, 'shape')
    # Obtaining the member '__getitem__' of a type (line 52)
    getitem___379456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 4), shape_379455, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 52)
    subscript_call_result_379457 = invoke(stypy.reporting.localization.Localization(__file__, 52, 4), getitem___379456, int_379453)
    
    # Assigning a type to the variable 'tuple_var_assignment_379385' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'tuple_var_assignment_379385', subscript_call_result_379457)
    
    # Assigning a Subscript to a Name (line 52):
    
    # Obtaining the type of the subscript
    int_379458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 4), 'int')
    # Getting the type of 'A' (line 52)
    A_379459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 10), 'A')
    # Obtaining the member 'shape' of a type (line 52)
    shape_379460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 10), A_379459, 'shape')
    # Obtaining the member '__getitem__' of a type (line 52)
    getitem___379461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 4), shape_379460, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 52)
    subscript_call_result_379462 = invoke(stypy.reporting.localization.Localization(__file__, 52, 4), getitem___379461, int_379458)
    
    # Assigning a type to the variable 'tuple_var_assignment_379386' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'tuple_var_assignment_379386', subscript_call_result_379462)
    
    # Assigning a Name to a Name (line 52):
    # Getting the type of 'tuple_var_assignment_379385' (line 52)
    tuple_var_assignment_379385_379463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'tuple_var_assignment_379385')
    # Assigning a type to the variable 'M' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'M', tuple_var_assignment_379385_379463)
    
    # Assigning a Name to a Name (line 52):
    # Getting the type of 'tuple_var_assignment_379386' (line 52)
    tuple_var_assignment_379386_379464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'tuple_var_assignment_379386')
    # Assigning a type to the variable 'N' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 6), 'N', tuple_var_assignment_379386_379464)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'M' (line 54)
    M_379465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 7), 'M')
    int_379466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 11), 'int')
    # Applying the binary operator '%' (line 54)
    result_mod_379467 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 7), '%', M_379465, int_379466)
    
    int_379468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 16), 'int')
    # Applying the binary operator '==' (line 54)
    result_eq_379469 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 7), '==', result_mod_379467, int_379468)
    
    
    # Getting the type of 'N' (line 54)
    N_379470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 22), 'N')
    int_379471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 26), 'int')
    # Applying the binary operator '%' (line 54)
    result_mod_379472 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 22), '%', N_379470, int_379471)
    
    int_379473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 31), 'int')
    # Applying the binary operator '==' (line 54)
    result_eq_379474 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 22), '==', result_mod_379472, int_379473)
    
    # Applying the binary operator 'and' (line 54)
    result_and_keyword_379475 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 7), 'and', result_eq_379469, result_eq_379474)
    
    # Testing the type of an if condition (line 54)
    if_condition_379476 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 4), result_and_keyword_379475)
    # Assigning a type to the variable 'if_condition_379476' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'if_condition_379476', if_condition_379476)
    # SSA begins for if statement (line 54)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 55):
    
    # Assigning a BinOp to a Name (line 55):
    # Getting the type of 'nnz' (line 55)
    nnz_379477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 14), 'nnz')
    int_379478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 21), 'int')
    
    # Call to count_blocks(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'A' (line 55)
    A_379480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 38), 'A', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 55)
    tuple_379481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 55)
    # Adding element type (line 55)
    int_379482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 41), tuple_379481, int_379482)
    # Adding element type (line 55)
    int_379483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 41), tuple_379481, int_379483)
    
    # Processing the call keyword arguments (line 55)
    kwargs_379484 = {}
    # Getting the type of 'count_blocks' (line 55)
    count_blocks_379479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 25), 'count_blocks', False)
    # Calling count_blocks(args, kwargs) (line 55)
    count_blocks_call_result_379485 = invoke(stypy.reporting.localization.Localization(__file__, 55, 25), count_blocks_379479, *[A_379480, tuple_379481], **kwargs_379484)
    
    # Applying the binary operator '*' (line 55)
    result_mul_379486 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 21), '*', int_379478, count_blocks_call_result_379485)
    
    # Applying the binary operator 'div' (line 55)
    result_div_379487 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 14), 'div', nnz_379477, result_mul_379486)
    
    # Assigning a type to the variable 'e22' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'e22', result_div_379487)
    # SSA branch for the else part of an if statement (line 54)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 57):
    
    # Assigning a Num to a Name (line 57):
    float_379488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 14), 'float')
    # Assigning a type to the variable 'e22' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'e22', float_379488)
    # SSA join for if statement (line 54)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'M' (line 59)
    M_379489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 7), 'M')
    int_379490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 11), 'int')
    # Applying the binary operator '%' (line 59)
    result_mod_379491 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 7), '%', M_379489, int_379490)
    
    int_379492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 16), 'int')
    # Applying the binary operator '==' (line 59)
    result_eq_379493 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 7), '==', result_mod_379491, int_379492)
    
    
    # Getting the type of 'N' (line 59)
    N_379494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 22), 'N')
    int_379495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 26), 'int')
    # Applying the binary operator '%' (line 59)
    result_mod_379496 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 22), '%', N_379494, int_379495)
    
    int_379497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 31), 'int')
    # Applying the binary operator '==' (line 59)
    result_eq_379498 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 22), '==', result_mod_379496, int_379497)
    
    # Applying the binary operator 'and' (line 59)
    result_and_keyword_379499 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 7), 'and', result_eq_379493, result_eq_379498)
    
    # Testing the type of an if condition (line 59)
    if_condition_379500 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 4), result_and_keyword_379499)
    # Assigning a type to the variable 'if_condition_379500' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'if_condition_379500', if_condition_379500)
    # SSA begins for if statement (line 59)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 60):
    
    # Assigning a BinOp to a Name (line 60):
    # Getting the type of 'nnz' (line 60)
    nnz_379501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 14), 'nnz')
    int_379502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 21), 'int')
    
    # Call to count_blocks(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'A' (line 60)
    A_379504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 38), 'A', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 60)
    tuple_379505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 60)
    # Adding element type (line 60)
    int_379506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 41), tuple_379505, int_379506)
    # Adding element type (line 60)
    int_379507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 41), tuple_379505, int_379507)
    
    # Processing the call keyword arguments (line 60)
    kwargs_379508 = {}
    # Getting the type of 'count_blocks' (line 60)
    count_blocks_379503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 25), 'count_blocks', False)
    # Calling count_blocks(args, kwargs) (line 60)
    count_blocks_call_result_379509 = invoke(stypy.reporting.localization.Localization(__file__, 60, 25), count_blocks_379503, *[A_379504, tuple_379505], **kwargs_379508)
    
    # Applying the binary operator '*' (line 60)
    result_mul_379510 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 21), '*', int_379502, count_blocks_call_result_379509)
    
    # Applying the binary operator 'div' (line 60)
    result_div_379511 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 14), 'div', nnz_379501, result_mul_379510)
    
    # Assigning a type to the variable 'e33' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'e33', result_div_379511)
    # SSA branch for the else part of an if statement (line 59)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 62):
    
    # Assigning a Num to a Name (line 62):
    float_379512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 14), 'float')
    # Assigning a type to the variable 'e33' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'e33', float_379512)
    # SSA join for if statement (line 59)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'e22' (line 64)
    e22_379513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 7), 'e22')
    # Getting the type of 'high_efficiency' (line 64)
    high_efficiency_379514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 13), 'high_efficiency')
    # Applying the binary operator '>' (line 64)
    result_gt_379515 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 7), '>', e22_379513, high_efficiency_379514)
    
    
    # Getting the type of 'e33' (line 64)
    e33_379516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 33), 'e33')
    # Getting the type of 'high_efficiency' (line 64)
    high_efficiency_379517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 39), 'high_efficiency')
    # Applying the binary operator '>' (line 64)
    result_gt_379518 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 33), '>', e33_379516, high_efficiency_379517)
    
    # Applying the binary operator 'and' (line 64)
    result_and_keyword_379519 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 7), 'and', result_gt_379515, result_gt_379518)
    
    # Testing the type of an if condition (line 64)
    if_condition_379520 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 4), result_and_keyword_379519)
    # Assigning a type to the variable 'if_condition_379520' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'if_condition_379520', if_condition_379520)
    # SSA begins for if statement (line 64)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 65):
    
    # Assigning a BinOp to a Name (line 65):
    # Getting the type of 'nnz' (line 65)
    nnz_379521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 14), 'nnz')
    int_379522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 21), 'int')
    
    # Call to count_blocks(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'A' (line 65)
    A_379524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 39), 'A', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 65)
    tuple_379525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 65)
    # Adding element type (line 65)
    int_379526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 42), tuple_379525, int_379526)
    # Adding element type (line 65)
    int_379527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 42), tuple_379525, int_379527)
    
    # Processing the call keyword arguments (line 65)
    kwargs_379528 = {}
    # Getting the type of 'count_blocks' (line 65)
    count_blocks_379523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 26), 'count_blocks', False)
    # Calling count_blocks(args, kwargs) (line 65)
    count_blocks_call_result_379529 = invoke(stypy.reporting.localization.Localization(__file__, 65, 26), count_blocks_379523, *[A_379524, tuple_379525], **kwargs_379528)
    
    # Applying the binary operator '*' (line 65)
    result_mul_379530 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 21), '*', int_379522, count_blocks_call_result_379529)
    
    # Applying the binary operator 'div' (line 65)
    result_div_379531 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 14), 'div', nnz_379521, result_mul_379530)
    
    # Assigning a type to the variable 'e66' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'e66', result_div_379531)
    
    
    # Getting the type of 'e66' (line 66)
    e66_379532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 11), 'e66')
    # Getting the type of 'efficiency' (line 66)
    efficiency_379533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 17), 'efficiency')
    # Applying the binary operator '>' (line 66)
    result_gt_379534 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 11), '>', e66_379532, efficiency_379533)
    
    # Testing the type of an if condition (line 66)
    if_condition_379535 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 8), result_gt_379534)
    # Assigning a type to the variable 'if_condition_379535' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'if_condition_379535', if_condition_379535)
    # SSA begins for if statement (line 66)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 67)
    tuple_379536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 67)
    # Adding element type (line 67)
    int_379537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 20), tuple_379536, int_379537)
    # Adding element type (line 67)
    int_379538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 20), tuple_379536, int_379538)
    
    # Assigning a type to the variable 'stypy_return_type' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'stypy_return_type', tuple_379536)
    # SSA branch for the else part of an if statement (line 66)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 69)
    tuple_379539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 69)
    # Adding element type (line 69)
    int_379540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 20), tuple_379539, int_379540)
    # Adding element type (line 69)
    int_379541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 20), tuple_379539, int_379541)
    
    # Assigning a type to the variable 'stypy_return_type' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'stypy_return_type', tuple_379539)
    # SSA join for if statement (line 66)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 64)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'M' (line 71)
    M_379542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 11), 'M')
    int_379543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 15), 'int')
    # Applying the binary operator '%' (line 71)
    result_mod_379544 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 11), '%', M_379542, int_379543)
    
    int_379545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 20), 'int')
    # Applying the binary operator '==' (line 71)
    result_eq_379546 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 11), '==', result_mod_379544, int_379545)
    
    
    # Getting the type of 'N' (line 71)
    N_379547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 26), 'N')
    int_379548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 30), 'int')
    # Applying the binary operator '%' (line 71)
    result_mod_379549 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 26), '%', N_379547, int_379548)
    
    int_379550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 35), 'int')
    # Applying the binary operator '==' (line 71)
    result_eq_379551 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 26), '==', result_mod_379549, int_379550)
    
    # Applying the binary operator 'and' (line 71)
    result_and_keyword_379552 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 11), 'and', result_eq_379546, result_eq_379551)
    
    # Testing the type of an if condition (line 71)
    if_condition_379553 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 8), result_and_keyword_379552)
    # Assigning a type to the variable 'if_condition_379553' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'if_condition_379553', if_condition_379553)
    # SSA begins for if statement (line 71)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 72):
    
    # Assigning a BinOp to a Name (line 72):
    # Getting the type of 'nnz' (line 72)
    nnz_379554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 18), 'nnz')
    int_379555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 25), 'int')
    
    # Call to count_blocks(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'A' (line 72)
    A_379557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 43), 'A', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 72)
    tuple_379558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 72)
    # Adding element type (line 72)
    int_379559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 46), tuple_379558, int_379559)
    # Adding element type (line 72)
    int_379560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 46), tuple_379558, int_379560)
    
    # Processing the call keyword arguments (line 72)
    kwargs_379561 = {}
    # Getting the type of 'count_blocks' (line 72)
    count_blocks_379556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 30), 'count_blocks', False)
    # Calling count_blocks(args, kwargs) (line 72)
    count_blocks_call_result_379562 = invoke(stypy.reporting.localization.Localization(__file__, 72, 30), count_blocks_379556, *[A_379557, tuple_379558], **kwargs_379561)
    
    # Applying the binary operator '*' (line 72)
    result_mul_379563 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 25), '*', int_379555, count_blocks_call_result_379562)
    
    # Applying the binary operator 'div' (line 72)
    result_div_379564 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 18), 'div', nnz_379554, result_mul_379563)
    
    # Assigning a type to the variable 'e44' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'e44', result_div_379564)
    # SSA branch for the else part of an if statement (line 71)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 74):
    
    # Assigning a Num to a Name (line 74):
    float_379565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 18), 'float')
    # Assigning a type to the variable 'e44' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'e44', float_379565)
    # SSA join for if statement (line 71)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'e44' (line 76)
    e44_379566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 11), 'e44')
    # Getting the type of 'efficiency' (line 76)
    efficiency_379567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 17), 'efficiency')
    # Applying the binary operator '>' (line 76)
    result_gt_379568 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 11), '>', e44_379566, efficiency_379567)
    
    # Testing the type of an if condition (line 76)
    if_condition_379569 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 8), result_gt_379568)
    # Assigning a type to the variable 'if_condition_379569' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'if_condition_379569', if_condition_379569)
    # SSA begins for if statement (line 76)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 77)
    tuple_379570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 77)
    # Adding element type (line 77)
    int_379571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 20), tuple_379570, int_379571)
    # Adding element type (line 77)
    int_379572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 20), tuple_379570, int_379572)
    
    # Assigning a type to the variable 'stypy_return_type' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'stypy_return_type', tuple_379570)
    # SSA branch for the else part of an if statement (line 76)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'e33' (line 78)
    e33_379573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 13), 'e33')
    # Getting the type of 'efficiency' (line 78)
    efficiency_379574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 19), 'efficiency')
    # Applying the binary operator '>' (line 78)
    result_gt_379575 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 13), '>', e33_379573, efficiency_379574)
    
    # Testing the type of an if condition (line 78)
    if_condition_379576 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 13), result_gt_379575)
    # Assigning a type to the variable 'if_condition_379576' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 13), 'if_condition_379576', if_condition_379576)
    # SSA begins for if statement (line 78)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 79)
    tuple_379577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 79)
    # Adding element type (line 79)
    int_379578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 20), tuple_379577, int_379578)
    # Adding element type (line 79)
    int_379579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 20), tuple_379577, int_379579)
    
    # Assigning a type to the variable 'stypy_return_type' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'stypy_return_type', tuple_379577)
    # SSA branch for the else part of an if statement (line 78)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'e22' (line 80)
    e22_379580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 13), 'e22')
    # Getting the type of 'efficiency' (line 80)
    efficiency_379581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 19), 'efficiency')
    # Applying the binary operator '>' (line 80)
    result_gt_379582 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 13), '>', e22_379580, efficiency_379581)
    
    # Testing the type of an if condition (line 80)
    if_condition_379583 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 13), result_gt_379582)
    # Assigning a type to the variable 'if_condition_379583' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 13), 'if_condition_379583', if_condition_379583)
    # SSA begins for if statement (line 80)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 81)
    tuple_379584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 81)
    # Adding element type (line 81)
    int_379585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 20), tuple_379584, int_379585)
    # Adding element type (line 81)
    int_379586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 20), tuple_379584, int_379586)
    
    # Assigning a type to the variable 'stypy_return_type' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'stypy_return_type', tuple_379584)
    # SSA branch for the else part of an if statement (line 80)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 83)
    tuple_379587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 83)
    # Adding element type (line 83)
    int_379588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 20), tuple_379587, int_379588)
    # Adding element type (line 83)
    int_379589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 20), tuple_379587, int_379589)
    
    # Assigning a type to the variable 'stypy_return_type' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'stypy_return_type', tuple_379587)
    # SSA join for if statement (line 80)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 78)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 76)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 64)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'estimate_blocksize(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'estimate_blocksize' in the type store
    # Getting the type of 'stypy_return_type' (line 35)
    stypy_return_type_379590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_379590)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'estimate_blocksize'
    return stypy_return_type_379590

# Assigning a type to the variable 'estimate_blocksize' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'estimate_blocksize', estimate_blocksize)

@norecursion
def count_blocks(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'count_blocks'
    module_type_store = module_type_store.open_function_context('count_blocks', 86, 0, False)
    
    # Passed parameters checking function
    count_blocks.stypy_localization = localization
    count_blocks.stypy_type_of_self = None
    count_blocks.stypy_type_store = module_type_store
    count_blocks.stypy_function_name = 'count_blocks'
    count_blocks.stypy_param_names_list = ['A', 'blocksize']
    count_blocks.stypy_varargs_param_name = None
    count_blocks.stypy_kwargs_param_name = None
    count_blocks.stypy_call_defaults = defaults
    count_blocks.stypy_call_varargs = varargs
    count_blocks.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'count_blocks', ['A', 'blocksize'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'count_blocks', localization, ['A', 'blocksize'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'count_blocks(...)' code ##################

    str_379591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, (-1)), 'str', 'For a given blocksize=(r,c) count the number of occupied\n    blocks in a sparse matrix A\n    ')
    
    # Assigning a Name to a Tuple (line 90):
    
    # Assigning a Subscript to a Name (line 90):
    
    # Obtaining the type of the subscript
    int_379592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 4), 'int')
    # Getting the type of 'blocksize' (line 90)
    blocksize_379593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 10), 'blocksize')
    # Obtaining the member '__getitem__' of a type (line 90)
    getitem___379594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 4), blocksize_379593, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 90)
    subscript_call_result_379595 = invoke(stypy.reporting.localization.Localization(__file__, 90, 4), getitem___379594, int_379592)
    
    # Assigning a type to the variable 'tuple_var_assignment_379387' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'tuple_var_assignment_379387', subscript_call_result_379595)
    
    # Assigning a Subscript to a Name (line 90):
    
    # Obtaining the type of the subscript
    int_379596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 4), 'int')
    # Getting the type of 'blocksize' (line 90)
    blocksize_379597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 10), 'blocksize')
    # Obtaining the member '__getitem__' of a type (line 90)
    getitem___379598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 4), blocksize_379597, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 90)
    subscript_call_result_379599 = invoke(stypy.reporting.localization.Localization(__file__, 90, 4), getitem___379598, int_379596)
    
    # Assigning a type to the variable 'tuple_var_assignment_379388' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'tuple_var_assignment_379388', subscript_call_result_379599)
    
    # Assigning a Name to a Name (line 90):
    # Getting the type of 'tuple_var_assignment_379387' (line 90)
    tuple_var_assignment_379387_379600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'tuple_var_assignment_379387')
    # Assigning a type to the variable 'r' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'r', tuple_var_assignment_379387_379600)
    
    # Assigning a Name to a Name (line 90):
    # Getting the type of 'tuple_var_assignment_379388' (line 90)
    tuple_var_assignment_379388_379601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'tuple_var_assignment_379388')
    # Assigning a type to the variable 'c' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 6), 'c', tuple_var_assignment_379388_379601)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'r' (line 91)
    r_379602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 7), 'r')
    int_379603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 11), 'int')
    # Applying the binary operator '<' (line 91)
    result_lt_379604 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 7), '<', r_379602, int_379603)
    
    
    # Getting the type of 'c' (line 91)
    c_379605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'c')
    int_379606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 20), 'int')
    # Applying the binary operator '<' (line 91)
    result_lt_379607 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 16), '<', c_379605, int_379606)
    
    # Applying the binary operator 'or' (line 91)
    result_or_keyword_379608 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 7), 'or', result_lt_379604, result_lt_379607)
    
    # Testing the type of an if condition (line 91)
    if_condition_379609 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 4), result_or_keyword_379608)
    # Assigning a type to the variable 'if_condition_379609' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'if_condition_379609', if_condition_379609)
    # SSA begins for if statement (line 91)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 92)
    # Processing the call arguments (line 92)
    str_379611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 25), 'str', 'r and c must be positive')
    # Processing the call keyword arguments (line 92)
    kwargs_379612 = {}
    # Getting the type of 'ValueError' (line 92)
    ValueError_379610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 92)
    ValueError_call_result_379613 = invoke(stypy.reporting.localization.Localization(__file__, 92, 14), ValueError_379610, *[str_379611], **kwargs_379612)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 92, 8), ValueError_call_result_379613, 'raise parameter', BaseException)
    # SSA join for if statement (line 91)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isspmatrix_csr(...): (line 94)
    # Processing the call arguments (line 94)
    # Getting the type of 'A' (line 94)
    A_379615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 22), 'A', False)
    # Processing the call keyword arguments (line 94)
    kwargs_379616 = {}
    # Getting the type of 'isspmatrix_csr' (line 94)
    isspmatrix_csr_379614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 7), 'isspmatrix_csr', False)
    # Calling isspmatrix_csr(args, kwargs) (line 94)
    isspmatrix_csr_call_result_379617 = invoke(stypy.reporting.localization.Localization(__file__, 94, 7), isspmatrix_csr_379614, *[A_379615], **kwargs_379616)
    
    # Testing the type of an if condition (line 94)
    if_condition_379618 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 4), isspmatrix_csr_call_result_379617)
    # Assigning a type to the variable 'if_condition_379618' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'if_condition_379618', if_condition_379618)
    # SSA begins for if statement (line 94)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Tuple (line 95):
    
    # Assigning a Subscript to a Name (line 95):
    
    # Obtaining the type of the subscript
    int_379619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 8), 'int')
    # Getting the type of 'A' (line 95)
    A_379620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 14), 'A')
    # Obtaining the member 'shape' of a type (line 95)
    shape_379621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 14), A_379620, 'shape')
    # Obtaining the member '__getitem__' of a type (line 95)
    getitem___379622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), shape_379621, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 95)
    subscript_call_result_379623 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), getitem___379622, int_379619)
    
    # Assigning a type to the variable 'tuple_var_assignment_379389' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'tuple_var_assignment_379389', subscript_call_result_379623)
    
    # Assigning a Subscript to a Name (line 95):
    
    # Obtaining the type of the subscript
    int_379624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 8), 'int')
    # Getting the type of 'A' (line 95)
    A_379625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 14), 'A')
    # Obtaining the member 'shape' of a type (line 95)
    shape_379626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 14), A_379625, 'shape')
    # Obtaining the member '__getitem__' of a type (line 95)
    getitem___379627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), shape_379626, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 95)
    subscript_call_result_379628 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), getitem___379627, int_379624)
    
    # Assigning a type to the variable 'tuple_var_assignment_379390' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'tuple_var_assignment_379390', subscript_call_result_379628)
    
    # Assigning a Name to a Name (line 95):
    # Getting the type of 'tuple_var_assignment_379389' (line 95)
    tuple_var_assignment_379389_379629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'tuple_var_assignment_379389')
    # Assigning a type to the variable 'M' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'M', tuple_var_assignment_379389_379629)
    
    # Assigning a Name to a Name (line 95):
    # Getting the type of 'tuple_var_assignment_379390' (line 95)
    tuple_var_assignment_379390_379630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'tuple_var_assignment_379390')
    # Assigning a type to the variable 'N' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 10), 'N', tuple_var_assignment_379390_379630)
    
    # Call to csr_count_blocks(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'M' (line 96)
    M_379632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 32), 'M', False)
    # Getting the type of 'N' (line 96)
    N_379633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 34), 'N', False)
    # Getting the type of 'r' (line 96)
    r_379634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 36), 'r', False)
    # Getting the type of 'c' (line 96)
    c_379635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 38), 'c', False)
    # Getting the type of 'A' (line 96)
    A_379636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 40), 'A', False)
    # Obtaining the member 'indptr' of a type (line 96)
    indptr_379637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 40), A_379636, 'indptr')
    # Getting the type of 'A' (line 96)
    A_379638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 49), 'A', False)
    # Obtaining the member 'indices' of a type (line 96)
    indices_379639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 49), A_379638, 'indices')
    # Processing the call keyword arguments (line 96)
    kwargs_379640 = {}
    # Getting the type of 'csr_count_blocks' (line 96)
    csr_count_blocks_379631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 15), 'csr_count_blocks', False)
    # Calling csr_count_blocks(args, kwargs) (line 96)
    csr_count_blocks_call_result_379641 = invoke(stypy.reporting.localization.Localization(__file__, 96, 15), csr_count_blocks_379631, *[M_379632, N_379633, r_379634, c_379635, indptr_379637, indices_379639], **kwargs_379640)
    
    # Assigning a type to the variable 'stypy_return_type' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'stypy_return_type', csr_count_blocks_call_result_379641)
    # SSA branch for the else part of an if statement (line 94)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isspmatrix_csc(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'A' (line 97)
    A_379643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 24), 'A', False)
    # Processing the call keyword arguments (line 97)
    kwargs_379644 = {}
    # Getting the type of 'isspmatrix_csc' (line 97)
    isspmatrix_csc_379642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 9), 'isspmatrix_csc', False)
    # Calling isspmatrix_csc(args, kwargs) (line 97)
    isspmatrix_csc_call_result_379645 = invoke(stypy.reporting.localization.Localization(__file__, 97, 9), isspmatrix_csc_379642, *[A_379643], **kwargs_379644)
    
    # Testing the type of an if condition (line 97)
    if_condition_379646 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 9), isspmatrix_csc_call_result_379645)
    # Assigning a type to the variable 'if_condition_379646' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 9), 'if_condition_379646', if_condition_379646)
    # SSA begins for if statement (line 97)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to count_blocks(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'A' (line 98)
    A_379648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 28), 'A', False)
    # Obtaining the member 'T' of a type (line 98)
    T_379649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 28), A_379648, 'T')
    
    # Obtaining an instance of the builtin type 'tuple' (line 98)
    tuple_379650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 98)
    # Adding element type (line 98)
    # Getting the type of 'c' (line 98)
    c_379651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 33), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 33), tuple_379650, c_379651)
    # Adding element type (line 98)
    # Getting the type of 'r' (line 98)
    r_379652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 35), 'r', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 33), tuple_379650, r_379652)
    
    # Processing the call keyword arguments (line 98)
    kwargs_379653 = {}
    # Getting the type of 'count_blocks' (line 98)
    count_blocks_379647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 15), 'count_blocks', False)
    # Calling count_blocks(args, kwargs) (line 98)
    count_blocks_call_result_379654 = invoke(stypy.reporting.localization.Localization(__file__, 98, 15), count_blocks_379647, *[T_379649, tuple_379650], **kwargs_379653)
    
    # Assigning a type to the variable 'stypy_return_type' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'stypy_return_type', count_blocks_call_result_379654)
    # SSA branch for the else part of an if statement (line 97)
    module_type_store.open_ssa_branch('else')
    
    # Call to count_blocks(...): (line 100)
    # Processing the call arguments (line 100)
    
    # Call to csr_matrix(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'A' (line 100)
    A_379657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 39), 'A', False)
    # Processing the call keyword arguments (line 100)
    kwargs_379658 = {}
    # Getting the type of 'csr_matrix' (line 100)
    csr_matrix_379656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 28), 'csr_matrix', False)
    # Calling csr_matrix(args, kwargs) (line 100)
    csr_matrix_call_result_379659 = invoke(stypy.reporting.localization.Localization(__file__, 100, 28), csr_matrix_379656, *[A_379657], **kwargs_379658)
    
    # Getting the type of 'blocksize' (line 100)
    blocksize_379660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 42), 'blocksize', False)
    # Processing the call keyword arguments (line 100)
    kwargs_379661 = {}
    # Getting the type of 'count_blocks' (line 100)
    count_blocks_379655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'count_blocks', False)
    # Calling count_blocks(args, kwargs) (line 100)
    count_blocks_call_result_379662 = invoke(stypy.reporting.localization.Localization(__file__, 100, 15), count_blocks_379655, *[csr_matrix_call_result_379659, blocksize_379660], **kwargs_379661)
    
    # Assigning a type to the variable 'stypy_return_type' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'stypy_return_type', count_blocks_call_result_379662)
    # SSA join for if statement (line 97)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 94)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'count_blocks(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'count_blocks' in the type store
    # Getting the type of 'stypy_return_type' (line 86)
    stypy_return_type_379663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_379663)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'count_blocks'
    return stypy_return_type_379663

# Assigning a type to the variable 'count_blocks' (line 86)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'count_blocks', count_blocks)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
