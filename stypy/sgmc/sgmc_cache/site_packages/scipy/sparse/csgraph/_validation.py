
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from scipy.sparse import csr_matrix, isspmatrix, isspmatrix_csc
5: from ._tools import csgraph_to_dense, csgraph_from_dense,\
6:     csgraph_masked_from_dense, csgraph_from_masked
7: 
8: DTYPE = np.float64
9: 
10: 
11: def validate_graph(csgraph, directed, dtype=DTYPE,
12:                    csr_output=True, dense_output=True,
13:                    copy_if_dense=False, copy_if_sparse=False,
14:                    null_value_in=0, null_value_out=np.inf,
15:                    infinity_null=True, nan_null=True):
16:     '''Routine for validation and conversion of csgraph inputs'''
17:     if not (csr_output or dense_output):
18:         raise ValueError("Internal: dense or csr output must be true")
19: 
20:     # if undirected and csc storage, then transposing in-place
21:     # is quicker than later converting to csr.
22:     if (not directed) and isspmatrix_csc(csgraph):
23:         csgraph = csgraph.T
24: 
25:     if isspmatrix(csgraph):
26:         if csr_output:
27:             csgraph = csr_matrix(csgraph, dtype=DTYPE, copy=copy_if_sparse)
28:         else:
29:             csgraph = csgraph_to_dense(csgraph, null_value=null_value_out)
30:     elif np.ma.isMaskedArray(csgraph):
31:         if dense_output:
32:             mask = csgraph.mask
33:             csgraph = np.array(csgraph.data, dtype=DTYPE, copy=copy_if_dense)
34:             csgraph[mask] = null_value_out
35:         else:
36:             csgraph = csgraph_from_masked(csgraph)
37:     else:
38:         if dense_output:
39:             csgraph = csgraph_masked_from_dense(csgraph,
40:                                                 copy=copy_if_dense,
41:                                                 null_value=null_value_in,
42:                                                 nan_null=nan_null,
43:                                                 infinity_null=infinity_null)
44:             mask = csgraph.mask
45:             csgraph = np.asarray(csgraph.data, dtype=DTYPE)
46:             csgraph[mask] = null_value_out
47:         else:
48:             csgraph = csgraph_from_dense(csgraph, null_value=null_value_in,
49:                                          infinity_null=infinity_null,
50:                                          nan_null=nan_null)
51: 
52:     if csgraph.ndim != 2:
53:         raise ValueError("compressed-sparse graph must be two dimensional")
54: 
55:     if csgraph.shape[0] != csgraph.shape[1]:
56:         raise ValueError("compressed-sparse graph must be shape (N, N)")
57: 
58:     return csgraph
59: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')
import_381487 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_381487) is not StypyTypeError):

    if (import_381487 != 'pyd_module'):
        __import__(import_381487)
        sys_modules_381488 = sys.modules[import_381487]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_381488.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_381487)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from scipy.sparse import csr_matrix, isspmatrix, isspmatrix_csc' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')
import_381489 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.sparse')

if (type(import_381489) is not StypyTypeError):

    if (import_381489 != 'pyd_module'):
        __import__(import_381489)
        sys_modules_381490 = sys.modules[import_381489]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.sparse', sys_modules_381490.module_type_store, module_type_store, ['csr_matrix', 'isspmatrix', 'isspmatrix_csc'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_381490, sys_modules_381490.module_type_store, module_type_store)
    else:
        from scipy.sparse import csr_matrix, isspmatrix, isspmatrix_csc

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.sparse', None, module_type_store, ['csr_matrix', 'isspmatrix', 'isspmatrix_csc'], [csr_matrix, isspmatrix, isspmatrix_csc])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.sparse', import_381489)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.sparse.csgraph._tools import csgraph_to_dense, csgraph_from_dense, csgraph_masked_from_dense, csgraph_from_masked' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')
import_381491 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse.csgraph._tools')

if (type(import_381491) is not StypyTypeError):

    if (import_381491 != 'pyd_module'):
        __import__(import_381491)
        sys_modules_381492 = sys.modules[import_381491]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse.csgraph._tools', sys_modules_381492.module_type_store, module_type_store, ['csgraph_to_dense', 'csgraph_from_dense', 'csgraph_masked_from_dense', 'csgraph_from_masked'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_381492, sys_modules_381492.module_type_store, module_type_store)
    else:
        from scipy.sparse.csgraph._tools import csgraph_to_dense, csgraph_from_dense, csgraph_masked_from_dense, csgraph_from_masked

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse.csgraph._tools', None, module_type_store, ['csgraph_to_dense', 'csgraph_from_dense', 'csgraph_masked_from_dense', 'csgraph_from_masked'], [csgraph_to_dense, csgraph_from_dense, csgraph_masked_from_dense, csgraph_from_masked])

else:
    # Assigning a type to the variable 'scipy.sparse.csgraph._tools' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse.csgraph._tools', import_381491)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')


# Assigning a Attribute to a Name (line 8):
# Getting the type of 'np' (line 8)
np_381493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'np')
# Obtaining the member 'float64' of a type (line 8)
float64_381494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 8), np_381493, 'float64')
# Assigning a type to the variable 'DTYPE' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'DTYPE', float64_381494)

@norecursion
def validate_graph(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'DTYPE' (line 11)
    DTYPE_381495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 44), 'DTYPE')
    # Getting the type of 'True' (line 12)
    True_381496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 30), 'True')
    # Getting the type of 'True' (line 12)
    True_381497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 49), 'True')
    # Getting the type of 'False' (line 13)
    False_381498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 33), 'False')
    # Getting the type of 'False' (line 13)
    False_381499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 55), 'False')
    int_381500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 33), 'int')
    # Getting the type of 'np' (line 14)
    np_381501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 51), 'np')
    # Obtaining the member 'inf' of a type (line 14)
    inf_381502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 51), np_381501, 'inf')
    # Getting the type of 'True' (line 15)
    True_381503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 33), 'True')
    # Getting the type of 'True' (line 15)
    True_381504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 48), 'True')
    defaults = [DTYPE_381495, True_381496, True_381497, False_381498, False_381499, int_381500, inf_381502, True_381503, True_381504]
    # Create a new context for function 'validate_graph'
    module_type_store = module_type_store.open_function_context('validate_graph', 11, 0, False)
    
    # Passed parameters checking function
    validate_graph.stypy_localization = localization
    validate_graph.stypy_type_of_self = None
    validate_graph.stypy_type_store = module_type_store
    validate_graph.stypy_function_name = 'validate_graph'
    validate_graph.stypy_param_names_list = ['csgraph', 'directed', 'dtype', 'csr_output', 'dense_output', 'copy_if_dense', 'copy_if_sparse', 'null_value_in', 'null_value_out', 'infinity_null', 'nan_null']
    validate_graph.stypy_varargs_param_name = None
    validate_graph.stypy_kwargs_param_name = None
    validate_graph.stypy_call_defaults = defaults
    validate_graph.stypy_call_varargs = varargs
    validate_graph.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'validate_graph', ['csgraph', 'directed', 'dtype', 'csr_output', 'dense_output', 'copy_if_dense', 'copy_if_sparse', 'null_value_in', 'null_value_out', 'infinity_null', 'nan_null'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'validate_graph', localization, ['csgraph', 'directed', 'dtype', 'csr_output', 'dense_output', 'copy_if_dense', 'copy_if_sparse', 'null_value_in', 'null_value_out', 'infinity_null', 'nan_null'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'validate_graph(...)' code ##################

    str_381505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'str', 'Routine for validation and conversion of csgraph inputs')
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'csr_output' (line 17)
    csr_output_381506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'csr_output')
    # Getting the type of 'dense_output' (line 17)
    dense_output_381507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 26), 'dense_output')
    # Applying the binary operator 'or' (line 17)
    result_or_keyword_381508 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 12), 'or', csr_output_381506, dense_output_381507)
    
    # Applying the 'not' unary operator (line 17)
    result_not__381509 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 7), 'not', result_or_keyword_381508)
    
    # Testing the type of an if condition (line 17)
    if_condition_381510 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 17, 4), result_not__381509)
    # Assigning a type to the variable 'if_condition_381510' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'if_condition_381510', if_condition_381510)
    # SSA begins for if statement (line 17)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 18)
    # Processing the call arguments (line 18)
    str_381512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'str', 'Internal: dense or csr output must be true')
    # Processing the call keyword arguments (line 18)
    kwargs_381513 = {}
    # Getting the type of 'ValueError' (line 18)
    ValueError_381511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 18)
    ValueError_call_result_381514 = invoke(stypy.reporting.localization.Localization(__file__, 18, 14), ValueError_381511, *[str_381512], **kwargs_381513)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 18, 8), ValueError_call_result_381514, 'raise parameter', BaseException)
    # SSA join for if statement (line 17)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'directed' (line 22)
    directed_381515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'directed')
    # Applying the 'not' unary operator (line 22)
    result_not__381516 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 8), 'not', directed_381515)
    
    
    # Call to isspmatrix_csc(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'csgraph' (line 22)
    csgraph_381518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 41), 'csgraph', False)
    # Processing the call keyword arguments (line 22)
    kwargs_381519 = {}
    # Getting the type of 'isspmatrix_csc' (line 22)
    isspmatrix_csc_381517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 26), 'isspmatrix_csc', False)
    # Calling isspmatrix_csc(args, kwargs) (line 22)
    isspmatrix_csc_call_result_381520 = invoke(stypy.reporting.localization.Localization(__file__, 22, 26), isspmatrix_csc_381517, *[csgraph_381518], **kwargs_381519)
    
    # Applying the binary operator 'and' (line 22)
    result_and_keyword_381521 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 7), 'and', result_not__381516, isspmatrix_csc_call_result_381520)
    
    # Testing the type of an if condition (line 22)
    if_condition_381522 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 22, 4), result_and_keyword_381521)
    # Assigning a type to the variable 'if_condition_381522' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'if_condition_381522', if_condition_381522)
    # SSA begins for if statement (line 22)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 23):
    # Getting the type of 'csgraph' (line 23)
    csgraph_381523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 18), 'csgraph')
    # Obtaining the member 'T' of a type (line 23)
    T_381524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 18), csgraph_381523, 'T')
    # Assigning a type to the variable 'csgraph' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'csgraph', T_381524)
    # SSA join for if statement (line 22)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isspmatrix(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'csgraph' (line 25)
    csgraph_381526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 18), 'csgraph', False)
    # Processing the call keyword arguments (line 25)
    kwargs_381527 = {}
    # Getting the type of 'isspmatrix' (line 25)
    isspmatrix_381525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 7), 'isspmatrix', False)
    # Calling isspmatrix(args, kwargs) (line 25)
    isspmatrix_call_result_381528 = invoke(stypy.reporting.localization.Localization(__file__, 25, 7), isspmatrix_381525, *[csgraph_381526], **kwargs_381527)
    
    # Testing the type of an if condition (line 25)
    if_condition_381529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 25, 4), isspmatrix_call_result_381528)
    # Assigning a type to the variable 'if_condition_381529' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'if_condition_381529', if_condition_381529)
    # SSA begins for if statement (line 25)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'csr_output' (line 26)
    csr_output_381530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'csr_output')
    # Testing the type of an if condition (line 26)
    if_condition_381531 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 26, 8), csr_output_381530)
    # Assigning a type to the variable 'if_condition_381531' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'if_condition_381531', if_condition_381531)
    # SSA begins for if statement (line 26)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 27):
    
    # Call to csr_matrix(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'csgraph' (line 27)
    csgraph_381533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 33), 'csgraph', False)
    # Processing the call keyword arguments (line 27)
    # Getting the type of 'DTYPE' (line 27)
    DTYPE_381534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 48), 'DTYPE', False)
    keyword_381535 = DTYPE_381534
    # Getting the type of 'copy_if_sparse' (line 27)
    copy_if_sparse_381536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 60), 'copy_if_sparse', False)
    keyword_381537 = copy_if_sparse_381536
    kwargs_381538 = {'dtype': keyword_381535, 'copy': keyword_381537}
    # Getting the type of 'csr_matrix' (line 27)
    csr_matrix_381532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 22), 'csr_matrix', False)
    # Calling csr_matrix(args, kwargs) (line 27)
    csr_matrix_call_result_381539 = invoke(stypy.reporting.localization.Localization(__file__, 27, 22), csr_matrix_381532, *[csgraph_381533], **kwargs_381538)
    
    # Assigning a type to the variable 'csgraph' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'csgraph', csr_matrix_call_result_381539)
    # SSA branch for the else part of an if statement (line 26)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 29):
    
    # Call to csgraph_to_dense(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'csgraph' (line 29)
    csgraph_381541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 39), 'csgraph', False)
    # Processing the call keyword arguments (line 29)
    # Getting the type of 'null_value_out' (line 29)
    null_value_out_381542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 59), 'null_value_out', False)
    keyword_381543 = null_value_out_381542
    kwargs_381544 = {'null_value': keyword_381543}
    # Getting the type of 'csgraph_to_dense' (line 29)
    csgraph_to_dense_381540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 22), 'csgraph_to_dense', False)
    # Calling csgraph_to_dense(args, kwargs) (line 29)
    csgraph_to_dense_call_result_381545 = invoke(stypy.reporting.localization.Localization(__file__, 29, 22), csgraph_to_dense_381540, *[csgraph_381541], **kwargs_381544)
    
    # Assigning a type to the variable 'csgraph' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'csgraph', csgraph_to_dense_call_result_381545)
    # SSA join for if statement (line 26)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 25)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isMaskedArray(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'csgraph' (line 30)
    csgraph_381549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 29), 'csgraph', False)
    # Processing the call keyword arguments (line 30)
    kwargs_381550 = {}
    # Getting the type of 'np' (line 30)
    np_381546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 9), 'np', False)
    # Obtaining the member 'ma' of a type (line 30)
    ma_381547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 9), np_381546, 'ma')
    # Obtaining the member 'isMaskedArray' of a type (line 30)
    isMaskedArray_381548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 9), ma_381547, 'isMaskedArray')
    # Calling isMaskedArray(args, kwargs) (line 30)
    isMaskedArray_call_result_381551 = invoke(stypy.reporting.localization.Localization(__file__, 30, 9), isMaskedArray_381548, *[csgraph_381549], **kwargs_381550)
    
    # Testing the type of an if condition (line 30)
    if_condition_381552 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 30, 9), isMaskedArray_call_result_381551)
    # Assigning a type to the variable 'if_condition_381552' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 9), 'if_condition_381552', if_condition_381552)
    # SSA begins for if statement (line 30)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'dense_output' (line 31)
    dense_output_381553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), 'dense_output')
    # Testing the type of an if condition (line 31)
    if_condition_381554 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 8), dense_output_381553)
    # Assigning a type to the variable 'if_condition_381554' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'if_condition_381554', if_condition_381554)
    # SSA begins for if statement (line 31)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 32):
    # Getting the type of 'csgraph' (line 32)
    csgraph_381555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 19), 'csgraph')
    # Obtaining the member 'mask' of a type (line 32)
    mask_381556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 19), csgraph_381555, 'mask')
    # Assigning a type to the variable 'mask' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'mask', mask_381556)
    
    # Assigning a Call to a Name (line 33):
    
    # Call to array(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'csgraph' (line 33)
    csgraph_381559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 31), 'csgraph', False)
    # Obtaining the member 'data' of a type (line 33)
    data_381560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 31), csgraph_381559, 'data')
    # Processing the call keyword arguments (line 33)
    # Getting the type of 'DTYPE' (line 33)
    DTYPE_381561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 51), 'DTYPE', False)
    keyword_381562 = DTYPE_381561
    # Getting the type of 'copy_if_dense' (line 33)
    copy_if_dense_381563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 63), 'copy_if_dense', False)
    keyword_381564 = copy_if_dense_381563
    kwargs_381565 = {'dtype': keyword_381562, 'copy': keyword_381564}
    # Getting the type of 'np' (line 33)
    np_381557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 22), 'np', False)
    # Obtaining the member 'array' of a type (line 33)
    array_381558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 22), np_381557, 'array')
    # Calling array(args, kwargs) (line 33)
    array_call_result_381566 = invoke(stypy.reporting.localization.Localization(__file__, 33, 22), array_381558, *[data_381560], **kwargs_381565)
    
    # Assigning a type to the variable 'csgraph' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'csgraph', array_call_result_381566)
    
    # Assigning a Name to a Subscript (line 34):
    # Getting the type of 'null_value_out' (line 34)
    null_value_out_381567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 28), 'null_value_out')
    # Getting the type of 'csgraph' (line 34)
    csgraph_381568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'csgraph')
    # Getting the type of 'mask' (line 34)
    mask_381569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 20), 'mask')
    # Storing an element on a container (line 34)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 12), csgraph_381568, (mask_381569, null_value_out_381567))
    # SSA branch for the else part of an if statement (line 31)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 36):
    
    # Call to csgraph_from_masked(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'csgraph' (line 36)
    csgraph_381571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 42), 'csgraph', False)
    # Processing the call keyword arguments (line 36)
    kwargs_381572 = {}
    # Getting the type of 'csgraph_from_masked' (line 36)
    csgraph_from_masked_381570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 22), 'csgraph_from_masked', False)
    # Calling csgraph_from_masked(args, kwargs) (line 36)
    csgraph_from_masked_call_result_381573 = invoke(stypy.reporting.localization.Localization(__file__, 36, 22), csgraph_from_masked_381570, *[csgraph_381571], **kwargs_381572)
    
    # Assigning a type to the variable 'csgraph' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'csgraph', csgraph_from_masked_call_result_381573)
    # SSA join for if statement (line 31)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 30)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'dense_output' (line 38)
    dense_output_381574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), 'dense_output')
    # Testing the type of an if condition (line 38)
    if_condition_381575 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 8), dense_output_381574)
    # Assigning a type to the variable 'if_condition_381575' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'if_condition_381575', if_condition_381575)
    # SSA begins for if statement (line 38)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 39):
    
    # Call to csgraph_masked_from_dense(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'csgraph' (line 39)
    csgraph_381577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 48), 'csgraph', False)
    # Processing the call keyword arguments (line 39)
    # Getting the type of 'copy_if_dense' (line 40)
    copy_if_dense_381578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 53), 'copy_if_dense', False)
    keyword_381579 = copy_if_dense_381578
    # Getting the type of 'null_value_in' (line 41)
    null_value_in_381580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 59), 'null_value_in', False)
    keyword_381581 = null_value_in_381580
    # Getting the type of 'nan_null' (line 42)
    nan_null_381582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 57), 'nan_null', False)
    keyword_381583 = nan_null_381582
    # Getting the type of 'infinity_null' (line 43)
    infinity_null_381584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 62), 'infinity_null', False)
    keyword_381585 = infinity_null_381584
    kwargs_381586 = {'infinity_null': keyword_381585, 'copy': keyword_381579, 'nan_null': keyword_381583, 'null_value': keyword_381581}
    # Getting the type of 'csgraph_masked_from_dense' (line 39)
    csgraph_masked_from_dense_381576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 22), 'csgraph_masked_from_dense', False)
    # Calling csgraph_masked_from_dense(args, kwargs) (line 39)
    csgraph_masked_from_dense_call_result_381587 = invoke(stypy.reporting.localization.Localization(__file__, 39, 22), csgraph_masked_from_dense_381576, *[csgraph_381577], **kwargs_381586)
    
    # Assigning a type to the variable 'csgraph' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'csgraph', csgraph_masked_from_dense_call_result_381587)
    
    # Assigning a Attribute to a Name (line 44):
    # Getting the type of 'csgraph' (line 44)
    csgraph_381588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 19), 'csgraph')
    # Obtaining the member 'mask' of a type (line 44)
    mask_381589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 19), csgraph_381588, 'mask')
    # Assigning a type to the variable 'mask' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'mask', mask_381589)
    
    # Assigning a Call to a Name (line 45):
    
    # Call to asarray(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'csgraph' (line 45)
    csgraph_381592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 33), 'csgraph', False)
    # Obtaining the member 'data' of a type (line 45)
    data_381593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 33), csgraph_381592, 'data')
    # Processing the call keyword arguments (line 45)
    # Getting the type of 'DTYPE' (line 45)
    DTYPE_381594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 53), 'DTYPE', False)
    keyword_381595 = DTYPE_381594
    kwargs_381596 = {'dtype': keyword_381595}
    # Getting the type of 'np' (line 45)
    np_381590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 22), 'np', False)
    # Obtaining the member 'asarray' of a type (line 45)
    asarray_381591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 22), np_381590, 'asarray')
    # Calling asarray(args, kwargs) (line 45)
    asarray_call_result_381597 = invoke(stypy.reporting.localization.Localization(__file__, 45, 22), asarray_381591, *[data_381593], **kwargs_381596)
    
    # Assigning a type to the variable 'csgraph' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'csgraph', asarray_call_result_381597)
    
    # Assigning a Name to a Subscript (line 46):
    # Getting the type of 'null_value_out' (line 46)
    null_value_out_381598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 28), 'null_value_out')
    # Getting the type of 'csgraph' (line 46)
    csgraph_381599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'csgraph')
    # Getting the type of 'mask' (line 46)
    mask_381600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'mask')
    # Storing an element on a container (line 46)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 12), csgraph_381599, (mask_381600, null_value_out_381598))
    # SSA branch for the else part of an if statement (line 38)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 48):
    
    # Call to csgraph_from_dense(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'csgraph' (line 48)
    csgraph_381602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 41), 'csgraph', False)
    # Processing the call keyword arguments (line 48)
    # Getting the type of 'null_value_in' (line 48)
    null_value_in_381603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 61), 'null_value_in', False)
    keyword_381604 = null_value_in_381603
    # Getting the type of 'infinity_null' (line 49)
    infinity_null_381605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 55), 'infinity_null', False)
    keyword_381606 = infinity_null_381605
    # Getting the type of 'nan_null' (line 50)
    nan_null_381607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 50), 'nan_null', False)
    keyword_381608 = nan_null_381607
    kwargs_381609 = {'nan_null': keyword_381608, 'infinity_null': keyword_381606, 'null_value': keyword_381604}
    # Getting the type of 'csgraph_from_dense' (line 48)
    csgraph_from_dense_381601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'csgraph_from_dense', False)
    # Calling csgraph_from_dense(args, kwargs) (line 48)
    csgraph_from_dense_call_result_381610 = invoke(stypy.reporting.localization.Localization(__file__, 48, 22), csgraph_from_dense_381601, *[csgraph_381602], **kwargs_381609)
    
    # Assigning a type to the variable 'csgraph' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'csgraph', csgraph_from_dense_call_result_381610)
    # SSA join for if statement (line 38)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 30)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 25)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'csgraph' (line 52)
    csgraph_381611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 7), 'csgraph')
    # Obtaining the member 'ndim' of a type (line 52)
    ndim_381612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 7), csgraph_381611, 'ndim')
    int_381613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 23), 'int')
    # Applying the binary operator '!=' (line 52)
    result_ne_381614 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 7), '!=', ndim_381612, int_381613)
    
    # Testing the type of an if condition (line 52)
    if_condition_381615 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 4), result_ne_381614)
    # Assigning a type to the variable 'if_condition_381615' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'if_condition_381615', if_condition_381615)
    # SSA begins for if statement (line 52)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 53)
    # Processing the call arguments (line 53)
    str_381617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 25), 'str', 'compressed-sparse graph must be two dimensional')
    # Processing the call keyword arguments (line 53)
    kwargs_381618 = {}
    # Getting the type of 'ValueError' (line 53)
    ValueError_381616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 53)
    ValueError_call_result_381619 = invoke(stypy.reporting.localization.Localization(__file__, 53, 14), ValueError_381616, *[str_381617], **kwargs_381618)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 53, 8), ValueError_call_result_381619, 'raise parameter', BaseException)
    # SSA join for if statement (line 52)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_381620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 21), 'int')
    # Getting the type of 'csgraph' (line 55)
    csgraph_381621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 7), 'csgraph')
    # Obtaining the member 'shape' of a type (line 55)
    shape_381622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 7), csgraph_381621, 'shape')
    # Obtaining the member '__getitem__' of a type (line 55)
    getitem___381623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 7), shape_381622, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 55)
    subscript_call_result_381624 = invoke(stypy.reporting.localization.Localization(__file__, 55, 7), getitem___381623, int_381620)
    
    
    # Obtaining the type of the subscript
    int_381625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 41), 'int')
    # Getting the type of 'csgraph' (line 55)
    csgraph_381626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 27), 'csgraph')
    # Obtaining the member 'shape' of a type (line 55)
    shape_381627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 27), csgraph_381626, 'shape')
    # Obtaining the member '__getitem__' of a type (line 55)
    getitem___381628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 27), shape_381627, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 55)
    subscript_call_result_381629 = invoke(stypy.reporting.localization.Localization(__file__, 55, 27), getitem___381628, int_381625)
    
    # Applying the binary operator '!=' (line 55)
    result_ne_381630 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 7), '!=', subscript_call_result_381624, subscript_call_result_381629)
    
    # Testing the type of an if condition (line 55)
    if_condition_381631 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 55, 4), result_ne_381630)
    # Assigning a type to the variable 'if_condition_381631' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'if_condition_381631', if_condition_381631)
    # SSA begins for if statement (line 55)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 56)
    # Processing the call arguments (line 56)
    str_381633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 25), 'str', 'compressed-sparse graph must be shape (N, N)')
    # Processing the call keyword arguments (line 56)
    kwargs_381634 = {}
    # Getting the type of 'ValueError' (line 56)
    ValueError_381632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 56)
    ValueError_call_result_381635 = invoke(stypy.reporting.localization.Localization(__file__, 56, 14), ValueError_381632, *[str_381633], **kwargs_381634)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 56, 8), ValueError_call_result_381635, 'raise parameter', BaseException)
    # SSA join for if statement (line 55)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'csgraph' (line 58)
    csgraph_381636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 11), 'csgraph')
    # Assigning a type to the variable 'stypy_return_type' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type', csgraph_381636)
    
    # ################# End of 'validate_graph(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'validate_graph' in the type store
    # Getting the type of 'stypy_return_type' (line 11)
    stypy_return_type_381637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_381637)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'validate_graph'
    return stypy_return_type_381637

# Assigning a type to the variable 'validate_graph' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'validate_graph', validate_graph)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
