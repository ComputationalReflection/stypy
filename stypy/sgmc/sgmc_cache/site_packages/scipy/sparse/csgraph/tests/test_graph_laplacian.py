
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
2: #         Jake Vanderplas <vanderplas@astro.washington.edu>
3: # License: BSD
4: from __future__ import division, print_function, absolute_import
5: 
6: import numpy as np
7: from numpy.testing import assert_allclose, assert_array_almost_equal
8: from pytest import raises as assert_raises
9: from scipy import sparse
10: 
11: from scipy.sparse import csgraph
12: 
13: 
14: def _explicit_laplacian(x, normed=False):
15:     if sparse.issparse(x):
16:         x = x.todense()
17:     x = np.asarray(x)
18:     y = -1.0 * x
19:     for j in range(y.shape[0]):
20:         y[j,j] = x[j,j+1:].sum() + x[j,:j].sum()
21:     if normed:
22:         d = np.diag(y).copy()
23:         d[d == 0] = 1.0
24:         y /= d[:,None]**.5
25:         y /= d[None,:]**.5
26:     return y
27: 
28: 
29: def _check_symmetric_graph_laplacian(mat, normed):
30:     if not hasattr(mat, 'shape'):
31:         mat = eval(mat, dict(np=np, sparse=sparse))
32: 
33:     if sparse.issparse(mat):
34:         sp_mat = mat
35:         mat = sp_mat.todense()
36:     else:
37:         sp_mat = sparse.csr_matrix(mat)
38: 
39:     laplacian = csgraph.laplacian(mat, normed=normed)
40:     n_nodes = mat.shape[0]
41:     if not normed:
42:         assert_array_almost_equal(laplacian.sum(axis=0), np.zeros(n_nodes))
43:     assert_array_almost_equal(laplacian.T, laplacian)
44:     assert_array_almost_equal(laplacian,
45:             csgraph.laplacian(sp_mat, normed=normed).todense())
46: 
47:     assert_array_almost_equal(laplacian,
48:             _explicit_laplacian(mat, normed=normed))
49: 
50: 
51: def test_laplacian_value_error():
52:     for t in int, float, complex:
53:         for m in ([1, 1],
54:                   [[[1]]],
55:                   [[1, 2, 3], [4, 5, 6]],
56:                   [[1, 2], [3, 4], [5, 5]]):
57:             A = np.array(m, dtype=t)
58:             assert_raises(ValueError, csgraph.laplacian, A)
59: 
60: 
61: def test_symmetric_graph_laplacian():
62:     symmetric_mats = ('np.arange(10) * np.arange(10)[:, np.newaxis]',
63:             'np.ones((7, 7))',
64:             'np.eye(19)',
65:             'sparse.diags([1, 1], [-1, 1], shape=(4,4))',
66:             'sparse.diags([1, 1], [-1, 1], shape=(4,4)).todense()',
67:             'np.asarray(sparse.diags([1, 1], [-1, 1], shape=(4,4)).todense())',
68:             'np.vander(np.arange(4)) + np.vander(np.arange(4)).T')
69:     for mat_str in symmetric_mats:
70:         for normed in True, False:
71:             _check_symmetric_graph_laplacian(mat_str, normed)
72: 
73: 
74: def _assert_allclose_sparse(a, b, **kwargs):
75:     # helper function that can deal with sparse matrices
76:     if sparse.issparse(a):
77:         a = a.toarray()
78:     if sparse.issparse(b):
79:         b = a.toarray()
80:     assert_allclose(a, b, **kwargs)
81: 
82: 
83: def _check_laplacian(A, desired_L, desired_d, normed, use_out_degree):
84:     for arr_type in np.array, sparse.csr_matrix, sparse.coo_matrix:
85:         for t in int, float, complex:
86:             adj = arr_type(A, dtype=t)
87:             L = csgraph.laplacian(adj, normed=normed, return_diag=False,
88:                                   use_out_degree=use_out_degree)
89:             _assert_allclose_sparse(L, desired_L, atol=1e-12)
90:             L, d = csgraph.laplacian(adj, normed=normed, return_diag=True,
91:                                   use_out_degree=use_out_degree)
92:             _assert_allclose_sparse(L, desired_L, atol=1e-12)
93:             _assert_allclose_sparse(d, desired_d, atol=1e-12)
94: 
95: 
96: def test_asymmetric_laplacian():
97:     # adjacency matrix
98:     A = [[0, 1, 0],
99:          [4, 2, 0],
100:          [0, 0, 0]]
101: 
102:     # Laplacian matrix using out-degree
103:     L = [[1, -1, 0],
104:          [-4, 4, 0],
105:          [0, 0, 0]]
106:     d = [1, 4, 0]
107:     _check_laplacian(A, L, d, normed=False, use_out_degree=True)
108: 
109:     # normalized Laplacian matrix using out-degree
110:     L = [[1, -0.5, 0],
111:          [-2, 1, 0],
112:          [0, 0, 0]]
113:     d = [1, 2, 1]
114:     _check_laplacian(A, L, d, normed=True, use_out_degree=True)
115: 
116:     # Laplacian matrix using in-degree
117:     L = [[4, -1, 0],
118:          [-4, 1, 0],
119:          [0, 0, 0]]
120:     d = [4, 1, 0]
121:     _check_laplacian(A, L, d, normed=False, use_out_degree=False)
122: 
123:     # normalized Laplacian matrix using in-degree
124:     L = [[1, -0.5, 0],
125:          [-2, 1, 0],
126:          [0, 0, 0]]
127:     d = [2, 1, 1]
128:     _check_laplacian(A, L, d, normed=True, use_out_degree=False)
129: 
130: 
131: def test_sparse_formats():
132:     for fmt in ('csr', 'csc', 'coo', 'lil', 'dok', 'dia', 'bsr'):
133:         mat = sparse.diags([1, 1], [-1, 1], shape=(4,4), format=fmt)
134:         for normed in True, False:
135:             _check_symmetric_graph_laplacian(mat, normed)
136: 
137: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_382439 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_382439) is not StypyTypeError):

    if (import_382439 != 'pyd_module'):
        __import__(import_382439)
        sys_modules_382440 = sys.modules[import_382439]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_382440.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_382439)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.testing import assert_allclose, assert_array_almost_equal' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_382441 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing')

if (type(import_382441) is not StypyTypeError):

    if (import_382441 != 'pyd_module'):
        __import__(import_382441)
        sys_modules_382442 = sys.modules[import_382441]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', sys_modules_382442.module_type_store, module_type_store, ['assert_allclose', 'assert_array_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_382442, sys_modules_382442.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_allclose, assert_array_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', None, module_type_store, ['assert_allclose', 'assert_array_almost_equal'], [assert_allclose, assert_array_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', import_382441)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from pytest import assert_raises' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_382443 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest')

if (type(import_382443) is not StypyTypeError):

    if (import_382443 != 'pyd_module'):
        __import__(import_382443)
        sys_modules_382444 = sys.modules[import_382443]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', sys_modules_382444.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_382444, sys_modules_382444.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', import_382443)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy import sparse' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_382445 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy')

if (type(import_382445) is not StypyTypeError):

    if (import_382445 != 'pyd_module'):
        __import__(import_382445)
        sys_modules_382446 = sys.modules[import_382445]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy', sys_modules_382446.module_type_store, module_type_store, ['sparse'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_382446, sys_modules_382446.module_type_store, module_type_store)
    else:
        from scipy import sparse

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy', None, module_type_store, ['sparse'], [sparse])

else:
    # Assigning a type to the variable 'scipy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy', import_382445)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.sparse import csgraph' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_382447 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse')

if (type(import_382447) is not StypyTypeError):

    if (import_382447 != 'pyd_module'):
        __import__(import_382447)
        sys_modules_382448 = sys.modules[import_382447]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse', sys_modules_382448.module_type_store, module_type_store, ['csgraph'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_382448, sys_modules_382448.module_type_store, module_type_store)
    else:
        from scipy.sparse import csgraph

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse', None, module_type_store, ['csgraph'], [csgraph])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse', import_382447)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')


@norecursion
def _explicit_laplacian(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 14)
    False_382449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 34), 'False')
    defaults = [False_382449]
    # Create a new context for function '_explicit_laplacian'
    module_type_store = module_type_store.open_function_context('_explicit_laplacian', 14, 0, False)
    
    # Passed parameters checking function
    _explicit_laplacian.stypy_localization = localization
    _explicit_laplacian.stypy_type_of_self = None
    _explicit_laplacian.stypy_type_store = module_type_store
    _explicit_laplacian.stypy_function_name = '_explicit_laplacian'
    _explicit_laplacian.stypy_param_names_list = ['x', 'normed']
    _explicit_laplacian.stypy_varargs_param_name = None
    _explicit_laplacian.stypy_kwargs_param_name = None
    _explicit_laplacian.stypy_call_defaults = defaults
    _explicit_laplacian.stypy_call_varargs = varargs
    _explicit_laplacian.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_explicit_laplacian', ['x', 'normed'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_explicit_laplacian', localization, ['x', 'normed'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_explicit_laplacian(...)' code ##################

    
    
    # Call to issparse(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'x' (line 15)
    x_382452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 23), 'x', False)
    # Processing the call keyword arguments (line 15)
    kwargs_382453 = {}
    # Getting the type of 'sparse' (line 15)
    sparse_382450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 7), 'sparse', False)
    # Obtaining the member 'issparse' of a type (line 15)
    issparse_382451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 7), sparse_382450, 'issparse')
    # Calling issparse(args, kwargs) (line 15)
    issparse_call_result_382454 = invoke(stypy.reporting.localization.Localization(__file__, 15, 7), issparse_382451, *[x_382452], **kwargs_382453)
    
    # Testing the type of an if condition (line 15)
    if_condition_382455 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 15, 4), issparse_call_result_382454)
    # Assigning a type to the variable 'if_condition_382455' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'if_condition_382455', if_condition_382455)
    # SSA begins for if statement (line 15)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 16):
    
    # Assigning a Call to a Name (line 16):
    
    # Call to todense(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_382458 = {}
    # Getting the type of 'x' (line 16)
    x_382456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'x', False)
    # Obtaining the member 'todense' of a type (line 16)
    todense_382457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 12), x_382456, 'todense')
    # Calling todense(args, kwargs) (line 16)
    todense_call_result_382459 = invoke(stypy.reporting.localization.Localization(__file__, 16, 12), todense_382457, *[], **kwargs_382458)
    
    # Assigning a type to the variable 'x' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'x', todense_call_result_382459)
    # SSA join for if statement (line 15)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 17):
    
    # Assigning a Call to a Name (line 17):
    
    # Call to asarray(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 'x' (line 17)
    x_382462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 19), 'x', False)
    # Processing the call keyword arguments (line 17)
    kwargs_382463 = {}
    # Getting the type of 'np' (line 17)
    np_382460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 17)
    asarray_382461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), np_382460, 'asarray')
    # Calling asarray(args, kwargs) (line 17)
    asarray_call_result_382464 = invoke(stypy.reporting.localization.Localization(__file__, 17, 8), asarray_382461, *[x_382462], **kwargs_382463)
    
    # Assigning a type to the variable 'x' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'x', asarray_call_result_382464)
    
    # Assigning a BinOp to a Name (line 18):
    
    # Assigning a BinOp to a Name (line 18):
    float_382465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 8), 'float')
    # Getting the type of 'x' (line 18)
    x_382466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 15), 'x')
    # Applying the binary operator '*' (line 18)
    result_mul_382467 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 8), '*', float_382465, x_382466)
    
    # Assigning a type to the variable 'y' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'y', result_mul_382467)
    
    
    # Call to range(...): (line 19)
    # Processing the call arguments (line 19)
    
    # Obtaining the type of the subscript
    int_382469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 27), 'int')
    # Getting the type of 'y' (line 19)
    y_382470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 19), 'y', False)
    # Obtaining the member 'shape' of a type (line 19)
    shape_382471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 19), y_382470, 'shape')
    # Obtaining the member '__getitem__' of a type (line 19)
    getitem___382472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 19), shape_382471, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 19)
    subscript_call_result_382473 = invoke(stypy.reporting.localization.Localization(__file__, 19, 19), getitem___382472, int_382469)
    
    # Processing the call keyword arguments (line 19)
    kwargs_382474 = {}
    # Getting the type of 'range' (line 19)
    range_382468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 13), 'range', False)
    # Calling range(args, kwargs) (line 19)
    range_call_result_382475 = invoke(stypy.reporting.localization.Localization(__file__, 19, 13), range_382468, *[subscript_call_result_382473], **kwargs_382474)
    
    # Testing the type of a for loop iterable (line 19)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 19, 4), range_call_result_382475)
    # Getting the type of the for loop variable (line 19)
    for_loop_var_382476 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 19, 4), range_call_result_382475)
    # Assigning a type to the variable 'j' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'j', for_loop_var_382476)
    # SSA begins for a for statement (line 19)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 20):
    
    # Assigning a BinOp to a Subscript (line 20):
    
    # Call to sum(...): (line 20)
    # Processing the call keyword arguments (line 20)
    kwargs_382486 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 20)
    j_382477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 19), 'j', False)
    # Getting the type of 'j' (line 20)
    j_382478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 21), 'j', False)
    int_382479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 23), 'int')
    # Applying the binary operator '+' (line 20)
    result_add_382480 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 21), '+', j_382478, int_382479)
    
    slice_382481 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 20, 17), result_add_382480, None, None)
    # Getting the type of 'x' (line 20)
    x_382482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 17), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 20)
    getitem___382483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 17), x_382482, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 20)
    subscript_call_result_382484 = invoke(stypy.reporting.localization.Localization(__file__, 20, 17), getitem___382483, (j_382477, slice_382481))
    
    # Obtaining the member 'sum' of a type (line 20)
    sum_382485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 17), subscript_call_result_382484, 'sum')
    # Calling sum(args, kwargs) (line 20)
    sum_call_result_382487 = invoke(stypy.reporting.localization.Localization(__file__, 20, 17), sum_382485, *[], **kwargs_382486)
    
    
    # Call to sum(...): (line 20)
    # Processing the call keyword arguments (line 20)
    kwargs_382495 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 20)
    j_382488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 37), 'j', False)
    # Getting the type of 'j' (line 20)
    j_382489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 40), 'j', False)
    slice_382490 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 20, 35), None, j_382489, None)
    # Getting the type of 'x' (line 20)
    x_382491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 35), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 20)
    getitem___382492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 35), x_382491, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 20)
    subscript_call_result_382493 = invoke(stypy.reporting.localization.Localization(__file__, 20, 35), getitem___382492, (j_382488, slice_382490))
    
    # Obtaining the member 'sum' of a type (line 20)
    sum_382494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 35), subscript_call_result_382493, 'sum')
    # Calling sum(args, kwargs) (line 20)
    sum_call_result_382496 = invoke(stypy.reporting.localization.Localization(__file__, 20, 35), sum_382494, *[], **kwargs_382495)
    
    # Applying the binary operator '+' (line 20)
    result_add_382497 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 17), '+', sum_call_result_382487, sum_call_result_382496)
    
    # Getting the type of 'y' (line 20)
    y_382498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'y')
    
    # Obtaining an instance of the builtin type 'tuple' (line 20)
    tuple_382499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 20)
    # Adding element type (line 20)
    # Getting the type of 'j' (line 20)
    j_382500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 10), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), tuple_382499, j_382500)
    # Adding element type (line 20)
    # Getting the type of 'j' (line 20)
    j_382501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), tuple_382499, j_382501)
    
    # Storing an element on a container (line 20)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 8), y_382498, (tuple_382499, result_add_382497))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'normed' (line 21)
    normed_382502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 7), 'normed')
    # Testing the type of an if condition (line 21)
    if_condition_382503 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 21, 4), normed_382502)
    # Assigning a type to the variable 'if_condition_382503' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'if_condition_382503', if_condition_382503)
    # SSA begins for if statement (line 21)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 22):
    
    # Assigning a Call to a Name (line 22):
    
    # Call to copy(...): (line 22)
    # Processing the call keyword arguments (line 22)
    kwargs_382510 = {}
    
    # Call to diag(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'y' (line 22)
    y_382506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 20), 'y', False)
    # Processing the call keyword arguments (line 22)
    kwargs_382507 = {}
    # Getting the type of 'np' (line 22)
    np_382504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'np', False)
    # Obtaining the member 'diag' of a type (line 22)
    diag_382505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 12), np_382504, 'diag')
    # Calling diag(args, kwargs) (line 22)
    diag_call_result_382508 = invoke(stypy.reporting.localization.Localization(__file__, 22, 12), diag_382505, *[y_382506], **kwargs_382507)
    
    # Obtaining the member 'copy' of a type (line 22)
    copy_382509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 12), diag_call_result_382508, 'copy')
    # Calling copy(args, kwargs) (line 22)
    copy_call_result_382511 = invoke(stypy.reporting.localization.Localization(__file__, 22, 12), copy_382509, *[], **kwargs_382510)
    
    # Assigning a type to the variable 'd' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'd', copy_call_result_382511)
    
    # Assigning a Num to a Subscript (line 23):
    
    # Assigning a Num to a Subscript (line 23):
    float_382512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 20), 'float')
    # Getting the type of 'd' (line 23)
    d_382513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'd')
    
    # Getting the type of 'd' (line 23)
    d_382514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'd')
    int_382515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 15), 'int')
    # Applying the binary operator '==' (line 23)
    result_eq_382516 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 10), '==', d_382514, int_382515)
    
    # Storing an element on a container (line 23)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 8), d_382513, (result_eq_382516, float_382512))
    
    # Getting the type of 'y' (line 24)
    y_382517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'y')
    
    # Obtaining the type of the subscript
    slice_382518 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 24, 13), None, None, None)
    # Getting the type of 'None' (line 24)
    None_382519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 17), 'None')
    # Getting the type of 'd' (line 24)
    d_382520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 13), 'd')
    # Obtaining the member '__getitem__' of a type (line 24)
    getitem___382521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 13), d_382520, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 24)
    subscript_call_result_382522 = invoke(stypy.reporting.localization.Localization(__file__, 24, 13), getitem___382521, (slice_382518, None_382519))
    
    float_382523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 24), 'float')
    # Applying the binary operator '**' (line 24)
    result_pow_382524 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 13), '**', subscript_call_result_382522, float_382523)
    
    # Applying the binary operator 'div=' (line 24)
    result_div_382525 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 8), 'div=', y_382517, result_pow_382524)
    # Assigning a type to the variable 'y' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'y', result_div_382525)
    
    
    # Getting the type of 'y' (line 25)
    y_382526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'y')
    
    # Obtaining the type of the subscript
    # Getting the type of 'None' (line 25)
    None_382527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'None')
    slice_382528 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 25, 13), None, None, None)
    # Getting the type of 'd' (line 25)
    d_382529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 13), 'd')
    # Obtaining the member '__getitem__' of a type (line 25)
    getitem___382530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 13), d_382529, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 25)
    subscript_call_result_382531 = invoke(stypy.reporting.localization.Localization(__file__, 25, 13), getitem___382530, (None_382527, slice_382528))
    
    float_382532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 24), 'float')
    # Applying the binary operator '**' (line 25)
    result_pow_382533 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 13), '**', subscript_call_result_382531, float_382532)
    
    # Applying the binary operator 'div=' (line 25)
    result_div_382534 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 8), 'div=', y_382526, result_pow_382533)
    # Assigning a type to the variable 'y' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'y', result_div_382534)
    
    # SSA join for if statement (line 21)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'y' (line 26)
    y_382535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'y')
    # Assigning a type to the variable 'stypy_return_type' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type', y_382535)
    
    # ################# End of '_explicit_laplacian(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_explicit_laplacian' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_382536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_382536)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_explicit_laplacian'
    return stypy_return_type_382536

# Assigning a type to the variable '_explicit_laplacian' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), '_explicit_laplacian', _explicit_laplacian)

@norecursion
def _check_symmetric_graph_laplacian(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_check_symmetric_graph_laplacian'
    module_type_store = module_type_store.open_function_context('_check_symmetric_graph_laplacian', 29, 0, False)
    
    # Passed parameters checking function
    _check_symmetric_graph_laplacian.stypy_localization = localization
    _check_symmetric_graph_laplacian.stypy_type_of_self = None
    _check_symmetric_graph_laplacian.stypy_type_store = module_type_store
    _check_symmetric_graph_laplacian.stypy_function_name = '_check_symmetric_graph_laplacian'
    _check_symmetric_graph_laplacian.stypy_param_names_list = ['mat', 'normed']
    _check_symmetric_graph_laplacian.stypy_varargs_param_name = None
    _check_symmetric_graph_laplacian.stypy_kwargs_param_name = None
    _check_symmetric_graph_laplacian.stypy_call_defaults = defaults
    _check_symmetric_graph_laplacian.stypy_call_varargs = varargs
    _check_symmetric_graph_laplacian.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_check_symmetric_graph_laplacian', ['mat', 'normed'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_check_symmetric_graph_laplacian', localization, ['mat', 'normed'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_check_symmetric_graph_laplacian(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 30)
    str_382537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 24), 'str', 'shape')
    # Getting the type of 'mat' (line 30)
    mat_382538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 19), 'mat')
    
    (may_be_382539, more_types_in_union_382540) = may_not_provide_member(str_382537, mat_382538)

    if may_be_382539:

        if more_types_in_union_382540:
            # Runtime conditional SSA (line 30)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'mat' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'mat', remove_member_provider_from_union(mat_382538, 'shape'))
        
        # Assigning a Call to a Name (line 31):
        
        # Assigning a Call to a Name (line 31):
        
        # Call to eval(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'mat' (line 31)
        mat_382542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 19), 'mat', False)
        
        # Call to dict(...): (line 31)
        # Processing the call keyword arguments (line 31)
        # Getting the type of 'np' (line 31)
        np_382544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 32), 'np', False)
        keyword_382545 = np_382544
        # Getting the type of 'sparse' (line 31)
        sparse_382546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 43), 'sparse', False)
        keyword_382547 = sparse_382546
        kwargs_382548 = {'np': keyword_382545, 'sparse': keyword_382547}
        # Getting the type of 'dict' (line 31)
        dict_382543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 24), 'dict', False)
        # Calling dict(args, kwargs) (line 31)
        dict_call_result_382549 = invoke(stypy.reporting.localization.Localization(__file__, 31, 24), dict_382543, *[], **kwargs_382548)
        
        # Processing the call keyword arguments (line 31)
        kwargs_382550 = {}
        # Getting the type of 'eval' (line 31)
        eval_382541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'eval', False)
        # Calling eval(args, kwargs) (line 31)
        eval_call_result_382551 = invoke(stypy.reporting.localization.Localization(__file__, 31, 14), eval_382541, *[mat_382542, dict_call_result_382549], **kwargs_382550)
        
        # Assigning a type to the variable 'mat' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'mat', eval_call_result_382551)

        if more_types_in_union_382540:
            # SSA join for if statement (line 30)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to issparse(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'mat' (line 33)
    mat_382554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 23), 'mat', False)
    # Processing the call keyword arguments (line 33)
    kwargs_382555 = {}
    # Getting the type of 'sparse' (line 33)
    sparse_382552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 7), 'sparse', False)
    # Obtaining the member 'issparse' of a type (line 33)
    issparse_382553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 7), sparse_382552, 'issparse')
    # Calling issparse(args, kwargs) (line 33)
    issparse_call_result_382556 = invoke(stypy.reporting.localization.Localization(__file__, 33, 7), issparse_382553, *[mat_382554], **kwargs_382555)
    
    # Testing the type of an if condition (line 33)
    if_condition_382557 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 4), issparse_call_result_382556)
    # Assigning a type to the variable 'if_condition_382557' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'if_condition_382557', if_condition_382557)
    # SSA begins for if statement (line 33)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 34):
    
    # Assigning a Name to a Name (line 34):
    # Getting the type of 'mat' (line 34)
    mat_382558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 17), 'mat')
    # Assigning a type to the variable 'sp_mat' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'sp_mat', mat_382558)
    
    # Assigning a Call to a Name (line 35):
    
    # Assigning a Call to a Name (line 35):
    
    # Call to todense(...): (line 35)
    # Processing the call keyword arguments (line 35)
    kwargs_382561 = {}
    # Getting the type of 'sp_mat' (line 35)
    sp_mat_382559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 14), 'sp_mat', False)
    # Obtaining the member 'todense' of a type (line 35)
    todense_382560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 14), sp_mat_382559, 'todense')
    # Calling todense(args, kwargs) (line 35)
    todense_call_result_382562 = invoke(stypy.reporting.localization.Localization(__file__, 35, 14), todense_382560, *[], **kwargs_382561)
    
    # Assigning a type to the variable 'mat' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'mat', todense_call_result_382562)
    # SSA branch for the else part of an if statement (line 33)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 37):
    
    # Assigning a Call to a Name (line 37):
    
    # Call to csr_matrix(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'mat' (line 37)
    mat_382565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 35), 'mat', False)
    # Processing the call keyword arguments (line 37)
    kwargs_382566 = {}
    # Getting the type of 'sparse' (line 37)
    sparse_382563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 17), 'sparse', False)
    # Obtaining the member 'csr_matrix' of a type (line 37)
    csr_matrix_382564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 17), sparse_382563, 'csr_matrix')
    # Calling csr_matrix(args, kwargs) (line 37)
    csr_matrix_call_result_382567 = invoke(stypy.reporting.localization.Localization(__file__, 37, 17), csr_matrix_382564, *[mat_382565], **kwargs_382566)
    
    # Assigning a type to the variable 'sp_mat' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'sp_mat', csr_matrix_call_result_382567)
    # SSA join for if statement (line 33)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 39):
    
    # Assigning a Call to a Name (line 39):
    
    # Call to laplacian(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'mat' (line 39)
    mat_382570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 34), 'mat', False)
    # Processing the call keyword arguments (line 39)
    # Getting the type of 'normed' (line 39)
    normed_382571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 46), 'normed', False)
    keyword_382572 = normed_382571
    kwargs_382573 = {'normed': keyword_382572}
    # Getting the type of 'csgraph' (line 39)
    csgraph_382568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'csgraph', False)
    # Obtaining the member 'laplacian' of a type (line 39)
    laplacian_382569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 16), csgraph_382568, 'laplacian')
    # Calling laplacian(args, kwargs) (line 39)
    laplacian_call_result_382574 = invoke(stypy.reporting.localization.Localization(__file__, 39, 16), laplacian_382569, *[mat_382570], **kwargs_382573)
    
    # Assigning a type to the variable 'laplacian' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'laplacian', laplacian_call_result_382574)
    
    # Assigning a Subscript to a Name (line 40):
    
    # Assigning a Subscript to a Name (line 40):
    
    # Obtaining the type of the subscript
    int_382575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 24), 'int')
    # Getting the type of 'mat' (line 40)
    mat_382576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 14), 'mat')
    # Obtaining the member 'shape' of a type (line 40)
    shape_382577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 14), mat_382576, 'shape')
    # Obtaining the member '__getitem__' of a type (line 40)
    getitem___382578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 14), shape_382577, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
    subscript_call_result_382579 = invoke(stypy.reporting.localization.Localization(__file__, 40, 14), getitem___382578, int_382575)
    
    # Assigning a type to the variable 'n_nodes' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'n_nodes', subscript_call_result_382579)
    
    
    # Getting the type of 'normed' (line 41)
    normed_382580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 11), 'normed')
    # Applying the 'not' unary operator (line 41)
    result_not__382581 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 7), 'not', normed_382580)
    
    # Testing the type of an if condition (line 41)
    if_condition_382582 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 4), result_not__382581)
    # Assigning a type to the variable 'if_condition_382582' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'if_condition_382582', if_condition_382582)
    # SSA begins for if statement (line 41)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to assert_array_almost_equal(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Call to sum(...): (line 42)
    # Processing the call keyword arguments (line 42)
    int_382586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 53), 'int')
    keyword_382587 = int_382586
    kwargs_382588 = {'axis': keyword_382587}
    # Getting the type of 'laplacian' (line 42)
    laplacian_382584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 34), 'laplacian', False)
    # Obtaining the member 'sum' of a type (line 42)
    sum_382585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 34), laplacian_382584, 'sum')
    # Calling sum(args, kwargs) (line 42)
    sum_call_result_382589 = invoke(stypy.reporting.localization.Localization(__file__, 42, 34), sum_382585, *[], **kwargs_382588)
    
    
    # Call to zeros(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'n_nodes' (line 42)
    n_nodes_382592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 66), 'n_nodes', False)
    # Processing the call keyword arguments (line 42)
    kwargs_382593 = {}
    # Getting the type of 'np' (line 42)
    np_382590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 57), 'np', False)
    # Obtaining the member 'zeros' of a type (line 42)
    zeros_382591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 57), np_382590, 'zeros')
    # Calling zeros(args, kwargs) (line 42)
    zeros_call_result_382594 = invoke(stypy.reporting.localization.Localization(__file__, 42, 57), zeros_382591, *[n_nodes_382592], **kwargs_382593)
    
    # Processing the call keyword arguments (line 42)
    kwargs_382595 = {}
    # Getting the type of 'assert_array_almost_equal' (line 42)
    assert_array_almost_equal_382583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 42)
    assert_array_almost_equal_call_result_382596 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), assert_array_almost_equal_382583, *[sum_call_result_382589, zeros_call_result_382594], **kwargs_382595)
    
    # SSA join for if statement (line 41)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to assert_array_almost_equal(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'laplacian' (line 43)
    laplacian_382598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 30), 'laplacian', False)
    # Obtaining the member 'T' of a type (line 43)
    T_382599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 30), laplacian_382598, 'T')
    # Getting the type of 'laplacian' (line 43)
    laplacian_382600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 43), 'laplacian', False)
    # Processing the call keyword arguments (line 43)
    kwargs_382601 = {}
    # Getting the type of 'assert_array_almost_equal' (line 43)
    assert_array_almost_equal_382597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 43)
    assert_array_almost_equal_call_result_382602 = invoke(stypy.reporting.localization.Localization(__file__, 43, 4), assert_array_almost_equal_382597, *[T_382599, laplacian_382600], **kwargs_382601)
    
    
    # Call to assert_array_almost_equal(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'laplacian' (line 44)
    laplacian_382604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 30), 'laplacian', False)
    
    # Call to todense(...): (line 45)
    # Processing the call keyword arguments (line 45)
    kwargs_382613 = {}
    
    # Call to laplacian(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'sp_mat' (line 45)
    sp_mat_382607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 30), 'sp_mat', False)
    # Processing the call keyword arguments (line 45)
    # Getting the type of 'normed' (line 45)
    normed_382608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 45), 'normed', False)
    keyword_382609 = normed_382608
    kwargs_382610 = {'normed': keyword_382609}
    # Getting the type of 'csgraph' (line 45)
    csgraph_382605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'csgraph', False)
    # Obtaining the member 'laplacian' of a type (line 45)
    laplacian_382606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 12), csgraph_382605, 'laplacian')
    # Calling laplacian(args, kwargs) (line 45)
    laplacian_call_result_382611 = invoke(stypy.reporting.localization.Localization(__file__, 45, 12), laplacian_382606, *[sp_mat_382607], **kwargs_382610)
    
    # Obtaining the member 'todense' of a type (line 45)
    todense_382612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 12), laplacian_call_result_382611, 'todense')
    # Calling todense(args, kwargs) (line 45)
    todense_call_result_382614 = invoke(stypy.reporting.localization.Localization(__file__, 45, 12), todense_382612, *[], **kwargs_382613)
    
    # Processing the call keyword arguments (line 44)
    kwargs_382615 = {}
    # Getting the type of 'assert_array_almost_equal' (line 44)
    assert_array_almost_equal_382603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 44)
    assert_array_almost_equal_call_result_382616 = invoke(stypy.reporting.localization.Localization(__file__, 44, 4), assert_array_almost_equal_382603, *[laplacian_382604, todense_call_result_382614], **kwargs_382615)
    
    
    # Call to assert_array_almost_equal(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'laplacian' (line 47)
    laplacian_382618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 30), 'laplacian', False)
    
    # Call to _explicit_laplacian(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'mat' (line 48)
    mat_382620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 32), 'mat', False)
    # Processing the call keyword arguments (line 48)
    # Getting the type of 'normed' (line 48)
    normed_382621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 44), 'normed', False)
    keyword_382622 = normed_382621
    kwargs_382623 = {'normed': keyword_382622}
    # Getting the type of '_explicit_laplacian' (line 48)
    _explicit_laplacian_382619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), '_explicit_laplacian', False)
    # Calling _explicit_laplacian(args, kwargs) (line 48)
    _explicit_laplacian_call_result_382624 = invoke(stypy.reporting.localization.Localization(__file__, 48, 12), _explicit_laplacian_382619, *[mat_382620], **kwargs_382623)
    
    # Processing the call keyword arguments (line 47)
    kwargs_382625 = {}
    # Getting the type of 'assert_array_almost_equal' (line 47)
    assert_array_almost_equal_382617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 47)
    assert_array_almost_equal_call_result_382626 = invoke(stypy.reporting.localization.Localization(__file__, 47, 4), assert_array_almost_equal_382617, *[laplacian_382618, _explicit_laplacian_call_result_382624], **kwargs_382625)
    
    
    # ################# End of '_check_symmetric_graph_laplacian(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_check_symmetric_graph_laplacian' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_382627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_382627)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_check_symmetric_graph_laplacian'
    return stypy_return_type_382627

# Assigning a type to the variable '_check_symmetric_graph_laplacian' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), '_check_symmetric_graph_laplacian', _check_symmetric_graph_laplacian)

@norecursion
def test_laplacian_value_error(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_laplacian_value_error'
    module_type_store = module_type_store.open_function_context('test_laplacian_value_error', 51, 0, False)
    
    # Passed parameters checking function
    test_laplacian_value_error.stypy_localization = localization
    test_laplacian_value_error.stypy_type_of_self = None
    test_laplacian_value_error.stypy_type_store = module_type_store
    test_laplacian_value_error.stypy_function_name = 'test_laplacian_value_error'
    test_laplacian_value_error.stypy_param_names_list = []
    test_laplacian_value_error.stypy_varargs_param_name = None
    test_laplacian_value_error.stypy_kwargs_param_name = None
    test_laplacian_value_error.stypy_call_defaults = defaults
    test_laplacian_value_error.stypy_call_varargs = varargs
    test_laplacian_value_error.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_laplacian_value_error', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_laplacian_value_error', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_laplacian_value_error(...)' code ##################

    
    
    # Obtaining an instance of the builtin type 'tuple' (line 52)
    tuple_382628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 52)
    # Adding element type (line 52)
    # Getting the type of 'int' (line 52)
    int_382629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 13), tuple_382628, int_382629)
    # Adding element type (line 52)
    # Getting the type of 'float' (line 52)
    float_382630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 13), tuple_382628, float_382630)
    # Adding element type (line 52)
    # Getting the type of 'complex' (line 52)
    complex_382631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 25), 'complex')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 13), tuple_382628, complex_382631)
    
    # Testing the type of a for loop iterable (line 52)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 52, 4), tuple_382628)
    # Getting the type of the for loop variable (line 52)
    for_loop_var_382632 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 52, 4), tuple_382628)
    # Assigning a type to the variable 't' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 't', for_loop_var_382632)
    # SSA begins for a for statement (line 52)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 53)
    tuple_382633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 53)
    # Adding element type (line 53)
    
    # Obtaining an instance of the builtin type 'list' (line 53)
    list_382634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 53)
    # Adding element type (line 53)
    int_382635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 18), list_382634, int_382635)
    # Adding element type (line 53)
    int_382636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 18), list_382634, int_382636)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 18), tuple_382633, list_382634)
    # Adding element type (line 53)
    
    # Obtaining an instance of the builtin type 'list' (line 54)
    list_382637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 54)
    # Adding element type (line 54)
    
    # Obtaining an instance of the builtin type 'list' (line 54)
    list_382638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 54)
    # Adding element type (line 54)
    
    # Obtaining an instance of the builtin type 'list' (line 54)
    list_382639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 54)
    # Adding element type (line 54)
    int_382640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 20), list_382639, int_382640)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 19), list_382638, list_382639)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 18), list_382637, list_382638)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 18), tuple_382633, list_382637)
    # Adding element type (line 53)
    
    # Obtaining an instance of the builtin type 'list' (line 55)
    list_382641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 55)
    # Adding element type (line 55)
    
    # Obtaining an instance of the builtin type 'list' (line 55)
    list_382642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 55)
    # Adding element type (line 55)
    int_382643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 19), list_382642, int_382643)
    # Adding element type (line 55)
    int_382644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 19), list_382642, int_382644)
    # Adding element type (line 55)
    int_382645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 19), list_382642, int_382645)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 18), list_382641, list_382642)
    # Adding element type (line 55)
    
    # Obtaining an instance of the builtin type 'list' (line 55)
    list_382646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 55)
    # Adding element type (line 55)
    int_382647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 30), list_382646, int_382647)
    # Adding element type (line 55)
    int_382648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 30), list_382646, int_382648)
    # Adding element type (line 55)
    int_382649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 30), list_382646, int_382649)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 18), list_382641, list_382646)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 18), tuple_382633, list_382641)
    # Adding element type (line 53)
    
    # Obtaining an instance of the builtin type 'list' (line 56)
    list_382650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 56)
    # Adding element type (line 56)
    
    # Obtaining an instance of the builtin type 'list' (line 56)
    list_382651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 56)
    # Adding element type (line 56)
    int_382652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 19), list_382651, int_382652)
    # Adding element type (line 56)
    int_382653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 19), list_382651, int_382653)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 18), list_382650, list_382651)
    # Adding element type (line 56)
    
    # Obtaining an instance of the builtin type 'list' (line 56)
    list_382654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 56)
    # Adding element type (line 56)
    int_382655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 27), list_382654, int_382655)
    # Adding element type (line 56)
    int_382656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 27), list_382654, int_382656)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 18), list_382650, list_382654)
    # Adding element type (line 56)
    
    # Obtaining an instance of the builtin type 'list' (line 56)
    list_382657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 56)
    # Adding element type (line 56)
    int_382658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 35), list_382657, int_382658)
    # Adding element type (line 56)
    int_382659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 35), list_382657, int_382659)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 18), list_382650, list_382657)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 18), tuple_382633, list_382650)
    
    # Testing the type of a for loop iterable (line 53)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 53, 8), tuple_382633)
    # Getting the type of the for loop variable (line 53)
    for_loop_var_382660 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 53, 8), tuple_382633)
    # Assigning a type to the variable 'm' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'm', for_loop_var_382660)
    # SSA begins for a for statement (line 53)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 57):
    
    # Assigning a Call to a Name (line 57):
    
    # Call to array(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'm' (line 57)
    m_382663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'm', False)
    # Processing the call keyword arguments (line 57)
    # Getting the type of 't' (line 57)
    t_382664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 34), 't', False)
    keyword_382665 = t_382664
    kwargs_382666 = {'dtype': keyword_382665}
    # Getting the type of 'np' (line 57)
    np_382661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'np', False)
    # Obtaining the member 'array' of a type (line 57)
    array_382662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 16), np_382661, 'array')
    # Calling array(args, kwargs) (line 57)
    array_call_result_382667 = invoke(stypy.reporting.localization.Localization(__file__, 57, 16), array_382662, *[m_382663], **kwargs_382666)
    
    # Assigning a type to the variable 'A' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'A', array_call_result_382667)
    
    # Call to assert_raises(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'ValueError' (line 58)
    ValueError_382669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 26), 'ValueError', False)
    # Getting the type of 'csgraph' (line 58)
    csgraph_382670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 38), 'csgraph', False)
    # Obtaining the member 'laplacian' of a type (line 58)
    laplacian_382671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 38), csgraph_382670, 'laplacian')
    # Getting the type of 'A' (line 58)
    A_382672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 57), 'A', False)
    # Processing the call keyword arguments (line 58)
    kwargs_382673 = {}
    # Getting the type of 'assert_raises' (line 58)
    assert_raises_382668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 58)
    assert_raises_call_result_382674 = invoke(stypy.reporting.localization.Localization(__file__, 58, 12), assert_raises_382668, *[ValueError_382669, laplacian_382671, A_382672], **kwargs_382673)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_laplacian_value_error(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_laplacian_value_error' in the type store
    # Getting the type of 'stypy_return_type' (line 51)
    stypy_return_type_382675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_382675)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_laplacian_value_error'
    return stypy_return_type_382675

# Assigning a type to the variable 'test_laplacian_value_error' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'test_laplacian_value_error', test_laplacian_value_error)

@norecursion
def test_symmetric_graph_laplacian(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_symmetric_graph_laplacian'
    module_type_store = module_type_store.open_function_context('test_symmetric_graph_laplacian', 61, 0, False)
    
    # Passed parameters checking function
    test_symmetric_graph_laplacian.stypy_localization = localization
    test_symmetric_graph_laplacian.stypy_type_of_self = None
    test_symmetric_graph_laplacian.stypy_type_store = module_type_store
    test_symmetric_graph_laplacian.stypy_function_name = 'test_symmetric_graph_laplacian'
    test_symmetric_graph_laplacian.stypy_param_names_list = []
    test_symmetric_graph_laplacian.stypy_varargs_param_name = None
    test_symmetric_graph_laplacian.stypy_kwargs_param_name = None
    test_symmetric_graph_laplacian.stypy_call_defaults = defaults
    test_symmetric_graph_laplacian.stypy_call_varargs = varargs
    test_symmetric_graph_laplacian.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_symmetric_graph_laplacian', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_symmetric_graph_laplacian', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_symmetric_graph_laplacian(...)' code ##################

    
    # Assigning a Tuple to a Name (line 62):
    
    # Assigning a Tuple to a Name (line 62):
    
    # Obtaining an instance of the builtin type 'tuple' (line 62)
    tuple_382676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 62)
    # Adding element type (line 62)
    str_382677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 22), 'str', 'np.arange(10) * np.arange(10)[:, np.newaxis]')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 22), tuple_382676, str_382677)
    # Adding element type (line 62)
    str_382678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 12), 'str', 'np.ones((7, 7))')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 22), tuple_382676, str_382678)
    # Adding element type (line 62)
    str_382679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 12), 'str', 'np.eye(19)')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 22), tuple_382676, str_382679)
    # Adding element type (line 62)
    str_382680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 12), 'str', 'sparse.diags([1, 1], [-1, 1], shape=(4,4))')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 22), tuple_382676, str_382680)
    # Adding element type (line 62)
    str_382681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 12), 'str', 'sparse.diags([1, 1], [-1, 1], shape=(4,4)).todense()')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 22), tuple_382676, str_382681)
    # Adding element type (line 62)
    str_382682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 12), 'str', 'np.asarray(sparse.diags([1, 1], [-1, 1], shape=(4,4)).todense())')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 22), tuple_382676, str_382682)
    # Adding element type (line 62)
    str_382683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 12), 'str', 'np.vander(np.arange(4)) + np.vander(np.arange(4)).T')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 22), tuple_382676, str_382683)
    
    # Assigning a type to the variable 'symmetric_mats' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'symmetric_mats', tuple_382676)
    
    # Getting the type of 'symmetric_mats' (line 69)
    symmetric_mats_382684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 19), 'symmetric_mats')
    # Testing the type of a for loop iterable (line 69)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 69, 4), symmetric_mats_382684)
    # Getting the type of the for loop variable (line 69)
    for_loop_var_382685 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 69, 4), symmetric_mats_382684)
    # Assigning a type to the variable 'mat_str' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'mat_str', for_loop_var_382685)
    # SSA begins for a for statement (line 69)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 70)
    tuple_382686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 70)
    # Adding element type (line 70)
    # Getting the type of 'True' (line 70)
    True_382687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 22), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 22), tuple_382686, True_382687)
    # Adding element type (line 70)
    # Getting the type of 'False' (line 70)
    False_382688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 28), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 22), tuple_382686, False_382688)
    
    # Testing the type of a for loop iterable (line 70)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 70, 8), tuple_382686)
    # Getting the type of the for loop variable (line 70)
    for_loop_var_382689 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 70, 8), tuple_382686)
    # Assigning a type to the variable 'normed' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'normed', for_loop_var_382689)
    # SSA begins for a for statement (line 70)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to _check_symmetric_graph_laplacian(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'mat_str' (line 71)
    mat_str_382691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 45), 'mat_str', False)
    # Getting the type of 'normed' (line 71)
    normed_382692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 54), 'normed', False)
    # Processing the call keyword arguments (line 71)
    kwargs_382693 = {}
    # Getting the type of '_check_symmetric_graph_laplacian' (line 71)
    _check_symmetric_graph_laplacian_382690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), '_check_symmetric_graph_laplacian', False)
    # Calling _check_symmetric_graph_laplacian(args, kwargs) (line 71)
    _check_symmetric_graph_laplacian_call_result_382694 = invoke(stypy.reporting.localization.Localization(__file__, 71, 12), _check_symmetric_graph_laplacian_382690, *[mat_str_382691, normed_382692], **kwargs_382693)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_symmetric_graph_laplacian(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_symmetric_graph_laplacian' in the type store
    # Getting the type of 'stypy_return_type' (line 61)
    stypy_return_type_382695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_382695)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_symmetric_graph_laplacian'
    return stypy_return_type_382695

# Assigning a type to the variable 'test_symmetric_graph_laplacian' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'test_symmetric_graph_laplacian', test_symmetric_graph_laplacian)

@norecursion
def _assert_allclose_sparse(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_assert_allclose_sparse'
    module_type_store = module_type_store.open_function_context('_assert_allclose_sparse', 74, 0, False)
    
    # Passed parameters checking function
    _assert_allclose_sparse.stypy_localization = localization
    _assert_allclose_sparse.stypy_type_of_self = None
    _assert_allclose_sparse.stypy_type_store = module_type_store
    _assert_allclose_sparse.stypy_function_name = '_assert_allclose_sparse'
    _assert_allclose_sparse.stypy_param_names_list = ['a', 'b']
    _assert_allclose_sparse.stypy_varargs_param_name = None
    _assert_allclose_sparse.stypy_kwargs_param_name = 'kwargs'
    _assert_allclose_sparse.stypy_call_defaults = defaults
    _assert_allclose_sparse.stypy_call_varargs = varargs
    _assert_allclose_sparse.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_assert_allclose_sparse', ['a', 'b'], None, 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_assert_allclose_sparse', localization, ['a', 'b'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_assert_allclose_sparse(...)' code ##################

    
    
    # Call to issparse(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'a' (line 76)
    a_382698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'a', False)
    # Processing the call keyword arguments (line 76)
    kwargs_382699 = {}
    # Getting the type of 'sparse' (line 76)
    sparse_382696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 7), 'sparse', False)
    # Obtaining the member 'issparse' of a type (line 76)
    issparse_382697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 7), sparse_382696, 'issparse')
    # Calling issparse(args, kwargs) (line 76)
    issparse_call_result_382700 = invoke(stypy.reporting.localization.Localization(__file__, 76, 7), issparse_382697, *[a_382698], **kwargs_382699)
    
    # Testing the type of an if condition (line 76)
    if_condition_382701 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 4), issparse_call_result_382700)
    # Assigning a type to the variable 'if_condition_382701' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'if_condition_382701', if_condition_382701)
    # SSA begins for if statement (line 76)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 77):
    
    # Assigning a Call to a Name (line 77):
    
    # Call to toarray(...): (line 77)
    # Processing the call keyword arguments (line 77)
    kwargs_382704 = {}
    # Getting the type of 'a' (line 77)
    a_382702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'a', False)
    # Obtaining the member 'toarray' of a type (line 77)
    toarray_382703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 12), a_382702, 'toarray')
    # Calling toarray(args, kwargs) (line 77)
    toarray_call_result_382705 = invoke(stypy.reporting.localization.Localization(__file__, 77, 12), toarray_382703, *[], **kwargs_382704)
    
    # Assigning a type to the variable 'a' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'a', toarray_call_result_382705)
    # SSA join for if statement (line 76)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to issparse(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'b' (line 78)
    b_382708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'b', False)
    # Processing the call keyword arguments (line 78)
    kwargs_382709 = {}
    # Getting the type of 'sparse' (line 78)
    sparse_382706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 7), 'sparse', False)
    # Obtaining the member 'issparse' of a type (line 78)
    issparse_382707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 7), sparse_382706, 'issparse')
    # Calling issparse(args, kwargs) (line 78)
    issparse_call_result_382710 = invoke(stypy.reporting.localization.Localization(__file__, 78, 7), issparse_382707, *[b_382708], **kwargs_382709)
    
    # Testing the type of an if condition (line 78)
    if_condition_382711 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 4), issparse_call_result_382710)
    # Assigning a type to the variable 'if_condition_382711' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'if_condition_382711', if_condition_382711)
    # SSA begins for if statement (line 78)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 79):
    
    # Assigning a Call to a Name (line 79):
    
    # Call to toarray(...): (line 79)
    # Processing the call keyword arguments (line 79)
    kwargs_382714 = {}
    # Getting the type of 'a' (line 79)
    a_382712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'a', False)
    # Obtaining the member 'toarray' of a type (line 79)
    toarray_382713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), a_382712, 'toarray')
    # Calling toarray(args, kwargs) (line 79)
    toarray_call_result_382715 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), toarray_382713, *[], **kwargs_382714)
    
    # Assigning a type to the variable 'b' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'b', toarray_call_result_382715)
    # SSA join for if statement (line 78)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to assert_allclose(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'a' (line 80)
    a_382717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), 'a', False)
    # Getting the type of 'b' (line 80)
    b_382718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 23), 'b', False)
    # Processing the call keyword arguments (line 80)
    # Getting the type of 'kwargs' (line 80)
    kwargs_382719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 28), 'kwargs', False)
    kwargs_382720 = {'kwargs_382719': kwargs_382719}
    # Getting the type of 'assert_allclose' (line 80)
    assert_allclose_382716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 80)
    assert_allclose_call_result_382721 = invoke(stypy.reporting.localization.Localization(__file__, 80, 4), assert_allclose_382716, *[a_382717, b_382718], **kwargs_382720)
    
    
    # ################# End of '_assert_allclose_sparse(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_assert_allclose_sparse' in the type store
    # Getting the type of 'stypy_return_type' (line 74)
    stypy_return_type_382722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_382722)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_assert_allclose_sparse'
    return stypy_return_type_382722

# Assigning a type to the variable '_assert_allclose_sparse' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), '_assert_allclose_sparse', _assert_allclose_sparse)

@norecursion
def _check_laplacian(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_check_laplacian'
    module_type_store = module_type_store.open_function_context('_check_laplacian', 83, 0, False)
    
    # Passed parameters checking function
    _check_laplacian.stypy_localization = localization
    _check_laplacian.stypy_type_of_self = None
    _check_laplacian.stypy_type_store = module_type_store
    _check_laplacian.stypy_function_name = '_check_laplacian'
    _check_laplacian.stypy_param_names_list = ['A', 'desired_L', 'desired_d', 'normed', 'use_out_degree']
    _check_laplacian.stypy_varargs_param_name = None
    _check_laplacian.stypy_kwargs_param_name = None
    _check_laplacian.stypy_call_defaults = defaults
    _check_laplacian.stypy_call_varargs = varargs
    _check_laplacian.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_check_laplacian', ['A', 'desired_L', 'desired_d', 'normed', 'use_out_degree'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_check_laplacian', localization, ['A', 'desired_L', 'desired_d', 'normed', 'use_out_degree'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_check_laplacian(...)' code ##################

    
    
    # Obtaining an instance of the builtin type 'tuple' (line 84)
    tuple_382723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 84)
    # Adding element type (line 84)
    # Getting the type of 'np' (line 84)
    np_382724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 20), 'np')
    # Obtaining the member 'array' of a type (line 84)
    array_382725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 20), np_382724, 'array')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 20), tuple_382723, array_382725)
    # Adding element type (line 84)
    # Getting the type of 'sparse' (line 84)
    sparse_382726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 30), 'sparse')
    # Obtaining the member 'csr_matrix' of a type (line 84)
    csr_matrix_382727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 30), sparse_382726, 'csr_matrix')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 20), tuple_382723, csr_matrix_382727)
    # Adding element type (line 84)
    # Getting the type of 'sparse' (line 84)
    sparse_382728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 49), 'sparse')
    # Obtaining the member 'coo_matrix' of a type (line 84)
    coo_matrix_382729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 49), sparse_382728, 'coo_matrix')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 20), tuple_382723, coo_matrix_382729)
    
    # Testing the type of a for loop iterable (line 84)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 84, 4), tuple_382723)
    # Getting the type of the for loop variable (line 84)
    for_loop_var_382730 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 84, 4), tuple_382723)
    # Assigning a type to the variable 'arr_type' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'arr_type', for_loop_var_382730)
    # SSA begins for a for statement (line 84)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 85)
    tuple_382731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 85)
    # Adding element type (line 85)
    # Getting the type of 'int' (line 85)
    int_382732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 17), tuple_382731, int_382732)
    # Adding element type (line 85)
    # Getting the type of 'float' (line 85)
    float_382733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 22), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 17), tuple_382731, float_382733)
    # Adding element type (line 85)
    # Getting the type of 'complex' (line 85)
    complex_382734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 29), 'complex')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 17), tuple_382731, complex_382734)
    
    # Testing the type of a for loop iterable (line 85)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 85, 8), tuple_382731)
    # Getting the type of the for loop variable (line 85)
    for_loop_var_382735 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 85, 8), tuple_382731)
    # Assigning a type to the variable 't' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 't', for_loop_var_382735)
    # SSA begins for a for statement (line 85)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 86):
    
    # Assigning a Call to a Name (line 86):
    
    # Call to arr_type(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'A' (line 86)
    A_382737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 27), 'A', False)
    # Processing the call keyword arguments (line 86)
    # Getting the type of 't' (line 86)
    t_382738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 36), 't', False)
    keyword_382739 = t_382738
    kwargs_382740 = {'dtype': keyword_382739}
    # Getting the type of 'arr_type' (line 86)
    arr_type_382736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 18), 'arr_type', False)
    # Calling arr_type(args, kwargs) (line 86)
    arr_type_call_result_382741 = invoke(stypy.reporting.localization.Localization(__file__, 86, 18), arr_type_382736, *[A_382737], **kwargs_382740)
    
    # Assigning a type to the variable 'adj' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'adj', arr_type_call_result_382741)
    
    # Assigning a Call to a Name (line 87):
    
    # Assigning a Call to a Name (line 87):
    
    # Call to laplacian(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'adj' (line 87)
    adj_382744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 34), 'adj', False)
    # Processing the call keyword arguments (line 87)
    # Getting the type of 'normed' (line 87)
    normed_382745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 46), 'normed', False)
    keyword_382746 = normed_382745
    # Getting the type of 'False' (line 87)
    False_382747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 66), 'False', False)
    keyword_382748 = False_382747
    # Getting the type of 'use_out_degree' (line 88)
    use_out_degree_382749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 49), 'use_out_degree', False)
    keyword_382750 = use_out_degree_382749
    kwargs_382751 = {'normed': keyword_382746, 'use_out_degree': keyword_382750, 'return_diag': keyword_382748}
    # Getting the type of 'csgraph' (line 87)
    csgraph_382742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'csgraph', False)
    # Obtaining the member 'laplacian' of a type (line 87)
    laplacian_382743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 16), csgraph_382742, 'laplacian')
    # Calling laplacian(args, kwargs) (line 87)
    laplacian_call_result_382752 = invoke(stypy.reporting.localization.Localization(__file__, 87, 16), laplacian_382743, *[adj_382744], **kwargs_382751)
    
    # Assigning a type to the variable 'L' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'L', laplacian_call_result_382752)
    
    # Call to _assert_allclose_sparse(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'L' (line 89)
    L_382754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 36), 'L', False)
    # Getting the type of 'desired_L' (line 89)
    desired_L_382755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 39), 'desired_L', False)
    # Processing the call keyword arguments (line 89)
    float_382756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 55), 'float')
    keyword_382757 = float_382756
    kwargs_382758 = {'atol': keyword_382757}
    # Getting the type of '_assert_allclose_sparse' (line 89)
    _assert_allclose_sparse_382753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), '_assert_allclose_sparse', False)
    # Calling _assert_allclose_sparse(args, kwargs) (line 89)
    _assert_allclose_sparse_call_result_382759 = invoke(stypy.reporting.localization.Localization(__file__, 89, 12), _assert_allclose_sparse_382753, *[L_382754, desired_L_382755], **kwargs_382758)
    
    
    # Assigning a Call to a Tuple (line 90):
    
    # Assigning a Subscript to a Name (line 90):
    
    # Obtaining the type of the subscript
    int_382760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 12), 'int')
    
    # Call to laplacian(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'adj' (line 90)
    adj_382763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 37), 'adj', False)
    # Processing the call keyword arguments (line 90)
    # Getting the type of 'normed' (line 90)
    normed_382764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 49), 'normed', False)
    keyword_382765 = normed_382764
    # Getting the type of 'True' (line 90)
    True_382766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 69), 'True', False)
    keyword_382767 = True_382766
    # Getting the type of 'use_out_degree' (line 91)
    use_out_degree_382768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 49), 'use_out_degree', False)
    keyword_382769 = use_out_degree_382768
    kwargs_382770 = {'normed': keyword_382765, 'use_out_degree': keyword_382769, 'return_diag': keyword_382767}
    # Getting the type of 'csgraph' (line 90)
    csgraph_382761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 19), 'csgraph', False)
    # Obtaining the member 'laplacian' of a type (line 90)
    laplacian_382762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 19), csgraph_382761, 'laplacian')
    # Calling laplacian(args, kwargs) (line 90)
    laplacian_call_result_382771 = invoke(stypy.reporting.localization.Localization(__file__, 90, 19), laplacian_382762, *[adj_382763], **kwargs_382770)
    
    # Obtaining the member '__getitem__' of a type (line 90)
    getitem___382772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), laplacian_call_result_382771, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 90)
    subscript_call_result_382773 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), getitem___382772, int_382760)
    
    # Assigning a type to the variable 'tuple_var_assignment_382437' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'tuple_var_assignment_382437', subscript_call_result_382773)
    
    # Assigning a Subscript to a Name (line 90):
    
    # Obtaining the type of the subscript
    int_382774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 12), 'int')
    
    # Call to laplacian(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'adj' (line 90)
    adj_382777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 37), 'adj', False)
    # Processing the call keyword arguments (line 90)
    # Getting the type of 'normed' (line 90)
    normed_382778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 49), 'normed', False)
    keyword_382779 = normed_382778
    # Getting the type of 'True' (line 90)
    True_382780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 69), 'True', False)
    keyword_382781 = True_382780
    # Getting the type of 'use_out_degree' (line 91)
    use_out_degree_382782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 49), 'use_out_degree', False)
    keyword_382783 = use_out_degree_382782
    kwargs_382784 = {'normed': keyword_382779, 'use_out_degree': keyword_382783, 'return_diag': keyword_382781}
    # Getting the type of 'csgraph' (line 90)
    csgraph_382775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 19), 'csgraph', False)
    # Obtaining the member 'laplacian' of a type (line 90)
    laplacian_382776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 19), csgraph_382775, 'laplacian')
    # Calling laplacian(args, kwargs) (line 90)
    laplacian_call_result_382785 = invoke(stypy.reporting.localization.Localization(__file__, 90, 19), laplacian_382776, *[adj_382777], **kwargs_382784)
    
    # Obtaining the member '__getitem__' of a type (line 90)
    getitem___382786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), laplacian_call_result_382785, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 90)
    subscript_call_result_382787 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), getitem___382786, int_382774)
    
    # Assigning a type to the variable 'tuple_var_assignment_382438' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'tuple_var_assignment_382438', subscript_call_result_382787)
    
    # Assigning a Name to a Name (line 90):
    # Getting the type of 'tuple_var_assignment_382437' (line 90)
    tuple_var_assignment_382437_382788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'tuple_var_assignment_382437')
    # Assigning a type to the variable 'L' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'L', tuple_var_assignment_382437_382788)
    
    # Assigning a Name to a Name (line 90):
    # Getting the type of 'tuple_var_assignment_382438' (line 90)
    tuple_var_assignment_382438_382789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'tuple_var_assignment_382438')
    # Assigning a type to the variable 'd' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 15), 'd', tuple_var_assignment_382438_382789)
    
    # Call to _assert_allclose_sparse(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'L' (line 92)
    L_382791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 36), 'L', False)
    # Getting the type of 'desired_L' (line 92)
    desired_L_382792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 39), 'desired_L', False)
    # Processing the call keyword arguments (line 92)
    float_382793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 55), 'float')
    keyword_382794 = float_382793
    kwargs_382795 = {'atol': keyword_382794}
    # Getting the type of '_assert_allclose_sparse' (line 92)
    _assert_allclose_sparse_382790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), '_assert_allclose_sparse', False)
    # Calling _assert_allclose_sparse(args, kwargs) (line 92)
    _assert_allclose_sparse_call_result_382796 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), _assert_allclose_sparse_382790, *[L_382791, desired_L_382792], **kwargs_382795)
    
    
    # Call to _assert_allclose_sparse(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'd' (line 93)
    d_382798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 36), 'd', False)
    # Getting the type of 'desired_d' (line 93)
    desired_d_382799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 39), 'desired_d', False)
    # Processing the call keyword arguments (line 93)
    float_382800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 55), 'float')
    keyword_382801 = float_382800
    kwargs_382802 = {'atol': keyword_382801}
    # Getting the type of '_assert_allclose_sparse' (line 93)
    _assert_allclose_sparse_382797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), '_assert_allclose_sparse', False)
    # Calling _assert_allclose_sparse(args, kwargs) (line 93)
    _assert_allclose_sparse_call_result_382803 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), _assert_allclose_sparse_382797, *[d_382798, desired_d_382799], **kwargs_382802)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_check_laplacian(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_check_laplacian' in the type store
    # Getting the type of 'stypy_return_type' (line 83)
    stypy_return_type_382804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_382804)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_check_laplacian'
    return stypy_return_type_382804

# Assigning a type to the variable '_check_laplacian' (line 83)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), '_check_laplacian', _check_laplacian)

@norecursion
def test_asymmetric_laplacian(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_asymmetric_laplacian'
    module_type_store = module_type_store.open_function_context('test_asymmetric_laplacian', 96, 0, False)
    
    # Passed parameters checking function
    test_asymmetric_laplacian.stypy_localization = localization
    test_asymmetric_laplacian.stypy_type_of_self = None
    test_asymmetric_laplacian.stypy_type_store = module_type_store
    test_asymmetric_laplacian.stypy_function_name = 'test_asymmetric_laplacian'
    test_asymmetric_laplacian.stypy_param_names_list = []
    test_asymmetric_laplacian.stypy_varargs_param_name = None
    test_asymmetric_laplacian.stypy_kwargs_param_name = None
    test_asymmetric_laplacian.stypy_call_defaults = defaults
    test_asymmetric_laplacian.stypy_call_varargs = varargs
    test_asymmetric_laplacian.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_asymmetric_laplacian', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_asymmetric_laplacian', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_asymmetric_laplacian(...)' code ##################

    
    # Assigning a List to a Name (line 98):
    
    # Assigning a List to a Name (line 98):
    
    # Obtaining an instance of the builtin type 'list' (line 98)
    list_382805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 98)
    # Adding element type (line 98)
    
    # Obtaining an instance of the builtin type 'list' (line 98)
    list_382806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 98)
    # Adding element type (line 98)
    int_382807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 9), list_382806, int_382807)
    # Adding element type (line 98)
    int_382808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 9), list_382806, int_382808)
    # Adding element type (line 98)
    int_382809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 9), list_382806, int_382809)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 8), list_382805, list_382806)
    # Adding element type (line 98)
    
    # Obtaining an instance of the builtin type 'list' (line 99)
    list_382810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 99)
    # Adding element type (line 99)
    int_382811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 9), list_382810, int_382811)
    # Adding element type (line 99)
    int_382812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 9), list_382810, int_382812)
    # Adding element type (line 99)
    int_382813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 9), list_382810, int_382813)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 8), list_382805, list_382810)
    # Adding element type (line 98)
    
    # Obtaining an instance of the builtin type 'list' (line 100)
    list_382814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 100)
    # Adding element type (line 100)
    int_382815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 9), list_382814, int_382815)
    # Adding element type (line 100)
    int_382816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 9), list_382814, int_382816)
    # Adding element type (line 100)
    int_382817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 9), list_382814, int_382817)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 8), list_382805, list_382814)
    
    # Assigning a type to the variable 'A' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'A', list_382805)
    
    # Assigning a List to a Name (line 103):
    
    # Assigning a List to a Name (line 103):
    
    # Obtaining an instance of the builtin type 'list' (line 103)
    list_382818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 103)
    # Adding element type (line 103)
    
    # Obtaining an instance of the builtin type 'list' (line 103)
    list_382819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 103)
    # Adding element type (line 103)
    int_382820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 9), list_382819, int_382820)
    # Adding element type (line 103)
    int_382821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 9), list_382819, int_382821)
    # Adding element type (line 103)
    int_382822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 9), list_382819, int_382822)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 8), list_382818, list_382819)
    # Adding element type (line 103)
    
    # Obtaining an instance of the builtin type 'list' (line 104)
    list_382823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 104)
    # Adding element type (line 104)
    int_382824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 9), list_382823, int_382824)
    # Adding element type (line 104)
    int_382825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 9), list_382823, int_382825)
    # Adding element type (line 104)
    int_382826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 9), list_382823, int_382826)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 8), list_382818, list_382823)
    # Adding element type (line 103)
    
    # Obtaining an instance of the builtin type 'list' (line 105)
    list_382827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 105)
    # Adding element type (line 105)
    int_382828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 9), list_382827, int_382828)
    # Adding element type (line 105)
    int_382829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 9), list_382827, int_382829)
    # Adding element type (line 105)
    int_382830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 9), list_382827, int_382830)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 8), list_382818, list_382827)
    
    # Assigning a type to the variable 'L' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'L', list_382818)
    
    # Assigning a List to a Name (line 106):
    
    # Assigning a List to a Name (line 106):
    
    # Obtaining an instance of the builtin type 'list' (line 106)
    list_382831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 106)
    # Adding element type (line 106)
    int_382832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 8), list_382831, int_382832)
    # Adding element type (line 106)
    int_382833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 8), list_382831, int_382833)
    # Adding element type (line 106)
    int_382834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 8), list_382831, int_382834)
    
    # Assigning a type to the variable 'd' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'd', list_382831)
    
    # Call to _check_laplacian(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'A' (line 107)
    A_382836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 21), 'A', False)
    # Getting the type of 'L' (line 107)
    L_382837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 24), 'L', False)
    # Getting the type of 'd' (line 107)
    d_382838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 27), 'd', False)
    # Processing the call keyword arguments (line 107)
    # Getting the type of 'False' (line 107)
    False_382839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 37), 'False', False)
    keyword_382840 = False_382839
    # Getting the type of 'True' (line 107)
    True_382841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 59), 'True', False)
    keyword_382842 = True_382841
    kwargs_382843 = {'normed': keyword_382840, 'use_out_degree': keyword_382842}
    # Getting the type of '_check_laplacian' (line 107)
    _check_laplacian_382835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), '_check_laplacian', False)
    # Calling _check_laplacian(args, kwargs) (line 107)
    _check_laplacian_call_result_382844 = invoke(stypy.reporting.localization.Localization(__file__, 107, 4), _check_laplacian_382835, *[A_382836, L_382837, d_382838], **kwargs_382843)
    
    
    # Assigning a List to a Name (line 110):
    
    # Assigning a List to a Name (line 110):
    
    # Obtaining an instance of the builtin type 'list' (line 110)
    list_382845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 110)
    # Adding element type (line 110)
    
    # Obtaining an instance of the builtin type 'list' (line 110)
    list_382846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 110)
    # Adding element type (line 110)
    int_382847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 9), list_382846, int_382847)
    # Adding element type (line 110)
    float_382848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 13), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 9), list_382846, float_382848)
    # Adding element type (line 110)
    int_382849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 9), list_382846, int_382849)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 8), list_382845, list_382846)
    # Adding element type (line 110)
    
    # Obtaining an instance of the builtin type 'list' (line 111)
    list_382850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 111)
    # Adding element type (line 111)
    int_382851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 9), list_382850, int_382851)
    # Adding element type (line 111)
    int_382852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 9), list_382850, int_382852)
    # Adding element type (line 111)
    int_382853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 9), list_382850, int_382853)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 8), list_382845, list_382850)
    # Adding element type (line 110)
    
    # Obtaining an instance of the builtin type 'list' (line 112)
    list_382854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 112)
    # Adding element type (line 112)
    int_382855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 9), list_382854, int_382855)
    # Adding element type (line 112)
    int_382856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 9), list_382854, int_382856)
    # Adding element type (line 112)
    int_382857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 9), list_382854, int_382857)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 8), list_382845, list_382854)
    
    # Assigning a type to the variable 'L' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'L', list_382845)
    
    # Assigning a List to a Name (line 113):
    
    # Assigning a List to a Name (line 113):
    
    # Obtaining an instance of the builtin type 'list' (line 113)
    list_382858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 113)
    # Adding element type (line 113)
    int_382859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 8), list_382858, int_382859)
    # Adding element type (line 113)
    int_382860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 8), list_382858, int_382860)
    # Adding element type (line 113)
    int_382861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 8), list_382858, int_382861)
    
    # Assigning a type to the variable 'd' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'd', list_382858)
    
    # Call to _check_laplacian(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'A' (line 114)
    A_382863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 21), 'A', False)
    # Getting the type of 'L' (line 114)
    L_382864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 24), 'L', False)
    # Getting the type of 'd' (line 114)
    d_382865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 27), 'd', False)
    # Processing the call keyword arguments (line 114)
    # Getting the type of 'True' (line 114)
    True_382866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 37), 'True', False)
    keyword_382867 = True_382866
    # Getting the type of 'True' (line 114)
    True_382868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 58), 'True', False)
    keyword_382869 = True_382868
    kwargs_382870 = {'normed': keyword_382867, 'use_out_degree': keyword_382869}
    # Getting the type of '_check_laplacian' (line 114)
    _check_laplacian_382862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), '_check_laplacian', False)
    # Calling _check_laplacian(args, kwargs) (line 114)
    _check_laplacian_call_result_382871 = invoke(stypy.reporting.localization.Localization(__file__, 114, 4), _check_laplacian_382862, *[A_382863, L_382864, d_382865], **kwargs_382870)
    
    
    # Assigning a List to a Name (line 117):
    
    # Assigning a List to a Name (line 117):
    
    # Obtaining an instance of the builtin type 'list' (line 117)
    list_382872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 117)
    # Adding element type (line 117)
    
    # Obtaining an instance of the builtin type 'list' (line 117)
    list_382873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 117)
    # Adding element type (line 117)
    int_382874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 9), list_382873, int_382874)
    # Adding element type (line 117)
    int_382875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 9), list_382873, int_382875)
    # Adding element type (line 117)
    int_382876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 9), list_382873, int_382876)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 8), list_382872, list_382873)
    # Adding element type (line 117)
    
    # Obtaining an instance of the builtin type 'list' (line 118)
    list_382877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 118)
    # Adding element type (line 118)
    int_382878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 9), list_382877, int_382878)
    # Adding element type (line 118)
    int_382879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 9), list_382877, int_382879)
    # Adding element type (line 118)
    int_382880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 9), list_382877, int_382880)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 8), list_382872, list_382877)
    # Adding element type (line 117)
    
    # Obtaining an instance of the builtin type 'list' (line 119)
    list_382881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 119)
    # Adding element type (line 119)
    int_382882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 9), list_382881, int_382882)
    # Adding element type (line 119)
    int_382883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 9), list_382881, int_382883)
    # Adding element type (line 119)
    int_382884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 9), list_382881, int_382884)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 8), list_382872, list_382881)
    
    # Assigning a type to the variable 'L' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'L', list_382872)
    
    # Assigning a List to a Name (line 120):
    
    # Assigning a List to a Name (line 120):
    
    # Obtaining an instance of the builtin type 'list' (line 120)
    list_382885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 120)
    # Adding element type (line 120)
    int_382886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 8), list_382885, int_382886)
    # Adding element type (line 120)
    int_382887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 8), list_382885, int_382887)
    # Adding element type (line 120)
    int_382888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 8), list_382885, int_382888)
    
    # Assigning a type to the variable 'd' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'd', list_382885)
    
    # Call to _check_laplacian(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'A' (line 121)
    A_382890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 21), 'A', False)
    # Getting the type of 'L' (line 121)
    L_382891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 24), 'L', False)
    # Getting the type of 'd' (line 121)
    d_382892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 27), 'd', False)
    # Processing the call keyword arguments (line 121)
    # Getting the type of 'False' (line 121)
    False_382893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 37), 'False', False)
    keyword_382894 = False_382893
    # Getting the type of 'False' (line 121)
    False_382895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 59), 'False', False)
    keyword_382896 = False_382895
    kwargs_382897 = {'normed': keyword_382894, 'use_out_degree': keyword_382896}
    # Getting the type of '_check_laplacian' (line 121)
    _check_laplacian_382889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), '_check_laplacian', False)
    # Calling _check_laplacian(args, kwargs) (line 121)
    _check_laplacian_call_result_382898 = invoke(stypy.reporting.localization.Localization(__file__, 121, 4), _check_laplacian_382889, *[A_382890, L_382891, d_382892], **kwargs_382897)
    
    
    # Assigning a List to a Name (line 124):
    
    # Assigning a List to a Name (line 124):
    
    # Obtaining an instance of the builtin type 'list' (line 124)
    list_382899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 124)
    # Adding element type (line 124)
    
    # Obtaining an instance of the builtin type 'list' (line 124)
    list_382900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 124)
    # Adding element type (line 124)
    int_382901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 9), list_382900, int_382901)
    # Adding element type (line 124)
    float_382902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 13), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 9), list_382900, float_382902)
    # Adding element type (line 124)
    int_382903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 9), list_382900, int_382903)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 8), list_382899, list_382900)
    # Adding element type (line 124)
    
    # Obtaining an instance of the builtin type 'list' (line 125)
    list_382904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 125)
    # Adding element type (line 125)
    int_382905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 9), list_382904, int_382905)
    # Adding element type (line 125)
    int_382906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 9), list_382904, int_382906)
    # Adding element type (line 125)
    int_382907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 9), list_382904, int_382907)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 8), list_382899, list_382904)
    # Adding element type (line 124)
    
    # Obtaining an instance of the builtin type 'list' (line 126)
    list_382908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 126)
    # Adding element type (line 126)
    int_382909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 9), list_382908, int_382909)
    # Adding element type (line 126)
    int_382910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 9), list_382908, int_382910)
    # Adding element type (line 126)
    int_382911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 9), list_382908, int_382911)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 8), list_382899, list_382908)
    
    # Assigning a type to the variable 'L' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'L', list_382899)
    
    # Assigning a List to a Name (line 127):
    
    # Assigning a List to a Name (line 127):
    
    # Obtaining an instance of the builtin type 'list' (line 127)
    list_382912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 127)
    # Adding element type (line 127)
    int_382913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 8), list_382912, int_382913)
    # Adding element type (line 127)
    int_382914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 8), list_382912, int_382914)
    # Adding element type (line 127)
    int_382915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 8), list_382912, int_382915)
    
    # Assigning a type to the variable 'd' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'd', list_382912)
    
    # Call to _check_laplacian(...): (line 128)
    # Processing the call arguments (line 128)
    # Getting the type of 'A' (line 128)
    A_382917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 21), 'A', False)
    # Getting the type of 'L' (line 128)
    L_382918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 24), 'L', False)
    # Getting the type of 'd' (line 128)
    d_382919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 27), 'd', False)
    # Processing the call keyword arguments (line 128)
    # Getting the type of 'True' (line 128)
    True_382920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 37), 'True', False)
    keyword_382921 = True_382920
    # Getting the type of 'False' (line 128)
    False_382922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 58), 'False', False)
    keyword_382923 = False_382922
    kwargs_382924 = {'normed': keyword_382921, 'use_out_degree': keyword_382923}
    # Getting the type of '_check_laplacian' (line 128)
    _check_laplacian_382916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), '_check_laplacian', False)
    # Calling _check_laplacian(args, kwargs) (line 128)
    _check_laplacian_call_result_382925 = invoke(stypy.reporting.localization.Localization(__file__, 128, 4), _check_laplacian_382916, *[A_382917, L_382918, d_382919], **kwargs_382924)
    
    
    # ################# End of 'test_asymmetric_laplacian(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_asymmetric_laplacian' in the type store
    # Getting the type of 'stypy_return_type' (line 96)
    stypy_return_type_382926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_382926)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_asymmetric_laplacian'
    return stypy_return_type_382926

# Assigning a type to the variable 'test_asymmetric_laplacian' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'test_asymmetric_laplacian', test_asymmetric_laplacian)

@norecursion
def test_sparse_formats(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_sparse_formats'
    module_type_store = module_type_store.open_function_context('test_sparse_formats', 131, 0, False)
    
    # Passed parameters checking function
    test_sparse_formats.stypy_localization = localization
    test_sparse_formats.stypy_type_of_self = None
    test_sparse_formats.stypy_type_store = module_type_store
    test_sparse_formats.stypy_function_name = 'test_sparse_formats'
    test_sparse_formats.stypy_param_names_list = []
    test_sparse_formats.stypy_varargs_param_name = None
    test_sparse_formats.stypy_kwargs_param_name = None
    test_sparse_formats.stypy_call_defaults = defaults
    test_sparse_formats.stypy_call_varargs = varargs
    test_sparse_formats.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_sparse_formats', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_sparse_formats', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_sparse_formats(...)' code ##################

    
    
    # Obtaining an instance of the builtin type 'tuple' (line 132)
    tuple_382927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 132)
    # Adding element type (line 132)
    str_382928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 16), 'str', 'csr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 16), tuple_382927, str_382928)
    # Adding element type (line 132)
    str_382929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 23), 'str', 'csc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 16), tuple_382927, str_382929)
    # Adding element type (line 132)
    str_382930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 30), 'str', 'coo')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 16), tuple_382927, str_382930)
    # Adding element type (line 132)
    str_382931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 37), 'str', 'lil')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 16), tuple_382927, str_382931)
    # Adding element type (line 132)
    str_382932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 44), 'str', 'dok')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 16), tuple_382927, str_382932)
    # Adding element type (line 132)
    str_382933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 51), 'str', 'dia')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 16), tuple_382927, str_382933)
    # Adding element type (line 132)
    str_382934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 58), 'str', 'bsr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 16), tuple_382927, str_382934)
    
    # Testing the type of a for loop iterable (line 132)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 132, 4), tuple_382927)
    # Getting the type of the for loop variable (line 132)
    for_loop_var_382935 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 132, 4), tuple_382927)
    # Assigning a type to the variable 'fmt' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'fmt', for_loop_var_382935)
    # SSA begins for a for statement (line 132)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 133):
    
    # Assigning a Call to a Name (line 133):
    
    # Call to diags(...): (line 133)
    # Processing the call arguments (line 133)
    
    # Obtaining an instance of the builtin type 'list' (line 133)
    list_382938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 133)
    # Adding element type (line 133)
    int_382939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 27), list_382938, int_382939)
    # Adding element type (line 133)
    int_382940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 27), list_382938, int_382940)
    
    
    # Obtaining an instance of the builtin type 'list' (line 133)
    list_382941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 133)
    # Adding element type (line 133)
    int_382942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 35), list_382941, int_382942)
    # Adding element type (line 133)
    int_382943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 35), list_382941, int_382943)
    
    # Processing the call keyword arguments (line 133)
    
    # Obtaining an instance of the builtin type 'tuple' (line 133)
    tuple_382944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 51), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 133)
    # Adding element type (line 133)
    int_382945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 51), tuple_382944, int_382945)
    # Adding element type (line 133)
    int_382946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 53), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 51), tuple_382944, int_382946)
    
    keyword_382947 = tuple_382944
    # Getting the type of 'fmt' (line 133)
    fmt_382948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 64), 'fmt', False)
    keyword_382949 = fmt_382948
    kwargs_382950 = {'shape': keyword_382947, 'format': keyword_382949}
    # Getting the type of 'sparse' (line 133)
    sparse_382936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 14), 'sparse', False)
    # Obtaining the member 'diags' of a type (line 133)
    diags_382937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 14), sparse_382936, 'diags')
    # Calling diags(args, kwargs) (line 133)
    diags_call_result_382951 = invoke(stypy.reporting.localization.Localization(__file__, 133, 14), diags_382937, *[list_382938, list_382941], **kwargs_382950)
    
    # Assigning a type to the variable 'mat' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'mat', diags_call_result_382951)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 134)
    tuple_382952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 134)
    # Adding element type (line 134)
    # Getting the type of 'True' (line 134)
    True_382953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 22), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 22), tuple_382952, True_382953)
    # Adding element type (line 134)
    # Getting the type of 'False' (line 134)
    False_382954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 28), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 22), tuple_382952, False_382954)
    
    # Testing the type of a for loop iterable (line 134)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 134, 8), tuple_382952)
    # Getting the type of the for loop variable (line 134)
    for_loop_var_382955 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 134, 8), tuple_382952)
    # Assigning a type to the variable 'normed' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'normed', for_loop_var_382955)
    # SSA begins for a for statement (line 134)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to _check_symmetric_graph_laplacian(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'mat' (line 135)
    mat_382957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 45), 'mat', False)
    # Getting the type of 'normed' (line 135)
    normed_382958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 50), 'normed', False)
    # Processing the call keyword arguments (line 135)
    kwargs_382959 = {}
    # Getting the type of '_check_symmetric_graph_laplacian' (line 135)
    _check_symmetric_graph_laplacian_382956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), '_check_symmetric_graph_laplacian', False)
    # Calling _check_symmetric_graph_laplacian(args, kwargs) (line 135)
    _check_symmetric_graph_laplacian_call_result_382960 = invoke(stypy.reporting.localization.Localization(__file__, 135, 12), _check_symmetric_graph_laplacian_382956, *[mat_382957, normed_382958], **kwargs_382959)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_sparse_formats(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_sparse_formats' in the type store
    # Getting the type of 'stypy_return_type' (line 131)
    stypy_return_type_382961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_382961)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_sparse_formats'
    return stypy_return_type_382961

# Assigning a type to the variable 'test_sparse_formats' (line 131)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 0), 'test_sparse_formats', test_sparse_formats)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
