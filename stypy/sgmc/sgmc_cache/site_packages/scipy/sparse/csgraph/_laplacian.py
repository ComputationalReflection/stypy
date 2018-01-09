
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Laplacian of a compressed-sparse graph
3: '''
4: 
5: # Authors: Aric Hagberg <hagberg@lanl.gov>
6: #          Gael Varoquaux <gael.varoquaux@normalesup.org>
7: #          Jake Vanderplas <vanderplas@astro.washington.edu>
8: # License: BSD
9: 
10: from __future__ import division, print_function, absolute_import
11: 
12: import numpy as np
13: from scipy.sparse import isspmatrix
14: 
15: 
16: ###############################################################################
17: # Graph laplacian
18: def laplacian(csgraph, normed=False, return_diag=False, use_out_degree=False):
19:     '''
20:     Return the Laplacian matrix of a directed graph.
21: 
22:     Parameters
23:     ----------
24:     csgraph : array_like or sparse matrix, 2 dimensions
25:         compressed-sparse graph, with shape (N, N).
26:     normed : bool, optional
27:         If True, then compute normalized Laplacian.
28:     return_diag : bool, optional
29:         If True, then also return an array related to vertex degrees.
30:     use_out_degree : bool, optional
31:         If True, then use out-degree instead of in-degree.
32:         This distinction matters only if the graph is asymmetric.
33:         Default: False.
34: 
35:     Returns
36:     -------
37:     lap : ndarray or sparse matrix
38:         The N x N laplacian matrix of csgraph. It will be a numpy array (dense)
39:         if the input was dense, or a sparse matrix otherwise.
40:     diag : ndarray, optional
41:         The length-N diagonal of the Laplacian matrix.
42:         For the normalized Laplacian, this is the array of square roots
43:         of vertex degrees or 1 if the degree is zero.
44: 
45:     Notes
46:     -----
47:     The Laplacian matrix of a graph is sometimes referred to as the
48:     "Kirchoff matrix" or the "admittance matrix", and is useful in many
49:     parts of spectral graph theory.  In particular, the eigen-decomposition
50:     of the laplacian matrix can give insight into many properties of the graph.
51: 
52:     Examples
53:     --------
54:     >>> from scipy.sparse import csgraph
55:     >>> G = np.arange(5) * np.arange(5)[:, np.newaxis]
56:     >>> G
57:     array([[ 0,  0,  0,  0,  0],
58:            [ 0,  1,  2,  3,  4],
59:            [ 0,  2,  4,  6,  8],
60:            [ 0,  3,  6,  9, 12],
61:            [ 0,  4,  8, 12, 16]])
62:     >>> csgraph.laplacian(G, normed=False)
63:     array([[  0,   0,   0,   0,   0],
64:            [  0,   9,  -2,  -3,  -4],
65:            [  0,  -2,  16,  -6,  -8],
66:            [  0,  -3,  -6,  21, -12],
67:            [  0,  -4,  -8, -12,  24]])
68:     '''
69:     if csgraph.ndim != 2 or csgraph.shape[0] != csgraph.shape[1]:
70:         raise ValueError('csgraph must be a square matrix or array')
71: 
72:     if normed and (np.issubdtype(csgraph.dtype, np.signedinteger)
73:                    or np.issubdtype(csgraph.dtype, np.uint)):
74:         csgraph = csgraph.astype(float)
75: 
76:     create_lap = _laplacian_sparse if isspmatrix(csgraph) else _laplacian_dense
77:     degree_axis = 1 if use_out_degree else 0
78:     lap, d = create_lap(csgraph, normed=normed, axis=degree_axis)
79:     if return_diag:
80:         return lap, d
81:     return lap
82: 
83: 
84: def _setdiag_dense(A, d):
85:     A.flat[::len(d)+1] = d
86: 
87: 
88: def _laplacian_sparse(graph, normed=False, axis=0):
89:     if graph.format in ('lil', 'dok'):
90:         m = graph.tocoo()
91:         needs_copy = False
92:     else:
93:         m = graph
94:         needs_copy = True
95:     w = m.sum(axis=axis).getA1() - m.diagonal()
96:     if normed:
97:         m = m.tocoo(copy=needs_copy)
98:         isolated_node_mask = (w == 0)
99:         w = np.where(isolated_node_mask, 1, np.sqrt(w))
100:         m.data /= w[m.row]
101:         m.data /= w[m.col]
102:         m.data *= -1
103:         m.setdiag(1 - isolated_node_mask)
104:     else:
105:         if m.format == 'dia':
106:             m = m.copy()
107:         else:
108:             m = m.tocoo(copy=needs_copy)
109:         m.data *= -1
110:         m.setdiag(w)
111:     return m, w
112: 
113: 
114: def _laplacian_dense(graph, normed=False, axis=0):
115:     m = np.array(graph)
116:     np.fill_diagonal(m, 0)
117:     w = m.sum(axis=axis)
118:     if normed:
119:         isolated_node_mask = (w == 0)
120:         w = np.where(isolated_node_mask, 1, np.sqrt(w))
121:         m /= w
122:         m /= w[:, np.newaxis]
123:         m *= -1
124:         _setdiag_dense(m, 1 - isolated_node_mask)
125:     else:
126:         m *= -1
127:         _setdiag_dense(m, w)
128:     return m, w
129: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_381200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', '\nLaplacian of a compressed-sparse graph\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import numpy' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')
import_381201 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy')

if (type(import_381201) is not StypyTypeError):

    if (import_381201 != 'pyd_module'):
        __import__(import_381201)
        sys_modules_381202 = sys.modules[import_381201]
        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'np', sys_modules_381202.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy', import_381201)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.sparse import isspmatrix' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')
import_381203 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse')

if (type(import_381203) is not StypyTypeError):

    if (import_381203 != 'pyd_module'):
        __import__(import_381203)
        sys_modules_381204 = sys.modules[import_381203]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse', sys_modules_381204.module_type_store, module_type_store, ['isspmatrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_381204, sys_modules_381204.module_type_store, module_type_store)
    else:
        from scipy.sparse import isspmatrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse', None, module_type_store, ['isspmatrix'], [isspmatrix])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse', import_381203)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')


@norecursion
def laplacian(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 18)
    False_381205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 30), 'False')
    # Getting the type of 'False' (line 18)
    False_381206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 49), 'False')
    # Getting the type of 'False' (line 18)
    False_381207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 71), 'False')
    defaults = [False_381205, False_381206, False_381207]
    # Create a new context for function 'laplacian'
    module_type_store = module_type_store.open_function_context('laplacian', 18, 0, False)
    
    # Passed parameters checking function
    laplacian.stypy_localization = localization
    laplacian.stypy_type_of_self = None
    laplacian.stypy_type_store = module_type_store
    laplacian.stypy_function_name = 'laplacian'
    laplacian.stypy_param_names_list = ['csgraph', 'normed', 'return_diag', 'use_out_degree']
    laplacian.stypy_varargs_param_name = None
    laplacian.stypy_kwargs_param_name = None
    laplacian.stypy_call_defaults = defaults
    laplacian.stypy_call_varargs = varargs
    laplacian.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'laplacian', ['csgraph', 'normed', 'return_diag', 'use_out_degree'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'laplacian', localization, ['csgraph', 'normed', 'return_diag', 'use_out_degree'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'laplacian(...)' code ##################

    str_381208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, (-1)), 'str', '\n    Return the Laplacian matrix of a directed graph.\n\n    Parameters\n    ----------\n    csgraph : array_like or sparse matrix, 2 dimensions\n        compressed-sparse graph, with shape (N, N).\n    normed : bool, optional\n        If True, then compute normalized Laplacian.\n    return_diag : bool, optional\n        If True, then also return an array related to vertex degrees.\n    use_out_degree : bool, optional\n        If True, then use out-degree instead of in-degree.\n        This distinction matters only if the graph is asymmetric.\n        Default: False.\n\n    Returns\n    -------\n    lap : ndarray or sparse matrix\n        The N x N laplacian matrix of csgraph. It will be a numpy array (dense)\n        if the input was dense, or a sparse matrix otherwise.\n    diag : ndarray, optional\n        The length-N diagonal of the Laplacian matrix.\n        For the normalized Laplacian, this is the array of square roots\n        of vertex degrees or 1 if the degree is zero.\n\n    Notes\n    -----\n    The Laplacian matrix of a graph is sometimes referred to as the\n    "Kirchoff matrix" or the "admittance matrix", and is useful in many\n    parts of spectral graph theory.  In particular, the eigen-decomposition\n    of the laplacian matrix can give insight into many properties of the graph.\n\n    Examples\n    --------\n    >>> from scipy.sparse import csgraph\n    >>> G = np.arange(5) * np.arange(5)[:, np.newaxis]\n    >>> G\n    array([[ 0,  0,  0,  0,  0],\n           [ 0,  1,  2,  3,  4],\n           [ 0,  2,  4,  6,  8],\n           [ 0,  3,  6,  9, 12],\n           [ 0,  4,  8, 12, 16]])\n    >>> csgraph.laplacian(G, normed=False)\n    array([[  0,   0,   0,   0,   0],\n           [  0,   9,  -2,  -3,  -4],\n           [  0,  -2,  16,  -6,  -8],\n           [  0,  -3,  -6,  21, -12],\n           [  0,  -4,  -8, -12,  24]])\n    ')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'csgraph' (line 69)
    csgraph_381209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 7), 'csgraph')
    # Obtaining the member 'ndim' of a type (line 69)
    ndim_381210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 7), csgraph_381209, 'ndim')
    int_381211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 23), 'int')
    # Applying the binary operator '!=' (line 69)
    result_ne_381212 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 7), '!=', ndim_381210, int_381211)
    
    
    
    # Obtaining the type of the subscript
    int_381213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 42), 'int')
    # Getting the type of 'csgraph' (line 69)
    csgraph_381214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 28), 'csgraph')
    # Obtaining the member 'shape' of a type (line 69)
    shape_381215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 28), csgraph_381214, 'shape')
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___381216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 28), shape_381215, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_381217 = invoke(stypy.reporting.localization.Localization(__file__, 69, 28), getitem___381216, int_381213)
    
    
    # Obtaining the type of the subscript
    int_381218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 62), 'int')
    # Getting the type of 'csgraph' (line 69)
    csgraph_381219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 48), 'csgraph')
    # Obtaining the member 'shape' of a type (line 69)
    shape_381220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 48), csgraph_381219, 'shape')
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___381221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 48), shape_381220, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_381222 = invoke(stypy.reporting.localization.Localization(__file__, 69, 48), getitem___381221, int_381218)
    
    # Applying the binary operator '!=' (line 69)
    result_ne_381223 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 28), '!=', subscript_call_result_381217, subscript_call_result_381222)
    
    # Applying the binary operator 'or' (line 69)
    result_or_keyword_381224 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 7), 'or', result_ne_381212, result_ne_381223)
    
    # Testing the type of an if condition (line 69)
    if_condition_381225 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 4), result_or_keyword_381224)
    # Assigning a type to the variable 'if_condition_381225' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'if_condition_381225', if_condition_381225)
    # SSA begins for if statement (line 69)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 70)
    # Processing the call arguments (line 70)
    str_381227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 25), 'str', 'csgraph must be a square matrix or array')
    # Processing the call keyword arguments (line 70)
    kwargs_381228 = {}
    # Getting the type of 'ValueError' (line 70)
    ValueError_381226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 70)
    ValueError_call_result_381229 = invoke(stypy.reporting.localization.Localization(__file__, 70, 14), ValueError_381226, *[str_381227], **kwargs_381228)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 70, 8), ValueError_call_result_381229, 'raise parameter', BaseException)
    # SSA join for if statement (line 69)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'normed' (line 72)
    normed_381230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 7), 'normed')
    
    # Evaluating a boolean operation
    
    # Call to issubdtype(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'csgraph' (line 72)
    csgraph_381233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 33), 'csgraph', False)
    # Obtaining the member 'dtype' of a type (line 72)
    dtype_381234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 33), csgraph_381233, 'dtype')
    # Getting the type of 'np' (line 72)
    np_381235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 48), 'np', False)
    # Obtaining the member 'signedinteger' of a type (line 72)
    signedinteger_381236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 48), np_381235, 'signedinteger')
    # Processing the call keyword arguments (line 72)
    kwargs_381237 = {}
    # Getting the type of 'np' (line 72)
    np_381231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 72)
    issubdtype_381232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 19), np_381231, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 72)
    issubdtype_call_result_381238 = invoke(stypy.reporting.localization.Localization(__file__, 72, 19), issubdtype_381232, *[dtype_381234, signedinteger_381236], **kwargs_381237)
    
    
    # Call to issubdtype(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'csgraph' (line 73)
    csgraph_381241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 36), 'csgraph', False)
    # Obtaining the member 'dtype' of a type (line 73)
    dtype_381242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 36), csgraph_381241, 'dtype')
    # Getting the type of 'np' (line 73)
    np_381243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 51), 'np', False)
    # Obtaining the member 'uint' of a type (line 73)
    uint_381244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 51), np_381243, 'uint')
    # Processing the call keyword arguments (line 73)
    kwargs_381245 = {}
    # Getting the type of 'np' (line 73)
    np_381239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 22), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 73)
    issubdtype_381240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 22), np_381239, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 73)
    issubdtype_call_result_381246 = invoke(stypy.reporting.localization.Localization(__file__, 73, 22), issubdtype_381240, *[dtype_381242, uint_381244], **kwargs_381245)
    
    # Applying the binary operator 'or' (line 72)
    result_or_keyword_381247 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 19), 'or', issubdtype_call_result_381238, issubdtype_call_result_381246)
    
    # Applying the binary operator 'and' (line 72)
    result_and_keyword_381248 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 7), 'and', normed_381230, result_or_keyword_381247)
    
    # Testing the type of an if condition (line 72)
    if_condition_381249 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 4), result_and_keyword_381248)
    # Assigning a type to the variable 'if_condition_381249' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'if_condition_381249', if_condition_381249)
    # SSA begins for if statement (line 72)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 74):
    
    # Assigning a Call to a Name (line 74):
    
    # Call to astype(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'float' (line 74)
    float_381252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 33), 'float', False)
    # Processing the call keyword arguments (line 74)
    kwargs_381253 = {}
    # Getting the type of 'csgraph' (line 74)
    csgraph_381250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 18), 'csgraph', False)
    # Obtaining the member 'astype' of a type (line 74)
    astype_381251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 18), csgraph_381250, 'astype')
    # Calling astype(args, kwargs) (line 74)
    astype_call_result_381254 = invoke(stypy.reporting.localization.Localization(__file__, 74, 18), astype_381251, *[float_381252], **kwargs_381253)
    
    # Assigning a type to the variable 'csgraph' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'csgraph', astype_call_result_381254)
    # SSA join for if statement (line 72)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a IfExp to a Name (line 76):
    
    # Assigning a IfExp to a Name (line 76):
    
    
    # Call to isspmatrix(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'csgraph' (line 76)
    csgraph_381256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 49), 'csgraph', False)
    # Processing the call keyword arguments (line 76)
    kwargs_381257 = {}
    # Getting the type of 'isspmatrix' (line 76)
    isspmatrix_381255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 38), 'isspmatrix', False)
    # Calling isspmatrix(args, kwargs) (line 76)
    isspmatrix_call_result_381258 = invoke(stypy.reporting.localization.Localization(__file__, 76, 38), isspmatrix_381255, *[csgraph_381256], **kwargs_381257)
    
    # Testing the type of an if expression (line 76)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 17), isspmatrix_call_result_381258)
    # SSA begins for if expression (line 76)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of '_laplacian_sparse' (line 76)
    _laplacian_sparse_381259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 17), '_laplacian_sparse')
    # SSA branch for the else part of an if expression (line 76)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of '_laplacian_dense' (line 76)
    _laplacian_dense_381260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 63), '_laplacian_dense')
    # SSA join for if expression (line 76)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_381261 = union_type.UnionType.add(_laplacian_sparse_381259, _laplacian_dense_381260)
    
    # Assigning a type to the variable 'create_lap' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'create_lap', if_exp_381261)
    
    # Assigning a IfExp to a Name (line 77):
    
    # Assigning a IfExp to a Name (line 77):
    
    # Getting the type of 'use_out_degree' (line 77)
    use_out_degree_381262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 23), 'use_out_degree')
    # Testing the type of an if expression (line 77)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 18), use_out_degree_381262)
    # SSA begins for if expression (line 77)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    int_381263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 18), 'int')
    # SSA branch for the else part of an if expression (line 77)
    module_type_store.open_ssa_branch('if expression else')
    int_381264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 43), 'int')
    # SSA join for if expression (line 77)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_381265 = union_type.UnionType.add(int_381263, int_381264)
    
    # Assigning a type to the variable 'degree_axis' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'degree_axis', if_exp_381265)
    
    # Assigning a Call to a Tuple (line 78):
    
    # Assigning a Subscript to a Name (line 78):
    
    # Obtaining the type of the subscript
    int_381266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 4), 'int')
    
    # Call to create_lap(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'csgraph' (line 78)
    csgraph_381268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 24), 'csgraph', False)
    # Processing the call keyword arguments (line 78)
    # Getting the type of 'normed' (line 78)
    normed_381269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 40), 'normed', False)
    keyword_381270 = normed_381269
    # Getting the type of 'degree_axis' (line 78)
    degree_axis_381271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 53), 'degree_axis', False)
    keyword_381272 = degree_axis_381271
    kwargs_381273 = {'normed': keyword_381270, 'axis': keyword_381272}
    # Getting the type of 'create_lap' (line 78)
    create_lap_381267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 13), 'create_lap', False)
    # Calling create_lap(args, kwargs) (line 78)
    create_lap_call_result_381274 = invoke(stypy.reporting.localization.Localization(__file__, 78, 13), create_lap_381267, *[csgraph_381268], **kwargs_381273)
    
    # Obtaining the member '__getitem__' of a type (line 78)
    getitem___381275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 4), create_lap_call_result_381274, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 78)
    subscript_call_result_381276 = invoke(stypy.reporting.localization.Localization(__file__, 78, 4), getitem___381275, int_381266)
    
    # Assigning a type to the variable 'tuple_var_assignment_381198' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'tuple_var_assignment_381198', subscript_call_result_381276)
    
    # Assigning a Subscript to a Name (line 78):
    
    # Obtaining the type of the subscript
    int_381277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 4), 'int')
    
    # Call to create_lap(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'csgraph' (line 78)
    csgraph_381279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 24), 'csgraph', False)
    # Processing the call keyword arguments (line 78)
    # Getting the type of 'normed' (line 78)
    normed_381280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 40), 'normed', False)
    keyword_381281 = normed_381280
    # Getting the type of 'degree_axis' (line 78)
    degree_axis_381282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 53), 'degree_axis', False)
    keyword_381283 = degree_axis_381282
    kwargs_381284 = {'normed': keyword_381281, 'axis': keyword_381283}
    # Getting the type of 'create_lap' (line 78)
    create_lap_381278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 13), 'create_lap', False)
    # Calling create_lap(args, kwargs) (line 78)
    create_lap_call_result_381285 = invoke(stypy.reporting.localization.Localization(__file__, 78, 13), create_lap_381278, *[csgraph_381279], **kwargs_381284)
    
    # Obtaining the member '__getitem__' of a type (line 78)
    getitem___381286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 4), create_lap_call_result_381285, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 78)
    subscript_call_result_381287 = invoke(stypy.reporting.localization.Localization(__file__, 78, 4), getitem___381286, int_381277)
    
    # Assigning a type to the variable 'tuple_var_assignment_381199' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'tuple_var_assignment_381199', subscript_call_result_381287)
    
    # Assigning a Name to a Name (line 78):
    # Getting the type of 'tuple_var_assignment_381198' (line 78)
    tuple_var_assignment_381198_381288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'tuple_var_assignment_381198')
    # Assigning a type to the variable 'lap' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'lap', tuple_var_assignment_381198_381288)
    
    # Assigning a Name to a Name (line 78):
    # Getting the type of 'tuple_var_assignment_381199' (line 78)
    tuple_var_assignment_381199_381289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'tuple_var_assignment_381199')
    # Assigning a type to the variable 'd' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 9), 'd', tuple_var_assignment_381199_381289)
    
    # Getting the type of 'return_diag' (line 79)
    return_diag_381290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 7), 'return_diag')
    # Testing the type of an if condition (line 79)
    if_condition_381291 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 4), return_diag_381290)
    # Assigning a type to the variable 'if_condition_381291' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'if_condition_381291', if_condition_381291)
    # SSA begins for if statement (line 79)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 80)
    tuple_381292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 80)
    # Adding element type (line 80)
    # Getting the type of 'lap' (line 80)
    lap_381293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'lap')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 15), tuple_381292, lap_381293)
    # Adding element type (line 80)
    # Getting the type of 'd' (line 80)
    d_381294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), 'd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 15), tuple_381292, d_381294)
    
    # Assigning a type to the variable 'stypy_return_type' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'stypy_return_type', tuple_381292)
    # SSA join for if statement (line 79)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'lap' (line 81)
    lap_381295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'lap')
    # Assigning a type to the variable 'stypy_return_type' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type', lap_381295)
    
    # ################# End of 'laplacian(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'laplacian' in the type store
    # Getting the type of 'stypy_return_type' (line 18)
    stypy_return_type_381296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_381296)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'laplacian'
    return stypy_return_type_381296

# Assigning a type to the variable 'laplacian' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'laplacian', laplacian)

@norecursion
def _setdiag_dense(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_setdiag_dense'
    module_type_store = module_type_store.open_function_context('_setdiag_dense', 84, 0, False)
    
    # Passed parameters checking function
    _setdiag_dense.stypy_localization = localization
    _setdiag_dense.stypy_type_of_self = None
    _setdiag_dense.stypy_type_store = module_type_store
    _setdiag_dense.stypy_function_name = '_setdiag_dense'
    _setdiag_dense.stypy_param_names_list = ['A', 'd']
    _setdiag_dense.stypy_varargs_param_name = None
    _setdiag_dense.stypy_kwargs_param_name = None
    _setdiag_dense.stypy_call_defaults = defaults
    _setdiag_dense.stypy_call_varargs = varargs
    _setdiag_dense.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_setdiag_dense', ['A', 'd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_setdiag_dense', localization, ['A', 'd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_setdiag_dense(...)' code ##################

    
    # Assigning a Name to a Subscript (line 85):
    
    # Assigning a Name to a Subscript (line 85):
    # Getting the type of 'd' (line 85)
    d_381297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 25), 'd')
    # Getting the type of 'A' (line 85)
    A_381298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'A')
    # Obtaining the member 'flat' of a type (line 85)
    flat_381299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 4), A_381298, 'flat')
    
    # Call to len(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'd' (line 85)
    d_381301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 17), 'd', False)
    # Processing the call keyword arguments (line 85)
    kwargs_381302 = {}
    # Getting the type of 'len' (line 85)
    len_381300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 13), 'len', False)
    # Calling len(args, kwargs) (line 85)
    len_call_result_381303 = invoke(stypy.reporting.localization.Localization(__file__, 85, 13), len_381300, *[d_381301], **kwargs_381302)
    
    int_381304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 20), 'int')
    # Applying the binary operator '+' (line 85)
    result_add_381305 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 13), '+', len_call_result_381303, int_381304)
    
    slice_381306 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 85, 4), None, None, result_add_381305)
    # Storing an element on a container (line 85)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 4), flat_381299, (slice_381306, d_381297))
    
    # ################# End of '_setdiag_dense(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_setdiag_dense' in the type store
    # Getting the type of 'stypy_return_type' (line 84)
    stypy_return_type_381307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_381307)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_setdiag_dense'
    return stypy_return_type_381307

# Assigning a type to the variable '_setdiag_dense' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), '_setdiag_dense', _setdiag_dense)

@norecursion
def _laplacian_sparse(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 88)
    False_381308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 36), 'False')
    int_381309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 48), 'int')
    defaults = [False_381308, int_381309]
    # Create a new context for function '_laplacian_sparse'
    module_type_store = module_type_store.open_function_context('_laplacian_sparse', 88, 0, False)
    
    # Passed parameters checking function
    _laplacian_sparse.stypy_localization = localization
    _laplacian_sparse.stypy_type_of_self = None
    _laplacian_sparse.stypy_type_store = module_type_store
    _laplacian_sparse.stypy_function_name = '_laplacian_sparse'
    _laplacian_sparse.stypy_param_names_list = ['graph', 'normed', 'axis']
    _laplacian_sparse.stypy_varargs_param_name = None
    _laplacian_sparse.stypy_kwargs_param_name = None
    _laplacian_sparse.stypy_call_defaults = defaults
    _laplacian_sparse.stypy_call_varargs = varargs
    _laplacian_sparse.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_laplacian_sparse', ['graph', 'normed', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_laplacian_sparse', localization, ['graph', 'normed', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_laplacian_sparse(...)' code ##################

    
    
    # Getting the type of 'graph' (line 89)
    graph_381310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 7), 'graph')
    # Obtaining the member 'format' of a type (line 89)
    format_381311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 7), graph_381310, 'format')
    
    # Obtaining an instance of the builtin type 'tuple' (line 89)
    tuple_381312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 89)
    # Adding element type (line 89)
    str_381313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 24), 'str', 'lil')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 24), tuple_381312, str_381313)
    # Adding element type (line 89)
    str_381314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 31), 'str', 'dok')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 24), tuple_381312, str_381314)
    
    # Applying the binary operator 'in' (line 89)
    result_contains_381315 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 7), 'in', format_381311, tuple_381312)
    
    # Testing the type of an if condition (line 89)
    if_condition_381316 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 4), result_contains_381315)
    # Assigning a type to the variable 'if_condition_381316' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'if_condition_381316', if_condition_381316)
    # SSA begins for if statement (line 89)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 90):
    
    # Assigning a Call to a Name (line 90):
    
    # Call to tocoo(...): (line 90)
    # Processing the call keyword arguments (line 90)
    kwargs_381319 = {}
    # Getting the type of 'graph' (line 90)
    graph_381317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'graph', False)
    # Obtaining the member 'tocoo' of a type (line 90)
    tocoo_381318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), graph_381317, 'tocoo')
    # Calling tocoo(args, kwargs) (line 90)
    tocoo_call_result_381320 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), tocoo_381318, *[], **kwargs_381319)
    
    # Assigning a type to the variable 'm' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'm', tocoo_call_result_381320)
    
    # Assigning a Name to a Name (line 91):
    
    # Assigning a Name to a Name (line 91):
    # Getting the type of 'False' (line 91)
    False_381321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 21), 'False')
    # Assigning a type to the variable 'needs_copy' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'needs_copy', False_381321)
    # SSA branch for the else part of an if statement (line 89)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 93):
    
    # Assigning a Name to a Name (line 93):
    # Getting the type of 'graph' (line 93)
    graph_381322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'graph')
    # Assigning a type to the variable 'm' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'm', graph_381322)
    
    # Assigning a Name to a Name (line 94):
    
    # Assigning a Name to a Name (line 94):
    # Getting the type of 'True' (line 94)
    True_381323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 21), 'True')
    # Assigning a type to the variable 'needs_copy' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'needs_copy', True_381323)
    # SSA join for if statement (line 89)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 95):
    
    # Assigning a BinOp to a Name (line 95):
    
    # Call to getA1(...): (line 95)
    # Processing the call keyword arguments (line 95)
    kwargs_381331 = {}
    
    # Call to sum(...): (line 95)
    # Processing the call keyword arguments (line 95)
    # Getting the type of 'axis' (line 95)
    axis_381326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 19), 'axis', False)
    keyword_381327 = axis_381326
    kwargs_381328 = {'axis': keyword_381327}
    # Getting the type of 'm' (line 95)
    m_381324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'm', False)
    # Obtaining the member 'sum' of a type (line 95)
    sum_381325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), m_381324, 'sum')
    # Calling sum(args, kwargs) (line 95)
    sum_call_result_381329 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), sum_381325, *[], **kwargs_381328)
    
    # Obtaining the member 'getA1' of a type (line 95)
    getA1_381330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), sum_call_result_381329, 'getA1')
    # Calling getA1(args, kwargs) (line 95)
    getA1_call_result_381332 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), getA1_381330, *[], **kwargs_381331)
    
    
    # Call to diagonal(...): (line 95)
    # Processing the call keyword arguments (line 95)
    kwargs_381335 = {}
    # Getting the type of 'm' (line 95)
    m_381333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 35), 'm', False)
    # Obtaining the member 'diagonal' of a type (line 95)
    diagonal_381334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 35), m_381333, 'diagonal')
    # Calling diagonal(args, kwargs) (line 95)
    diagonal_call_result_381336 = invoke(stypy.reporting.localization.Localization(__file__, 95, 35), diagonal_381334, *[], **kwargs_381335)
    
    # Applying the binary operator '-' (line 95)
    result_sub_381337 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 8), '-', getA1_call_result_381332, diagonal_call_result_381336)
    
    # Assigning a type to the variable 'w' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'w', result_sub_381337)
    
    # Getting the type of 'normed' (line 96)
    normed_381338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 7), 'normed')
    # Testing the type of an if condition (line 96)
    if_condition_381339 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 4), normed_381338)
    # Assigning a type to the variable 'if_condition_381339' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'if_condition_381339', if_condition_381339)
    # SSA begins for if statement (line 96)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 97):
    
    # Assigning a Call to a Name (line 97):
    
    # Call to tocoo(...): (line 97)
    # Processing the call keyword arguments (line 97)
    # Getting the type of 'needs_copy' (line 97)
    needs_copy_381342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 25), 'needs_copy', False)
    keyword_381343 = needs_copy_381342
    kwargs_381344 = {'copy': keyword_381343}
    # Getting the type of 'm' (line 97)
    m_381340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'm', False)
    # Obtaining the member 'tocoo' of a type (line 97)
    tocoo_381341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 12), m_381340, 'tocoo')
    # Calling tocoo(args, kwargs) (line 97)
    tocoo_call_result_381345 = invoke(stypy.reporting.localization.Localization(__file__, 97, 12), tocoo_381341, *[], **kwargs_381344)
    
    # Assigning a type to the variable 'm' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'm', tocoo_call_result_381345)
    
    # Assigning a Compare to a Name (line 98):
    
    # Assigning a Compare to a Name (line 98):
    
    # Getting the type of 'w' (line 98)
    w_381346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 30), 'w')
    int_381347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 35), 'int')
    # Applying the binary operator '==' (line 98)
    result_eq_381348 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 30), '==', w_381346, int_381347)
    
    # Assigning a type to the variable 'isolated_node_mask' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'isolated_node_mask', result_eq_381348)
    
    # Assigning a Call to a Name (line 99):
    
    # Assigning a Call to a Name (line 99):
    
    # Call to where(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'isolated_node_mask' (line 99)
    isolated_node_mask_381351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 21), 'isolated_node_mask', False)
    int_381352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 41), 'int')
    
    # Call to sqrt(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'w' (line 99)
    w_381355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 52), 'w', False)
    # Processing the call keyword arguments (line 99)
    kwargs_381356 = {}
    # Getting the type of 'np' (line 99)
    np_381353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 44), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 99)
    sqrt_381354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 44), np_381353, 'sqrt')
    # Calling sqrt(args, kwargs) (line 99)
    sqrt_call_result_381357 = invoke(stypy.reporting.localization.Localization(__file__, 99, 44), sqrt_381354, *[w_381355], **kwargs_381356)
    
    # Processing the call keyword arguments (line 99)
    kwargs_381358 = {}
    # Getting the type of 'np' (line 99)
    np_381349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'np', False)
    # Obtaining the member 'where' of a type (line 99)
    where_381350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 12), np_381349, 'where')
    # Calling where(args, kwargs) (line 99)
    where_call_result_381359 = invoke(stypy.reporting.localization.Localization(__file__, 99, 12), where_381350, *[isolated_node_mask_381351, int_381352, sqrt_call_result_381357], **kwargs_381358)
    
    # Assigning a type to the variable 'w' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'w', where_call_result_381359)
    
    # Getting the type of 'm' (line 100)
    m_381360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'm')
    # Obtaining the member 'data' of a type (line 100)
    data_381361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), m_381360, 'data')
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 100)
    m_381362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 20), 'm')
    # Obtaining the member 'row' of a type (line 100)
    row_381363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 20), m_381362, 'row')
    # Getting the type of 'w' (line 100)
    w_381364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 18), 'w')
    # Obtaining the member '__getitem__' of a type (line 100)
    getitem___381365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 18), w_381364, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 100)
    subscript_call_result_381366 = invoke(stypy.reporting.localization.Localization(__file__, 100, 18), getitem___381365, row_381363)
    
    # Applying the binary operator 'div=' (line 100)
    result_div_381367 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 8), 'div=', data_381361, subscript_call_result_381366)
    # Getting the type of 'm' (line 100)
    m_381368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'm')
    # Setting the type of the member 'data' of a type (line 100)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), m_381368, 'data', result_div_381367)
    
    
    # Getting the type of 'm' (line 101)
    m_381369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'm')
    # Obtaining the member 'data' of a type (line 101)
    data_381370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), m_381369, 'data')
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 101)
    m_381371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 20), 'm')
    # Obtaining the member 'col' of a type (line 101)
    col_381372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 20), m_381371, 'col')
    # Getting the type of 'w' (line 101)
    w_381373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 18), 'w')
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___381374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 18), w_381373, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_381375 = invoke(stypy.reporting.localization.Localization(__file__, 101, 18), getitem___381374, col_381372)
    
    # Applying the binary operator 'div=' (line 101)
    result_div_381376 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 8), 'div=', data_381370, subscript_call_result_381375)
    # Getting the type of 'm' (line 101)
    m_381377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'm')
    # Setting the type of the member 'data' of a type (line 101)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), m_381377, 'data', result_div_381376)
    
    
    # Getting the type of 'm' (line 102)
    m_381378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'm')
    # Obtaining the member 'data' of a type (line 102)
    data_381379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), m_381378, 'data')
    int_381380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 18), 'int')
    # Applying the binary operator '*=' (line 102)
    result_imul_381381 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 8), '*=', data_381379, int_381380)
    # Getting the type of 'm' (line 102)
    m_381382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'm')
    # Setting the type of the member 'data' of a type (line 102)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), m_381382, 'data', result_imul_381381)
    
    
    # Call to setdiag(...): (line 103)
    # Processing the call arguments (line 103)
    int_381385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 18), 'int')
    # Getting the type of 'isolated_node_mask' (line 103)
    isolated_node_mask_381386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 22), 'isolated_node_mask', False)
    # Applying the binary operator '-' (line 103)
    result_sub_381387 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 18), '-', int_381385, isolated_node_mask_381386)
    
    # Processing the call keyword arguments (line 103)
    kwargs_381388 = {}
    # Getting the type of 'm' (line 103)
    m_381383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'm', False)
    # Obtaining the member 'setdiag' of a type (line 103)
    setdiag_381384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), m_381383, 'setdiag')
    # Calling setdiag(args, kwargs) (line 103)
    setdiag_call_result_381389 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), setdiag_381384, *[result_sub_381387], **kwargs_381388)
    
    # SSA branch for the else part of an if statement (line 96)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'm' (line 105)
    m_381390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 'm')
    # Obtaining the member 'format' of a type (line 105)
    format_381391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 11), m_381390, 'format')
    str_381392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 23), 'str', 'dia')
    # Applying the binary operator '==' (line 105)
    result_eq_381393 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 11), '==', format_381391, str_381392)
    
    # Testing the type of an if condition (line 105)
    if_condition_381394 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 8), result_eq_381393)
    # Assigning a type to the variable 'if_condition_381394' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'if_condition_381394', if_condition_381394)
    # SSA begins for if statement (line 105)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 106):
    
    # Assigning a Call to a Name (line 106):
    
    # Call to copy(...): (line 106)
    # Processing the call keyword arguments (line 106)
    kwargs_381397 = {}
    # Getting the type of 'm' (line 106)
    m_381395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'm', False)
    # Obtaining the member 'copy' of a type (line 106)
    copy_381396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 16), m_381395, 'copy')
    # Calling copy(args, kwargs) (line 106)
    copy_call_result_381398 = invoke(stypy.reporting.localization.Localization(__file__, 106, 16), copy_381396, *[], **kwargs_381397)
    
    # Assigning a type to the variable 'm' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'm', copy_call_result_381398)
    # SSA branch for the else part of an if statement (line 105)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 108):
    
    # Assigning a Call to a Name (line 108):
    
    # Call to tocoo(...): (line 108)
    # Processing the call keyword arguments (line 108)
    # Getting the type of 'needs_copy' (line 108)
    needs_copy_381401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 'needs_copy', False)
    keyword_381402 = needs_copy_381401
    kwargs_381403 = {'copy': keyword_381402}
    # Getting the type of 'm' (line 108)
    m_381399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'm', False)
    # Obtaining the member 'tocoo' of a type (line 108)
    tocoo_381400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 16), m_381399, 'tocoo')
    # Calling tocoo(args, kwargs) (line 108)
    tocoo_call_result_381404 = invoke(stypy.reporting.localization.Localization(__file__, 108, 16), tocoo_381400, *[], **kwargs_381403)
    
    # Assigning a type to the variable 'm' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'm', tocoo_call_result_381404)
    # SSA join for if statement (line 105)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'm' (line 109)
    m_381405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'm')
    # Obtaining the member 'data' of a type (line 109)
    data_381406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), m_381405, 'data')
    int_381407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 18), 'int')
    # Applying the binary operator '*=' (line 109)
    result_imul_381408 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 8), '*=', data_381406, int_381407)
    # Getting the type of 'm' (line 109)
    m_381409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'm')
    # Setting the type of the member 'data' of a type (line 109)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), m_381409, 'data', result_imul_381408)
    
    
    # Call to setdiag(...): (line 110)
    # Processing the call arguments (line 110)
    # Getting the type of 'w' (line 110)
    w_381412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 18), 'w', False)
    # Processing the call keyword arguments (line 110)
    kwargs_381413 = {}
    # Getting the type of 'm' (line 110)
    m_381410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'm', False)
    # Obtaining the member 'setdiag' of a type (line 110)
    setdiag_381411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), m_381410, 'setdiag')
    # Calling setdiag(args, kwargs) (line 110)
    setdiag_call_result_381414 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), setdiag_381411, *[w_381412], **kwargs_381413)
    
    # SSA join for if statement (line 96)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 111)
    tuple_381415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 111)
    # Adding element type (line 111)
    # Getting the type of 'm' (line 111)
    m_381416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 11), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 11), tuple_381415, m_381416)
    # Adding element type (line 111)
    # Getting the type of 'w' (line 111)
    w_381417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 14), 'w')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 11), tuple_381415, w_381417)
    
    # Assigning a type to the variable 'stypy_return_type' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type', tuple_381415)
    
    # ################# End of '_laplacian_sparse(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_laplacian_sparse' in the type store
    # Getting the type of 'stypy_return_type' (line 88)
    stypy_return_type_381418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_381418)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_laplacian_sparse'
    return stypy_return_type_381418

# Assigning a type to the variable '_laplacian_sparse' (line 88)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), '_laplacian_sparse', _laplacian_sparse)

@norecursion
def _laplacian_dense(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 114)
    False_381419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 35), 'False')
    int_381420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 47), 'int')
    defaults = [False_381419, int_381420]
    # Create a new context for function '_laplacian_dense'
    module_type_store = module_type_store.open_function_context('_laplacian_dense', 114, 0, False)
    
    # Passed parameters checking function
    _laplacian_dense.stypy_localization = localization
    _laplacian_dense.stypy_type_of_self = None
    _laplacian_dense.stypy_type_store = module_type_store
    _laplacian_dense.stypy_function_name = '_laplacian_dense'
    _laplacian_dense.stypy_param_names_list = ['graph', 'normed', 'axis']
    _laplacian_dense.stypy_varargs_param_name = None
    _laplacian_dense.stypy_kwargs_param_name = None
    _laplacian_dense.stypy_call_defaults = defaults
    _laplacian_dense.stypy_call_varargs = varargs
    _laplacian_dense.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_laplacian_dense', ['graph', 'normed', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_laplacian_dense', localization, ['graph', 'normed', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_laplacian_dense(...)' code ##################

    
    # Assigning a Call to a Name (line 115):
    
    # Assigning a Call to a Name (line 115):
    
    # Call to array(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'graph' (line 115)
    graph_381423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 17), 'graph', False)
    # Processing the call keyword arguments (line 115)
    kwargs_381424 = {}
    # Getting the type of 'np' (line 115)
    np_381421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 115)
    array_381422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 8), np_381421, 'array')
    # Calling array(args, kwargs) (line 115)
    array_call_result_381425 = invoke(stypy.reporting.localization.Localization(__file__, 115, 8), array_381422, *[graph_381423], **kwargs_381424)
    
    # Assigning a type to the variable 'm' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'm', array_call_result_381425)
    
    # Call to fill_diagonal(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'm' (line 116)
    m_381428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 21), 'm', False)
    int_381429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 24), 'int')
    # Processing the call keyword arguments (line 116)
    kwargs_381430 = {}
    # Getting the type of 'np' (line 116)
    np_381426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'np', False)
    # Obtaining the member 'fill_diagonal' of a type (line 116)
    fill_diagonal_381427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 4), np_381426, 'fill_diagonal')
    # Calling fill_diagonal(args, kwargs) (line 116)
    fill_diagonal_call_result_381431 = invoke(stypy.reporting.localization.Localization(__file__, 116, 4), fill_diagonal_381427, *[m_381428, int_381429], **kwargs_381430)
    
    
    # Assigning a Call to a Name (line 117):
    
    # Assigning a Call to a Name (line 117):
    
    # Call to sum(...): (line 117)
    # Processing the call keyword arguments (line 117)
    # Getting the type of 'axis' (line 117)
    axis_381434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 19), 'axis', False)
    keyword_381435 = axis_381434
    kwargs_381436 = {'axis': keyword_381435}
    # Getting the type of 'm' (line 117)
    m_381432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'm', False)
    # Obtaining the member 'sum' of a type (line 117)
    sum_381433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), m_381432, 'sum')
    # Calling sum(args, kwargs) (line 117)
    sum_call_result_381437 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), sum_381433, *[], **kwargs_381436)
    
    # Assigning a type to the variable 'w' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'w', sum_call_result_381437)
    
    # Getting the type of 'normed' (line 118)
    normed_381438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 7), 'normed')
    # Testing the type of an if condition (line 118)
    if_condition_381439 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 4), normed_381438)
    # Assigning a type to the variable 'if_condition_381439' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'if_condition_381439', if_condition_381439)
    # SSA begins for if statement (line 118)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Compare to a Name (line 119):
    
    # Assigning a Compare to a Name (line 119):
    
    # Getting the type of 'w' (line 119)
    w_381440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 30), 'w')
    int_381441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 35), 'int')
    # Applying the binary operator '==' (line 119)
    result_eq_381442 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 30), '==', w_381440, int_381441)
    
    # Assigning a type to the variable 'isolated_node_mask' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'isolated_node_mask', result_eq_381442)
    
    # Assigning a Call to a Name (line 120):
    
    # Assigning a Call to a Name (line 120):
    
    # Call to where(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'isolated_node_mask' (line 120)
    isolated_node_mask_381445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 21), 'isolated_node_mask', False)
    int_381446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 41), 'int')
    
    # Call to sqrt(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'w' (line 120)
    w_381449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 52), 'w', False)
    # Processing the call keyword arguments (line 120)
    kwargs_381450 = {}
    # Getting the type of 'np' (line 120)
    np_381447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 44), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 120)
    sqrt_381448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 44), np_381447, 'sqrt')
    # Calling sqrt(args, kwargs) (line 120)
    sqrt_call_result_381451 = invoke(stypy.reporting.localization.Localization(__file__, 120, 44), sqrt_381448, *[w_381449], **kwargs_381450)
    
    # Processing the call keyword arguments (line 120)
    kwargs_381452 = {}
    # Getting the type of 'np' (line 120)
    np_381443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'np', False)
    # Obtaining the member 'where' of a type (line 120)
    where_381444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 12), np_381443, 'where')
    # Calling where(args, kwargs) (line 120)
    where_call_result_381453 = invoke(stypy.reporting.localization.Localization(__file__, 120, 12), where_381444, *[isolated_node_mask_381445, int_381446, sqrt_call_result_381451], **kwargs_381452)
    
    # Assigning a type to the variable 'w' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'w', where_call_result_381453)
    
    # Getting the type of 'm' (line 121)
    m_381454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'm')
    # Getting the type of 'w' (line 121)
    w_381455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 13), 'w')
    # Applying the binary operator 'div=' (line 121)
    result_div_381456 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 8), 'div=', m_381454, w_381455)
    # Assigning a type to the variable 'm' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'm', result_div_381456)
    
    
    # Getting the type of 'm' (line 122)
    m_381457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'm')
    
    # Obtaining the type of the subscript
    slice_381458 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 122, 13), None, None, None)
    # Getting the type of 'np' (line 122)
    np_381459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 18), 'np')
    # Obtaining the member 'newaxis' of a type (line 122)
    newaxis_381460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 18), np_381459, 'newaxis')
    # Getting the type of 'w' (line 122)
    w_381461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 13), 'w')
    # Obtaining the member '__getitem__' of a type (line 122)
    getitem___381462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 13), w_381461, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 122)
    subscript_call_result_381463 = invoke(stypy.reporting.localization.Localization(__file__, 122, 13), getitem___381462, (slice_381458, newaxis_381460))
    
    # Applying the binary operator 'div=' (line 122)
    result_div_381464 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 8), 'div=', m_381457, subscript_call_result_381463)
    # Assigning a type to the variable 'm' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'm', result_div_381464)
    
    
    # Getting the type of 'm' (line 123)
    m_381465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'm')
    int_381466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 13), 'int')
    # Applying the binary operator '*=' (line 123)
    result_imul_381467 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 8), '*=', m_381465, int_381466)
    # Assigning a type to the variable 'm' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'm', result_imul_381467)
    
    
    # Call to _setdiag_dense(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'm' (line 124)
    m_381469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 23), 'm', False)
    int_381470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 26), 'int')
    # Getting the type of 'isolated_node_mask' (line 124)
    isolated_node_mask_381471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 30), 'isolated_node_mask', False)
    # Applying the binary operator '-' (line 124)
    result_sub_381472 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 26), '-', int_381470, isolated_node_mask_381471)
    
    # Processing the call keyword arguments (line 124)
    kwargs_381473 = {}
    # Getting the type of '_setdiag_dense' (line 124)
    _setdiag_dense_381468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), '_setdiag_dense', False)
    # Calling _setdiag_dense(args, kwargs) (line 124)
    _setdiag_dense_call_result_381474 = invoke(stypy.reporting.localization.Localization(__file__, 124, 8), _setdiag_dense_381468, *[m_381469, result_sub_381472], **kwargs_381473)
    
    # SSA branch for the else part of an if statement (line 118)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'm' (line 126)
    m_381475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'm')
    int_381476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 13), 'int')
    # Applying the binary operator '*=' (line 126)
    result_imul_381477 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 8), '*=', m_381475, int_381476)
    # Assigning a type to the variable 'm' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'm', result_imul_381477)
    
    
    # Call to _setdiag_dense(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'm' (line 127)
    m_381479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'm', False)
    # Getting the type of 'w' (line 127)
    w_381480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 26), 'w', False)
    # Processing the call keyword arguments (line 127)
    kwargs_381481 = {}
    # Getting the type of '_setdiag_dense' (line 127)
    _setdiag_dense_381478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), '_setdiag_dense', False)
    # Calling _setdiag_dense(args, kwargs) (line 127)
    _setdiag_dense_call_result_381482 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), _setdiag_dense_381478, *[m_381479, w_381480], **kwargs_381481)
    
    # SSA join for if statement (line 118)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 128)
    tuple_381483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 128)
    # Adding element type (line 128)
    # Getting the type of 'm' (line 128)
    m_381484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 11), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 11), tuple_381483, m_381484)
    # Adding element type (line 128)
    # Getting the type of 'w' (line 128)
    w_381485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 14), 'w')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 11), tuple_381483, w_381485)
    
    # Assigning a type to the variable 'stypy_return_type' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'stypy_return_type', tuple_381483)
    
    # ################# End of '_laplacian_dense(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_laplacian_dense' in the type store
    # Getting the type of 'stypy_return_type' (line 114)
    stypy_return_type_381486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_381486)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_laplacian_dense'
    return stypy_return_type_381486

# Assigning a type to the variable '_laplacian_dense' (line 114)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), '_laplacian_dense', _laplacian_dense)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
