
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from numpy.lib import add_newdoc
2: 
3: add_newdoc('scipy.sparse.linalg.dsolve._superlu', 'SuperLU',
4:     '''
5:     LU factorization of a sparse matrix.
6: 
7:     Factorization is represented as::
8: 
9:         Pr * A * Pc = L * U
10: 
11:     To construct these `SuperLU` objects, call the `splu` and `spilu`
12:     functions.
13: 
14:     Attributes
15:     ----------
16:     shape
17:     nnz
18:     perm_c
19:     perm_r
20:     L
21:     U
22: 
23:     Methods
24:     -------
25:     solve
26: 
27:     Notes
28:     -----
29: 
30:     .. versionadded:: 0.14.0
31: 
32:     Examples
33:     --------
34:     The LU decomposition can be used to solve matrix equations. Consider:
35: 
36:     >>> import numpy as np
37:     >>> from scipy.sparse import csc_matrix, linalg as sla
38:     >>> A = csc_matrix([[1,2,0,4],[1,0,0,1],[1,0,2,1],[2,2,1,0.]])
39: 
40:     This can be solved for a given right-hand side:
41: 
42:     >>> lu = sla.splu(A)
43:     >>> b = np.array([1, 2, 3, 4])
44:     >>> x = lu.solve(b)
45:     >>> A.dot(x)
46:     array([ 1.,  2.,  3.,  4.])
47: 
48:     The ``lu`` object also contains an explicit representation of the
49:     decomposition. The permutations are represented as mappings of
50:     indices:
51: 
52:     >>> lu.perm_r
53:     array([0, 2, 1, 3], dtype=int32)
54:     >>> lu.perm_c
55:     array([2, 0, 1, 3], dtype=int32)
56: 
57:     The L and U factors are sparse matrices in CSC format:
58: 
59:     >>> lu.L.A
60:     array([[ 1. ,  0. ,  0. ,  0. ],
61:            [ 0. ,  1. ,  0. ,  0. ],
62:            [ 0. ,  0. ,  1. ,  0. ],
63:            [ 1. ,  0.5,  0.5,  1. ]])
64:     >>> lu.U.A
65:     array([[ 2.,  0.,  1.,  4.],
66:            [ 0.,  2.,  1.,  1.],
67:            [ 0.,  0.,  1.,  1.],
68:            [ 0.,  0.,  0., -5.]])
69: 
70:     The permutation matrices can be constructed:
71: 
72:     >>> Pr = csc_matrix((4, 4))
73:     >>> Pr[lu.perm_r, np.arange(4)] = 1
74:     >>> Pc = csc_matrix((4, 4))
75:     >>> Pc[np.arange(4), lu.perm_c] = 1
76: 
77:     We can reassemble the original matrix:
78: 
79:     >>> (Pr.T * (lu.L * lu.U) * Pc.T).A
80:     array([[ 1.,  2.,  0.,  4.],
81:            [ 1.,  0.,  0.,  1.],
82:            [ 1.,  0.,  2.,  1.],
83:            [ 2.,  2.,  1.,  0.]])
84:     ''')
85: 
86: add_newdoc('scipy.sparse.linalg.dsolve._superlu', 'SuperLU', ('solve',
87:     '''
88:     solve(rhs[, trans])
89: 
90:     Solves linear system of equations with one or several right-hand sides.
91: 
92:     Parameters
93:     ----------
94:     rhs : ndarray, shape (n,) or (n, k)
95:         Right hand side(s) of equation
96:     trans : {'N', 'T', 'H'}, optional
97:         Type of system to solve::
98: 
99:             'N':   A   * x == rhs  (default)
100:             'T':   A^T * x == rhs
101:             'H':   A^H * x == rhs
102: 
103:         i.e., normal, transposed, and hermitian conjugate.
104: 
105:     Returns
106:     -------
107:     x : ndarray, shape ``rhs.shape``
108:         Solution vector(s)
109:     '''))
110: 
111: add_newdoc('scipy.sparse.linalg.dsolve._superlu', 'SuperLU', ('L',
112:     '''
113:     Lower triangular factor with unit diagonal as a
114:     `scipy.sparse.csc_matrix`.
115: 
116:     .. versionadded:: 0.14.0
117:     '''))
118: 
119: add_newdoc('scipy.sparse.linalg.dsolve._superlu', 'SuperLU', ('U',
120:     '''
121:     Upper triangular factor as a `scipy.sparse.csc_matrix`.
122: 
123:     .. versionadded:: 0.14.0
124:     '''))
125: 
126: add_newdoc('scipy.sparse.linalg.dsolve._superlu', 'SuperLU', ('shape',
127:     '''
128:     Shape of the original matrix as a tuple of ints.
129:     '''))
130: 
131: add_newdoc('scipy.sparse.linalg.dsolve._superlu', 'SuperLU', ('nnz',
132:     '''
133:     Number of nonzero elements in the matrix.
134:     '''))
135: 
136: add_newdoc('scipy.sparse.linalg.dsolve._superlu', 'SuperLU', ('perm_c',
137:     '''
138:     Permutation Pc represented as an array of indices.
139: 
140:     The column permutation matrix can be reconstructed via:
141: 
142:     >>> Pc = np.zeros((n, n))
143:     >>> Pc[np.arange(n), perm_c] = 1
144:     '''))
145: 
146: add_newdoc('scipy.sparse.linalg.dsolve._superlu', 'SuperLU', ('perm_r',
147:     '''
148:     Permutation Pr represented as an array of indices.
149: 
150:     The row permutation matrix can be reconstructed via:
151: 
152:     >>> Pr = np.zeros((n, n))
153:     >>> Pr[perm_r, np.arange(n)] = 1
154:     '''))
155: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from numpy.lib import add_newdoc' statement (line 1)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')
import_392843 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy.lib')

if (type(import_392843) is not StypyTypeError):

    if (import_392843 != 'pyd_module'):
        __import__(import_392843)
        sys_modules_392844 = sys.modules[import_392843]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy.lib', sys_modules_392844.module_type_store, module_type_store, ['add_newdoc'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_392844, sys_modules_392844.module_type_store, module_type_store)
    else:
        from numpy.lib import add_newdoc

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy.lib', None, module_type_store, ['add_newdoc'], [add_newdoc])

else:
    # Assigning a type to the variable 'numpy.lib' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy.lib', import_392843)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')


# Call to add_newdoc(...): (line 3)
# Processing the call arguments (line 3)
str_392846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 11), 'str', 'scipy.sparse.linalg.dsolve._superlu')
str_392847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 50), 'str', 'SuperLU')
str_392848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, (-1)), 'str', '\n    LU factorization of a sparse matrix.\n\n    Factorization is represented as::\n\n        Pr * A * Pc = L * U\n\n    To construct these `SuperLU` objects, call the `splu` and `spilu`\n    functions.\n\n    Attributes\n    ----------\n    shape\n    nnz\n    perm_c\n    perm_r\n    L\n    U\n\n    Methods\n    -------\n    solve\n\n    Notes\n    -----\n\n    .. versionadded:: 0.14.0\n\n    Examples\n    --------\n    The LU decomposition can be used to solve matrix equations. Consider:\n\n    >>> import numpy as np\n    >>> from scipy.sparse import csc_matrix, linalg as sla\n    >>> A = csc_matrix([[1,2,0,4],[1,0,0,1],[1,0,2,1],[2,2,1,0.]])\n\n    This can be solved for a given right-hand side:\n\n    >>> lu = sla.splu(A)\n    >>> b = np.array([1, 2, 3, 4])\n    >>> x = lu.solve(b)\n    >>> A.dot(x)\n    array([ 1.,  2.,  3.,  4.])\n\n    The ``lu`` object also contains an explicit representation of the\n    decomposition. The permutations are represented as mappings of\n    indices:\n\n    >>> lu.perm_r\n    array([0, 2, 1, 3], dtype=int32)\n    >>> lu.perm_c\n    array([2, 0, 1, 3], dtype=int32)\n\n    The L and U factors are sparse matrices in CSC format:\n\n    >>> lu.L.A\n    array([[ 1. ,  0. ,  0. ,  0. ],\n           [ 0. ,  1. ,  0. ,  0. ],\n           [ 0. ,  0. ,  1. ,  0. ],\n           [ 1. ,  0.5,  0.5,  1. ]])\n    >>> lu.U.A\n    array([[ 2.,  0.,  1.,  4.],\n           [ 0.,  2.,  1.,  1.],\n           [ 0.,  0.,  1.,  1.],\n           [ 0.,  0.,  0., -5.]])\n\n    The permutation matrices can be constructed:\n\n    >>> Pr = csc_matrix((4, 4))\n    >>> Pr[lu.perm_r, np.arange(4)] = 1\n    >>> Pc = csc_matrix((4, 4))\n    >>> Pc[np.arange(4), lu.perm_c] = 1\n\n    We can reassemble the original matrix:\n\n    >>> (Pr.T * (lu.L * lu.U) * Pc.T).A\n    array([[ 1.,  2.,  0.,  4.],\n           [ 1.,  0.,  0.,  1.],\n           [ 1.,  0.,  2.,  1.],\n           [ 2.,  2.,  1.,  0.]])\n    ')
# Processing the call keyword arguments (line 3)
kwargs_392849 = {}
# Getting the type of 'add_newdoc' (line 3)
add_newdoc_392845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3)
add_newdoc_call_result_392850 = invoke(stypy.reporting.localization.Localization(__file__, 3, 0), add_newdoc_392845, *[str_392846, str_392847, str_392848], **kwargs_392849)


# Call to add_newdoc(...): (line 86)
# Processing the call arguments (line 86)
str_392852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 11), 'str', 'scipy.sparse.linalg.dsolve._superlu')
str_392853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 50), 'str', 'SuperLU')

# Obtaining an instance of the builtin type 'tuple' (line 86)
tuple_392854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 62), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 86)
# Adding element type (line 86)
str_392855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 62), 'str', 'solve')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 62), tuple_392854, str_392855)
# Adding element type (line 86)
str_392856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, (-1)), 'str', "\n    solve(rhs[, trans])\n\n    Solves linear system of equations with one or several right-hand sides.\n\n    Parameters\n    ----------\n    rhs : ndarray, shape (n,) or (n, k)\n        Right hand side(s) of equation\n    trans : {'N', 'T', 'H'}, optional\n        Type of system to solve::\n\n            'N':   A   * x == rhs  (default)\n            'T':   A^T * x == rhs\n            'H':   A^H * x == rhs\n\n        i.e., normal, transposed, and hermitian conjugate.\n\n    Returns\n    -------\n    x : ndarray, shape ``rhs.shape``\n        Solution vector(s)\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 62), tuple_392854, str_392856)

# Processing the call keyword arguments (line 86)
kwargs_392857 = {}
# Getting the type of 'add_newdoc' (line 86)
add_newdoc_392851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 86)
add_newdoc_call_result_392858 = invoke(stypy.reporting.localization.Localization(__file__, 86, 0), add_newdoc_392851, *[str_392852, str_392853, tuple_392854], **kwargs_392857)


# Call to add_newdoc(...): (line 111)
# Processing the call arguments (line 111)
str_392860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 11), 'str', 'scipy.sparse.linalg.dsolve._superlu')
str_392861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 50), 'str', 'SuperLU')

# Obtaining an instance of the builtin type 'tuple' (line 111)
tuple_392862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 62), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 111)
# Adding element type (line 111)
str_392863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 62), 'str', 'L')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 62), tuple_392862, str_392863)
# Adding element type (line 111)
str_392864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, (-1)), 'str', '\n    Lower triangular factor with unit diagonal as a\n    `scipy.sparse.csc_matrix`.\n\n    .. versionadded:: 0.14.0\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 62), tuple_392862, str_392864)

# Processing the call keyword arguments (line 111)
kwargs_392865 = {}
# Getting the type of 'add_newdoc' (line 111)
add_newdoc_392859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 111)
add_newdoc_call_result_392866 = invoke(stypy.reporting.localization.Localization(__file__, 111, 0), add_newdoc_392859, *[str_392860, str_392861, tuple_392862], **kwargs_392865)


# Call to add_newdoc(...): (line 119)
# Processing the call arguments (line 119)
str_392868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 11), 'str', 'scipy.sparse.linalg.dsolve._superlu')
str_392869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 50), 'str', 'SuperLU')

# Obtaining an instance of the builtin type 'tuple' (line 119)
tuple_392870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 62), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 119)
# Adding element type (line 119)
str_392871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 62), 'str', 'U')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 62), tuple_392870, str_392871)
# Adding element type (line 119)
str_392872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, (-1)), 'str', '\n    Upper triangular factor as a `scipy.sparse.csc_matrix`.\n\n    .. versionadded:: 0.14.0\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 62), tuple_392870, str_392872)

# Processing the call keyword arguments (line 119)
kwargs_392873 = {}
# Getting the type of 'add_newdoc' (line 119)
add_newdoc_392867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 119)
add_newdoc_call_result_392874 = invoke(stypy.reporting.localization.Localization(__file__, 119, 0), add_newdoc_392867, *[str_392868, str_392869, tuple_392870], **kwargs_392873)


# Call to add_newdoc(...): (line 126)
# Processing the call arguments (line 126)
str_392876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 11), 'str', 'scipy.sparse.linalg.dsolve._superlu')
str_392877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 50), 'str', 'SuperLU')

# Obtaining an instance of the builtin type 'tuple' (line 126)
tuple_392878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 62), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 126)
# Adding element type (line 126)
str_392879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 62), 'str', 'shape')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 62), tuple_392878, str_392879)
# Adding element type (line 126)
str_392880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, (-1)), 'str', '\n    Shape of the original matrix as a tuple of ints.\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 62), tuple_392878, str_392880)

# Processing the call keyword arguments (line 126)
kwargs_392881 = {}
# Getting the type of 'add_newdoc' (line 126)
add_newdoc_392875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 126)
add_newdoc_call_result_392882 = invoke(stypy.reporting.localization.Localization(__file__, 126, 0), add_newdoc_392875, *[str_392876, str_392877, tuple_392878], **kwargs_392881)


# Call to add_newdoc(...): (line 131)
# Processing the call arguments (line 131)
str_392884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 11), 'str', 'scipy.sparse.linalg.dsolve._superlu')
str_392885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 50), 'str', 'SuperLU')

# Obtaining an instance of the builtin type 'tuple' (line 131)
tuple_392886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 62), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 131)
# Adding element type (line 131)
str_392887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 62), 'str', 'nnz')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 62), tuple_392886, str_392887)
# Adding element type (line 131)
str_392888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, (-1)), 'str', '\n    Number of nonzero elements in the matrix.\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 62), tuple_392886, str_392888)

# Processing the call keyword arguments (line 131)
kwargs_392889 = {}
# Getting the type of 'add_newdoc' (line 131)
add_newdoc_392883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 131)
add_newdoc_call_result_392890 = invoke(stypy.reporting.localization.Localization(__file__, 131, 0), add_newdoc_392883, *[str_392884, str_392885, tuple_392886], **kwargs_392889)


# Call to add_newdoc(...): (line 136)
# Processing the call arguments (line 136)
str_392892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 11), 'str', 'scipy.sparse.linalg.dsolve._superlu')
str_392893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 50), 'str', 'SuperLU')

# Obtaining an instance of the builtin type 'tuple' (line 136)
tuple_392894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 62), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 136)
# Adding element type (line 136)
str_392895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 62), 'str', 'perm_c')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 62), tuple_392894, str_392895)
# Adding element type (line 136)
str_392896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, (-1)), 'str', '\n    Permutation Pc represented as an array of indices.\n\n    The column permutation matrix can be reconstructed via:\n\n    >>> Pc = np.zeros((n, n))\n    >>> Pc[np.arange(n), perm_c] = 1\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 62), tuple_392894, str_392896)

# Processing the call keyword arguments (line 136)
kwargs_392897 = {}
# Getting the type of 'add_newdoc' (line 136)
add_newdoc_392891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 136)
add_newdoc_call_result_392898 = invoke(stypy.reporting.localization.Localization(__file__, 136, 0), add_newdoc_392891, *[str_392892, str_392893, tuple_392894], **kwargs_392897)


# Call to add_newdoc(...): (line 146)
# Processing the call arguments (line 146)
str_392900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 11), 'str', 'scipy.sparse.linalg.dsolve._superlu')
str_392901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 50), 'str', 'SuperLU')

# Obtaining an instance of the builtin type 'tuple' (line 146)
tuple_392902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 62), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 146)
# Adding element type (line 146)
str_392903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 62), 'str', 'perm_r')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 62), tuple_392902, str_392903)
# Adding element type (line 146)
str_392904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, (-1)), 'str', '\n    Permutation Pr represented as an array of indices.\n\n    The row permutation matrix can be reconstructed via:\n\n    >>> Pr = np.zeros((n, n))\n    >>> Pr[perm_r, np.arange(n)] = 1\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 62), tuple_392902, str_392904)

# Processing the call keyword arguments (line 146)
kwargs_392905 = {}
# Getting the type of 'add_newdoc' (line 146)
add_newdoc_392899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 146)
add_newdoc_call_result_392906 = invoke(stypy.reporting.localization.Localization(__file__, 146, 0), add_newdoc_392899, *[str_392900, str_392901, tuple_392902], **kwargs_392905)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
