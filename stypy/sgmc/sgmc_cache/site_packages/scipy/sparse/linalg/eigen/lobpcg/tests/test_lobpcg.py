
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Test functions for the sparse.linalg.eigen.lobpcg module
2: '''
3: from __future__ import division, print_function, absolute_import
4: 
5: import numpy as np
6: from numpy.testing import (assert_almost_equal, assert_equal,
7:         assert_allclose, assert_array_less, assert_)
8: 
9: from scipy import ones, rand, r_, diag, linalg, eye
10: from scipy.linalg import eig, eigh, toeplitz
11: import scipy.sparse
12: from scipy.sparse.linalg.eigen.lobpcg import lobpcg
13: 
14: 
15: def ElasticRod(n):
16:     # Fixed-free elastic rod
17:     L = 1.0
18:     le = L/n
19:     rho = 7.85e3
20:     S = 1.e-4
21:     E = 2.1e11
22:     mass = rho*S*le/6.
23:     k = E*S/le
24:     A = k*(diag(r_[2.*ones(n-1),1])-diag(ones(n-1),1)-diag(ones(n-1),-1))
25:     B = mass*(diag(r_[4.*ones(n-1),2])+diag(ones(n-1),1)+diag(ones(n-1),-1))
26:     return A,B
27: 
28: 
29: def MikotaPair(n):
30:     # Mikota pair acts as a nice test since the eigenvalues
31:     # are the squares of the integers n, n=1,2,...
32:     x = np.arange(1,n+1)
33:     B = diag(1./x)
34:     y = np.arange(n-1,0,-1)
35:     z = np.arange(2*n-1,0,-2)
36:     A = diag(z)-diag(y,-1)-diag(y,1)
37:     return A,B
38: 
39: 
40: def compare_solutions(A,B,m):
41:     n = A.shape[0]
42: 
43:     np.random.seed(0)
44: 
45:     V = rand(n,m)
46:     X = linalg.orth(V)
47: 
48:     eigs,vecs = lobpcg(A, X, B=B, tol=1e-5, maxiter=30)
49:     eigs.sort()
50: 
51:     w,v = eig(A,b=B)
52:     w.sort()
53: 
54:     assert_almost_equal(w[:int(m/2)],eigs[:int(m/2)],decimal=2)
55: 
56: 
57: def test_Small():
58:     A,B = ElasticRod(10)
59:     compare_solutions(A,B,10)
60:     A,B = MikotaPair(10)
61:     compare_solutions(A,B,10)
62: 
63: 
64: def test_ElasticRod():
65:     A,B = ElasticRod(100)
66:     compare_solutions(A,B,20)
67: 
68: 
69: def test_MikotaPair():
70:     A,B = MikotaPair(100)
71:     compare_solutions(A,B,20)
72: 
73: 
74: def test_trivial():
75:     n = 5
76:     X = ones((n, 1))
77:     A = eye(n)
78:     compare_solutions(A, None, n)
79: 
80: 
81: def test_regression():
82:     # https://mail.python.org/pipermail/scipy-user/2010-October/026944.html
83:     n = 10
84:     X = np.ones((n, 1))
85:     A = np.identity(n)
86:     w, V = lobpcg(A, X)
87:     assert_allclose(w, [1])
88: 
89: 
90: def test_diagonal():
91:     # This test was moved from '__main__' in lobpcg.py.
92:     # Coincidentally or not, this is the same eigensystem
93:     # required to reproduce arpack bug
94:     # http://forge.scilab.org/index.php/p/arpack-ng/issues/1397/
95:     # even using the same n=100.
96: 
97:     np.random.seed(1234)
98: 
99:     # The system of interest is of size n x n.
100:     n = 100
101: 
102:     # We care about only m eigenpairs.
103:     m = 4
104: 
105:     # Define the generalized eigenvalue problem Av = cBv
106:     # where (c, v) is a generalized eigenpair,
107:     # and where we choose A to be the diagonal matrix whose entries are 1..n
108:     # and where B is chosen to be the identity matrix.
109:     vals = np.arange(1, n+1, dtype=float)
110:     A = scipy.sparse.diags([vals], [0], (n, n))
111:     B = scipy.sparse.eye(n)
112: 
113:     # Let the preconditioner M be the inverse of A.
114:     M = scipy.sparse.diags([np.reciprocal(vals)], [0], (n, n))
115: 
116:     # Pick random initial vectors.
117:     X = np.random.rand(n, m)
118: 
119:     # Require that the returned eigenvectors be in the orthogonal complement
120:     # of the first few standard basis vectors.
121:     m_excluded = 3
122:     Y = np.eye(n, m_excluded)
123: 
124:     eigs, vecs = lobpcg(A, X, B, M=M, Y=Y, tol=1e-4, maxiter=40, largest=False)
125: 
126:     assert_allclose(eigs, np.arange(1+m_excluded, 1+m_excluded+m))
127:     _check_eigen(A, eigs, vecs, rtol=1e-3, atol=1e-3)
128: 
129: 
130: def _check_eigen(M, w, V, rtol=1e-8, atol=1e-14):
131:     mult_wV = np.multiply(w, V)
132:     dot_MV = M.dot(V)
133:     assert_allclose(mult_wV, dot_MV, rtol=rtol, atol=atol)
134: 
135: 
136: def _check_fiedler(n, p):
137:     # This is not necessarily the recommended way to find the Fiedler vector.
138:     np.random.seed(1234)
139:     col = np.zeros(n)
140:     col[1] = 1
141:     A = toeplitz(col)
142:     D = np.diag(A.sum(axis=1))
143:     L = D - A
144:     # Compute the full eigendecomposition using tricks, e.g.
145:     # http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
146:     tmp = np.pi * np.arange(n) / n
147:     analytic_w = 2 * (1 - np.cos(tmp))
148:     analytic_V = np.cos(np.outer(np.arange(n) + 1/2, tmp))
149:     _check_eigen(L, analytic_w, analytic_V)
150:     # Compute the full eigendecomposition using eigh.
151:     eigh_w, eigh_V = eigh(L)
152:     _check_eigen(L, eigh_w, eigh_V)
153:     # Check that the first eigenvalue is near zero and that the rest agree.
154:     assert_array_less(np.abs([eigh_w[0], analytic_w[0]]), 1e-14)
155:     assert_allclose(eigh_w[1:], analytic_w[1:])
156: 
157:     # Check small lobpcg eigenvalues.
158:     X = analytic_V[:, :p]
159:     lobpcg_w, lobpcg_V = lobpcg(L, X, largest=False)
160:     assert_equal(lobpcg_w.shape, (p,))
161:     assert_equal(lobpcg_V.shape, (n, p))
162:     _check_eigen(L, lobpcg_w, lobpcg_V)
163:     assert_array_less(np.abs(np.min(lobpcg_w)), 1e-14)
164:     assert_allclose(np.sort(lobpcg_w)[1:], analytic_w[1:p])
165: 
166:     # Check large lobpcg eigenvalues.
167:     X = analytic_V[:, -p:]
168:     lobpcg_w, lobpcg_V = lobpcg(L, X, largest=True)
169:     assert_equal(lobpcg_w.shape, (p,))
170:     assert_equal(lobpcg_V.shape, (n, p))
171:     _check_eigen(L, lobpcg_w, lobpcg_V)
172:     assert_allclose(np.sort(lobpcg_w), analytic_w[-p:])
173: 
174:     # Look for the Fiedler vector using good but not exactly correct guesses.
175:     fiedler_guess = np.concatenate((np.ones(n//2), -np.ones(n-n//2)))
176:     X = np.vstack((np.ones(n), fiedler_guess)).T
177:     lobpcg_w, lobpcg_V = lobpcg(L, X, largest=False)
178:     # Mathematically, the smaller eigenvalue should be zero
179:     # and the larger should be the algebraic connectivity.
180:     lobpcg_w = np.sort(lobpcg_w)
181:     assert_allclose(lobpcg_w, analytic_w[:2], atol=1e-14)
182: 
183: 
184: def test_fiedler_small_8():
185:     # This triggers the dense path because 8 < 2*5.
186:     _check_fiedler(8, 2)
187: 
188: 
189: def test_fiedler_large_12():
190:     # This does not trigger the dense path, because 2*5 <= 12.
191:     _check_fiedler(12, 2)
192: 
193: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_406954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', ' Test functions for the sparse.linalg.eigen.lobpcg module\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/tests/')
import_406955 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_406955) is not StypyTypeError):

    if (import_406955 != 'pyd_module'):
        __import__(import_406955)
        sys_modules_406956 = sys.modules[import_406955]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_406956.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_406955)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.testing import assert_almost_equal, assert_equal, assert_allclose, assert_array_less, assert_' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/tests/')
import_406957 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing')

if (type(import_406957) is not StypyTypeError):

    if (import_406957 != 'pyd_module'):
        __import__(import_406957)
        sys_modules_406958 = sys.modules[import_406957]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', sys_modules_406958.module_type_store, module_type_store, ['assert_almost_equal', 'assert_equal', 'assert_allclose', 'assert_array_less', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_406958, sys_modules_406958.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_almost_equal, assert_equal, assert_allclose, assert_array_less, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', None, module_type_store, ['assert_almost_equal', 'assert_equal', 'assert_allclose', 'assert_array_less', 'assert_'], [assert_almost_equal, assert_equal, assert_allclose, assert_array_less, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', import_406957)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy import ones, rand, r_, diag, linalg, eye' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/tests/')
import_406959 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy')

if (type(import_406959) is not StypyTypeError):

    if (import_406959 != 'pyd_module'):
        __import__(import_406959)
        sys_modules_406960 = sys.modules[import_406959]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy', sys_modules_406960.module_type_store, module_type_store, ['ones', 'rand', 'r_', 'diag', 'linalg', 'eye'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_406960, sys_modules_406960.module_type_store, module_type_store)
    else:
        from scipy import ones, rand, r_, diag, linalg, eye

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy', None, module_type_store, ['ones', 'rand', 'r_', 'diag', 'linalg', 'eye'], [ones, rand, r_, diag, linalg, eye])

else:
    # Assigning a type to the variable 'scipy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy', import_406959)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.linalg import eig, eigh, toeplitz' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/tests/')
import_406961 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.linalg')

if (type(import_406961) is not StypyTypeError):

    if (import_406961 != 'pyd_module'):
        __import__(import_406961)
        sys_modules_406962 = sys.modules[import_406961]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.linalg', sys_modules_406962.module_type_store, module_type_store, ['eig', 'eigh', 'toeplitz'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_406962, sys_modules_406962.module_type_store, module_type_store)
    else:
        from scipy.linalg import eig, eigh, toeplitz

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.linalg', None, module_type_store, ['eig', 'eigh', 'toeplitz'], [eig, eigh, toeplitz])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.linalg', import_406961)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import scipy.sparse' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/tests/')
import_406963 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse')

if (type(import_406963) is not StypyTypeError):

    if (import_406963 != 'pyd_module'):
        __import__(import_406963)
        sys_modules_406964 = sys.modules[import_406963]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse', sys_modules_406964.module_type_store, module_type_store)
    else:
        import scipy.sparse

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse', scipy.sparse, module_type_store)

else:
    # Assigning a type to the variable 'scipy.sparse' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse', import_406963)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.sparse.linalg.eigen.lobpcg import lobpcg' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/tests/')
import_406965 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse.linalg.eigen.lobpcg')

if (type(import_406965) is not StypyTypeError):

    if (import_406965 != 'pyd_module'):
        __import__(import_406965)
        sys_modules_406966 = sys.modules[import_406965]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse.linalg.eigen.lobpcg', sys_modules_406966.module_type_store, module_type_store, ['lobpcg'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_406966, sys_modules_406966.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.eigen.lobpcg import lobpcg

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse.linalg.eigen.lobpcg', None, module_type_store, ['lobpcg'], [lobpcg])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.eigen.lobpcg' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse.linalg.eigen.lobpcg', import_406965)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/lobpcg/tests/')


@norecursion
def ElasticRod(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ElasticRod'
    module_type_store = module_type_store.open_function_context('ElasticRod', 15, 0, False)
    
    # Passed parameters checking function
    ElasticRod.stypy_localization = localization
    ElasticRod.stypy_type_of_self = None
    ElasticRod.stypy_type_store = module_type_store
    ElasticRod.stypy_function_name = 'ElasticRod'
    ElasticRod.stypy_param_names_list = ['n']
    ElasticRod.stypy_varargs_param_name = None
    ElasticRod.stypy_kwargs_param_name = None
    ElasticRod.stypy_call_defaults = defaults
    ElasticRod.stypy_call_varargs = varargs
    ElasticRod.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ElasticRod', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ElasticRod', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ElasticRod(...)' code ##################

    
    # Assigning a Num to a Name (line 17):
    
    # Assigning a Num to a Name (line 17):
    float_406967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 8), 'float')
    # Assigning a type to the variable 'L' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'L', float_406967)
    
    # Assigning a BinOp to a Name (line 18):
    
    # Assigning a BinOp to a Name (line 18):
    # Getting the type of 'L' (line 18)
    L_406968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 9), 'L')
    # Getting the type of 'n' (line 18)
    n_406969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'n')
    # Applying the binary operator 'div' (line 18)
    result_div_406970 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 9), 'div', L_406968, n_406969)
    
    # Assigning a type to the variable 'le' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'le', result_div_406970)
    
    # Assigning a Num to a Name (line 19):
    
    # Assigning a Num to a Name (line 19):
    float_406971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 10), 'float')
    # Assigning a type to the variable 'rho' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'rho', float_406971)
    
    # Assigning a Num to a Name (line 20):
    
    # Assigning a Num to a Name (line 20):
    float_406972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 8), 'float')
    # Assigning a type to the variable 'S' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'S', float_406972)
    
    # Assigning a Num to a Name (line 21):
    
    # Assigning a Num to a Name (line 21):
    float_406973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 8), 'float')
    # Assigning a type to the variable 'E' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'E', float_406973)
    
    # Assigning a BinOp to a Name (line 22):
    
    # Assigning a BinOp to a Name (line 22):
    # Getting the type of 'rho' (line 22)
    rho_406974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'rho')
    # Getting the type of 'S' (line 22)
    S_406975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 15), 'S')
    # Applying the binary operator '*' (line 22)
    result_mul_406976 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 11), '*', rho_406974, S_406975)
    
    # Getting the type of 'le' (line 22)
    le_406977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'le')
    # Applying the binary operator '*' (line 22)
    result_mul_406978 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 16), '*', result_mul_406976, le_406977)
    
    float_406979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 20), 'float')
    # Applying the binary operator 'div' (line 22)
    result_div_406980 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 19), 'div', result_mul_406978, float_406979)
    
    # Assigning a type to the variable 'mass' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'mass', result_div_406980)
    
    # Assigning a BinOp to a Name (line 23):
    
    # Assigning a BinOp to a Name (line 23):
    # Getting the type of 'E' (line 23)
    E_406981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'E')
    # Getting the type of 'S' (line 23)
    S_406982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'S')
    # Applying the binary operator '*' (line 23)
    result_mul_406983 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 8), '*', E_406981, S_406982)
    
    # Getting the type of 'le' (line 23)
    le_406984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'le')
    # Applying the binary operator 'div' (line 23)
    result_div_406985 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 11), 'div', result_mul_406983, le_406984)
    
    # Assigning a type to the variable 'k' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'k', result_div_406985)
    
    # Assigning a BinOp to a Name (line 24):
    
    # Assigning a BinOp to a Name (line 24):
    # Getting the type of 'k' (line 24)
    k_406986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'k')
    
    # Call to diag(...): (line 24)
    # Processing the call arguments (line 24)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 24)
    tuple_406988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 24)
    # Adding element type (line 24)
    float_406989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 19), 'float')
    
    # Call to ones(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'n' (line 24)
    n_406991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 27), 'n', False)
    int_406992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 29), 'int')
    # Applying the binary operator '-' (line 24)
    result_sub_406993 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 27), '-', n_406991, int_406992)
    
    # Processing the call keyword arguments (line 24)
    kwargs_406994 = {}
    # Getting the type of 'ones' (line 24)
    ones_406990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 22), 'ones', False)
    # Calling ones(args, kwargs) (line 24)
    ones_call_result_406995 = invoke(stypy.reporting.localization.Localization(__file__, 24, 22), ones_406990, *[result_sub_406993], **kwargs_406994)
    
    # Applying the binary operator '*' (line 24)
    result_mul_406996 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 19), '*', float_406989, ones_call_result_406995)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), tuple_406988, result_mul_406996)
    # Adding element type (line 24)
    int_406997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), tuple_406988, int_406997)
    
    # Getting the type of 'r_' (line 24)
    r__406998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'r_', False)
    # Obtaining the member '__getitem__' of a type (line 24)
    getitem___406999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 16), r__406998, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 24)
    subscript_call_result_407000 = invoke(stypy.reporting.localization.Localization(__file__, 24, 16), getitem___406999, tuple_406988)
    
    # Processing the call keyword arguments (line 24)
    kwargs_407001 = {}
    # Getting the type of 'diag' (line 24)
    diag_406987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 11), 'diag', False)
    # Calling diag(args, kwargs) (line 24)
    diag_call_result_407002 = invoke(stypy.reporting.localization.Localization(__file__, 24, 11), diag_406987, *[subscript_call_result_407000], **kwargs_407001)
    
    
    # Call to diag(...): (line 24)
    # Processing the call arguments (line 24)
    
    # Call to ones(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'n' (line 24)
    n_407005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 46), 'n', False)
    int_407006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 48), 'int')
    # Applying the binary operator '-' (line 24)
    result_sub_407007 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 46), '-', n_407005, int_407006)
    
    # Processing the call keyword arguments (line 24)
    kwargs_407008 = {}
    # Getting the type of 'ones' (line 24)
    ones_407004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 41), 'ones', False)
    # Calling ones(args, kwargs) (line 24)
    ones_call_result_407009 = invoke(stypy.reporting.localization.Localization(__file__, 24, 41), ones_407004, *[result_sub_407007], **kwargs_407008)
    
    int_407010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 51), 'int')
    # Processing the call keyword arguments (line 24)
    kwargs_407011 = {}
    # Getting the type of 'diag' (line 24)
    diag_407003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 36), 'diag', False)
    # Calling diag(args, kwargs) (line 24)
    diag_call_result_407012 = invoke(stypy.reporting.localization.Localization(__file__, 24, 36), diag_407003, *[ones_call_result_407009, int_407010], **kwargs_407011)
    
    # Applying the binary operator '-' (line 24)
    result_sub_407013 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 11), '-', diag_call_result_407002, diag_call_result_407012)
    
    
    # Call to diag(...): (line 24)
    # Processing the call arguments (line 24)
    
    # Call to ones(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'n' (line 24)
    n_407016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 64), 'n', False)
    int_407017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 66), 'int')
    # Applying the binary operator '-' (line 24)
    result_sub_407018 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 64), '-', n_407016, int_407017)
    
    # Processing the call keyword arguments (line 24)
    kwargs_407019 = {}
    # Getting the type of 'ones' (line 24)
    ones_407015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 59), 'ones', False)
    # Calling ones(args, kwargs) (line 24)
    ones_call_result_407020 = invoke(stypy.reporting.localization.Localization(__file__, 24, 59), ones_407015, *[result_sub_407018], **kwargs_407019)
    
    int_407021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 69), 'int')
    # Processing the call keyword arguments (line 24)
    kwargs_407022 = {}
    # Getting the type of 'diag' (line 24)
    diag_407014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 54), 'diag', False)
    # Calling diag(args, kwargs) (line 24)
    diag_call_result_407023 = invoke(stypy.reporting.localization.Localization(__file__, 24, 54), diag_407014, *[ones_call_result_407020, int_407021], **kwargs_407022)
    
    # Applying the binary operator '-' (line 24)
    result_sub_407024 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 53), '-', result_sub_407013, diag_call_result_407023)
    
    # Applying the binary operator '*' (line 24)
    result_mul_407025 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 8), '*', k_406986, result_sub_407024)
    
    # Assigning a type to the variable 'A' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'A', result_mul_407025)
    
    # Assigning a BinOp to a Name (line 25):
    
    # Assigning a BinOp to a Name (line 25):
    # Getting the type of 'mass' (line 25)
    mass_407026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'mass')
    
    # Call to diag(...): (line 25)
    # Processing the call arguments (line 25)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 25)
    tuple_407028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 25)
    # Adding element type (line 25)
    float_407029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 22), 'float')
    
    # Call to ones(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'n' (line 25)
    n_407031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 30), 'n', False)
    int_407032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 32), 'int')
    # Applying the binary operator '-' (line 25)
    result_sub_407033 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 30), '-', n_407031, int_407032)
    
    # Processing the call keyword arguments (line 25)
    kwargs_407034 = {}
    # Getting the type of 'ones' (line 25)
    ones_407030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 25), 'ones', False)
    # Calling ones(args, kwargs) (line 25)
    ones_call_result_407035 = invoke(stypy.reporting.localization.Localization(__file__, 25, 25), ones_407030, *[result_sub_407033], **kwargs_407034)
    
    # Applying the binary operator '*' (line 25)
    result_mul_407036 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 22), '*', float_407029, ones_call_result_407035)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 22), tuple_407028, result_mul_407036)
    # Adding element type (line 25)
    int_407037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 22), tuple_407028, int_407037)
    
    # Getting the type of 'r_' (line 25)
    r__407038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 19), 'r_', False)
    # Obtaining the member '__getitem__' of a type (line 25)
    getitem___407039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 19), r__407038, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 25)
    subscript_call_result_407040 = invoke(stypy.reporting.localization.Localization(__file__, 25, 19), getitem___407039, tuple_407028)
    
    # Processing the call keyword arguments (line 25)
    kwargs_407041 = {}
    # Getting the type of 'diag' (line 25)
    diag_407027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 14), 'diag', False)
    # Calling diag(args, kwargs) (line 25)
    diag_call_result_407042 = invoke(stypy.reporting.localization.Localization(__file__, 25, 14), diag_407027, *[subscript_call_result_407040], **kwargs_407041)
    
    
    # Call to diag(...): (line 25)
    # Processing the call arguments (line 25)
    
    # Call to ones(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'n' (line 25)
    n_407045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 49), 'n', False)
    int_407046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 51), 'int')
    # Applying the binary operator '-' (line 25)
    result_sub_407047 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 49), '-', n_407045, int_407046)
    
    # Processing the call keyword arguments (line 25)
    kwargs_407048 = {}
    # Getting the type of 'ones' (line 25)
    ones_407044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 44), 'ones', False)
    # Calling ones(args, kwargs) (line 25)
    ones_call_result_407049 = invoke(stypy.reporting.localization.Localization(__file__, 25, 44), ones_407044, *[result_sub_407047], **kwargs_407048)
    
    int_407050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 54), 'int')
    # Processing the call keyword arguments (line 25)
    kwargs_407051 = {}
    # Getting the type of 'diag' (line 25)
    diag_407043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 39), 'diag', False)
    # Calling diag(args, kwargs) (line 25)
    diag_call_result_407052 = invoke(stypy.reporting.localization.Localization(__file__, 25, 39), diag_407043, *[ones_call_result_407049, int_407050], **kwargs_407051)
    
    # Applying the binary operator '+' (line 25)
    result_add_407053 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 14), '+', diag_call_result_407042, diag_call_result_407052)
    
    
    # Call to diag(...): (line 25)
    # Processing the call arguments (line 25)
    
    # Call to ones(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'n' (line 25)
    n_407056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 67), 'n', False)
    int_407057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 69), 'int')
    # Applying the binary operator '-' (line 25)
    result_sub_407058 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 67), '-', n_407056, int_407057)
    
    # Processing the call keyword arguments (line 25)
    kwargs_407059 = {}
    # Getting the type of 'ones' (line 25)
    ones_407055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 62), 'ones', False)
    # Calling ones(args, kwargs) (line 25)
    ones_call_result_407060 = invoke(stypy.reporting.localization.Localization(__file__, 25, 62), ones_407055, *[result_sub_407058], **kwargs_407059)
    
    int_407061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 72), 'int')
    # Processing the call keyword arguments (line 25)
    kwargs_407062 = {}
    # Getting the type of 'diag' (line 25)
    diag_407054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 57), 'diag', False)
    # Calling diag(args, kwargs) (line 25)
    diag_call_result_407063 = invoke(stypy.reporting.localization.Localization(__file__, 25, 57), diag_407054, *[ones_call_result_407060, int_407061], **kwargs_407062)
    
    # Applying the binary operator '+' (line 25)
    result_add_407064 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 56), '+', result_add_407053, diag_call_result_407063)
    
    # Applying the binary operator '*' (line 25)
    result_mul_407065 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 8), '*', mass_407026, result_add_407064)
    
    # Assigning a type to the variable 'B' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'B', result_mul_407065)
    
    # Obtaining an instance of the builtin type 'tuple' (line 26)
    tuple_407066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 26)
    # Adding element type (line 26)
    # Getting the type of 'A' (line 26)
    A_407067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'A')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 11), tuple_407066, A_407067)
    # Adding element type (line 26)
    # Getting the type of 'B' (line 26)
    B_407068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 13), 'B')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 11), tuple_407066, B_407068)
    
    # Assigning a type to the variable 'stypy_return_type' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type', tuple_407066)
    
    # ################# End of 'ElasticRod(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ElasticRod' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_407069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_407069)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ElasticRod'
    return stypy_return_type_407069

# Assigning a type to the variable 'ElasticRod' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'ElasticRod', ElasticRod)

@norecursion
def MikotaPair(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'MikotaPair'
    module_type_store = module_type_store.open_function_context('MikotaPair', 29, 0, False)
    
    # Passed parameters checking function
    MikotaPair.stypy_localization = localization
    MikotaPair.stypy_type_of_self = None
    MikotaPair.stypy_type_store = module_type_store
    MikotaPair.stypy_function_name = 'MikotaPair'
    MikotaPair.stypy_param_names_list = ['n']
    MikotaPair.stypy_varargs_param_name = None
    MikotaPair.stypy_kwargs_param_name = None
    MikotaPair.stypy_call_defaults = defaults
    MikotaPair.stypy_call_varargs = varargs
    MikotaPair.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'MikotaPair', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'MikotaPair', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'MikotaPair(...)' code ##################

    
    # Assigning a Call to a Name (line 32):
    
    # Assigning a Call to a Name (line 32):
    
    # Call to arange(...): (line 32)
    # Processing the call arguments (line 32)
    int_407072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 18), 'int')
    # Getting the type of 'n' (line 32)
    n_407073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 20), 'n', False)
    int_407074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 22), 'int')
    # Applying the binary operator '+' (line 32)
    result_add_407075 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 20), '+', n_407073, int_407074)
    
    # Processing the call keyword arguments (line 32)
    kwargs_407076 = {}
    # Getting the type of 'np' (line 32)
    np_407070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 32)
    arange_407071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), np_407070, 'arange')
    # Calling arange(args, kwargs) (line 32)
    arange_call_result_407077 = invoke(stypy.reporting.localization.Localization(__file__, 32, 8), arange_407071, *[int_407072, result_add_407075], **kwargs_407076)
    
    # Assigning a type to the variable 'x' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'x', arange_call_result_407077)
    
    # Assigning a Call to a Name (line 33):
    
    # Assigning a Call to a Name (line 33):
    
    # Call to diag(...): (line 33)
    # Processing the call arguments (line 33)
    float_407079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 13), 'float')
    # Getting the type of 'x' (line 33)
    x_407080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 16), 'x', False)
    # Applying the binary operator 'div' (line 33)
    result_div_407081 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 13), 'div', float_407079, x_407080)
    
    # Processing the call keyword arguments (line 33)
    kwargs_407082 = {}
    # Getting the type of 'diag' (line 33)
    diag_407078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'diag', False)
    # Calling diag(args, kwargs) (line 33)
    diag_call_result_407083 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), diag_407078, *[result_div_407081], **kwargs_407082)
    
    # Assigning a type to the variable 'B' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'B', diag_call_result_407083)
    
    # Assigning a Call to a Name (line 34):
    
    # Assigning a Call to a Name (line 34):
    
    # Call to arange(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'n' (line 34)
    n_407086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 18), 'n', False)
    int_407087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 20), 'int')
    # Applying the binary operator '-' (line 34)
    result_sub_407088 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 18), '-', n_407086, int_407087)
    
    int_407089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 22), 'int')
    int_407090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 24), 'int')
    # Processing the call keyword arguments (line 34)
    kwargs_407091 = {}
    # Getting the type of 'np' (line 34)
    np_407084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 34)
    arange_407085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), np_407084, 'arange')
    # Calling arange(args, kwargs) (line 34)
    arange_call_result_407092 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), arange_407085, *[result_sub_407088, int_407089, int_407090], **kwargs_407091)
    
    # Assigning a type to the variable 'y' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'y', arange_call_result_407092)
    
    # Assigning a Call to a Name (line 35):
    
    # Assigning a Call to a Name (line 35):
    
    # Call to arange(...): (line 35)
    # Processing the call arguments (line 35)
    int_407095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 18), 'int')
    # Getting the type of 'n' (line 35)
    n_407096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 20), 'n', False)
    # Applying the binary operator '*' (line 35)
    result_mul_407097 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 18), '*', int_407095, n_407096)
    
    int_407098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 22), 'int')
    # Applying the binary operator '-' (line 35)
    result_sub_407099 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 18), '-', result_mul_407097, int_407098)
    
    int_407100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 24), 'int')
    int_407101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 26), 'int')
    # Processing the call keyword arguments (line 35)
    kwargs_407102 = {}
    # Getting the type of 'np' (line 35)
    np_407093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 35)
    arange_407094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), np_407093, 'arange')
    # Calling arange(args, kwargs) (line 35)
    arange_call_result_407103 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), arange_407094, *[result_sub_407099, int_407100, int_407101], **kwargs_407102)
    
    # Assigning a type to the variable 'z' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'z', arange_call_result_407103)
    
    # Assigning a BinOp to a Name (line 36):
    
    # Assigning a BinOp to a Name (line 36):
    
    # Call to diag(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'z' (line 36)
    z_407105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 13), 'z', False)
    # Processing the call keyword arguments (line 36)
    kwargs_407106 = {}
    # Getting the type of 'diag' (line 36)
    diag_407104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'diag', False)
    # Calling diag(args, kwargs) (line 36)
    diag_call_result_407107 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), diag_407104, *[z_407105], **kwargs_407106)
    
    
    # Call to diag(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'y' (line 36)
    y_407109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 21), 'y', False)
    int_407110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 23), 'int')
    # Processing the call keyword arguments (line 36)
    kwargs_407111 = {}
    # Getting the type of 'diag' (line 36)
    diag_407108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'diag', False)
    # Calling diag(args, kwargs) (line 36)
    diag_call_result_407112 = invoke(stypy.reporting.localization.Localization(__file__, 36, 16), diag_407108, *[y_407109, int_407110], **kwargs_407111)
    
    # Applying the binary operator '-' (line 36)
    result_sub_407113 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 8), '-', diag_call_result_407107, diag_call_result_407112)
    
    
    # Call to diag(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'y' (line 36)
    y_407115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 32), 'y', False)
    int_407116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 34), 'int')
    # Processing the call keyword arguments (line 36)
    kwargs_407117 = {}
    # Getting the type of 'diag' (line 36)
    diag_407114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 27), 'diag', False)
    # Calling diag(args, kwargs) (line 36)
    diag_call_result_407118 = invoke(stypy.reporting.localization.Localization(__file__, 36, 27), diag_407114, *[y_407115, int_407116], **kwargs_407117)
    
    # Applying the binary operator '-' (line 36)
    result_sub_407119 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 26), '-', result_sub_407113, diag_call_result_407118)
    
    # Assigning a type to the variable 'A' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'A', result_sub_407119)
    
    # Obtaining an instance of the builtin type 'tuple' (line 37)
    tuple_407120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 37)
    # Adding element type (line 37)
    # Getting the type of 'A' (line 37)
    A_407121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 11), 'A')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 11), tuple_407120, A_407121)
    # Adding element type (line 37)
    # Getting the type of 'B' (line 37)
    B_407122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 13), 'B')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 11), tuple_407120, B_407122)
    
    # Assigning a type to the variable 'stypy_return_type' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type', tuple_407120)
    
    # ################# End of 'MikotaPair(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'MikotaPair' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_407123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_407123)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'MikotaPair'
    return stypy_return_type_407123

# Assigning a type to the variable 'MikotaPair' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'MikotaPair', MikotaPair)

@norecursion
def compare_solutions(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'compare_solutions'
    module_type_store = module_type_store.open_function_context('compare_solutions', 40, 0, False)
    
    # Passed parameters checking function
    compare_solutions.stypy_localization = localization
    compare_solutions.stypy_type_of_self = None
    compare_solutions.stypy_type_store = module_type_store
    compare_solutions.stypy_function_name = 'compare_solutions'
    compare_solutions.stypy_param_names_list = ['A', 'B', 'm']
    compare_solutions.stypy_varargs_param_name = None
    compare_solutions.stypy_kwargs_param_name = None
    compare_solutions.stypy_call_defaults = defaults
    compare_solutions.stypy_call_varargs = varargs
    compare_solutions.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'compare_solutions', ['A', 'B', 'm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'compare_solutions', localization, ['A', 'B', 'm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'compare_solutions(...)' code ##################

    
    # Assigning a Subscript to a Name (line 41):
    
    # Assigning a Subscript to a Name (line 41):
    
    # Obtaining the type of the subscript
    int_407124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 16), 'int')
    # Getting the type of 'A' (line 41)
    A_407125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'A')
    # Obtaining the member 'shape' of a type (line 41)
    shape_407126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), A_407125, 'shape')
    # Obtaining the member '__getitem__' of a type (line 41)
    getitem___407127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), shape_407126, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 41)
    subscript_call_result_407128 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), getitem___407127, int_407124)
    
    # Assigning a type to the variable 'n' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'n', subscript_call_result_407128)
    
    # Call to seed(...): (line 43)
    # Processing the call arguments (line 43)
    int_407132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 19), 'int')
    # Processing the call keyword arguments (line 43)
    kwargs_407133 = {}
    # Getting the type of 'np' (line 43)
    np_407129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 43)
    random_407130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 4), np_407129, 'random')
    # Obtaining the member 'seed' of a type (line 43)
    seed_407131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 4), random_407130, 'seed')
    # Calling seed(args, kwargs) (line 43)
    seed_call_result_407134 = invoke(stypy.reporting.localization.Localization(__file__, 43, 4), seed_407131, *[int_407132], **kwargs_407133)
    
    
    # Assigning a Call to a Name (line 45):
    
    # Assigning a Call to a Name (line 45):
    
    # Call to rand(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'n' (line 45)
    n_407136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 13), 'n', False)
    # Getting the type of 'm' (line 45)
    m_407137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'm', False)
    # Processing the call keyword arguments (line 45)
    kwargs_407138 = {}
    # Getting the type of 'rand' (line 45)
    rand_407135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'rand', False)
    # Calling rand(args, kwargs) (line 45)
    rand_call_result_407139 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), rand_407135, *[n_407136, m_407137], **kwargs_407138)
    
    # Assigning a type to the variable 'V' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'V', rand_call_result_407139)
    
    # Assigning a Call to a Name (line 46):
    
    # Assigning a Call to a Name (line 46):
    
    # Call to orth(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'V' (line 46)
    V_407142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'V', False)
    # Processing the call keyword arguments (line 46)
    kwargs_407143 = {}
    # Getting the type of 'linalg' (line 46)
    linalg_407140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'linalg', False)
    # Obtaining the member 'orth' of a type (line 46)
    orth_407141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), linalg_407140, 'orth')
    # Calling orth(args, kwargs) (line 46)
    orth_call_result_407144 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), orth_407141, *[V_407142], **kwargs_407143)
    
    # Assigning a type to the variable 'X' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'X', orth_call_result_407144)
    
    # Assigning a Call to a Tuple (line 48):
    
    # Assigning a Subscript to a Name (line 48):
    
    # Obtaining the type of the subscript
    int_407145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 4), 'int')
    
    # Call to lobpcg(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'A' (line 48)
    A_407147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 23), 'A', False)
    # Getting the type of 'X' (line 48)
    X_407148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 26), 'X', False)
    # Processing the call keyword arguments (line 48)
    # Getting the type of 'B' (line 48)
    B_407149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 31), 'B', False)
    keyword_407150 = B_407149
    float_407151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 38), 'float')
    keyword_407152 = float_407151
    int_407153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 52), 'int')
    keyword_407154 = int_407153
    kwargs_407155 = {'B': keyword_407150, 'tol': keyword_407152, 'maxiter': keyword_407154}
    # Getting the type of 'lobpcg' (line 48)
    lobpcg_407146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'lobpcg', False)
    # Calling lobpcg(args, kwargs) (line 48)
    lobpcg_call_result_407156 = invoke(stypy.reporting.localization.Localization(__file__, 48, 16), lobpcg_407146, *[A_407147, X_407148], **kwargs_407155)
    
    # Obtaining the member '__getitem__' of a type (line 48)
    getitem___407157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 4), lobpcg_call_result_407156, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 48)
    subscript_call_result_407158 = invoke(stypy.reporting.localization.Localization(__file__, 48, 4), getitem___407157, int_407145)
    
    # Assigning a type to the variable 'tuple_var_assignment_406930' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'tuple_var_assignment_406930', subscript_call_result_407158)
    
    # Assigning a Subscript to a Name (line 48):
    
    # Obtaining the type of the subscript
    int_407159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 4), 'int')
    
    # Call to lobpcg(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'A' (line 48)
    A_407161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 23), 'A', False)
    # Getting the type of 'X' (line 48)
    X_407162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 26), 'X', False)
    # Processing the call keyword arguments (line 48)
    # Getting the type of 'B' (line 48)
    B_407163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 31), 'B', False)
    keyword_407164 = B_407163
    float_407165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 38), 'float')
    keyword_407166 = float_407165
    int_407167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 52), 'int')
    keyword_407168 = int_407167
    kwargs_407169 = {'B': keyword_407164, 'tol': keyword_407166, 'maxiter': keyword_407168}
    # Getting the type of 'lobpcg' (line 48)
    lobpcg_407160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'lobpcg', False)
    # Calling lobpcg(args, kwargs) (line 48)
    lobpcg_call_result_407170 = invoke(stypy.reporting.localization.Localization(__file__, 48, 16), lobpcg_407160, *[A_407161, X_407162], **kwargs_407169)
    
    # Obtaining the member '__getitem__' of a type (line 48)
    getitem___407171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 4), lobpcg_call_result_407170, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 48)
    subscript_call_result_407172 = invoke(stypy.reporting.localization.Localization(__file__, 48, 4), getitem___407171, int_407159)
    
    # Assigning a type to the variable 'tuple_var_assignment_406931' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'tuple_var_assignment_406931', subscript_call_result_407172)
    
    # Assigning a Name to a Name (line 48):
    # Getting the type of 'tuple_var_assignment_406930' (line 48)
    tuple_var_assignment_406930_407173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'tuple_var_assignment_406930')
    # Assigning a type to the variable 'eigs' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'eigs', tuple_var_assignment_406930_407173)
    
    # Assigning a Name to a Name (line 48):
    # Getting the type of 'tuple_var_assignment_406931' (line 48)
    tuple_var_assignment_406931_407174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'tuple_var_assignment_406931')
    # Assigning a type to the variable 'vecs' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 9), 'vecs', tuple_var_assignment_406931_407174)
    
    # Call to sort(...): (line 49)
    # Processing the call keyword arguments (line 49)
    kwargs_407177 = {}
    # Getting the type of 'eigs' (line 49)
    eigs_407175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'eigs', False)
    # Obtaining the member 'sort' of a type (line 49)
    sort_407176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 4), eigs_407175, 'sort')
    # Calling sort(args, kwargs) (line 49)
    sort_call_result_407178 = invoke(stypy.reporting.localization.Localization(__file__, 49, 4), sort_407176, *[], **kwargs_407177)
    
    
    # Assigning a Call to a Tuple (line 51):
    
    # Assigning a Subscript to a Name (line 51):
    
    # Obtaining the type of the subscript
    int_407179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 4), 'int')
    
    # Call to eig(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'A' (line 51)
    A_407181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 14), 'A', False)
    # Processing the call keyword arguments (line 51)
    # Getting the type of 'B' (line 51)
    B_407182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 18), 'B', False)
    keyword_407183 = B_407182
    kwargs_407184 = {'b': keyword_407183}
    # Getting the type of 'eig' (line 51)
    eig_407180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 10), 'eig', False)
    # Calling eig(args, kwargs) (line 51)
    eig_call_result_407185 = invoke(stypy.reporting.localization.Localization(__file__, 51, 10), eig_407180, *[A_407181], **kwargs_407184)
    
    # Obtaining the member '__getitem__' of a type (line 51)
    getitem___407186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 4), eig_call_result_407185, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 51)
    subscript_call_result_407187 = invoke(stypy.reporting.localization.Localization(__file__, 51, 4), getitem___407186, int_407179)
    
    # Assigning a type to the variable 'tuple_var_assignment_406932' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'tuple_var_assignment_406932', subscript_call_result_407187)
    
    # Assigning a Subscript to a Name (line 51):
    
    # Obtaining the type of the subscript
    int_407188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 4), 'int')
    
    # Call to eig(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'A' (line 51)
    A_407190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 14), 'A', False)
    # Processing the call keyword arguments (line 51)
    # Getting the type of 'B' (line 51)
    B_407191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 18), 'B', False)
    keyword_407192 = B_407191
    kwargs_407193 = {'b': keyword_407192}
    # Getting the type of 'eig' (line 51)
    eig_407189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 10), 'eig', False)
    # Calling eig(args, kwargs) (line 51)
    eig_call_result_407194 = invoke(stypy.reporting.localization.Localization(__file__, 51, 10), eig_407189, *[A_407190], **kwargs_407193)
    
    # Obtaining the member '__getitem__' of a type (line 51)
    getitem___407195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 4), eig_call_result_407194, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 51)
    subscript_call_result_407196 = invoke(stypy.reporting.localization.Localization(__file__, 51, 4), getitem___407195, int_407188)
    
    # Assigning a type to the variable 'tuple_var_assignment_406933' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'tuple_var_assignment_406933', subscript_call_result_407196)
    
    # Assigning a Name to a Name (line 51):
    # Getting the type of 'tuple_var_assignment_406932' (line 51)
    tuple_var_assignment_406932_407197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'tuple_var_assignment_406932')
    # Assigning a type to the variable 'w' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'w', tuple_var_assignment_406932_407197)
    
    # Assigning a Name to a Name (line 51):
    # Getting the type of 'tuple_var_assignment_406933' (line 51)
    tuple_var_assignment_406933_407198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'tuple_var_assignment_406933')
    # Assigning a type to the variable 'v' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 6), 'v', tuple_var_assignment_406933_407198)
    
    # Call to sort(...): (line 52)
    # Processing the call keyword arguments (line 52)
    kwargs_407201 = {}
    # Getting the type of 'w' (line 52)
    w_407199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'w', False)
    # Obtaining the member 'sort' of a type (line 52)
    sort_407200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 4), w_407199, 'sort')
    # Calling sort(args, kwargs) (line 52)
    sort_call_result_407202 = invoke(stypy.reporting.localization.Localization(__file__, 52, 4), sort_407200, *[], **kwargs_407201)
    
    
    # Call to assert_almost_equal(...): (line 54)
    # Processing the call arguments (line 54)
    
    # Obtaining the type of the subscript
    
    # Call to int(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'm' (line 54)
    m_407205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 31), 'm', False)
    int_407206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 33), 'int')
    # Applying the binary operator 'div' (line 54)
    result_div_407207 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 31), 'div', m_407205, int_407206)
    
    # Processing the call keyword arguments (line 54)
    kwargs_407208 = {}
    # Getting the type of 'int' (line 54)
    int_407204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 27), 'int', False)
    # Calling int(args, kwargs) (line 54)
    int_call_result_407209 = invoke(stypy.reporting.localization.Localization(__file__, 54, 27), int_407204, *[result_div_407207], **kwargs_407208)
    
    slice_407210 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 54, 24), None, int_call_result_407209, None)
    # Getting the type of 'w' (line 54)
    w_407211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'w', False)
    # Obtaining the member '__getitem__' of a type (line 54)
    getitem___407212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 24), w_407211, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 54)
    subscript_call_result_407213 = invoke(stypy.reporting.localization.Localization(__file__, 54, 24), getitem___407212, slice_407210)
    
    
    # Obtaining the type of the subscript
    
    # Call to int(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'm' (line 54)
    m_407215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 47), 'm', False)
    int_407216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 49), 'int')
    # Applying the binary operator 'div' (line 54)
    result_div_407217 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 47), 'div', m_407215, int_407216)
    
    # Processing the call keyword arguments (line 54)
    kwargs_407218 = {}
    # Getting the type of 'int' (line 54)
    int_407214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 43), 'int', False)
    # Calling int(args, kwargs) (line 54)
    int_call_result_407219 = invoke(stypy.reporting.localization.Localization(__file__, 54, 43), int_407214, *[result_div_407217], **kwargs_407218)
    
    slice_407220 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 54, 37), None, int_call_result_407219, None)
    # Getting the type of 'eigs' (line 54)
    eigs_407221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 37), 'eigs', False)
    # Obtaining the member '__getitem__' of a type (line 54)
    getitem___407222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 37), eigs_407221, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 54)
    subscript_call_result_407223 = invoke(stypy.reporting.localization.Localization(__file__, 54, 37), getitem___407222, slice_407220)
    
    # Processing the call keyword arguments (line 54)
    int_407224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 61), 'int')
    keyword_407225 = int_407224
    kwargs_407226 = {'decimal': keyword_407225}
    # Getting the type of 'assert_almost_equal' (line 54)
    assert_almost_equal_407203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 54)
    assert_almost_equal_call_result_407227 = invoke(stypy.reporting.localization.Localization(__file__, 54, 4), assert_almost_equal_407203, *[subscript_call_result_407213, subscript_call_result_407223], **kwargs_407226)
    
    
    # ################# End of 'compare_solutions(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'compare_solutions' in the type store
    # Getting the type of 'stypy_return_type' (line 40)
    stypy_return_type_407228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_407228)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'compare_solutions'
    return stypy_return_type_407228

# Assigning a type to the variable 'compare_solutions' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'compare_solutions', compare_solutions)

@norecursion
def test_Small(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_Small'
    module_type_store = module_type_store.open_function_context('test_Small', 57, 0, False)
    
    # Passed parameters checking function
    test_Small.stypy_localization = localization
    test_Small.stypy_type_of_self = None
    test_Small.stypy_type_store = module_type_store
    test_Small.stypy_function_name = 'test_Small'
    test_Small.stypy_param_names_list = []
    test_Small.stypy_varargs_param_name = None
    test_Small.stypy_kwargs_param_name = None
    test_Small.stypy_call_defaults = defaults
    test_Small.stypy_call_varargs = varargs
    test_Small.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_Small', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_Small', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_Small(...)' code ##################

    
    # Assigning a Call to a Tuple (line 58):
    
    # Assigning a Subscript to a Name (line 58):
    
    # Obtaining the type of the subscript
    int_407229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 4), 'int')
    
    # Call to ElasticRod(...): (line 58)
    # Processing the call arguments (line 58)
    int_407231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 21), 'int')
    # Processing the call keyword arguments (line 58)
    kwargs_407232 = {}
    # Getting the type of 'ElasticRod' (line 58)
    ElasticRod_407230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 10), 'ElasticRod', False)
    # Calling ElasticRod(args, kwargs) (line 58)
    ElasticRod_call_result_407233 = invoke(stypy.reporting.localization.Localization(__file__, 58, 10), ElasticRod_407230, *[int_407231], **kwargs_407232)
    
    # Obtaining the member '__getitem__' of a type (line 58)
    getitem___407234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 4), ElasticRod_call_result_407233, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 58)
    subscript_call_result_407235 = invoke(stypy.reporting.localization.Localization(__file__, 58, 4), getitem___407234, int_407229)
    
    # Assigning a type to the variable 'tuple_var_assignment_406934' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'tuple_var_assignment_406934', subscript_call_result_407235)
    
    # Assigning a Subscript to a Name (line 58):
    
    # Obtaining the type of the subscript
    int_407236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 4), 'int')
    
    # Call to ElasticRod(...): (line 58)
    # Processing the call arguments (line 58)
    int_407238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 21), 'int')
    # Processing the call keyword arguments (line 58)
    kwargs_407239 = {}
    # Getting the type of 'ElasticRod' (line 58)
    ElasticRod_407237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 10), 'ElasticRod', False)
    # Calling ElasticRod(args, kwargs) (line 58)
    ElasticRod_call_result_407240 = invoke(stypy.reporting.localization.Localization(__file__, 58, 10), ElasticRod_407237, *[int_407238], **kwargs_407239)
    
    # Obtaining the member '__getitem__' of a type (line 58)
    getitem___407241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 4), ElasticRod_call_result_407240, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 58)
    subscript_call_result_407242 = invoke(stypy.reporting.localization.Localization(__file__, 58, 4), getitem___407241, int_407236)
    
    # Assigning a type to the variable 'tuple_var_assignment_406935' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'tuple_var_assignment_406935', subscript_call_result_407242)
    
    # Assigning a Name to a Name (line 58):
    # Getting the type of 'tuple_var_assignment_406934' (line 58)
    tuple_var_assignment_406934_407243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'tuple_var_assignment_406934')
    # Assigning a type to the variable 'A' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'A', tuple_var_assignment_406934_407243)
    
    # Assigning a Name to a Name (line 58):
    # Getting the type of 'tuple_var_assignment_406935' (line 58)
    tuple_var_assignment_406935_407244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'tuple_var_assignment_406935')
    # Assigning a type to the variable 'B' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 6), 'B', tuple_var_assignment_406935_407244)
    
    # Call to compare_solutions(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'A' (line 59)
    A_407246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 22), 'A', False)
    # Getting the type of 'B' (line 59)
    B_407247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 24), 'B', False)
    int_407248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 26), 'int')
    # Processing the call keyword arguments (line 59)
    kwargs_407249 = {}
    # Getting the type of 'compare_solutions' (line 59)
    compare_solutions_407245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'compare_solutions', False)
    # Calling compare_solutions(args, kwargs) (line 59)
    compare_solutions_call_result_407250 = invoke(stypy.reporting.localization.Localization(__file__, 59, 4), compare_solutions_407245, *[A_407246, B_407247, int_407248], **kwargs_407249)
    
    
    # Assigning a Call to a Tuple (line 60):
    
    # Assigning a Subscript to a Name (line 60):
    
    # Obtaining the type of the subscript
    int_407251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 4), 'int')
    
    # Call to MikotaPair(...): (line 60)
    # Processing the call arguments (line 60)
    int_407253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 21), 'int')
    # Processing the call keyword arguments (line 60)
    kwargs_407254 = {}
    # Getting the type of 'MikotaPair' (line 60)
    MikotaPair_407252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 10), 'MikotaPair', False)
    # Calling MikotaPair(args, kwargs) (line 60)
    MikotaPair_call_result_407255 = invoke(stypy.reporting.localization.Localization(__file__, 60, 10), MikotaPair_407252, *[int_407253], **kwargs_407254)
    
    # Obtaining the member '__getitem__' of a type (line 60)
    getitem___407256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 4), MikotaPair_call_result_407255, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 60)
    subscript_call_result_407257 = invoke(stypy.reporting.localization.Localization(__file__, 60, 4), getitem___407256, int_407251)
    
    # Assigning a type to the variable 'tuple_var_assignment_406936' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'tuple_var_assignment_406936', subscript_call_result_407257)
    
    # Assigning a Subscript to a Name (line 60):
    
    # Obtaining the type of the subscript
    int_407258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 4), 'int')
    
    # Call to MikotaPair(...): (line 60)
    # Processing the call arguments (line 60)
    int_407260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 21), 'int')
    # Processing the call keyword arguments (line 60)
    kwargs_407261 = {}
    # Getting the type of 'MikotaPair' (line 60)
    MikotaPair_407259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 10), 'MikotaPair', False)
    # Calling MikotaPair(args, kwargs) (line 60)
    MikotaPair_call_result_407262 = invoke(stypy.reporting.localization.Localization(__file__, 60, 10), MikotaPair_407259, *[int_407260], **kwargs_407261)
    
    # Obtaining the member '__getitem__' of a type (line 60)
    getitem___407263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 4), MikotaPair_call_result_407262, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 60)
    subscript_call_result_407264 = invoke(stypy.reporting.localization.Localization(__file__, 60, 4), getitem___407263, int_407258)
    
    # Assigning a type to the variable 'tuple_var_assignment_406937' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'tuple_var_assignment_406937', subscript_call_result_407264)
    
    # Assigning a Name to a Name (line 60):
    # Getting the type of 'tuple_var_assignment_406936' (line 60)
    tuple_var_assignment_406936_407265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'tuple_var_assignment_406936')
    # Assigning a type to the variable 'A' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'A', tuple_var_assignment_406936_407265)
    
    # Assigning a Name to a Name (line 60):
    # Getting the type of 'tuple_var_assignment_406937' (line 60)
    tuple_var_assignment_406937_407266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'tuple_var_assignment_406937')
    # Assigning a type to the variable 'B' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 6), 'B', tuple_var_assignment_406937_407266)
    
    # Call to compare_solutions(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'A' (line 61)
    A_407268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 22), 'A', False)
    # Getting the type of 'B' (line 61)
    B_407269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'B', False)
    int_407270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 26), 'int')
    # Processing the call keyword arguments (line 61)
    kwargs_407271 = {}
    # Getting the type of 'compare_solutions' (line 61)
    compare_solutions_407267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'compare_solutions', False)
    # Calling compare_solutions(args, kwargs) (line 61)
    compare_solutions_call_result_407272 = invoke(stypy.reporting.localization.Localization(__file__, 61, 4), compare_solutions_407267, *[A_407268, B_407269, int_407270], **kwargs_407271)
    
    
    # ################# End of 'test_Small(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_Small' in the type store
    # Getting the type of 'stypy_return_type' (line 57)
    stypy_return_type_407273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_407273)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_Small'
    return stypy_return_type_407273

# Assigning a type to the variable 'test_Small' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'test_Small', test_Small)

@norecursion
def test_ElasticRod(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_ElasticRod'
    module_type_store = module_type_store.open_function_context('test_ElasticRod', 64, 0, False)
    
    # Passed parameters checking function
    test_ElasticRod.stypy_localization = localization
    test_ElasticRod.stypy_type_of_self = None
    test_ElasticRod.stypy_type_store = module_type_store
    test_ElasticRod.stypy_function_name = 'test_ElasticRod'
    test_ElasticRod.stypy_param_names_list = []
    test_ElasticRod.stypy_varargs_param_name = None
    test_ElasticRod.stypy_kwargs_param_name = None
    test_ElasticRod.stypy_call_defaults = defaults
    test_ElasticRod.stypy_call_varargs = varargs
    test_ElasticRod.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_ElasticRod', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_ElasticRod', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_ElasticRod(...)' code ##################

    
    # Assigning a Call to a Tuple (line 65):
    
    # Assigning a Subscript to a Name (line 65):
    
    # Obtaining the type of the subscript
    int_407274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 4), 'int')
    
    # Call to ElasticRod(...): (line 65)
    # Processing the call arguments (line 65)
    int_407276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 21), 'int')
    # Processing the call keyword arguments (line 65)
    kwargs_407277 = {}
    # Getting the type of 'ElasticRod' (line 65)
    ElasticRod_407275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 10), 'ElasticRod', False)
    # Calling ElasticRod(args, kwargs) (line 65)
    ElasticRod_call_result_407278 = invoke(stypy.reporting.localization.Localization(__file__, 65, 10), ElasticRod_407275, *[int_407276], **kwargs_407277)
    
    # Obtaining the member '__getitem__' of a type (line 65)
    getitem___407279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 4), ElasticRod_call_result_407278, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 65)
    subscript_call_result_407280 = invoke(stypy.reporting.localization.Localization(__file__, 65, 4), getitem___407279, int_407274)
    
    # Assigning a type to the variable 'tuple_var_assignment_406938' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_406938', subscript_call_result_407280)
    
    # Assigning a Subscript to a Name (line 65):
    
    # Obtaining the type of the subscript
    int_407281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 4), 'int')
    
    # Call to ElasticRod(...): (line 65)
    # Processing the call arguments (line 65)
    int_407283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 21), 'int')
    # Processing the call keyword arguments (line 65)
    kwargs_407284 = {}
    # Getting the type of 'ElasticRod' (line 65)
    ElasticRod_407282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 10), 'ElasticRod', False)
    # Calling ElasticRod(args, kwargs) (line 65)
    ElasticRod_call_result_407285 = invoke(stypy.reporting.localization.Localization(__file__, 65, 10), ElasticRod_407282, *[int_407283], **kwargs_407284)
    
    # Obtaining the member '__getitem__' of a type (line 65)
    getitem___407286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 4), ElasticRod_call_result_407285, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 65)
    subscript_call_result_407287 = invoke(stypy.reporting.localization.Localization(__file__, 65, 4), getitem___407286, int_407281)
    
    # Assigning a type to the variable 'tuple_var_assignment_406939' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_406939', subscript_call_result_407287)
    
    # Assigning a Name to a Name (line 65):
    # Getting the type of 'tuple_var_assignment_406938' (line 65)
    tuple_var_assignment_406938_407288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_406938')
    # Assigning a type to the variable 'A' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'A', tuple_var_assignment_406938_407288)
    
    # Assigning a Name to a Name (line 65):
    # Getting the type of 'tuple_var_assignment_406939' (line 65)
    tuple_var_assignment_406939_407289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_406939')
    # Assigning a type to the variable 'B' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 6), 'B', tuple_var_assignment_406939_407289)
    
    # Call to compare_solutions(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'A' (line 66)
    A_407291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 22), 'A', False)
    # Getting the type of 'B' (line 66)
    B_407292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 24), 'B', False)
    int_407293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 26), 'int')
    # Processing the call keyword arguments (line 66)
    kwargs_407294 = {}
    # Getting the type of 'compare_solutions' (line 66)
    compare_solutions_407290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'compare_solutions', False)
    # Calling compare_solutions(args, kwargs) (line 66)
    compare_solutions_call_result_407295 = invoke(stypy.reporting.localization.Localization(__file__, 66, 4), compare_solutions_407290, *[A_407291, B_407292, int_407293], **kwargs_407294)
    
    
    # ################# End of 'test_ElasticRod(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_ElasticRod' in the type store
    # Getting the type of 'stypy_return_type' (line 64)
    stypy_return_type_407296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_407296)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_ElasticRod'
    return stypy_return_type_407296

# Assigning a type to the variable 'test_ElasticRod' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'test_ElasticRod', test_ElasticRod)

@norecursion
def test_MikotaPair(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_MikotaPair'
    module_type_store = module_type_store.open_function_context('test_MikotaPair', 69, 0, False)
    
    # Passed parameters checking function
    test_MikotaPair.stypy_localization = localization
    test_MikotaPair.stypy_type_of_self = None
    test_MikotaPair.stypy_type_store = module_type_store
    test_MikotaPair.stypy_function_name = 'test_MikotaPair'
    test_MikotaPair.stypy_param_names_list = []
    test_MikotaPair.stypy_varargs_param_name = None
    test_MikotaPair.stypy_kwargs_param_name = None
    test_MikotaPair.stypy_call_defaults = defaults
    test_MikotaPair.stypy_call_varargs = varargs
    test_MikotaPair.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_MikotaPair', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_MikotaPair', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_MikotaPair(...)' code ##################

    
    # Assigning a Call to a Tuple (line 70):
    
    # Assigning a Subscript to a Name (line 70):
    
    # Obtaining the type of the subscript
    int_407297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 4), 'int')
    
    # Call to MikotaPair(...): (line 70)
    # Processing the call arguments (line 70)
    int_407299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 21), 'int')
    # Processing the call keyword arguments (line 70)
    kwargs_407300 = {}
    # Getting the type of 'MikotaPair' (line 70)
    MikotaPair_407298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 10), 'MikotaPair', False)
    # Calling MikotaPair(args, kwargs) (line 70)
    MikotaPair_call_result_407301 = invoke(stypy.reporting.localization.Localization(__file__, 70, 10), MikotaPair_407298, *[int_407299], **kwargs_407300)
    
    # Obtaining the member '__getitem__' of a type (line 70)
    getitem___407302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 4), MikotaPair_call_result_407301, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 70)
    subscript_call_result_407303 = invoke(stypy.reporting.localization.Localization(__file__, 70, 4), getitem___407302, int_407297)
    
    # Assigning a type to the variable 'tuple_var_assignment_406940' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'tuple_var_assignment_406940', subscript_call_result_407303)
    
    # Assigning a Subscript to a Name (line 70):
    
    # Obtaining the type of the subscript
    int_407304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 4), 'int')
    
    # Call to MikotaPair(...): (line 70)
    # Processing the call arguments (line 70)
    int_407306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 21), 'int')
    # Processing the call keyword arguments (line 70)
    kwargs_407307 = {}
    # Getting the type of 'MikotaPair' (line 70)
    MikotaPair_407305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 10), 'MikotaPair', False)
    # Calling MikotaPair(args, kwargs) (line 70)
    MikotaPair_call_result_407308 = invoke(stypy.reporting.localization.Localization(__file__, 70, 10), MikotaPair_407305, *[int_407306], **kwargs_407307)
    
    # Obtaining the member '__getitem__' of a type (line 70)
    getitem___407309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 4), MikotaPair_call_result_407308, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 70)
    subscript_call_result_407310 = invoke(stypy.reporting.localization.Localization(__file__, 70, 4), getitem___407309, int_407304)
    
    # Assigning a type to the variable 'tuple_var_assignment_406941' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'tuple_var_assignment_406941', subscript_call_result_407310)
    
    # Assigning a Name to a Name (line 70):
    # Getting the type of 'tuple_var_assignment_406940' (line 70)
    tuple_var_assignment_406940_407311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'tuple_var_assignment_406940')
    # Assigning a type to the variable 'A' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'A', tuple_var_assignment_406940_407311)
    
    # Assigning a Name to a Name (line 70):
    # Getting the type of 'tuple_var_assignment_406941' (line 70)
    tuple_var_assignment_406941_407312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'tuple_var_assignment_406941')
    # Assigning a type to the variable 'B' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 6), 'B', tuple_var_assignment_406941_407312)
    
    # Call to compare_solutions(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'A' (line 71)
    A_407314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), 'A', False)
    # Getting the type of 'B' (line 71)
    B_407315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 24), 'B', False)
    int_407316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 26), 'int')
    # Processing the call keyword arguments (line 71)
    kwargs_407317 = {}
    # Getting the type of 'compare_solutions' (line 71)
    compare_solutions_407313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'compare_solutions', False)
    # Calling compare_solutions(args, kwargs) (line 71)
    compare_solutions_call_result_407318 = invoke(stypy.reporting.localization.Localization(__file__, 71, 4), compare_solutions_407313, *[A_407314, B_407315, int_407316], **kwargs_407317)
    
    
    # ################# End of 'test_MikotaPair(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_MikotaPair' in the type store
    # Getting the type of 'stypy_return_type' (line 69)
    stypy_return_type_407319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_407319)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_MikotaPair'
    return stypy_return_type_407319

# Assigning a type to the variable 'test_MikotaPair' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'test_MikotaPair', test_MikotaPair)

@norecursion
def test_trivial(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_trivial'
    module_type_store = module_type_store.open_function_context('test_trivial', 74, 0, False)
    
    # Passed parameters checking function
    test_trivial.stypy_localization = localization
    test_trivial.stypy_type_of_self = None
    test_trivial.stypy_type_store = module_type_store
    test_trivial.stypy_function_name = 'test_trivial'
    test_trivial.stypy_param_names_list = []
    test_trivial.stypy_varargs_param_name = None
    test_trivial.stypy_kwargs_param_name = None
    test_trivial.stypy_call_defaults = defaults
    test_trivial.stypy_call_varargs = varargs
    test_trivial.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_trivial', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_trivial', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_trivial(...)' code ##################

    
    # Assigning a Num to a Name (line 75):
    
    # Assigning a Num to a Name (line 75):
    int_407320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 8), 'int')
    # Assigning a type to the variable 'n' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'n', int_407320)
    
    # Assigning a Call to a Name (line 76):
    
    # Assigning a Call to a Name (line 76):
    
    # Call to ones(...): (line 76)
    # Processing the call arguments (line 76)
    
    # Obtaining an instance of the builtin type 'tuple' (line 76)
    tuple_407322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 76)
    # Adding element type (line 76)
    # Getting the type of 'n' (line 76)
    n_407323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 14), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 14), tuple_407322, n_407323)
    # Adding element type (line 76)
    int_407324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 14), tuple_407322, int_407324)
    
    # Processing the call keyword arguments (line 76)
    kwargs_407325 = {}
    # Getting the type of 'ones' (line 76)
    ones_407321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'ones', False)
    # Calling ones(args, kwargs) (line 76)
    ones_call_result_407326 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), ones_407321, *[tuple_407322], **kwargs_407325)
    
    # Assigning a type to the variable 'X' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'X', ones_call_result_407326)
    
    # Assigning a Call to a Name (line 77):
    
    # Assigning a Call to a Name (line 77):
    
    # Call to eye(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'n' (line 77)
    n_407328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'n', False)
    # Processing the call keyword arguments (line 77)
    kwargs_407329 = {}
    # Getting the type of 'eye' (line 77)
    eye_407327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'eye', False)
    # Calling eye(args, kwargs) (line 77)
    eye_call_result_407330 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), eye_407327, *[n_407328], **kwargs_407329)
    
    # Assigning a type to the variable 'A' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'A', eye_call_result_407330)
    
    # Call to compare_solutions(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'A' (line 78)
    A_407332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 22), 'A', False)
    # Getting the type of 'None' (line 78)
    None_407333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 25), 'None', False)
    # Getting the type of 'n' (line 78)
    n_407334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 31), 'n', False)
    # Processing the call keyword arguments (line 78)
    kwargs_407335 = {}
    # Getting the type of 'compare_solutions' (line 78)
    compare_solutions_407331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'compare_solutions', False)
    # Calling compare_solutions(args, kwargs) (line 78)
    compare_solutions_call_result_407336 = invoke(stypy.reporting.localization.Localization(__file__, 78, 4), compare_solutions_407331, *[A_407332, None_407333, n_407334], **kwargs_407335)
    
    
    # ################# End of 'test_trivial(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_trivial' in the type store
    # Getting the type of 'stypy_return_type' (line 74)
    stypy_return_type_407337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_407337)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_trivial'
    return stypy_return_type_407337

# Assigning a type to the variable 'test_trivial' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'test_trivial', test_trivial)

@norecursion
def test_regression(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_regression'
    module_type_store = module_type_store.open_function_context('test_regression', 81, 0, False)
    
    # Passed parameters checking function
    test_regression.stypy_localization = localization
    test_regression.stypy_type_of_self = None
    test_regression.stypy_type_store = module_type_store
    test_regression.stypy_function_name = 'test_regression'
    test_regression.stypy_param_names_list = []
    test_regression.stypy_varargs_param_name = None
    test_regression.stypy_kwargs_param_name = None
    test_regression.stypy_call_defaults = defaults
    test_regression.stypy_call_varargs = varargs
    test_regression.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_regression', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_regression', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_regression(...)' code ##################

    
    # Assigning a Num to a Name (line 83):
    
    # Assigning a Num to a Name (line 83):
    int_407338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 8), 'int')
    # Assigning a type to the variable 'n' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'n', int_407338)
    
    # Assigning a Call to a Name (line 84):
    
    # Assigning a Call to a Name (line 84):
    
    # Call to ones(...): (line 84)
    # Processing the call arguments (line 84)
    
    # Obtaining an instance of the builtin type 'tuple' (line 84)
    tuple_407341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 84)
    # Adding element type (line 84)
    # Getting the type of 'n' (line 84)
    n_407342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 17), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 17), tuple_407341, n_407342)
    # Adding element type (line 84)
    int_407343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 17), tuple_407341, int_407343)
    
    # Processing the call keyword arguments (line 84)
    kwargs_407344 = {}
    # Getting the type of 'np' (line 84)
    np_407339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'np', False)
    # Obtaining the member 'ones' of a type (line 84)
    ones_407340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), np_407339, 'ones')
    # Calling ones(args, kwargs) (line 84)
    ones_call_result_407345 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), ones_407340, *[tuple_407341], **kwargs_407344)
    
    # Assigning a type to the variable 'X' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'X', ones_call_result_407345)
    
    # Assigning a Call to a Name (line 85):
    
    # Assigning a Call to a Name (line 85):
    
    # Call to identity(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'n' (line 85)
    n_407348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 20), 'n', False)
    # Processing the call keyword arguments (line 85)
    kwargs_407349 = {}
    # Getting the type of 'np' (line 85)
    np_407346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'np', False)
    # Obtaining the member 'identity' of a type (line 85)
    identity_407347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), np_407346, 'identity')
    # Calling identity(args, kwargs) (line 85)
    identity_call_result_407350 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), identity_407347, *[n_407348], **kwargs_407349)
    
    # Assigning a type to the variable 'A' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'A', identity_call_result_407350)
    
    # Assigning a Call to a Tuple (line 86):
    
    # Assigning a Subscript to a Name (line 86):
    
    # Obtaining the type of the subscript
    int_407351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 4), 'int')
    
    # Call to lobpcg(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'A' (line 86)
    A_407353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 18), 'A', False)
    # Getting the type of 'X' (line 86)
    X_407354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 21), 'X', False)
    # Processing the call keyword arguments (line 86)
    kwargs_407355 = {}
    # Getting the type of 'lobpcg' (line 86)
    lobpcg_407352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 11), 'lobpcg', False)
    # Calling lobpcg(args, kwargs) (line 86)
    lobpcg_call_result_407356 = invoke(stypy.reporting.localization.Localization(__file__, 86, 11), lobpcg_407352, *[A_407353, X_407354], **kwargs_407355)
    
    # Obtaining the member '__getitem__' of a type (line 86)
    getitem___407357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 4), lobpcg_call_result_407356, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 86)
    subscript_call_result_407358 = invoke(stypy.reporting.localization.Localization(__file__, 86, 4), getitem___407357, int_407351)
    
    # Assigning a type to the variable 'tuple_var_assignment_406942' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'tuple_var_assignment_406942', subscript_call_result_407358)
    
    # Assigning a Subscript to a Name (line 86):
    
    # Obtaining the type of the subscript
    int_407359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 4), 'int')
    
    # Call to lobpcg(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'A' (line 86)
    A_407361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 18), 'A', False)
    # Getting the type of 'X' (line 86)
    X_407362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 21), 'X', False)
    # Processing the call keyword arguments (line 86)
    kwargs_407363 = {}
    # Getting the type of 'lobpcg' (line 86)
    lobpcg_407360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 11), 'lobpcg', False)
    # Calling lobpcg(args, kwargs) (line 86)
    lobpcg_call_result_407364 = invoke(stypy.reporting.localization.Localization(__file__, 86, 11), lobpcg_407360, *[A_407361, X_407362], **kwargs_407363)
    
    # Obtaining the member '__getitem__' of a type (line 86)
    getitem___407365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 4), lobpcg_call_result_407364, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 86)
    subscript_call_result_407366 = invoke(stypy.reporting.localization.Localization(__file__, 86, 4), getitem___407365, int_407359)
    
    # Assigning a type to the variable 'tuple_var_assignment_406943' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'tuple_var_assignment_406943', subscript_call_result_407366)
    
    # Assigning a Name to a Name (line 86):
    # Getting the type of 'tuple_var_assignment_406942' (line 86)
    tuple_var_assignment_406942_407367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'tuple_var_assignment_406942')
    # Assigning a type to the variable 'w' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'w', tuple_var_assignment_406942_407367)
    
    # Assigning a Name to a Name (line 86):
    # Getting the type of 'tuple_var_assignment_406943' (line 86)
    tuple_var_assignment_406943_407368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'tuple_var_assignment_406943')
    # Assigning a type to the variable 'V' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 7), 'V', tuple_var_assignment_406943_407368)
    
    # Call to assert_allclose(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'w' (line 87)
    w_407370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'w', False)
    
    # Obtaining an instance of the builtin type 'list' (line 87)
    list_407371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 87)
    # Adding element type (line 87)
    int_407372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 23), list_407371, int_407372)
    
    # Processing the call keyword arguments (line 87)
    kwargs_407373 = {}
    # Getting the type of 'assert_allclose' (line 87)
    assert_allclose_407369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 87)
    assert_allclose_call_result_407374 = invoke(stypy.reporting.localization.Localization(__file__, 87, 4), assert_allclose_407369, *[w_407370, list_407371], **kwargs_407373)
    
    
    # ################# End of 'test_regression(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_regression' in the type store
    # Getting the type of 'stypy_return_type' (line 81)
    stypy_return_type_407375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_407375)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_regression'
    return stypy_return_type_407375

# Assigning a type to the variable 'test_regression' (line 81)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'test_regression', test_regression)

@norecursion
def test_diagonal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_diagonal'
    module_type_store = module_type_store.open_function_context('test_diagonal', 90, 0, False)
    
    # Passed parameters checking function
    test_diagonal.stypy_localization = localization
    test_diagonal.stypy_type_of_self = None
    test_diagonal.stypy_type_store = module_type_store
    test_diagonal.stypy_function_name = 'test_diagonal'
    test_diagonal.stypy_param_names_list = []
    test_diagonal.stypy_varargs_param_name = None
    test_diagonal.stypy_kwargs_param_name = None
    test_diagonal.stypy_call_defaults = defaults
    test_diagonal.stypy_call_varargs = varargs
    test_diagonal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_diagonal', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_diagonal', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_diagonal(...)' code ##################

    
    # Call to seed(...): (line 97)
    # Processing the call arguments (line 97)
    int_407379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 19), 'int')
    # Processing the call keyword arguments (line 97)
    kwargs_407380 = {}
    # Getting the type of 'np' (line 97)
    np_407376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 97)
    random_407377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 4), np_407376, 'random')
    # Obtaining the member 'seed' of a type (line 97)
    seed_407378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 4), random_407377, 'seed')
    # Calling seed(args, kwargs) (line 97)
    seed_call_result_407381 = invoke(stypy.reporting.localization.Localization(__file__, 97, 4), seed_407378, *[int_407379], **kwargs_407380)
    
    
    # Assigning a Num to a Name (line 100):
    
    # Assigning a Num to a Name (line 100):
    int_407382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 8), 'int')
    # Assigning a type to the variable 'n' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'n', int_407382)
    
    # Assigning a Num to a Name (line 103):
    
    # Assigning a Num to a Name (line 103):
    int_407383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 8), 'int')
    # Assigning a type to the variable 'm' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'm', int_407383)
    
    # Assigning a Call to a Name (line 109):
    
    # Assigning a Call to a Name (line 109):
    
    # Call to arange(...): (line 109)
    # Processing the call arguments (line 109)
    int_407386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 21), 'int')
    # Getting the type of 'n' (line 109)
    n_407387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 24), 'n', False)
    int_407388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 26), 'int')
    # Applying the binary operator '+' (line 109)
    result_add_407389 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 24), '+', n_407387, int_407388)
    
    # Processing the call keyword arguments (line 109)
    # Getting the type of 'float' (line 109)
    float_407390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 35), 'float', False)
    keyword_407391 = float_407390
    kwargs_407392 = {'dtype': keyword_407391}
    # Getting the type of 'np' (line 109)
    np_407384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 'np', False)
    # Obtaining the member 'arange' of a type (line 109)
    arange_407385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 11), np_407384, 'arange')
    # Calling arange(args, kwargs) (line 109)
    arange_call_result_407393 = invoke(stypy.reporting.localization.Localization(__file__, 109, 11), arange_407385, *[int_407386, result_add_407389], **kwargs_407392)
    
    # Assigning a type to the variable 'vals' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'vals', arange_call_result_407393)
    
    # Assigning a Call to a Name (line 110):
    
    # Assigning a Call to a Name (line 110):
    
    # Call to diags(...): (line 110)
    # Processing the call arguments (line 110)
    
    # Obtaining an instance of the builtin type 'list' (line 110)
    list_407397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 110)
    # Adding element type (line 110)
    # Getting the type of 'vals' (line 110)
    vals_407398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 28), 'vals', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 27), list_407397, vals_407398)
    
    
    # Obtaining an instance of the builtin type 'list' (line 110)
    list_407399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 110)
    # Adding element type (line 110)
    int_407400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 35), list_407399, int_407400)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 110)
    tuple_407401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 110)
    # Adding element type (line 110)
    # Getting the type of 'n' (line 110)
    n_407402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 41), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 41), tuple_407401, n_407402)
    # Adding element type (line 110)
    # Getting the type of 'n' (line 110)
    n_407403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 44), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 41), tuple_407401, n_407403)
    
    # Processing the call keyword arguments (line 110)
    kwargs_407404 = {}
    # Getting the type of 'scipy' (line 110)
    scipy_407394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'scipy', False)
    # Obtaining the member 'sparse' of a type (line 110)
    sparse_407395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), scipy_407394, 'sparse')
    # Obtaining the member 'diags' of a type (line 110)
    diags_407396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), sparse_407395, 'diags')
    # Calling diags(args, kwargs) (line 110)
    diags_call_result_407405 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), diags_407396, *[list_407397, list_407399, tuple_407401], **kwargs_407404)
    
    # Assigning a type to the variable 'A' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'A', diags_call_result_407405)
    
    # Assigning a Call to a Name (line 111):
    
    # Assigning a Call to a Name (line 111):
    
    # Call to eye(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'n' (line 111)
    n_407409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 25), 'n', False)
    # Processing the call keyword arguments (line 111)
    kwargs_407410 = {}
    # Getting the type of 'scipy' (line 111)
    scipy_407406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'scipy', False)
    # Obtaining the member 'sparse' of a type (line 111)
    sparse_407407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), scipy_407406, 'sparse')
    # Obtaining the member 'eye' of a type (line 111)
    eye_407408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), sparse_407407, 'eye')
    # Calling eye(args, kwargs) (line 111)
    eye_call_result_407411 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), eye_407408, *[n_407409], **kwargs_407410)
    
    # Assigning a type to the variable 'B' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'B', eye_call_result_407411)
    
    # Assigning a Call to a Name (line 114):
    
    # Assigning a Call to a Name (line 114):
    
    # Call to diags(...): (line 114)
    # Processing the call arguments (line 114)
    
    # Obtaining an instance of the builtin type 'list' (line 114)
    list_407415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 114)
    # Adding element type (line 114)
    
    # Call to reciprocal(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'vals' (line 114)
    vals_407418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 42), 'vals', False)
    # Processing the call keyword arguments (line 114)
    kwargs_407419 = {}
    # Getting the type of 'np' (line 114)
    np_407416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 28), 'np', False)
    # Obtaining the member 'reciprocal' of a type (line 114)
    reciprocal_407417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 28), np_407416, 'reciprocal')
    # Calling reciprocal(args, kwargs) (line 114)
    reciprocal_call_result_407420 = invoke(stypy.reporting.localization.Localization(__file__, 114, 28), reciprocal_407417, *[vals_407418], **kwargs_407419)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 27), list_407415, reciprocal_call_result_407420)
    
    
    # Obtaining an instance of the builtin type 'list' (line 114)
    list_407421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 50), 'list')
    # Adding type elements to the builtin type 'list' instance (line 114)
    # Adding element type (line 114)
    int_407422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 50), list_407421, int_407422)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 114)
    tuple_407423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 56), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 114)
    # Adding element type (line 114)
    # Getting the type of 'n' (line 114)
    n_407424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 56), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 56), tuple_407423, n_407424)
    # Adding element type (line 114)
    # Getting the type of 'n' (line 114)
    n_407425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 59), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 56), tuple_407423, n_407425)
    
    # Processing the call keyword arguments (line 114)
    kwargs_407426 = {}
    # Getting the type of 'scipy' (line 114)
    scipy_407412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'scipy', False)
    # Obtaining the member 'sparse' of a type (line 114)
    sparse_407413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), scipy_407412, 'sparse')
    # Obtaining the member 'diags' of a type (line 114)
    diags_407414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), sparse_407413, 'diags')
    # Calling diags(args, kwargs) (line 114)
    diags_call_result_407427 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), diags_407414, *[list_407415, list_407421, tuple_407423], **kwargs_407426)
    
    # Assigning a type to the variable 'M' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'M', diags_call_result_407427)
    
    # Assigning a Call to a Name (line 117):
    
    # Assigning a Call to a Name (line 117):
    
    # Call to rand(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'n' (line 117)
    n_407431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 23), 'n', False)
    # Getting the type of 'm' (line 117)
    m_407432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 26), 'm', False)
    # Processing the call keyword arguments (line 117)
    kwargs_407433 = {}
    # Getting the type of 'np' (line 117)
    np_407428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 117)
    random_407429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), np_407428, 'random')
    # Obtaining the member 'rand' of a type (line 117)
    rand_407430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), random_407429, 'rand')
    # Calling rand(args, kwargs) (line 117)
    rand_call_result_407434 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), rand_407430, *[n_407431, m_407432], **kwargs_407433)
    
    # Assigning a type to the variable 'X' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'X', rand_call_result_407434)
    
    # Assigning a Num to a Name (line 121):
    
    # Assigning a Num to a Name (line 121):
    int_407435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 17), 'int')
    # Assigning a type to the variable 'm_excluded' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'm_excluded', int_407435)
    
    # Assigning a Call to a Name (line 122):
    
    # Assigning a Call to a Name (line 122):
    
    # Call to eye(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'n' (line 122)
    n_407438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 15), 'n', False)
    # Getting the type of 'm_excluded' (line 122)
    m_excluded_407439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 18), 'm_excluded', False)
    # Processing the call keyword arguments (line 122)
    kwargs_407440 = {}
    # Getting the type of 'np' (line 122)
    np_407436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'np', False)
    # Obtaining the member 'eye' of a type (line 122)
    eye_407437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), np_407436, 'eye')
    # Calling eye(args, kwargs) (line 122)
    eye_call_result_407441 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), eye_407437, *[n_407438, m_excluded_407439], **kwargs_407440)
    
    # Assigning a type to the variable 'Y' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'Y', eye_call_result_407441)
    
    # Assigning a Call to a Tuple (line 124):
    
    # Assigning a Subscript to a Name (line 124):
    
    # Obtaining the type of the subscript
    int_407442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 4), 'int')
    
    # Call to lobpcg(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'A' (line 124)
    A_407444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 24), 'A', False)
    # Getting the type of 'X' (line 124)
    X_407445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 27), 'X', False)
    # Getting the type of 'B' (line 124)
    B_407446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 30), 'B', False)
    # Processing the call keyword arguments (line 124)
    # Getting the type of 'M' (line 124)
    M_407447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 35), 'M', False)
    keyword_407448 = M_407447
    # Getting the type of 'Y' (line 124)
    Y_407449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 40), 'Y', False)
    keyword_407450 = Y_407449
    float_407451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 47), 'float')
    keyword_407452 = float_407451
    int_407453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 61), 'int')
    keyword_407454 = int_407453
    # Getting the type of 'False' (line 124)
    False_407455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 73), 'False', False)
    keyword_407456 = False_407455
    kwargs_407457 = {'Y': keyword_407450, 'largest': keyword_407456, 'M': keyword_407448, 'tol': keyword_407452, 'maxiter': keyword_407454}
    # Getting the type of 'lobpcg' (line 124)
    lobpcg_407443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 17), 'lobpcg', False)
    # Calling lobpcg(args, kwargs) (line 124)
    lobpcg_call_result_407458 = invoke(stypy.reporting.localization.Localization(__file__, 124, 17), lobpcg_407443, *[A_407444, X_407445, B_407446], **kwargs_407457)
    
    # Obtaining the member '__getitem__' of a type (line 124)
    getitem___407459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 4), lobpcg_call_result_407458, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 124)
    subscript_call_result_407460 = invoke(stypy.reporting.localization.Localization(__file__, 124, 4), getitem___407459, int_407442)
    
    # Assigning a type to the variable 'tuple_var_assignment_406944' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'tuple_var_assignment_406944', subscript_call_result_407460)
    
    # Assigning a Subscript to a Name (line 124):
    
    # Obtaining the type of the subscript
    int_407461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 4), 'int')
    
    # Call to lobpcg(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'A' (line 124)
    A_407463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 24), 'A', False)
    # Getting the type of 'X' (line 124)
    X_407464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 27), 'X', False)
    # Getting the type of 'B' (line 124)
    B_407465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 30), 'B', False)
    # Processing the call keyword arguments (line 124)
    # Getting the type of 'M' (line 124)
    M_407466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 35), 'M', False)
    keyword_407467 = M_407466
    # Getting the type of 'Y' (line 124)
    Y_407468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 40), 'Y', False)
    keyword_407469 = Y_407468
    float_407470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 47), 'float')
    keyword_407471 = float_407470
    int_407472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 61), 'int')
    keyword_407473 = int_407472
    # Getting the type of 'False' (line 124)
    False_407474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 73), 'False', False)
    keyword_407475 = False_407474
    kwargs_407476 = {'Y': keyword_407469, 'largest': keyword_407475, 'M': keyword_407467, 'tol': keyword_407471, 'maxiter': keyword_407473}
    # Getting the type of 'lobpcg' (line 124)
    lobpcg_407462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 17), 'lobpcg', False)
    # Calling lobpcg(args, kwargs) (line 124)
    lobpcg_call_result_407477 = invoke(stypy.reporting.localization.Localization(__file__, 124, 17), lobpcg_407462, *[A_407463, X_407464, B_407465], **kwargs_407476)
    
    # Obtaining the member '__getitem__' of a type (line 124)
    getitem___407478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 4), lobpcg_call_result_407477, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 124)
    subscript_call_result_407479 = invoke(stypy.reporting.localization.Localization(__file__, 124, 4), getitem___407478, int_407461)
    
    # Assigning a type to the variable 'tuple_var_assignment_406945' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'tuple_var_assignment_406945', subscript_call_result_407479)
    
    # Assigning a Name to a Name (line 124):
    # Getting the type of 'tuple_var_assignment_406944' (line 124)
    tuple_var_assignment_406944_407480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'tuple_var_assignment_406944')
    # Assigning a type to the variable 'eigs' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'eigs', tuple_var_assignment_406944_407480)
    
    # Assigning a Name to a Name (line 124):
    # Getting the type of 'tuple_var_assignment_406945' (line 124)
    tuple_var_assignment_406945_407481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'tuple_var_assignment_406945')
    # Assigning a type to the variable 'vecs' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 10), 'vecs', tuple_var_assignment_406945_407481)
    
    # Call to assert_allclose(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'eigs' (line 126)
    eigs_407483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), 'eigs', False)
    
    # Call to arange(...): (line 126)
    # Processing the call arguments (line 126)
    int_407486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 36), 'int')
    # Getting the type of 'm_excluded' (line 126)
    m_excluded_407487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 38), 'm_excluded', False)
    # Applying the binary operator '+' (line 126)
    result_add_407488 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 36), '+', int_407486, m_excluded_407487)
    
    int_407489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 50), 'int')
    # Getting the type of 'm_excluded' (line 126)
    m_excluded_407490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 52), 'm_excluded', False)
    # Applying the binary operator '+' (line 126)
    result_add_407491 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 50), '+', int_407489, m_excluded_407490)
    
    # Getting the type of 'm' (line 126)
    m_407492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 63), 'm', False)
    # Applying the binary operator '+' (line 126)
    result_add_407493 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 62), '+', result_add_407491, m_407492)
    
    # Processing the call keyword arguments (line 126)
    kwargs_407494 = {}
    # Getting the type of 'np' (line 126)
    np_407484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 26), 'np', False)
    # Obtaining the member 'arange' of a type (line 126)
    arange_407485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 26), np_407484, 'arange')
    # Calling arange(args, kwargs) (line 126)
    arange_call_result_407495 = invoke(stypy.reporting.localization.Localization(__file__, 126, 26), arange_407485, *[result_add_407488, result_add_407493], **kwargs_407494)
    
    # Processing the call keyword arguments (line 126)
    kwargs_407496 = {}
    # Getting the type of 'assert_allclose' (line 126)
    assert_allclose_407482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 126)
    assert_allclose_call_result_407497 = invoke(stypy.reporting.localization.Localization(__file__, 126, 4), assert_allclose_407482, *[eigs_407483, arange_call_result_407495], **kwargs_407496)
    
    
    # Call to _check_eigen(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'A' (line 127)
    A_407499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 17), 'A', False)
    # Getting the type of 'eigs' (line 127)
    eigs_407500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 20), 'eigs', False)
    # Getting the type of 'vecs' (line 127)
    vecs_407501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 26), 'vecs', False)
    # Processing the call keyword arguments (line 127)
    float_407502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 37), 'float')
    keyword_407503 = float_407502
    float_407504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 48), 'float')
    keyword_407505 = float_407504
    kwargs_407506 = {'rtol': keyword_407503, 'atol': keyword_407505}
    # Getting the type of '_check_eigen' (line 127)
    _check_eigen_407498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), '_check_eigen', False)
    # Calling _check_eigen(args, kwargs) (line 127)
    _check_eigen_call_result_407507 = invoke(stypy.reporting.localization.Localization(__file__, 127, 4), _check_eigen_407498, *[A_407499, eigs_407500, vecs_407501], **kwargs_407506)
    
    
    # ################# End of 'test_diagonal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_diagonal' in the type store
    # Getting the type of 'stypy_return_type' (line 90)
    stypy_return_type_407508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_407508)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_diagonal'
    return stypy_return_type_407508

# Assigning a type to the variable 'test_diagonal' (line 90)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'test_diagonal', test_diagonal)

@norecursion
def _check_eigen(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_407509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 31), 'float')
    float_407510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 42), 'float')
    defaults = [float_407509, float_407510]
    # Create a new context for function '_check_eigen'
    module_type_store = module_type_store.open_function_context('_check_eigen', 130, 0, False)
    
    # Passed parameters checking function
    _check_eigen.stypy_localization = localization
    _check_eigen.stypy_type_of_self = None
    _check_eigen.stypy_type_store = module_type_store
    _check_eigen.stypy_function_name = '_check_eigen'
    _check_eigen.stypy_param_names_list = ['M', 'w', 'V', 'rtol', 'atol']
    _check_eigen.stypy_varargs_param_name = None
    _check_eigen.stypy_kwargs_param_name = None
    _check_eigen.stypy_call_defaults = defaults
    _check_eigen.stypy_call_varargs = varargs
    _check_eigen.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_check_eigen', ['M', 'w', 'V', 'rtol', 'atol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_check_eigen', localization, ['M', 'w', 'V', 'rtol', 'atol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_check_eigen(...)' code ##################

    
    # Assigning a Call to a Name (line 131):
    
    # Assigning a Call to a Name (line 131):
    
    # Call to multiply(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'w' (line 131)
    w_407513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 26), 'w', False)
    # Getting the type of 'V' (line 131)
    V_407514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 29), 'V', False)
    # Processing the call keyword arguments (line 131)
    kwargs_407515 = {}
    # Getting the type of 'np' (line 131)
    np_407511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 14), 'np', False)
    # Obtaining the member 'multiply' of a type (line 131)
    multiply_407512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 14), np_407511, 'multiply')
    # Calling multiply(args, kwargs) (line 131)
    multiply_call_result_407516 = invoke(stypy.reporting.localization.Localization(__file__, 131, 14), multiply_407512, *[w_407513, V_407514], **kwargs_407515)
    
    # Assigning a type to the variable 'mult_wV' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'mult_wV', multiply_call_result_407516)
    
    # Assigning a Call to a Name (line 132):
    
    # Assigning a Call to a Name (line 132):
    
    # Call to dot(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'V' (line 132)
    V_407519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 19), 'V', False)
    # Processing the call keyword arguments (line 132)
    kwargs_407520 = {}
    # Getting the type of 'M' (line 132)
    M_407517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 13), 'M', False)
    # Obtaining the member 'dot' of a type (line 132)
    dot_407518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 13), M_407517, 'dot')
    # Calling dot(args, kwargs) (line 132)
    dot_call_result_407521 = invoke(stypy.reporting.localization.Localization(__file__, 132, 13), dot_407518, *[V_407519], **kwargs_407520)
    
    # Assigning a type to the variable 'dot_MV' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'dot_MV', dot_call_result_407521)
    
    # Call to assert_allclose(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'mult_wV' (line 133)
    mult_wV_407523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 20), 'mult_wV', False)
    # Getting the type of 'dot_MV' (line 133)
    dot_MV_407524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 29), 'dot_MV', False)
    # Processing the call keyword arguments (line 133)
    # Getting the type of 'rtol' (line 133)
    rtol_407525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 42), 'rtol', False)
    keyword_407526 = rtol_407525
    # Getting the type of 'atol' (line 133)
    atol_407527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 53), 'atol', False)
    keyword_407528 = atol_407527
    kwargs_407529 = {'rtol': keyword_407526, 'atol': keyword_407528}
    # Getting the type of 'assert_allclose' (line 133)
    assert_allclose_407522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 133)
    assert_allclose_call_result_407530 = invoke(stypy.reporting.localization.Localization(__file__, 133, 4), assert_allclose_407522, *[mult_wV_407523, dot_MV_407524], **kwargs_407529)
    
    
    # ################# End of '_check_eigen(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_check_eigen' in the type store
    # Getting the type of 'stypy_return_type' (line 130)
    stypy_return_type_407531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_407531)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_check_eigen'
    return stypy_return_type_407531

# Assigning a type to the variable '_check_eigen' (line 130)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 0), '_check_eigen', _check_eigen)

@norecursion
def _check_fiedler(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_check_fiedler'
    module_type_store = module_type_store.open_function_context('_check_fiedler', 136, 0, False)
    
    # Passed parameters checking function
    _check_fiedler.stypy_localization = localization
    _check_fiedler.stypy_type_of_self = None
    _check_fiedler.stypy_type_store = module_type_store
    _check_fiedler.stypy_function_name = '_check_fiedler'
    _check_fiedler.stypy_param_names_list = ['n', 'p']
    _check_fiedler.stypy_varargs_param_name = None
    _check_fiedler.stypy_kwargs_param_name = None
    _check_fiedler.stypy_call_defaults = defaults
    _check_fiedler.stypy_call_varargs = varargs
    _check_fiedler.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_check_fiedler', ['n', 'p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_check_fiedler', localization, ['n', 'p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_check_fiedler(...)' code ##################

    
    # Call to seed(...): (line 138)
    # Processing the call arguments (line 138)
    int_407535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 19), 'int')
    # Processing the call keyword arguments (line 138)
    kwargs_407536 = {}
    # Getting the type of 'np' (line 138)
    np_407532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 138)
    random_407533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 4), np_407532, 'random')
    # Obtaining the member 'seed' of a type (line 138)
    seed_407534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 4), random_407533, 'seed')
    # Calling seed(args, kwargs) (line 138)
    seed_call_result_407537 = invoke(stypy.reporting.localization.Localization(__file__, 138, 4), seed_407534, *[int_407535], **kwargs_407536)
    
    
    # Assigning a Call to a Name (line 139):
    
    # Assigning a Call to a Name (line 139):
    
    # Call to zeros(...): (line 139)
    # Processing the call arguments (line 139)
    # Getting the type of 'n' (line 139)
    n_407540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 19), 'n', False)
    # Processing the call keyword arguments (line 139)
    kwargs_407541 = {}
    # Getting the type of 'np' (line 139)
    np_407538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 10), 'np', False)
    # Obtaining the member 'zeros' of a type (line 139)
    zeros_407539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 10), np_407538, 'zeros')
    # Calling zeros(args, kwargs) (line 139)
    zeros_call_result_407542 = invoke(stypy.reporting.localization.Localization(__file__, 139, 10), zeros_407539, *[n_407540], **kwargs_407541)
    
    # Assigning a type to the variable 'col' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'col', zeros_call_result_407542)
    
    # Assigning a Num to a Subscript (line 140):
    
    # Assigning a Num to a Subscript (line 140):
    int_407543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 13), 'int')
    # Getting the type of 'col' (line 140)
    col_407544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'col')
    int_407545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 8), 'int')
    # Storing an element on a container (line 140)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 4), col_407544, (int_407545, int_407543))
    
    # Assigning a Call to a Name (line 141):
    
    # Assigning a Call to a Name (line 141):
    
    # Call to toeplitz(...): (line 141)
    # Processing the call arguments (line 141)
    # Getting the type of 'col' (line 141)
    col_407547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 17), 'col', False)
    # Processing the call keyword arguments (line 141)
    kwargs_407548 = {}
    # Getting the type of 'toeplitz' (line 141)
    toeplitz_407546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'toeplitz', False)
    # Calling toeplitz(args, kwargs) (line 141)
    toeplitz_call_result_407549 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), toeplitz_407546, *[col_407547], **kwargs_407548)
    
    # Assigning a type to the variable 'A' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'A', toeplitz_call_result_407549)
    
    # Assigning a Call to a Name (line 142):
    
    # Assigning a Call to a Name (line 142):
    
    # Call to diag(...): (line 142)
    # Processing the call arguments (line 142)
    
    # Call to sum(...): (line 142)
    # Processing the call keyword arguments (line 142)
    int_407554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 27), 'int')
    keyword_407555 = int_407554
    kwargs_407556 = {'axis': keyword_407555}
    # Getting the type of 'A' (line 142)
    A_407552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 16), 'A', False)
    # Obtaining the member 'sum' of a type (line 142)
    sum_407553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 16), A_407552, 'sum')
    # Calling sum(args, kwargs) (line 142)
    sum_call_result_407557 = invoke(stypy.reporting.localization.Localization(__file__, 142, 16), sum_407553, *[], **kwargs_407556)
    
    # Processing the call keyword arguments (line 142)
    kwargs_407558 = {}
    # Getting the type of 'np' (line 142)
    np_407550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'np', False)
    # Obtaining the member 'diag' of a type (line 142)
    diag_407551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), np_407550, 'diag')
    # Calling diag(args, kwargs) (line 142)
    diag_call_result_407559 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), diag_407551, *[sum_call_result_407557], **kwargs_407558)
    
    # Assigning a type to the variable 'D' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'D', diag_call_result_407559)
    
    # Assigning a BinOp to a Name (line 143):
    
    # Assigning a BinOp to a Name (line 143):
    # Getting the type of 'D' (line 143)
    D_407560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'D')
    # Getting the type of 'A' (line 143)
    A_407561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'A')
    # Applying the binary operator '-' (line 143)
    result_sub_407562 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 8), '-', D_407560, A_407561)
    
    # Assigning a type to the variable 'L' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'L', result_sub_407562)
    
    # Assigning a BinOp to a Name (line 146):
    
    # Assigning a BinOp to a Name (line 146):
    # Getting the type of 'np' (line 146)
    np_407563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 10), 'np')
    # Obtaining the member 'pi' of a type (line 146)
    pi_407564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 10), np_407563, 'pi')
    
    # Call to arange(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'n' (line 146)
    n_407567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 28), 'n', False)
    # Processing the call keyword arguments (line 146)
    kwargs_407568 = {}
    # Getting the type of 'np' (line 146)
    np_407565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 18), 'np', False)
    # Obtaining the member 'arange' of a type (line 146)
    arange_407566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 18), np_407565, 'arange')
    # Calling arange(args, kwargs) (line 146)
    arange_call_result_407569 = invoke(stypy.reporting.localization.Localization(__file__, 146, 18), arange_407566, *[n_407567], **kwargs_407568)
    
    # Applying the binary operator '*' (line 146)
    result_mul_407570 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 10), '*', pi_407564, arange_call_result_407569)
    
    # Getting the type of 'n' (line 146)
    n_407571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 33), 'n')
    # Applying the binary operator 'div' (line 146)
    result_div_407572 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 31), 'div', result_mul_407570, n_407571)
    
    # Assigning a type to the variable 'tmp' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'tmp', result_div_407572)
    
    # Assigning a BinOp to a Name (line 147):
    
    # Assigning a BinOp to a Name (line 147):
    int_407573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 17), 'int')
    int_407574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 22), 'int')
    
    # Call to cos(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 'tmp' (line 147)
    tmp_407577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 33), 'tmp', False)
    # Processing the call keyword arguments (line 147)
    kwargs_407578 = {}
    # Getting the type of 'np' (line 147)
    np_407575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 26), 'np', False)
    # Obtaining the member 'cos' of a type (line 147)
    cos_407576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 26), np_407575, 'cos')
    # Calling cos(args, kwargs) (line 147)
    cos_call_result_407579 = invoke(stypy.reporting.localization.Localization(__file__, 147, 26), cos_407576, *[tmp_407577], **kwargs_407578)
    
    # Applying the binary operator '-' (line 147)
    result_sub_407580 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 22), '-', int_407574, cos_call_result_407579)
    
    # Applying the binary operator '*' (line 147)
    result_mul_407581 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 17), '*', int_407573, result_sub_407580)
    
    # Assigning a type to the variable 'analytic_w' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'analytic_w', result_mul_407581)
    
    # Assigning a Call to a Name (line 148):
    
    # Assigning a Call to a Name (line 148):
    
    # Call to cos(...): (line 148)
    # Processing the call arguments (line 148)
    
    # Call to outer(...): (line 148)
    # Processing the call arguments (line 148)
    
    # Call to arange(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'n' (line 148)
    n_407588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 43), 'n', False)
    # Processing the call keyword arguments (line 148)
    kwargs_407589 = {}
    # Getting the type of 'np' (line 148)
    np_407586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 33), 'np', False)
    # Obtaining the member 'arange' of a type (line 148)
    arange_407587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 33), np_407586, 'arange')
    # Calling arange(args, kwargs) (line 148)
    arange_call_result_407590 = invoke(stypy.reporting.localization.Localization(__file__, 148, 33), arange_407587, *[n_407588], **kwargs_407589)
    
    int_407591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 48), 'int')
    int_407592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 50), 'int')
    # Applying the binary operator 'div' (line 148)
    result_div_407593 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 48), 'div', int_407591, int_407592)
    
    # Applying the binary operator '+' (line 148)
    result_add_407594 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 33), '+', arange_call_result_407590, result_div_407593)
    
    # Getting the type of 'tmp' (line 148)
    tmp_407595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 53), 'tmp', False)
    # Processing the call keyword arguments (line 148)
    kwargs_407596 = {}
    # Getting the type of 'np' (line 148)
    np_407584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 24), 'np', False)
    # Obtaining the member 'outer' of a type (line 148)
    outer_407585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 24), np_407584, 'outer')
    # Calling outer(args, kwargs) (line 148)
    outer_call_result_407597 = invoke(stypy.reporting.localization.Localization(__file__, 148, 24), outer_407585, *[result_add_407594, tmp_407595], **kwargs_407596)
    
    # Processing the call keyword arguments (line 148)
    kwargs_407598 = {}
    # Getting the type of 'np' (line 148)
    np_407582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 17), 'np', False)
    # Obtaining the member 'cos' of a type (line 148)
    cos_407583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 17), np_407582, 'cos')
    # Calling cos(args, kwargs) (line 148)
    cos_call_result_407599 = invoke(stypy.reporting.localization.Localization(__file__, 148, 17), cos_407583, *[outer_call_result_407597], **kwargs_407598)
    
    # Assigning a type to the variable 'analytic_V' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'analytic_V', cos_call_result_407599)
    
    # Call to _check_eigen(...): (line 149)
    # Processing the call arguments (line 149)
    # Getting the type of 'L' (line 149)
    L_407601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 17), 'L', False)
    # Getting the type of 'analytic_w' (line 149)
    analytic_w_407602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'analytic_w', False)
    # Getting the type of 'analytic_V' (line 149)
    analytic_V_407603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 32), 'analytic_V', False)
    # Processing the call keyword arguments (line 149)
    kwargs_407604 = {}
    # Getting the type of '_check_eigen' (line 149)
    _check_eigen_407600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), '_check_eigen', False)
    # Calling _check_eigen(args, kwargs) (line 149)
    _check_eigen_call_result_407605 = invoke(stypy.reporting.localization.Localization(__file__, 149, 4), _check_eigen_407600, *[L_407601, analytic_w_407602, analytic_V_407603], **kwargs_407604)
    
    
    # Assigning a Call to a Tuple (line 151):
    
    # Assigning a Subscript to a Name (line 151):
    
    # Obtaining the type of the subscript
    int_407606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 4), 'int')
    
    # Call to eigh(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'L' (line 151)
    L_407608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 26), 'L', False)
    # Processing the call keyword arguments (line 151)
    kwargs_407609 = {}
    # Getting the type of 'eigh' (line 151)
    eigh_407607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 21), 'eigh', False)
    # Calling eigh(args, kwargs) (line 151)
    eigh_call_result_407610 = invoke(stypy.reporting.localization.Localization(__file__, 151, 21), eigh_407607, *[L_407608], **kwargs_407609)
    
    # Obtaining the member '__getitem__' of a type (line 151)
    getitem___407611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 4), eigh_call_result_407610, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 151)
    subscript_call_result_407612 = invoke(stypy.reporting.localization.Localization(__file__, 151, 4), getitem___407611, int_407606)
    
    # Assigning a type to the variable 'tuple_var_assignment_406946' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'tuple_var_assignment_406946', subscript_call_result_407612)
    
    # Assigning a Subscript to a Name (line 151):
    
    # Obtaining the type of the subscript
    int_407613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 4), 'int')
    
    # Call to eigh(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'L' (line 151)
    L_407615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 26), 'L', False)
    # Processing the call keyword arguments (line 151)
    kwargs_407616 = {}
    # Getting the type of 'eigh' (line 151)
    eigh_407614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 21), 'eigh', False)
    # Calling eigh(args, kwargs) (line 151)
    eigh_call_result_407617 = invoke(stypy.reporting.localization.Localization(__file__, 151, 21), eigh_407614, *[L_407615], **kwargs_407616)
    
    # Obtaining the member '__getitem__' of a type (line 151)
    getitem___407618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 4), eigh_call_result_407617, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 151)
    subscript_call_result_407619 = invoke(stypy.reporting.localization.Localization(__file__, 151, 4), getitem___407618, int_407613)
    
    # Assigning a type to the variable 'tuple_var_assignment_406947' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'tuple_var_assignment_406947', subscript_call_result_407619)
    
    # Assigning a Name to a Name (line 151):
    # Getting the type of 'tuple_var_assignment_406946' (line 151)
    tuple_var_assignment_406946_407620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'tuple_var_assignment_406946')
    # Assigning a type to the variable 'eigh_w' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'eigh_w', tuple_var_assignment_406946_407620)
    
    # Assigning a Name to a Name (line 151):
    # Getting the type of 'tuple_var_assignment_406947' (line 151)
    tuple_var_assignment_406947_407621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'tuple_var_assignment_406947')
    # Assigning a type to the variable 'eigh_V' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'eigh_V', tuple_var_assignment_406947_407621)
    
    # Call to _check_eigen(...): (line 152)
    # Processing the call arguments (line 152)
    # Getting the type of 'L' (line 152)
    L_407623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 17), 'L', False)
    # Getting the type of 'eigh_w' (line 152)
    eigh_w_407624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 20), 'eigh_w', False)
    # Getting the type of 'eigh_V' (line 152)
    eigh_V_407625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 28), 'eigh_V', False)
    # Processing the call keyword arguments (line 152)
    kwargs_407626 = {}
    # Getting the type of '_check_eigen' (line 152)
    _check_eigen_407622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), '_check_eigen', False)
    # Calling _check_eigen(args, kwargs) (line 152)
    _check_eigen_call_result_407627 = invoke(stypy.reporting.localization.Localization(__file__, 152, 4), _check_eigen_407622, *[L_407623, eigh_w_407624, eigh_V_407625], **kwargs_407626)
    
    
    # Call to assert_array_less(...): (line 154)
    # Processing the call arguments (line 154)
    
    # Call to abs(...): (line 154)
    # Processing the call arguments (line 154)
    
    # Obtaining an instance of the builtin type 'list' (line 154)
    list_407631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 154)
    # Adding element type (line 154)
    
    # Obtaining the type of the subscript
    int_407632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 37), 'int')
    # Getting the type of 'eigh_w' (line 154)
    eigh_w_407633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 30), 'eigh_w', False)
    # Obtaining the member '__getitem__' of a type (line 154)
    getitem___407634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 30), eigh_w_407633, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 154)
    subscript_call_result_407635 = invoke(stypy.reporting.localization.Localization(__file__, 154, 30), getitem___407634, int_407632)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 29), list_407631, subscript_call_result_407635)
    # Adding element type (line 154)
    
    # Obtaining the type of the subscript
    int_407636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 52), 'int')
    # Getting the type of 'analytic_w' (line 154)
    analytic_w_407637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 41), 'analytic_w', False)
    # Obtaining the member '__getitem__' of a type (line 154)
    getitem___407638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 41), analytic_w_407637, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 154)
    subscript_call_result_407639 = invoke(stypy.reporting.localization.Localization(__file__, 154, 41), getitem___407638, int_407636)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 29), list_407631, subscript_call_result_407639)
    
    # Processing the call keyword arguments (line 154)
    kwargs_407640 = {}
    # Getting the type of 'np' (line 154)
    np_407629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 22), 'np', False)
    # Obtaining the member 'abs' of a type (line 154)
    abs_407630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 22), np_407629, 'abs')
    # Calling abs(args, kwargs) (line 154)
    abs_call_result_407641 = invoke(stypy.reporting.localization.Localization(__file__, 154, 22), abs_407630, *[list_407631], **kwargs_407640)
    
    float_407642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 58), 'float')
    # Processing the call keyword arguments (line 154)
    kwargs_407643 = {}
    # Getting the type of 'assert_array_less' (line 154)
    assert_array_less_407628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'assert_array_less', False)
    # Calling assert_array_less(args, kwargs) (line 154)
    assert_array_less_call_result_407644 = invoke(stypy.reporting.localization.Localization(__file__, 154, 4), assert_array_less_407628, *[abs_call_result_407641, float_407642], **kwargs_407643)
    
    
    # Call to assert_allclose(...): (line 155)
    # Processing the call arguments (line 155)
    
    # Obtaining the type of the subscript
    int_407646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 27), 'int')
    slice_407647 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 155, 20), int_407646, None, None)
    # Getting the type of 'eigh_w' (line 155)
    eigh_w_407648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 20), 'eigh_w', False)
    # Obtaining the member '__getitem__' of a type (line 155)
    getitem___407649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 20), eigh_w_407648, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 155)
    subscript_call_result_407650 = invoke(stypy.reporting.localization.Localization(__file__, 155, 20), getitem___407649, slice_407647)
    
    
    # Obtaining the type of the subscript
    int_407651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 43), 'int')
    slice_407652 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 155, 32), int_407651, None, None)
    # Getting the type of 'analytic_w' (line 155)
    analytic_w_407653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 32), 'analytic_w', False)
    # Obtaining the member '__getitem__' of a type (line 155)
    getitem___407654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 32), analytic_w_407653, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 155)
    subscript_call_result_407655 = invoke(stypy.reporting.localization.Localization(__file__, 155, 32), getitem___407654, slice_407652)
    
    # Processing the call keyword arguments (line 155)
    kwargs_407656 = {}
    # Getting the type of 'assert_allclose' (line 155)
    assert_allclose_407645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 155)
    assert_allclose_call_result_407657 = invoke(stypy.reporting.localization.Localization(__file__, 155, 4), assert_allclose_407645, *[subscript_call_result_407650, subscript_call_result_407655], **kwargs_407656)
    
    
    # Assigning a Subscript to a Name (line 158):
    
    # Assigning a Subscript to a Name (line 158):
    
    # Obtaining the type of the subscript
    slice_407658 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 158, 8), None, None, None)
    # Getting the type of 'p' (line 158)
    p_407659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 23), 'p')
    slice_407660 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 158, 8), None, p_407659, None)
    # Getting the type of 'analytic_V' (line 158)
    analytic_V_407661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'analytic_V')
    # Obtaining the member '__getitem__' of a type (line 158)
    getitem___407662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), analytic_V_407661, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 158)
    subscript_call_result_407663 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), getitem___407662, (slice_407658, slice_407660))
    
    # Assigning a type to the variable 'X' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'X', subscript_call_result_407663)
    
    # Assigning a Call to a Tuple (line 159):
    
    # Assigning a Subscript to a Name (line 159):
    
    # Obtaining the type of the subscript
    int_407664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 4), 'int')
    
    # Call to lobpcg(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of 'L' (line 159)
    L_407666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 32), 'L', False)
    # Getting the type of 'X' (line 159)
    X_407667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 35), 'X', False)
    # Processing the call keyword arguments (line 159)
    # Getting the type of 'False' (line 159)
    False_407668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 46), 'False', False)
    keyword_407669 = False_407668
    kwargs_407670 = {'largest': keyword_407669}
    # Getting the type of 'lobpcg' (line 159)
    lobpcg_407665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 25), 'lobpcg', False)
    # Calling lobpcg(args, kwargs) (line 159)
    lobpcg_call_result_407671 = invoke(stypy.reporting.localization.Localization(__file__, 159, 25), lobpcg_407665, *[L_407666, X_407667], **kwargs_407670)
    
    # Obtaining the member '__getitem__' of a type (line 159)
    getitem___407672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 4), lobpcg_call_result_407671, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 159)
    subscript_call_result_407673 = invoke(stypy.reporting.localization.Localization(__file__, 159, 4), getitem___407672, int_407664)
    
    # Assigning a type to the variable 'tuple_var_assignment_406948' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'tuple_var_assignment_406948', subscript_call_result_407673)
    
    # Assigning a Subscript to a Name (line 159):
    
    # Obtaining the type of the subscript
    int_407674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 4), 'int')
    
    # Call to lobpcg(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of 'L' (line 159)
    L_407676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 32), 'L', False)
    # Getting the type of 'X' (line 159)
    X_407677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 35), 'X', False)
    # Processing the call keyword arguments (line 159)
    # Getting the type of 'False' (line 159)
    False_407678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 46), 'False', False)
    keyword_407679 = False_407678
    kwargs_407680 = {'largest': keyword_407679}
    # Getting the type of 'lobpcg' (line 159)
    lobpcg_407675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 25), 'lobpcg', False)
    # Calling lobpcg(args, kwargs) (line 159)
    lobpcg_call_result_407681 = invoke(stypy.reporting.localization.Localization(__file__, 159, 25), lobpcg_407675, *[L_407676, X_407677], **kwargs_407680)
    
    # Obtaining the member '__getitem__' of a type (line 159)
    getitem___407682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 4), lobpcg_call_result_407681, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 159)
    subscript_call_result_407683 = invoke(stypy.reporting.localization.Localization(__file__, 159, 4), getitem___407682, int_407674)
    
    # Assigning a type to the variable 'tuple_var_assignment_406949' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'tuple_var_assignment_406949', subscript_call_result_407683)
    
    # Assigning a Name to a Name (line 159):
    # Getting the type of 'tuple_var_assignment_406948' (line 159)
    tuple_var_assignment_406948_407684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'tuple_var_assignment_406948')
    # Assigning a type to the variable 'lobpcg_w' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'lobpcg_w', tuple_var_assignment_406948_407684)
    
    # Assigning a Name to a Name (line 159):
    # Getting the type of 'tuple_var_assignment_406949' (line 159)
    tuple_var_assignment_406949_407685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'tuple_var_assignment_406949')
    # Assigning a type to the variable 'lobpcg_V' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 14), 'lobpcg_V', tuple_var_assignment_406949_407685)
    
    # Call to assert_equal(...): (line 160)
    # Processing the call arguments (line 160)
    # Getting the type of 'lobpcg_w' (line 160)
    lobpcg_w_407687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 17), 'lobpcg_w', False)
    # Obtaining the member 'shape' of a type (line 160)
    shape_407688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 17), lobpcg_w_407687, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 160)
    tuple_407689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 160)
    # Adding element type (line 160)
    # Getting the type of 'p' (line 160)
    p_407690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 34), 'p', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 34), tuple_407689, p_407690)
    
    # Processing the call keyword arguments (line 160)
    kwargs_407691 = {}
    # Getting the type of 'assert_equal' (line 160)
    assert_equal_407686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 160)
    assert_equal_call_result_407692 = invoke(stypy.reporting.localization.Localization(__file__, 160, 4), assert_equal_407686, *[shape_407688, tuple_407689], **kwargs_407691)
    
    
    # Call to assert_equal(...): (line 161)
    # Processing the call arguments (line 161)
    # Getting the type of 'lobpcg_V' (line 161)
    lobpcg_V_407694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 17), 'lobpcg_V', False)
    # Obtaining the member 'shape' of a type (line 161)
    shape_407695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 17), lobpcg_V_407694, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 161)
    tuple_407696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 161)
    # Adding element type (line 161)
    # Getting the type of 'n' (line 161)
    n_407697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 34), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 34), tuple_407696, n_407697)
    # Adding element type (line 161)
    # Getting the type of 'p' (line 161)
    p_407698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 37), 'p', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 34), tuple_407696, p_407698)
    
    # Processing the call keyword arguments (line 161)
    kwargs_407699 = {}
    # Getting the type of 'assert_equal' (line 161)
    assert_equal_407693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 161)
    assert_equal_call_result_407700 = invoke(stypy.reporting.localization.Localization(__file__, 161, 4), assert_equal_407693, *[shape_407695, tuple_407696], **kwargs_407699)
    
    
    # Call to _check_eigen(...): (line 162)
    # Processing the call arguments (line 162)
    # Getting the type of 'L' (line 162)
    L_407702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 17), 'L', False)
    # Getting the type of 'lobpcg_w' (line 162)
    lobpcg_w_407703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 20), 'lobpcg_w', False)
    # Getting the type of 'lobpcg_V' (line 162)
    lobpcg_V_407704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 30), 'lobpcg_V', False)
    # Processing the call keyword arguments (line 162)
    kwargs_407705 = {}
    # Getting the type of '_check_eigen' (line 162)
    _check_eigen_407701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), '_check_eigen', False)
    # Calling _check_eigen(args, kwargs) (line 162)
    _check_eigen_call_result_407706 = invoke(stypy.reporting.localization.Localization(__file__, 162, 4), _check_eigen_407701, *[L_407702, lobpcg_w_407703, lobpcg_V_407704], **kwargs_407705)
    
    
    # Call to assert_array_less(...): (line 163)
    # Processing the call arguments (line 163)
    
    # Call to abs(...): (line 163)
    # Processing the call arguments (line 163)
    
    # Call to min(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'lobpcg_w' (line 163)
    lobpcg_w_407712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 36), 'lobpcg_w', False)
    # Processing the call keyword arguments (line 163)
    kwargs_407713 = {}
    # Getting the type of 'np' (line 163)
    np_407710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 29), 'np', False)
    # Obtaining the member 'min' of a type (line 163)
    min_407711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 29), np_407710, 'min')
    # Calling min(args, kwargs) (line 163)
    min_call_result_407714 = invoke(stypy.reporting.localization.Localization(__file__, 163, 29), min_407711, *[lobpcg_w_407712], **kwargs_407713)
    
    # Processing the call keyword arguments (line 163)
    kwargs_407715 = {}
    # Getting the type of 'np' (line 163)
    np_407708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 22), 'np', False)
    # Obtaining the member 'abs' of a type (line 163)
    abs_407709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 22), np_407708, 'abs')
    # Calling abs(args, kwargs) (line 163)
    abs_call_result_407716 = invoke(stypy.reporting.localization.Localization(__file__, 163, 22), abs_407709, *[min_call_result_407714], **kwargs_407715)
    
    float_407717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 48), 'float')
    # Processing the call keyword arguments (line 163)
    kwargs_407718 = {}
    # Getting the type of 'assert_array_less' (line 163)
    assert_array_less_407707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'assert_array_less', False)
    # Calling assert_array_less(args, kwargs) (line 163)
    assert_array_less_call_result_407719 = invoke(stypy.reporting.localization.Localization(__file__, 163, 4), assert_array_less_407707, *[abs_call_result_407716, float_407717], **kwargs_407718)
    
    
    # Call to assert_allclose(...): (line 164)
    # Processing the call arguments (line 164)
    
    # Obtaining the type of the subscript
    int_407721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 38), 'int')
    slice_407722 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 164, 20), int_407721, None, None)
    
    # Call to sort(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'lobpcg_w' (line 164)
    lobpcg_w_407725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'lobpcg_w', False)
    # Processing the call keyword arguments (line 164)
    kwargs_407726 = {}
    # Getting the type of 'np' (line 164)
    np_407723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 20), 'np', False)
    # Obtaining the member 'sort' of a type (line 164)
    sort_407724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 20), np_407723, 'sort')
    # Calling sort(args, kwargs) (line 164)
    sort_call_result_407727 = invoke(stypy.reporting.localization.Localization(__file__, 164, 20), sort_407724, *[lobpcg_w_407725], **kwargs_407726)
    
    # Obtaining the member '__getitem__' of a type (line 164)
    getitem___407728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 20), sort_call_result_407727, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 164)
    subscript_call_result_407729 = invoke(stypy.reporting.localization.Localization(__file__, 164, 20), getitem___407728, slice_407722)
    
    
    # Obtaining the type of the subscript
    int_407730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 54), 'int')
    # Getting the type of 'p' (line 164)
    p_407731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 56), 'p', False)
    slice_407732 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 164, 43), int_407730, p_407731, None)
    # Getting the type of 'analytic_w' (line 164)
    analytic_w_407733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 43), 'analytic_w', False)
    # Obtaining the member '__getitem__' of a type (line 164)
    getitem___407734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 43), analytic_w_407733, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 164)
    subscript_call_result_407735 = invoke(stypy.reporting.localization.Localization(__file__, 164, 43), getitem___407734, slice_407732)
    
    # Processing the call keyword arguments (line 164)
    kwargs_407736 = {}
    # Getting the type of 'assert_allclose' (line 164)
    assert_allclose_407720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 164)
    assert_allclose_call_result_407737 = invoke(stypy.reporting.localization.Localization(__file__, 164, 4), assert_allclose_407720, *[subscript_call_result_407729, subscript_call_result_407735], **kwargs_407736)
    
    
    # Assigning a Subscript to a Name (line 167):
    
    # Assigning a Subscript to a Name (line 167):
    
    # Obtaining the type of the subscript
    slice_407738 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 167, 8), None, None, None)
    
    # Getting the type of 'p' (line 167)
    p_407739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 23), 'p')
    # Applying the 'usub' unary operator (line 167)
    result___neg___407740 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 22), 'usub', p_407739)
    
    slice_407741 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 167, 8), result___neg___407740, None, None)
    # Getting the type of 'analytic_V' (line 167)
    analytic_V_407742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'analytic_V')
    # Obtaining the member '__getitem__' of a type (line 167)
    getitem___407743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), analytic_V_407742, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 167)
    subscript_call_result_407744 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), getitem___407743, (slice_407738, slice_407741))
    
    # Assigning a type to the variable 'X' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'X', subscript_call_result_407744)
    
    # Assigning a Call to a Tuple (line 168):
    
    # Assigning a Subscript to a Name (line 168):
    
    # Obtaining the type of the subscript
    int_407745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 4), 'int')
    
    # Call to lobpcg(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'L' (line 168)
    L_407747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 32), 'L', False)
    # Getting the type of 'X' (line 168)
    X_407748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 35), 'X', False)
    # Processing the call keyword arguments (line 168)
    # Getting the type of 'True' (line 168)
    True_407749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 46), 'True', False)
    keyword_407750 = True_407749
    kwargs_407751 = {'largest': keyword_407750}
    # Getting the type of 'lobpcg' (line 168)
    lobpcg_407746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 25), 'lobpcg', False)
    # Calling lobpcg(args, kwargs) (line 168)
    lobpcg_call_result_407752 = invoke(stypy.reporting.localization.Localization(__file__, 168, 25), lobpcg_407746, *[L_407747, X_407748], **kwargs_407751)
    
    # Obtaining the member '__getitem__' of a type (line 168)
    getitem___407753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 4), lobpcg_call_result_407752, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 168)
    subscript_call_result_407754 = invoke(stypy.reporting.localization.Localization(__file__, 168, 4), getitem___407753, int_407745)
    
    # Assigning a type to the variable 'tuple_var_assignment_406950' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'tuple_var_assignment_406950', subscript_call_result_407754)
    
    # Assigning a Subscript to a Name (line 168):
    
    # Obtaining the type of the subscript
    int_407755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 4), 'int')
    
    # Call to lobpcg(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'L' (line 168)
    L_407757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 32), 'L', False)
    # Getting the type of 'X' (line 168)
    X_407758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 35), 'X', False)
    # Processing the call keyword arguments (line 168)
    # Getting the type of 'True' (line 168)
    True_407759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 46), 'True', False)
    keyword_407760 = True_407759
    kwargs_407761 = {'largest': keyword_407760}
    # Getting the type of 'lobpcg' (line 168)
    lobpcg_407756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 25), 'lobpcg', False)
    # Calling lobpcg(args, kwargs) (line 168)
    lobpcg_call_result_407762 = invoke(stypy.reporting.localization.Localization(__file__, 168, 25), lobpcg_407756, *[L_407757, X_407758], **kwargs_407761)
    
    # Obtaining the member '__getitem__' of a type (line 168)
    getitem___407763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 4), lobpcg_call_result_407762, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 168)
    subscript_call_result_407764 = invoke(stypy.reporting.localization.Localization(__file__, 168, 4), getitem___407763, int_407755)
    
    # Assigning a type to the variable 'tuple_var_assignment_406951' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'tuple_var_assignment_406951', subscript_call_result_407764)
    
    # Assigning a Name to a Name (line 168):
    # Getting the type of 'tuple_var_assignment_406950' (line 168)
    tuple_var_assignment_406950_407765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'tuple_var_assignment_406950')
    # Assigning a type to the variable 'lobpcg_w' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'lobpcg_w', tuple_var_assignment_406950_407765)
    
    # Assigning a Name to a Name (line 168):
    # Getting the type of 'tuple_var_assignment_406951' (line 168)
    tuple_var_assignment_406951_407766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'tuple_var_assignment_406951')
    # Assigning a type to the variable 'lobpcg_V' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 14), 'lobpcg_V', tuple_var_assignment_406951_407766)
    
    # Call to assert_equal(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'lobpcg_w' (line 169)
    lobpcg_w_407768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 17), 'lobpcg_w', False)
    # Obtaining the member 'shape' of a type (line 169)
    shape_407769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 17), lobpcg_w_407768, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 169)
    tuple_407770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 169)
    # Adding element type (line 169)
    # Getting the type of 'p' (line 169)
    p_407771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 34), 'p', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 34), tuple_407770, p_407771)
    
    # Processing the call keyword arguments (line 169)
    kwargs_407772 = {}
    # Getting the type of 'assert_equal' (line 169)
    assert_equal_407767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 169)
    assert_equal_call_result_407773 = invoke(stypy.reporting.localization.Localization(__file__, 169, 4), assert_equal_407767, *[shape_407769, tuple_407770], **kwargs_407772)
    
    
    # Call to assert_equal(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'lobpcg_V' (line 170)
    lobpcg_V_407775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 17), 'lobpcg_V', False)
    # Obtaining the member 'shape' of a type (line 170)
    shape_407776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 17), lobpcg_V_407775, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 170)
    tuple_407777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 170)
    # Adding element type (line 170)
    # Getting the type of 'n' (line 170)
    n_407778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 34), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 34), tuple_407777, n_407778)
    # Adding element type (line 170)
    # Getting the type of 'p' (line 170)
    p_407779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 37), 'p', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 34), tuple_407777, p_407779)
    
    # Processing the call keyword arguments (line 170)
    kwargs_407780 = {}
    # Getting the type of 'assert_equal' (line 170)
    assert_equal_407774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 170)
    assert_equal_call_result_407781 = invoke(stypy.reporting.localization.Localization(__file__, 170, 4), assert_equal_407774, *[shape_407776, tuple_407777], **kwargs_407780)
    
    
    # Call to _check_eigen(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'L' (line 171)
    L_407783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 17), 'L', False)
    # Getting the type of 'lobpcg_w' (line 171)
    lobpcg_w_407784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 20), 'lobpcg_w', False)
    # Getting the type of 'lobpcg_V' (line 171)
    lobpcg_V_407785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 30), 'lobpcg_V', False)
    # Processing the call keyword arguments (line 171)
    kwargs_407786 = {}
    # Getting the type of '_check_eigen' (line 171)
    _check_eigen_407782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), '_check_eigen', False)
    # Calling _check_eigen(args, kwargs) (line 171)
    _check_eigen_call_result_407787 = invoke(stypy.reporting.localization.Localization(__file__, 171, 4), _check_eigen_407782, *[L_407783, lobpcg_w_407784, lobpcg_V_407785], **kwargs_407786)
    
    
    # Call to assert_allclose(...): (line 172)
    # Processing the call arguments (line 172)
    
    # Call to sort(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'lobpcg_w' (line 172)
    lobpcg_w_407791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 28), 'lobpcg_w', False)
    # Processing the call keyword arguments (line 172)
    kwargs_407792 = {}
    # Getting the type of 'np' (line 172)
    np_407789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 20), 'np', False)
    # Obtaining the member 'sort' of a type (line 172)
    sort_407790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 20), np_407789, 'sort')
    # Calling sort(args, kwargs) (line 172)
    sort_call_result_407793 = invoke(stypy.reporting.localization.Localization(__file__, 172, 20), sort_407790, *[lobpcg_w_407791], **kwargs_407792)
    
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'p' (line 172)
    p_407794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 51), 'p', False)
    # Applying the 'usub' unary operator (line 172)
    result___neg___407795 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 50), 'usub', p_407794)
    
    slice_407796 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 172, 39), result___neg___407795, None, None)
    # Getting the type of 'analytic_w' (line 172)
    analytic_w_407797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 39), 'analytic_w', False)
    # Obtaining the member '__getitem__' of a type (line 172)
    getitem___407798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 39), analytic_w_407797, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 172)
    subscript_call_result_407799 = invoke(stypy.reporting.localization.Localization(__file__, 172, 39), getitem___407798, slice_407796)
    
    # Processing the call keyword arguments (line 172)
    kwargs_407800 = {}
    # Getting the type of 'assert_allclose' (line 172)
    assert_allclose_407788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 172)
    assert_allclose_call_result_407801 = invoke(stypy.reporting.localization.Localization(__file__, 172, 4), assert_allclose_407788, *[sort_call_result_407793, subscript_call_result_407799], **kwargs_407800)
    
    
    # Assigning a Call to a Name (line 175):
    
    # Assigning a Call to a Name (line 175):
    
    # Call to concatenate(...): (line 175)
    # Processing the call arguments (line 175)
    
    # Obtaining an instance of the builtin type 'tuple' (line 175)
    tuple_407804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 175)
    # Adding element type (line 175)
    
    # Call to ones(...): (line 175)
    # Processing the call arguments (line 175)
    # Getting the type of 'n' (line 175)
    n_407807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 44), 'n', False)
    int_407808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 47), 'int')
    # Applying the binary operator '//' (line 175)
    result_floordiv_407809 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 44), '//', n_407807, int_407808)
    
    # Processing the call keyword arguments (line 175)
    kwargs_407810 = {}
    # Getting the type of 'np' (line 175)
    np_407805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 36), 'np', False)
    # Obtaining the member 'ones' of a type (line 175)
    ones_407806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 36), np_407805, 'ones')
    # Calling ones(args, kwargs) (line 175)
    ones_call_result_407811 = invoke(stypy.reporting.localization.Localization(__file__, 175, 36), ones_407806, *[result_floordiv_407809], **kwargs_407810)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 36), tuple_407804, ones_call_result_407811)
    # Adding element type (line 175)
    
    
    # Call to ones(...): (line 175)
    # Processing the call arguments (line 175)
    # Getting the type of 'n' (line 175)
    n_407814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 60), 'n', False)
    # Getting the type of 'n' (line 175)
    n_407815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 62), 'n', False)
    int_407816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 65), 'int')
    # Applying the binary operator '//' (line 175)
    result_floordiv_407817 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 62), '//', n_407815, int_407816)
    
    # Applying the binary operator '-' (line 175)
    result_sub_407818 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 60), '-', n_407814, result_floordiv_407817)
    
    # Processing the call keyword arguments (line 175)
    kwargs_407819 = {}
    # Getting the type of 'np' (line 175)
    np_407812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 52), 'np', False)
    # Obtaining the member 'ones' of a type (line 175)
    ones_407813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 52), np_407812, 'ones')
    # Calling ones(args, kwargs) (line 175)
    ones_call_result_407820 = invoke(stypy.reporting.localization.Localization(__file__, 175, 52), ones_407813, *[result_sub_407818], **kwargs_407819)
    
    # Applying the 'usub' unary operator (line 175)
    result___neg___407821 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 51), 'usub', ones_call_result_407820)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 36), tuple_407804, result___neg___407821)
    
    # Processing the call keyword arguments (line 175)
    kwargs_407822 = {}
    # Getting the type of 'np' (line 175)
    np_407802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 20), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 175)
    concatenate_407803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 20), np_407802, 'concatenate')
    # Calling concatenate(args, kwargs) (line 175)
    concatenate_call_result_407823 = invoke(stypy.reporting.localization.Localization(__file__, 175, 20), concatenate_407803, *[tuple_407804], **kwargs_407822)
    
    # Assigning a type to the variable 'fiedler_guess' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'fiedler_guess', concatenate_call_result_407823)
    
    # Assigning a Attribute to a Name (line 176):
    
    # Assigning a Attribute to a Name (line 176):
    
    # Call to vstack(...): (line 176)
    # Processing the call arguments (line 176)
    
    # Obtaining an instance of the builtin type 'tuple' (line 176)
    tuple_407826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 176)
    # Adding element type (line 176)
    
    # Call to ones(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'n' (line 176)
    n_407829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 27), 'n', False)
    # Processing the call keyword arguments (line 176)
    kwargs_407830 = {}
    # Getting the type of 'np' (line 176)
    np_407827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 19), 'np', False)
    # Obtaining the member 'ones' of a type (line 176)
    ones_407828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 19), np_407827, 'ones')
    # Calling ones(args, kwargs) (line 176)
    ones_call_result_407831 = invoke(stypy.reporting.localization.Localization(__file__, 176, 19), ones_407828, *[n_407829], **kwargs_407830)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 19), tuple_407826, ones_call_result_407831)
    # Adding element type (line 176)
    # Getting the type of 'fiedler_guess' (line 176)
    fiedler_guess_407832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 31), 'fiedler_guess', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 19), tuple_407826, fiedler_guess_407832)
    
    # Processing the call keyword arguments (line 176)
    kwargs_407833 = {}
    # Getting the type of 'np' (line 176)
    np_407824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'np', False)
    # Obtaining the member 'vstack' of a type (line 176)
    vstack_407825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), np_407824, 'vstack')
    # Calling vstack(args, kwargs) (line 176)
    vstack_call_result_407834 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), vstack_407825, *[tuple_407826], **kwargs_407833)
    
    # Obtaining the member 'T' of a type (line 176)
    T_407835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), vstack_call_result_407834, 'T')
    # Assigning a type to the variable 'X' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'X', T_407835)
    
    # Assigning a Call to a Tuple (line 177):
    
    # Assigning a Subscript to a Name (line 177):
    
    # Obtaining the type of the subscript
    int_407836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 4), 'int')
    
    # Call to lobpcg(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'L' (line 177)
    L_407838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 32), 'L', False)
    # Getting the type of 'X' (line 177)
    X_407839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 35), 'X', False)
    # Processing the call keyword arguments (line 177)
    # Getting the type of 'False' (line 177)
    False_407840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 46), 'False', False)
    keyword_407841 = False_407840
    kwargs_407842 = {'largest': keyword_407841}
    # Getting the type of 'lobpcg' (line 177)
    lobpcg_407837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 25), 'lobpcg', False)
    # Calling lobpcg(args, kwargs) (line 177)
    lobpcg_call_result_407843 = invoke(stypy.reporting.localization.Localization(__file__, 177, 25), lobpcg_407837, *[L_407838, X_407839], **kwargs_407842)
    
    # Obtaining the member '__getitem__' of a type (line 177)
    getitem___407844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 4), lobpcg_call_result_407843, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 177)
    subscript_call_result_407845 = invoke(stypy.reporting.localization.Localization(__file__, 177, 4), getitem___407844, int_407836)
    
    # Assigning a type to the variable 'tuple_var_assignment_406952' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'tuple_var_assignment_406952', subscript_call_result_407845)
    
    # Assigning a Subscript to a Name (line 177):
    
    # Obtaining the type of the subscript
    int_407846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 4), 'int')
    
    # Call to lobpcg(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'L' (line 177)
    L_407848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 32), 'L', False)
    # Getting the type of 'X' (line 177)
    X_407849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 35), 'X', False)
    # Processing the call keyword arguments (line 177)
    # Getting the type of 'False' (line 177)
    False_407850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 46), 'False', False)
    keyword_407851 = False_407850
    kwargs_407852 = {'largest': keyword_407851}
    # Getting the type of 'lobpcg' (line 177)
    lobpcg_407847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 25), 'lobpcg', False)
    # Calling lobpcg(args, kwargs) (line 177)
    lobpcg_call_result_407853 = invoke(stypy.reporting.localization.Localization(__file__, 177, 25), lobpcg_407847, *[L_407848, X_407849], **kwargs_407852)
    
    # Obtaining the member '__getitem__' of a type (line 177)
    getitem___407854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 4), lobpcg_call_result_407853, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 177)
    subscript_call_result_407855 = invoke(stypy.reporting.localization.Localization(__file__, 177, 4), getitem___407854, int_407846)
    
    # Assigning a type to the variable 'tuple_var_assignment_406953' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'tuple_var_assignment_406953', subscript_call_result_407855)
    
    # Assigning a Name to a Name (line 177):
    # Getting the type of 'tuple_var_assignment_406952' (line 177)
    tuple_var_assignment_406952_407856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'tuple_var_assignment_406952')
    # Assigning a type to the variable 'lobpcg_w' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'lobpcg_w', tuple_var_assignment_406952_407856)
    
    # Assigning a Name to a Name (line 177):
    # Getting the type of 'tuple_var_assignment_406953' (line 177)
    tuple_var_assignment_406953_407857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'tuple_var_assignment_406953')
    # Assigning a type to the variable 'lobpcg_V' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 14), 'lobpcg_V', tuple_var_assignment_406953_407857)
    
    # Assigning a Call to a Name (line 180):
    
    # Assigning a Call to a Name (line 180):
    
    # Call to sort(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'lobpcg_w' (line 180)
    lobpcg_w_407860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 23), 'lobpcg_w', False)
    # Processing the call keyword arguments (line 180)
    kwargs_407861 = {}
    # Getting the type of 'np' (line 180)
    np_407858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 15), 'np', False)
    # Obtaining the member 'sort' of a type (line 180)
    sort_407859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 15), np_407858, 'sort')
    # Calling sort(args, kwargs) (line 180)
    sort_call_result_407862 = invoke(stypy.reporting.localization.Localization(__file__, 180, 15), sort_407859, *[lobpcg_w_407860], **kwargs_407861)
    
    # Assigning a type to the variable 'lobpcg_w' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'lobpcg_w', sort_call_result_407862)
    
    # Call to assert_allclose(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'lobpcg_w' (line 181)
    lobpcg_w_407864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 20), 'lobpcg_w', False)
    
    # Obtaining the type of the subscript
    int_407865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 42), 'int')
    slice_407866 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 181, 30), None, int_407865, None)
    # Getting the type of 'analytic_w' (line 181)
    analytic_w_407867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 30), 'analytic_w', False)
    # Obtaining the member '__getitem__' of a type (line 181)
    getitem___407868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 30), analytic_w_407867, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 181)
    subscript_call_result_407869 = invoke(stypy.reporting.localization.Localization(__file__, 181, 30), getitem___407868, slice_407866)
    
    # Processing the call keyword arguments (line 181)
    float_407870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 51), 'float')
    keyword_407871 = float_407870
    kwargs_407872 = {'atol': keyword_407871}
    # Getting the type of 'assert_allclose' (line 181)
    assert_allclose_407863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 181)
    assert_allclose_call_result_407873 = invoke(stypy.reporting.localization.Localization(__file__, 181, 4), assert_allclose_407863, *[lobpcg_w_407864, subscript_call_result_407869], **kwargs_407872)
    
    
    # ################# End of '_check_fiedler(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_check_fiedler' in the type store
    # Getting the type of 'stypy_return_type' (line 136)
    stypy_return_type_407874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_407874)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_check_fiedler'
    return stypy_return_type_407874

# Assigning a type to the variable '_check_fiedler' (line 136)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), '_check_fiedler', _check_fiedler)

@norecursion
def test_fiedler_small_8(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_fiedler_small_8'
    module_type_store = module_type_store.open_function_context('test_fiedler_small_8', 184, 0, False)
    
    # Passed parameters checking function
    test_fiedler_small_8.stypy_localization = localization
    test_fiedler_small_8.stypy_type_of_self = None
    test_fiedler_small_8.stypy_type_store = module_type_store
    test_fiedler_small_8.stypy_function_name = 'test_fiedler_small_8'
    test_fiedler_small_8.stypy_param_names_list = []
    test_fiedler_small_8.stypy_varargs_param_name = None
    test_fiedler_small_8.stypy_kwargs_param_name = None
    test_fiedler_small_8.stypy_call_defaults = defaults
    test_fiedler_small_8.stypy_call_varargs = varargs
    test_fiedler_small_8.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_fiedler_small_8', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_fiedler_small_8', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_fiedler_small_8(...)' code ##################

    
    # Call to _check_fiedler(...): (line 186)
    # Processing the call arguments (line 186)
    int_407876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 19), 'int')
    int_407877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 22), 'int')
    # Processing the call keyword arguments (line 186)
    kwargs_407878 = {}
    # Getting the type of '_check_fiedler' (line 186)
    _check_fiedler_407875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), '_check_fiedler', False)
    # Calling _check_fiedler(args, kwargs) (line 186)
    _check_fiedler_call_result_407879 = invoke(stypy.reporting.localization.Localization(__file__, 186, 4), _check_fiedler_407875, *[int_407876, int_407877], **kwargs_407878)
    
    
    # ################# End of 'test_fiedler_small_8(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_fiedler_small_8' in the type store
    # Getting the type of 'stypy_return_type' (line 184)
    stypy_return_type_407880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_407880)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_fiedler_small_8'
    return stypy_return_type_407880

# Assigning a type to the variable 'test_fiedler_small_8' (line 184)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 0), 'test_fiedler_small_8', test_fiedler_small_8)

@norecursion
def test_fiedler_large_12(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_fiedler_large_12'
    module_type_store = module_type_store.open_function_context('test_fiedler_large_12', 189, 0, False)
    
    # Passed parameters checking function
    test_fiedler_large_12.stypy_localization = localization
    test_fiedler_large_12.stypy_type_of_self = None
    test_fiedler_large_12.stypy_type_store = module_type_store
    test_fiedler_large_12.stypy_function_name = 'test_fiedler_large_12'
    test_fiedler_large_12.stypy_param_names_list = []
    test_fiedler_large_12.stypy_varargs_param_name = None
    test_fiedler_large_12.stypy_kwargs_param_name = None
    test_fiedler_large_12.stypy_call_defaults = defaults
    test_fiedler_large_12.stypy_call_varargs = varargs
    test_fiedler_large_12.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_fiedler_large_12', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_fiedler_large_12', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_fiedler_large_12(...)' code ##################

    
    # Call to _check_fiedler(...): (line 191)
    # Processing the call arguments (line 191)
    int_407882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 19), 'int')
    int_407883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 23), 'int')
    # Processing the call keyword arguments (line 191)
    kwargs_407884 = {}
    # Getting the type of '_check_fiedler' (line 191)
    _check_fiedler_407881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), '_check_fiedler', False)
    # Calling _check_fiedler(args, kwargs) (line 191)
    _check_fiedler_call_result_407885 = invoke(stypy.reporting.localization.Localization(__file__, 191, 4), _check_fiedler_407881, *[int_407882, int_407883], **kwargs_407884)
    
    
    # ################# End of 'test_fiedler_large_12(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_fiedler_large_12' in the type store
    # Getting the type of 'stypy_return_type' (line 189)
    stypy_return_type_407886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_407886)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_fiedler_large_12'
    return stypy_return_type_407886

# Assigning a type to the variable 'test_fiedler_large_12' (line 189)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 0), 'test_fiedler_large_12', test_fiedler_large_12)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
