
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from itertools import product, permutations
2: 
3: import numpy as np
4: from numpy.testing import assert_array_less, assert_allclose
5: from pytest import raises as assert_raises
6: 
7: from scipy.linalg import inv, eigh, norm
8: from scipy.linalg import orthogonal_procrustes
9: 
10: 
11: def test_orthogonal_procrustes_ndim_too_large():
12:     np.random.seed(1234)
13:     A = np.random.randn(3, 4, 5)
14:     B = np.random.randn(3, 4, 5)
15:     assert_raises(ValueError, orthogonal_procrustes, A, B)
16: 
17: 
18: def test_orthogonal_procrustes_ndim_too_small():
19:     np.random.seed(1234)
20:     A = np.random.randn(3)
21:     B = np.random.randn(3)
22:     assert_raises(ValueError, orthogonal_procrustes, A, B)
23: 
24: 
25: def test_orthogonal_procrustes_shape_mismatch():
26:     np.random.seed(1234)
27:     shapes = ((3, 3), (3, 4), (4, 3), (4, 4))
28:     for a, b in permutations(shapes, 2):
29:         A = np.random.randn(*a)
30:         B = np.random.randn(*b)
31:         assert_raises(ValueError, orthogonal_procrustes, A, B)
32: 
33: 
34: def test_orthogonal_procrustes_checkfinite_exception():
35:     np.random.seed(1234)
36:     m, n = 2, 3
37:     A_good = np.random.randn(m, n)
38:     B_good = np.random.randn(m, n)
39:     for bad_value in np.inf, -np.inf, np.nan:
40:         A_bad = A_good.copy()
41:         A_bad[1, 2] = bad_value
42:         B_bad = B_good.copy()
43:         B_bad[1, 2] = bad_value
44:         for A, B in ((A_good, B_bad), (A_bad, B_good), (A_bad, B_bad)):
45:             assert_raises(ValueError, orthogonal_procrustes, A, B)
46: 
47: 
48: def test_orthogonal_procrustes_scale_invariance():
49:     np.random.seed(1234)
50:     m, n = 4, 3
51:     for i in range(3):
52:         A_orig = np.random.randn(m, n)
53:         B_orig = np.random.randn(m, n)
54:         R_orig, s = orthogonal_procrustes(A_orig, B_orig)
55:         for A_scale in np.square(np.random.randn(3)):
56:             for B_scale in np.square(np.random.randn(3)):
57:                 R, s = orthogonal_procrustes(A_orig * A_scale, B_orig * B_scale)
58:                 assert_allclose(R, R_orig)
59: 
60: 
61: def test_orthogonal_procrustes_array_conversion():
62:     np.random.seed(1234)
63:     for m, n in ((6, 4), (4, 4), (4, 6)):
64:         A_arr = np.random.randn(m, n)
65:         B_arr = np.random.randn(m, n)
66:         As = (A_arr, A_arr.tolist(), np.matrix(A_arr))
67:         Bs = (B_arr, B_arr.tolist(), np.matrix(B_arr))
68:         R_arr, s = orthogonal_procrustes(A_arr, B_arr)
69:         AR_arr = A_arr.dot(R_arr)
70:         for A, B in product(As, Bs):
71:             R, s = orthogonal_procrustes(A, B)
72:             AR = A_arr.dot(R)
73:             assert_allclose(AR, AR_arr)
74: 
75: 
76: def test_orthogonal_procrustes():
77:     np.random.seed(1234)
78:     for m, n in ((6, 4), (4, 4), (4, 6)):
79:         # Sample a random target matrix.
80:         B = np.random.randn(m, n)
81:         # Sample a random orthogonal matrix
82:         # by computing eigh of a sampled symmetric matrix.
83:         X = np.random.randn(n, n)
84:         w, V = eigh(X.T + X)
85:         assert_allclose(inv(V), V.T)
86:         # Compute a matrix with a known orthogonal transformation that gives B.
87:         A = np.dot(B, V.T)
88:         # Check that an orthogonal transformation from A to B can be recovered.
89:         R, s = orthogonal_procrustes(A, B)
90:         assert_allclose(inv(R), R.T)
91:         assert_allclose(A.dot(R), B)
92:         # Create a perturbed input matrix.
93:         A_perturbed = A + 1e-2 * np.random.randn(m, n)
94:         # Check that the orthogonal procrustes function can find an orthogonal
95:         # transformation that is better than the orthogonal transformation
96:         # computed from the original input matrix.
97:         R_prime, s = orthogonal_procrustes(A_perturbed, B)
98:         assert_allclose(inv(R_prime), R_prime.T)
99:         # Compute the naive and optimal transformations of the perturbed input.
100:         naive_approx = A_perturbed.dot(R)
101:         optim_approx = A_perturbed.dot(R_prime)
102:         # Compute the Frobenius norm errors of the matrix approximations.
103:         naive_approx_error = norm(naive_approx - B, ord='fro')
104:         optim_approx_error = norm(optim_approx - B, ord='fro')
105:         # Check that the orthogonal Procrustes approximation is better.
106:         assert_array_less(optim_approx_error, naive_approx_error)
107: 
108: 
109: def _centered(A):
110:     mu = A.mean(axis=0)
111:     return A - mu, mu
112: 
113: 
114: def test_orthogonal_procrustes_exact_example():
115:     # Check a small application.
116:     # It uses translation, scaling, reflection, and rotation.
117:     #
118:     #         |
119:     #   a  b  |
120:     #         |
121:     #   d  c  |        w
122:     #         |
123:     # --------+--- x ----- z ---
124:     #         |
125:     #         |        y
126:     #         |
127:     #
128:     A_orig = np.array([[-3, 3], [-2, 3], [-2, 2], [-3, 2]], dtype=float)
129:     B_orig = np.array([[3, 2], [1, 0], [3, -2], [5, 0]], dtype=float)
130:     A, A_mu = _centered(A_orig)
131:     B, B_mu = _centered(B_orig)
132:     R, s = orthogonal_procrustes(A, B)
133:     scale = s / np.square(norm(A))
134:     B_approx = scale * np.dot(A, R) + B_mu
135:     assert_allclose(B_approx, B_orig, atol=1e-8)
136: 
137: 
138: def test_orthogonal_procrustes_stretched_example():
139:     # Try again with a target with a stretched y axis.
140:     A_orig = np.array([[-3, 3], [-2, 3], [-2, 2], [-3, 2]], dtype=float)
141:     B_orig = np.array([[3, 40], [1, 0], [3, -40], [5, 0]], dtype=float)
142:     A, A_mu = _centered(A_orig)
143:     B, B_mu = _centered(B_orig)
144:     R, s = orthogonal_procrustes(A, B)
145:     scale = s / np.square(norm(A))
146:     B_approx = scale * np.dot(A, R) + B_mu
147:     expected = np.array([[3, 21], [-18, 0], [3, -21], [24, 0]], dtype=float)
148:     assert_allclose(B_approx, expected, atol=1e-8)
149:     # Check disparity symmetry.
150:     expected_disparity = 0.4501246882793018
151:     AB_disparity = np.square(norm(B_approx - B_orig) / norm(B))
152:     assert_allclose(AB_disparity, expected_disparity)
153:     R, s = orthogonal_procrustes(B, A)
154:     scale = s / np.square(norm(B))
155:     A_approx = scale * np.dot(B, R) + A_mu
156:     BA_disparity = np.square(norm(A_approx - A_orig) / norm(A))
157:     assert_allclose(BA_disparity, expected_disparity)
158: 
159: 
160: def test_orthogonal_procrustes_skbio_example():
161:     # This transformation is also exact.
162:     # It uses translation, scaling, and reflection.
163:     #
164:     #   |
165:     #   | a
166:     #   | b
167:     #   | c d
168:     # --+---------
169:     #   |
170:     #   |       w
171:     #   |
172:     #   |       x
173:     #   |
174:     #   |   z   y
175:     #   |
176:     #
177:     A_orig = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], dtype=float)
178:     B_orig = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], dtype=float)
179:     B_standardized = np.array([
180:         [-0.13363062, 0.6681531],
181:         [-0.13363062, 0.13363062],
182:         [-0.13363062, -0.40089186],
183:         [0.40089186, -0.40089186]])
184:     A, A_mu = _centered(A_orig)
185:     B, B_mu = _centered(B_orig)
186:     R, s = orthogonal_procrustes(A, B)
187:     scale = s / np.square(norm(A))
188:     B_approx = scale * np.dot(A, R) + B_mu
189:     assert_allclose(B_approx, B_orig)
190:     assert_allclose(B / norm(B), B_standardized)
191: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from itertools import product, permutations' statement (line 1)
try:
    from itertools import product, permutations

except:
    product = UndefinedType
    permutations = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'itertools', None, module_type_store, ['product', 'permutations'], [product, permutations])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_104590 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_104590) is not StypyTypeError):

    if (import_104590 != 'pyd_module'):
        __import__(import_104590)
        sys_modules_104591 = sys.modules[import_104590]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_104591.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_104590)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_array_less, assert_allclose' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_104592 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_104592) is not StypyTypeError):

    if (import_104592 != 'pyd_module'):
        __import__(import_104592)
        sys_modules_104593 = sys.modules[import_104592]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_104593.module_type_store, module_type_store, ['assert_array_less', 'assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_104593, sys_modules_104593.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_array_less, assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_array_less', 'assert_allclose'], [assert_array_less, assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_104592)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from pytest import assert_raises' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_104594 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest')

if (type(import_104594) is not StypyTypeError):

    if (import_104594 != 'pyd_module'):
        __import__(import_104594)
        sys_modules_104595 = sys.modules[import_104594]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', sys_modules_104595.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_104595, sys_modules_104595.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', import_104594)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.linalg import inv, eigh, norm' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_104596 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg')

if (type(import_104596) is not StypyTypeError):

    if (import_104596 != 'pyd_module'):
        __import__(import_104596)
        sys_modules_104597 = sys.modules[import_104596]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg', sys_modules_104597.module_type_store, module_type_store, ['inv', 'eigh', 'norm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_104597, sys_modules_104597.module_type_store, module_type_store)
    else:
        from scipy.linalg import inv, eigh, norm

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg', None, module_type_store, ['inv', 'eigh', 'norm'], [inv, eigh, norm])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg', import_104596)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.linalg import orthogonal_procrustes' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_104598 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg')

if (type(import_104598) is not StypyTypeError):

    if (import_104598 != 'pyd_module'):
        __import__(import_104598)
        sys_modules_104599 = sys.modules[import_104598]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg', sys_modules_104599.module_type_store, module_type_store, ['orthogonal_procrustes'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_104599, sys_modules_104599.module_type_store, module_type_store)
    else:
        from scipy.linalg import orthogonal_procrustes

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg', None, module_type_store, ['orthogonal_procrustes'], [orthogonal_procrustes])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg', import_104598)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')


@norecursion
def test_orthogonal_procrustes_ndim_too_large(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_orthogonal_procrustes_ndim_too_large'
    module_type_store = module_type_store.open_function_context('test_orthogonal_procrustes_ndim_too_large', 11, 0, False)
    
    # Passed parameters checking function
    test_orthogonal_procrustes_ndim_too_large.stypy_localization = localization
    test_orthogonal_procrustes_ndim_too_large.stypy_type_of_self = None
    test_orthogonal_procrustes_ndim_too_large.stypy_type_store = module_type_store
    test_orthogonal_procrustes_ndim_too_large.stypy_function_name = 'test_orthogonal_procrustes_ndim_too_large'
    test_orthogonal_procrustes_ndim_too_large.stypy_param_names_list = []
    test_orthogonal_procrustes_ndim_too_large.stypy_varargs_param_name = None
    test_orthogonal_procrustes_ndim_too_large.stypy_kwargs_param_name = None
    test_orthogonal_procrustes_ndim_too_large.stypy_call_defaults = defaults
    test_orthogonal_procrustes_ndim_too_large.stypy_call_varargs = varargs
    test_orthogonal_procrustes_ndim_too_large.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_orthogonal_procrustes_ndim_too_large', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_orthogonal_procrustes_ndim_too_large', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_orthogonal_procrustes_ndim_too_large(...)' code ##################

    
    # Call to seed(...): (line 12)
    # Processing the call arguments (line 12)
    int_104603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 19), 'int')
    # Processing the call keyword arguments (line 12)
    kwargs_104604 = {}
    # Getting the type of 'np' (line 12)
    np_104600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 12)
    random_104601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), np_104600, 'random')
    # Obtaining the member 'seed' of a type (line 12)
    seed_104602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), random_104601, 'seed')
    # Calling seed(args, kwargs) (line 12)
    seed_call_result_104605 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), seed_104602, *[int_104603], **kwargs_104604)
    
    
    # Assigning a Call to a Name (line 13):
    
    # Assigning a Call to a Name (line 13):
    
    # Call to randn(...): (line 13)
    # Processing the call arguments (line 13)
    int_104609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 24), 'int')
    int_104610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 27), 'int')
    int_104611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 30), 'int')
    # Processing the call keyword arguments (line 13)
    kwargs_104612 = {}
    # Getting the type of 'np' (line 13)
    np_104606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 13)
    random_104607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 8), np_104606, 'random')
    # Obtaining the member 'randn' of a type (line 13)
    randn_104608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 8), random_104607, 'randn')
    # Calling randn(args, kwargs) (line 13)
    randn_call_result_104613 = invoke(stypy.reporting.localization.Localization(__file__, 13, 8), randn_104608, *[int_104609, int_104610, int_104611], **kwargs_104612)
    
    # Assigning a type to the variable 'A' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'A', randn_call_result_104613)
    
    # Assigning a Call to a Name (line 14):
    
    # Assigning a Call to a Name (line 14):
    
    # Call to randn(...): (line 14)
    # Processing the call arguments (line 14)
    int_104617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 24), 'int')
    int_104618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 27), 'int')
    int_104619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 30), 'int')
    # Processing the call keyword arguments (line 14)
    kwargs_104620 = {}
    # Getting the type of 'np' (line 14)
    np_104614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 14)
    random_104615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), np_104614, 'random')
    # Obtaining the member 'randn' of a type (line 14)
    randn_104616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), random_104615, 'randn')
    # Calling randn(args, kwargs) (line 14)
    randn_call_result_104621 = invoke(stypy.reporting.localization.Localization(__file__, 14, 8), randn_104616, *[int_104617, int_104618, int_104619], **kwargs_104620)
    
    # Assigning a type to the variable 'B' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'B', randn_call_result_104621)
    
    # Call to assert_raises(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'ValueError' (line 15)
    ValueError_104623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 18), 'ValueError', False)
    # Getting the type of 'orthogonal_procrustes' (line 15)
    orthogonal_procrustes_104624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 30), 'orthogonal_procrustes', False)
    # Getting the type of 'A' (line 15)
    A_104625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 53), 'A', False)
    # Getting the type of 'B' (line 15)
    B_104626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 56), 'B', False)
    # Processing the call keyword arguments (line 15)
    kwargs_104627 = {}
    # Getting the type of 'assert_raises' (line 15)
    assert_raises_104622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 15)
    assert_raises_call_result_104628 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), assert_raises_104622, *[ValueError_104623, orthogonal_procrustes_104624, A_104625, B_104626], **kwargs_104627)
    
    
    # ################# End of 'test_orthogonal_procrustes_ndim_too_large(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_orthogonal_procrustes_ndim_too_large' in the type store
    # Getting the type of 'stypy_return_type' (line 11)
    stypy_return_type_104629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_104629)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_orthogonal_procrustes_ndim_too_large'
    return stypy_return_type_104629

# Assigning a type to the variable 'test_orthogonal_procrustes_ndim_too_large' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'test_orthogonal_procrustes_ndim_too_large', test_orthogonal_procrustes_ndim_too_large)

@norecursion
def test_orthogonal_procrustes_ndim_too_small(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_orthogonal_procrustes_ndim_too_small'
    module_type_store = module_type_store.open_function_context('test_orthogonal_procrustes_ndim_too_small', 18, 0, False)
    
    # Passed parameters checking function
    test_orthogonal_procrustes_ndim_too_small.stypy_localization = localization
    test_orthogonal_procrustes_ndim_too_small.stypy_type_of_self = None
    test_orthogonal_procrustes_ndim_too_small.stypy_type_store = module_type_store
    test_orthogonal_procrustes_ndim_too_small.stypy_function_name = 'test_orthogonal_procrustes_ndim_too_small'
    test_orthogonal_procrustes_ndim_too_small.stypy_param_names_list = []
    test_orthogonal_procrustes_ndim_too_small.stypy_varargs_param_name = None
    test_orthogonal_procrustes_ndim_too_small.stypy_kwargs_param_name = None
    test_orthogonal_procrustes_ndim_too_small.stypy_call_defaults = defaults
    test_orthogonal_procrustes_ndim_too_small.stypy_call_varargs = varargs
    test_orthogonal_procrustes_ndim_too_small.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_orthogonal_procrustes_ndim_too_small', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_orthogonal_procrustes_ndim_too_small', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_orthogonal_procrustes_ndim_too_small(...)' code ##################

    
    # Call to seed(...): (line 19)
    # Processing the call arguments (line 19)
    int_104633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 19), 'int')
    # Processing the call keyword arguments (line 19)
    kwargs_104634 = {}
    # Getting the type of 'np' (line 19)
    np_104630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 19)
    random_104631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 4), np_104630, 'random')
    # Obtaining the member 'seed' of a type (line 19)
    seed_104632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 4), random_104631, 'seed')
    # Calling seed(args, kwargs) (line 19)
    seed_call_result_104635 = invoke(stypy.reporting.localization.Localization(__file__, 19, 4), seed_104632, *[int_104633], **kwargs_104634)
    
    
    # Assigning a Call to a Name (line 20):
    
    # Assigning a Call to a Name (line 20):
    
    # Call to randn(...): (line 20)
    # Processing the call arguments (line 20)
    int_104639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 24), 'int')
    # Processing the call keyword arguments (line 20)
    kwargs_104640 = {}
    # Getting the type of 'np' (line 20)
    np_104636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 20)
    random_104637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), np_104636, 'random')
    # Obtaining the member 'randn' of a type (line 20)
    randn_104638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), random_104637, 'randn')
    # Calling randn(args, kwargs) (line 20)
    randn_call_result_104641 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), randn_104638, *[int_104639], **kwargs_104640)
    
    # Assigning a type to the variable 'A' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'A', randn_call_result_104641)
    
    # Assigning a Call to a Name (line 21):
    
    # Assigning a Call to a Name (line 21):
    
    # Call to randn(...): (line 21)
    # Processing the call arguments (line 21)
    int_104645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 24), 'int')
    # Processing the call keyword arguments (line 21)
    kwargs_104646 = {}
    # Getting the type of 'np' (line 21)
    np_104642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 21)
    random_104643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), np_104642, 'random')
    # Obtaining the member 'randn' of a type (line 21)
    randn_104644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), random_104643, 'randn')
    # Calling randn(args, kwargs) (line 21)
    randn_call_result_104647 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), randn_104644, *[int_104645], **kwargs_104646)
    
    # Assigning a type to the variable 'B' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'B', randn_call_result_104647)
    
    # Call to assert_raises(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'ValueError' (line 22)
    ValueError_104649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 18), 'ValueError', False)
    # Getting the type of 'orthogonal_procrustes' (line 22)
    orthogonal_procrustes_104650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 30), 'orthogonal_procrustes', False)
    # Getting the type of 'A' (line 22)
    A_104651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 53), 'A', False)
    # Getting the type of 'B' (line 22)
    B_104652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 56), 'B', False)
    # Processing the call keyword arguments (line 22)
    kwargs_104653 = {}
    # Getting the type of 'assert_raises' (line 22)
    assert_raises_104648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 22)
    assert_raises_call_result_104654 = invoke(stypy.reporting.localization.Localization(__file__, 22, 4), assert_raises_104648, *[ValueError_104649, orthogonal_procrustes_104650, A_104651, B_104652], **kwargs_104653)
    
    
    # ################# End of 'test_orthogonal_procrustes_ndim_too_small(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_orthogonal_procrustes_ndim_too_small' in the type store
    # Getting the type of 'stypy_return_type' (line 18)
    stypy_return_type_104655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_104655)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_orthogonal_procrustes_ndim_too_small'
    return stypy_return_type_104655

# Assigning a type to the variable 'test_orthogonal_procrustes_ndim_too_small' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'test_orthogonal_procrustes_ndim_too_small', test_orthogonal_procrustes_ndim_too_small)

@norecursion
def test_orthogonal_procrustes_shape_mismatch(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_orthogonal_procrustes_shape_mismatch'
    module_type_store = module_type_store.open_function_context('test_orthogonal_procrustes_shape_mismatch', 25, 0, False)
    
    # Passed parameters checking function
    test_orthogonal_procrustes_shape_mismatch.stypy_localization = localization
    test_orthogonal_procrustes_shape_mismatch.stypy_type_of_self = None
    test_orthogonal_procrustes_shape_mismatch.stypy_type_store = module_type_store
    test_orthogonal_procrustes_shape_mismatch.stypy_function_name = 'test_orthogonal_procrustes_shape_mismatch'
    test_orthogonal_procrustes_shape_mismatch.stypy_param_names_list = []
    test_orthogonal_procrustes_shape_mismatch.stypy_varargs_param_name = None
    test_orthogonal_procrustes_shape_mismatch.stypy_kwargs_param_name = None
    test_orthogonal_procrustes_shape_mismatch.stypy_call_defaults = defaults
    test_orthogonal_procrustes_shape_mismatch.stypy_call_varargs = varargs
    test_orthogonal_procrustes_shape_mismatch.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_orthogonal_procrustes_shape_mismatch', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_orthogonal_procrustes_shape_mismatch', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_orthogonal_procrustes_shape_mismatch(...)' code ##################

    
    # Call to seed(...): (line 26)
    # Processing the call arguments (line 26)
    int_104659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'int')
    # Processing the call keyword arguments (line 26)
    kwargs_104660 = {}
    # Getting the type of 'np' (line 26)
    np_104656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 26)
    random_104657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 4), np_104656, 'random')
    # Obtaining the member 'seed' of a type (line 26)
    seed_104658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 4), random_104657, 'seed')
    # Calling seed(args, kwargs) (line 26)
    seed_call_result_104661 = invoke(stypy.reporting.localization.Localization(__file__, 26, 4), seed_104658, *[int_104659], **kwargs_104660)
    
    
    # Assigning a Tuple to a Name (line 27):
    
    # Assigning a Tuple to a Name (line 27):
    
    # Obtaining an instance of the builtin type 'tuple' (line 27)
    tuple_104662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 27)
    # Adding element type (line 27)
    
    # Obtaining an instance of the builtin type 'tuple' (line 27)
    tuple_104663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 27)
    # Adding element type (line 27)
    int_104664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 15), tuple_104663, int_104664)
    # Adding element type (line 27)
    int_104665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 15), tuple_104663, int_104665)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 14), tuple_104662, tuple_104663)
    # Adding element type (line 27)
    
    # Obtaining an instance of the builtin type 'tuple' (line 27)
    tuple_104666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 27)
    # Adding element type (line 27)
    int_104667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 23), tuple_104666, int_104667)
    # Adding element type (line 27)
    int_104668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 23), tuple_104666, int_104668)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 14), tuple_104662, tuple_104666)
    # Adding element type (line 27)
    
    # Obtaining an instance of the builtin type 'tuple' (line 27)
    tuple_104669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 27)
    # Adding element type (line 27)
    int_104670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 31), tuple_104669, int_104670)
    # Adding element type (line 27)
    int_104671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 31), tuple_104669, int_104671)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 14), tuple_104662, tuple_104669)
    # Adding element type (line 27)
    
    # Obtaining an instance of the builtin type 'tuple' (line 27)
    tuple_104672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 27)
    # Adding element type (line 27)
    int_104673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 39), tuple_104672, int_104673)
    # Adding element type (line 27)
    int_104674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 39), tuple_104672, int_104674)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 14), tuple_104662, tuple_104672)
    
    # Assigning a type to the variable 'shapes' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'shapes', tuple_104662)
    
    
    # Call to permutations(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'shapes' (line 28)
    shapes_104676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 29), 'shapes', False)
    int_104677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 37), 'int')
    # Processing the call keyword arguments (line 28)
    kwargs_104678 = {}
    # Getting the type of 'permutations' (line 28)
    permutations_104675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'permutations', False)
    # Calling permutations(args, kwargs) (line 28)
    permutations_call_result_104679 = invoke(stypy.reporting.localization.Localization(__file__, 28, 16), permutations_104675, *[shapes_104676, int_104677], **kwargs_104678)
    
    # Testing the type of a for loop iterable (line 28)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 28, 4), permutations_call_result_104679)
    # Getting the type of the for loop variable (line 28)
    for_loop_var_104680 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 28, 4), permutations_call_result_104679)
    # Assigning a type to the variable 'a' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 4), for_loop_var_104680))
    # Assigning a type to the variable 'b' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'b', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 4), for_loop_var_104680))
    # SSA begins for a for statement (line 28)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 29):
    
    # Assigning a Call to a Name (line 29):
    
    # Call to randn(...): (line 29)
    # Getting the type of 'a' (line 29)
    a_104684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 29), 'a', False)
    # Processing the call keyword arguments (line 29)
    kwargs_104685 = {}
    # Getting the type of 'np' (line 29)
    np_104681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'np', False)
    # Obtaining the member 'random' of a type (line 29)
    random_104682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 12), np_104681, 'random')
    # Obtaining the member 'randn' of a type (line 29)
    randn_104683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 12), random_104682, 'randn')
    # Calling randn(args, kwargs) (line 29)
    randn_call_result_104686 = invoke(stypy.reporting.localization.Localization(__file__, 29, 12), randn_104683, *[a_104684], **kwargs_104685)
    
    # Assigning a type to the variable 'A' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'A', randn_call_result_104686)
    
    # Assigning a Call to a Name (line 30):
    
    # Assigning a Call to a Name (line 30):
    
    # Call to randn(...): (line 30)
    # Getting the type of 'b' (line 30)
    b_104690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 29), 'b', False)
    # Processing the call keyword arguments (line 30)
    kwargs_104691 = {}
    # Getting the type of 'np' (line 30)
    np_104687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'np', False)
    # Obtaining the member 'random' of a type (line 30)
    random_104688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 12), np_104687, 'random')
    # Obtaining the member 'randn' of a type (line 30)
    randn_104689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 12), random_104688, 'randn')
    # Calling randn(args, kwargs) (line 30)
    randn_call_result_104692 = invoke(stypy.reporting.localization.Localization(__file__, 30, 12), randn_104689, *[b_104690], **kwargs_104691)
    
    # Assigning a type to the variable 'B' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'B', randn_call_result_104692)
    
    # Call to assert_raises(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'ValueError' (line 31)
    ValueError_104694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 22), 'ValueError', False)
    # Getting the type of 'orthogonal_procrustes' (line 31)
    orthogonal_procrustes_104695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 34), 'orthogonal_procrustes', False)
    # Getting the type of 'A' (line 31)
    A_104696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 57), 'A', False)
    # Getting the type of 'B' (line 31)
    B_104697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 60), 'B', False)
    # Processing the call keyword arguments (line 31)
    kwargs_104698 = {}
    # Getting the type of 'assert_raises' (line 31)
    assert_raises_104693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 31)
    assert_raises_call_result_104699 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), assert_raises_104693, *[ValueError_104694, orthogonal_procrustes_104695, A_104696, B_104697], **kwargs_104698)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_orthogonal_procrustes_shape_mismatch(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_orthogonal_procrustes_shape_mismatch' in the type store
    # Getting the type of 'stypy_return_type' (line 25)
    stypy_return_type_104700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_104700)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_orthogonal_procrustes_shape_mismatch'
    return stypy_return_type_104700

# Assigning a type to the variable 'test_orthogonal_procrustes_shape_mismatch' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'test_orthogonal_procrustes_shape_mismatch', test_orthogonal_procrustes_shape_mismatch)

@norecursion
def test_orthogonal_procrustes_checkfinite_exception(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_orthogonal_procrustes_checkfinite_exception'
    module_type_store = module_type_store.open_function_context('test_orthogonal_procrustes_checkfinite_exception', 34, 0, False)
    
    # Passed parameters checking function
    test_orthogonal_procrustes_checkfinite_exception.stypy_localization = localization
    test_orthogonal_procrustes_checkfinite_exception.stypy_type_of_self = None
    test_orthogonal_procrustes_checkfinite_exception.stypy_type_store = module_type_store
    test_orthogonal_procrustes_checkfinite_exception.stypy_function_name = 'test_orthogonal_procrustes_checkfinite_exception'
    test_orthogonal_procrustes_checkfinite_exception.stypy_param_names_list = []
    test_orthogonal_procrustes_checkfinite_exception.stypy_varargs_param_name = None
    test_orthogonal_procrustes_checkfinite_exception.stypy_kwargs_param_name = None
    test_orthogonal_procrustes_checkfinite_exception.stypy_call_defaults = defaults
    test_orthogonal_procrustes_checkfinite_exception.stypy_call_varargs = varargs
    test_orthogonal_procrustes_checkfinite_exception.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_orthogonal_procrustes_checkfinite_exception', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_orthogonal_procrustes_checkfinite_exception', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_orthogonal_procrustes_checkfinite_exception(...)' code ##################

    
    # Call to seed(...): (line 35)
    # Processing the call arguments (line 35)
    int_104704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 19), 'int')
    # Processing the call keyword arguments (line 35)
    kwargs_104705 = {}
    # Getting the type of 'np' (line 35)
    np_104701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 35)
    random_104702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 4), np_104701, 'random')
    # Obtaining the member 'seed' of a type (line 35)
    seed_104703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 4), random_104702, 'seed')
    # Calling seed(args, kwargs) (line 35)
    seed_call_result_104706 = invoke(stypy.reporting.localization.Localization(__file__, 35, 4), seed_104703, *[int_104704], **kwargs_104705)
    
    
    # Assigning a Tuple to a Tuple (line 36):
    
    # Assigning a Num to a Name (line 36):
    int_104707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 11), 'int')
    # Assigning a type to the variable 'tuple_assignment_104552' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'tuple_assignment_104552', int_104707)
    
    # Assigning a Num to a Name (line 36):
    int_104708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 14), 'int')
    # Assigning a type to the variable 'tuple_assignment_104553' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'tuple_assignment_104553', int_104708)
    
    # Assigning a Name to a Name (line 36):
    # Getting the type of 'tuple_assignment_104552' (line 36)
    tuple_assignment_104552_104709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'tuple_assignment_104552')
    # Assigning a type to the variable 'm' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'm', tuple_assignment_104552_104709)
    
    # Assigning a Name to a Name (line 36):
    # Getting the type of 'tuple_assignment_104553' (line 36)
    tuple_assignment_104553_104710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'tuple_assignment_104553')
    # Assigning a type to the variable 'n' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 7), 'n', tuple_assignment_104553_104710)
    
    # Assigning a Call to a Name (line 37):
    
    # Assigning a Call to a Name (line 37):
    
    # Call to randn(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'm' (line 37)
    m_104714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 29), 'm', False)
    # Getting the type of 'n' (line 37)
    n_104715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 32), 'n', False)
    # Processing the call keyword arguments (line 37)
    kwargs_104716 = {}
    # Getting the type of 'np' (line 37)
    np_104711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 13), 'np', False)
    # Obtaining the member 'random' of a type (line 37)
    random_104712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 13), np_104711, 'random')
    # Obtaining the member 'randn' of a type (line 37)
    randn_104713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 13), random_104712, 'randn')
    # Calling randn(args, kwargs) (line 37)
    randn_call_result_104717 = invoke(stypy.reporting.localization.Localization(__file__, 37, 13), randn_104713, *[m_104714, n_104715], **kwargs_104716)
    
    # Assigning a type to the variable 'A_good' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'A_good', randn_call_result_104717)
    
    # Assigning a Call to a Name (line 38):
    
    # Assigning a Call to a Name (line 38):
    
    # Call to randn(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'm' (line 38)
    m_104721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 29), 'm', False)
    # Getting the type of 'n' (line 38)
    n_104722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 32), 'n', False)
    # Processing the call keyword arguments (line 38)
    kwargs_104723 = {}
    # Getting the type of 'np' (line 38)
    np_104718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 13), 'np', False)
    # Obtaining the member 'random' of a type (line 38)
    random_104719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 13), np_104718, 'random')
    # Obtaining the member 'randn' of a type (line 38)
    randn_104720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 13), random_104719, 'randn')
    # Calling randn(args, kwargs) (line 38)
    randn_call_result_104724 = invoke(stypy.reporting.localization.Localization(__file__, 38, 13), randn_104720, *[m_104721, n_104722], **kwargs_104723)
    
    # Assigning a type to the variable 'B_good' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'B_good', randn_call_result_104724)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 39)
    tuple_104725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 39)
    # Adding element type (line 39)
    # Getting the type of 'np' (line 39)
    np_104726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 21), 'np')
    # Obtaining the member 'inf' of a type (line 39)
    inf_104727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 21), np_104726, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 21), tuple_104725, inf_104727)
    # Adding element type (line 39)
    
    # Getting the type of 'np' (line 39)
    np_104728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 30), 'np')
    # Obtaining the member 'inf' of a type (line 39)
    inf_104729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 30), np_104728, 'inf')
    # Applying the 'usub' unary operator (line 39)
    result___neg___104730 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 29), 'usub', inf_104729)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 21), tuple_104725, result___neg___104730)
    # Adding element type (line 39)
    # Getting the type of 'np' (line 39)
    np_104731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 38), 'np')
    # Obtaining the member 'nan' of a type (line 39)
    nan_104732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 38), np_104731, 'nan')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 21), tuple_104725, nan_104732)
    
    # Testing the type of a for loop iterable (line 39)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 39, 4), tuple_104725)
    # Getting the type of the for loop variable (line 39)
    for_loop_var_104733 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 39, 4), tuple_104725)
    # Assigning a type to the variable 'bad_value' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'bad_value', for_loop_var_104733)
    # SSA begins for a for statement (line 39)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 40):
    
    # Assigning a Call to a Name (line 40):
    
    # Call to copy(...): (line 40)
    # Processing the call keyword arguments (line 40)
    kwargs_104736 = {}
    # Getting the type of 'A_good' (line 40)
    A_good_104734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'A_good', False)
    # Obtaining the member 'copy' of a type (line 40)
    copy_104735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 16), A_good_104734, 'copy')
    # Calling copy(args, kwargs) (line 40)
    copy_call_result_104737 = invoke(stypy.reporting.localization.Localization(__file__, 40, 16), copy_104735, *[], **kwargs_104736)
    
    # Assigning a type to the variable 'A_bad' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'A_bad', copy_call_result_104737)
    
    # Assigning a Name to a Subscript (line 41):
    
    # Assigning a Name to a Subscript (line 41):
    # Getting the type of 'bad_value' (line 41)
    bad_value_104738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 22), 'bad_value')
    # Getting the type of 'A_bad' (line 41)
    A_bad_104739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'A_bad')
    
    # Obtaining an instance of the builtin type 'tuple' (line 41)
    tuple_104740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 41)
    # Adding element type (line 41)
    int_104741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 14), tuple_104740, int_104741)
    # Adding element type (line 41)
    int_104742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 14), tuple_104740, int_104742)
    
    # Storing an element on a container (line 41)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 8), A_bad_104739, (tuple_104740, bad_value_104738))
    
    # Assigning a Call to a Name (line 42):
    
    # Assigning a Call to a Name (line 42):
    
    # Call to copy(...): (line 42)
    # Processing the call keyword arguments (line 42)
    kwargs_104745 = {}
    # Getting the type of 'B_good' (line 42)
    B_good_104743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'B_good', False)
    # Obtaining the member 'copy' of a type (line 42)
    copy_104744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 16), B_good_104743, 'copy')
    # Calling copy(args, kwargs) (line 42)
    copy_call_result_104746 = invoke(stypy.reporting.localization.Localization(__file__, 42, 16), copy_104744, *[], **kwargs_104745)
    
    # Assigning a type to the variable 'B_bad' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'B_bad', copy_call_result_104746)
    
    # Assigning a Name to a Subscript (line 43):
    
    # Assigning a Name to a Subscript (line 43):
    # Getting the type of 'bad_value' (line 43)
    bad_value_104747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 22), 'bad_value')
    # Getting the type of 'B_bad' (line 43)
    B_bad_104748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'B_bad')
    
    # Obtaining an instance of the builtin type 'tuple' (line 43)
    tuple_104749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 43)
    # Adding element type (line 43)
    int_104750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), tuple_104749, int_104750)
    # Adding element type (line 43)
    int_104751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), tuple_104749, int_104751)
    
    # Storing an element on a container (line 43)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 8), B_bad_104748, (tuple_104749, bad_value_104747))
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 44)
    tuple_104752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 44)
    # Adding element type (line 44)
    
    # Obtaining an instance of the builtin type 'tuple' (line 44)
    tuple_104753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 44)
    # Adding element type (line 44)
    # Getting the type of 'A_good' (line 44)
    A_good_104754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 22), 'A_good')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 22), tuple_104753, A_good_104754)
    # Adding element type (line 44)
    # Getting the type of 'B_bad' (line 44)
    B_bad_104755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 30), 'B_bad')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 22), tuple_104753, B_bad_104755)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 21), tuple_104752, tuple_104753)
    # Adding element type (line 44)
    
    # Obtaining an instance of the builtin type 'tuple' (line 44)
    tuple_104756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 44)
    # Adding element type (line 44)
    # Getting the type of 'A_bad' (line 44)
    A_bad_104757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 39), 'A_bad')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 39), tuple_104756, A_bad_104757)
    # Adding element type (line 44)
    # Getting the type of 'B_good' (line 44)
    B_good_104758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 46), 'B_good')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 39), tuple_104756, B_good_104758)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 21), tuple_104752, tuple_104756)
    # Adding element type (line 44)
    
    # Obtaining an instance of the builtin type 'tuple' (line 44)
    tuple_104759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 56), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 44)
    # Adding element type (line 44)
    # Getting the type of 'A_bad' (line 44)
    A_bad_104760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 56), 'A_bad')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 56), tuple_104759, A_bad_104760)
    # Adding element type (line 44)
    # Getting the type of 'B_bad' (line 44)
    B_bad_104761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 63), 'B_bad')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 56), tuple_104759, B_bad_104761)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 21), tuple_104752, tuple_104759)
    
    # Testing the type of a for loop iterable (line 44)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 44, 8), tuple_104752)
    # Getting the type of the for loop variable (line 44)
    for_loop_var_104762 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 44, 8), tuple_104752)
    # Assigning a type to the variable 'A' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'A', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 8), for_loop_var_104762))
    # Assigning a type to the variable 'B' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'B', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 8), for_loop_var_104762))
    # SSA begins for a for statement (line 44)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to assert_raises(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'ValueError' (line 45)
    ValueError_104764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 26), 'ValueError', False)
    # Getting the type of 'orthogonal_procrustes' (line 45)
    orthogonal_procrustes_104765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 38), 'orthogonal_procrustes', False)
    # Getting the type of 'A' (line 45)
    A_104766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 61), 'A', False)
    # Getting the type of 'B' (line 45)
    B_104767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 64), 'B', False)
    # Processing the call keyword arguments (line 45)
    kwargs_104768 = {}
    # Getting the type of 'assert_raises' (line 45)
    assert_raises_104763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 45)
    assert_raises_call_result_104769 = invoke(stypy.reporting.localization.Localization(__file__, 45, 12), assert_raises_104763, *[ValueError_104764, orthogonal_procrustes_104765, A_104766, B_104767], **kwargs_104768)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_orthogonal_procrustes_checkfinite_exception(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_orthogonal_procrustes_checkfinite_exception' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_104770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_104770)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_orthogonal_procrustes_checkfinite_exception'
    return stypy_return_type_104770

# Assigning a type to the variable 'test_orthogonal_procrustes_checkfinite_exception' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'test_orthogonal_procrustes_checkfinite_exception', test_orthogonal_procrustes_checkfinite_exception)

@norecursion
def test_orthogonal_procrustes_scale_invariance(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_orthogonal_procrustes_scale_invariance'
    module_type_store = module_type_store.open_function_context('test_orthogonal_procrustes_scale_invariance', 48, 0, False)
    
    # Passed parameters checking function
    test_orthogonal_procrustes_scale_invariance.stypy_localization = localization
    test_orthogonal_procrustes_scale_invariance.stypy_type_of_self = None
    test_orthogonal_procrustes_scale_invariance.stypy_type_store = module_type_store
    test_orthogonal_procrustes_scale_invariance.stypy_function_name = 'test_orthogonal_procrustes_scale_invariance'
    test_orthogonal_procrustes_scale_invariance.stypy_param_names_list = []
    test_orthogonal_procrustes_scale_invariance.stypy_varargs_param_name = None
    test_orthogonal_procrustes_scale_invariance.stypy_kwargs_param_name = None
    test_orthogonal_procrustes_scale_invariance.stypy_call_defaults = defaults
    test_orthogonal_procrustes_scale_invariance.stypy_call_varargs = varargs
    test_orthogonal_procrustes_scale_invariance.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_orthogonal_procrustes_scale_invariance', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_orthogonal_procrustes_scale_invariance', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_orthogonal_procrustes_scale_invariance(...)' code ##################

    
    # Call to seed(...): (line 49)
    # Processing the call arguments (line 49)
    int_104774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 19), 'int')
    # Processing the call keyword arguments (line 49)
    kwargs_104775 = {}
    # Getting the type of 'np' (line 49)
    np_104771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 49)
    random_104772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 4), np_104771, 'random')
    # Obtaining the member 'seed' of a type (line 49)
    seed_104773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 4), random_104772, 'seed')
    # Calling seed(args, kwargs) (line 49)
    seed_call_result_104776 = invoke(stypy.reporting.localization.Localization(__file__, 49, 4), seed_104773, *[int_104774], **kwargs_104775)
    
    
    # Assigning a Tuple to a Tuple (line 50):
    
    # Assigning a Num to a Name (line 50):
    int_104777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 11), 'int')
    # Assigning a type to the variable 'tuple_assignment_104554' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'tuple_assignment_104554', int_104777)
    
    # Assigning a Num to a Name (line 50):
    int_104778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 14), 'int')
    # Assigning a type to the variable 'tuple_assignment_104555' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'tuple_assignment_104555', int_104778)
    
    # Assigning a Name to a Name (line 50):
    # Getting the type of 'tuple_assignment_104554' (line 50)
    tuple_assignment_104554_104779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'tuple_assignment_104554')
    # Assigning a type to the variable 'm' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'm', tuple_assignment_104554_104779)
    
    # Assigning a Name to a Name (line 50):
    # Getting the type of 'tuple_assignment_104555' (line 50)
    tuple_assignment_104555_104780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'tuple_assignment_104555')
    # Assigning a type to the variable 'n' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 7), 'n', tuple_assignment_104555_104780)
    
    
    # Call to range(...): (line 51)
    # Processing the call arguments (line 51)
    int_104782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 19), 'int')
    # Processing the call keyword arguments (line 51)
    kwargs_104783 = {}
    # Getting the type of 'range' (line 51)
    range_104781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 13), 'range', False)
    # Calling range(args, kwargs) (line 51)
    range_call_result_104784 = invoke(stypy.reporting.localization.Localization(__file__, 51, 13), range_104781, *[int_104782], **kwargs_104783)
    
    # Testing the type of a for loop iterable (line 51)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 51, 4), range_call_result_104784)
    # Getting the type of the for loop variable (line 51)
    for_loop_var_104785 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 51, 4), range_call_result_104784)
    # Assigning a type to the variable 'i' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'i', for_loop_var_104785)
    # SSA begins for a for statement (line 51)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 52):
    
    # Assigning a Call to a Name (line 52):
    
    # Call to randn(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'm' (line 52)
    m_104789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 33), 'm', False)
    # Getting the type of 'n' (line 52)
    n_104790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 36), 'n', False)
    # Processing the call keyword arguments (line 52)
    kwargs_104791 = {}
    # Getting the type of 'np' (line 52)
    np_104786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 17), 'np', False)
    # Obtaining the member 'random' of a type (line 52)
    random_104787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 17), np_104786, 'random')
    # Obtaining the member 'randn' of a type (line 52)
    randn_104788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 17), random_104787, 'randn')
    # Calling randn(args, kwargs) (line 52)
    randn_call_result_104792 = invoke(stypy.reporting.localization.Localization(__file__, 52, 17), randn_104788, *[m_104789, n_104790], **kwargs_104791)
    
    # Assigning a type to the variable 'A_orig' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'A_orig', randn_call_result_104792)
    
    # Assigning a Call to a Name (line 53):
    
    # Assigning a Call to a Name (line 53):
    
    # Call to randn(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'm' (line 53)
    m_104796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 33), 'm', False)
    # Getting the type of 'n' (line 53)
    n_104797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 36), 'n', False)
    # Processing the call keyword arguments (line 53)
    kwargs_104798 = {}
    # Getting the type of 'np' (line 53)
    np_104793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 17), 'np', False)
    # Obtaining the member 'random' of a type (line 53)
    random_104794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 17), np_104793, 'random')
    # Obtaining the member 'randn' of a type (line 53)
    randn_104795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 17), random_104794, 'randn')
    # Calling randn(args, kwargs) (line 53)
    randn_call_result_104799 = invoke(stypy.reporting.localization.Localization(__file__, 53, 17), randn_104795, *[m_104796, n_104797], **kwargs_104798)
    
    # Assigning a type to the variable 'B_orig' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'B_orig', randn_call_result_104799)
    
    # Assigning a Call to a Tuple (line 54):
    
    # Assigning a Subscript to a Name (line 54):
    
    # Obtaining the type of the subscript
    int_104800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 8), 'int')
    
    # Call to orthogonal_procrustes(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'A_orig' (line 54)
    A_orig_104802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 42), 'A_orig', False)
    # Getting the type of 'B_orig' (line 54)
    B_orig_104803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 50), 'B_orig', False)
    # Processing the call keyword arguments (line 54)
    kwargs_104804 = {}
    # Getting the type of 'orthogonal_procrustes' (line 54)
    orthogonal_procrustes_104801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'orthogonal_procrustes', False)
    # Calling orthogonal_procrustes(args, kwargs) (line 54)
    orthogonal_procrustes_call_result_104805 = invoke(stypy.reporting.localization.Localization(__file__, 54, 20), orthogonal_procrustes_104801, *[A_orig_104802, B_orig_104803], **kwargs_104804)
    
    # Obtaining the member '__getitem__' of a type (line 54)
    getitem___104806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), orthogonal_procrustes_call_result_104805, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 54)
    subscript_call_result_104807 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), getitem___104806, int_104800)
    
    # Assigning a type to the variable 'tuple_var_assignment_104556' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tuple_var_assignment_104556', subscript_call_result_104807)
    
    # Assigning a Subscript to a Name (line 54):
    
    # Obtaining the type of the subscript
    int_104808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 8), 'int')
    
    # Call to orthogonal_procrustes(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'A_orig' (line 54)
    A_orig_104810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 42), 'A_orig', False)
    # Getting the type of 'B_orig' (line 54)
    B_orig_104811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 50), 'B_orig', False)
    # Processing the call keyword arguments (line 54)
    kwargs_104812 = {}
    # Getting the type of 'orthogonal_procrustes' (line 54)
    orthogonal_procrustes_104809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'orthogonal_procrustes', False)
    # Calling orthogonal_procrustes(args, kwargs) (line 54)
    orthogonal_procrustes_call_result_104813 = invoke(stypy.reporting.localization.Localization(__file__, 54, 20), orthogonal_procrustes_104809, *[A_orig_104810, B_orig_104811], **kwargs_104812)
    
    # Obtaining the member '__getitem__' of a type (line 54)
    getitem___104814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), orthogonal_procrustes_call_result_104813, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 54)
    subscript_call_result_104815 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), getitem___104814, int_104808)
    
    # Assigning a type to the variable 'tuple_var_assignment_104557' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tuple_var_assignment_104557', subscript_call_result_104815)
    
    # Assigning a Name to a Name (line 54):
    # Getting the type of 'tuple_var_assignment_104556' (line 54)
    tuple_var_assignment_104556_104816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tuple_var_assignment_104556')
    # Assigning a type to the variable 'R_orig' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'R_orig', tuple_var_assignment_104556_104816)
    
    # Assigning a Name to a Name (line 54):
    # Getting the type of 'tuple_var_assignment_104557' (line 54)
    tuple_var_assignment_104557_104817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tuple_var_assignment_104557')
    # Assigning a type to the variable 's' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 's', tuple_var_assignment_104557_104817)
    
    
    # Call to square(...): (line 55)
    # Processing the call arguments (line 55)
    
    # Call to randn(...): (line 55)
    # Processing the call arguments (line 55)
    int_104823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 49), 'int')
    # Processing the call keyword arguments (line 55)
    kwargs_104824 = {}
    # Getting the type of 'np' (line 55)
    np_104820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 33), 'np', False)
    # Obtaining the member 'random' of a type (line 55)
    random_104821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 33), np_104820, 'random')
    # Obtaining the member 'randn' of a type (line 55)
    randn_104822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 33), random_104821, 'randn')
    # Calling randn(args, kwargs) (line 55)
    randn_call_result_104825 = invoke(stypy.reporting.localization.Localization(__file__, 55, 33), randn_104822, *[int_104823], **kwargs_104824)
    
    # Processing the call keyword arguments (line 55)
    kwargs_104826 = {}
    # Getting the type of 'np' (line 55)
    np_104818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 23), 'np', False)
    # Obtaining the member 'square' of a type (line 55)
    square_104819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 23), np_104818, 'square')
    # Calling square(args, kwargs) (line 55)
    square_call_result_104827 = invoke(stypy.reporting.localization.Localization(__file__, 55, 23), square_104819, *[randn_call_result_104825], **kwargs_104826)
    
    # Testing the type of a for loop iterable (line 55)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 55, 8), square_call_result_104827)
    # Getting the type of the for loop variable (line 55)
    for_loop_var_104828 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 55, 8), square_call_result_104827)
    # Assigning a type to the variable 'A_scale' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'A_scale', for_loop_var_104828)
    # SSA begins for a for statement (line 55)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to square(...): (line 56)
    # Processing the call arguments (line 56)
    
    # Call to randn(...): (line 56)
    # Processing the call arguments (line 56)
    int_104834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 53), 'int')
    # Processing the call keyword arguments (line 56)
    kwargs_104835 = {}
    # Getting the type of 'np' (line 56)
    np_104831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 37), 'np', False)
    # Obtaining the member 'random' of a type (line 56)
    random_104832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 37), np_104831, 'random')
    # Obtaining the member 'randn' of a type (line 56)
    randn_104833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 37), random_104832, 'randn')
    # Calling randn(args, kwargs) (line 56)
    randn_call_result_104836 = invoke(stypy.reporting.localization.Localization(__file__, 56, 37), randn_104833, *[int_104834], **kwargs_104835)
    
    # Processing the call keyword arguments (line 56)
    kwargs_104837 = {}
    # Getting the type of 'np' (line 56)
    np_104829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 27), 'np', False)
    # Obtaining the member 'square' of a type (line 56)
    square_104830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 27), np_104829, 'square')
    # Calling square(args, kwargs) (line 56)
    square_call_result_104838 = invoke(stypy.reporting.localization.Localization(__file__, 56, 27), square_104830, *[randn_call_result_104836], **kwargs_104837)
    
    # Testing the type of a for loop iterable (line 56)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 56, 12), square_call_result_104838)
    # Getting the type of the for loop variable (line 56)
    for_loop_var_104839 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 56, 12), square_call_result_104838)
    # Assigning a type to the variable 'B_scale' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'B_scale', for_loop_var_104839)
    # SSA begins for a for statement (line 56)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Tuple (line 57):
    
    # Assigning a Subscript to a Name (line 57):
    
    # Obtaining the type of the subscript
    int_104840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 16), 'int')
    
    # Call to orthogonal_procrustes(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'A_orig' (line 57)
    A_orig_104842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 45), 'A_orig', False)
    # Getting the type of 'A_scale' (line 57)
    A_scale_104843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 54), 'A_scale', False)
    # Applying the binary operator '*' (line 57)
    result_mul_104844 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 45), '*', A_orig_104842, A_scale_104843)
    
    # Getting the type of 'B_orig' (line 57)
    B_orig_104845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 63), 'B_orig', False)
    # Getting the type of 'B_scale' (line 57)
    B_scale_104846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 72), 'B_scale', False)
    # Applying the binary operator '*' (line 57)
    result_mul_104847 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 63), '*', B_orig_104845, B_scale_104846)
    
    # Processing the call keyword arguments (line 57)
    kwargs_104848 = {}
    # Getting the type of 'orthogonal_procrustes' (line 57)
    orthogonal_procrustes_104841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 23), 'orthogonal_procrustes', False)
    # Calling orthogonal_procrustes(args, kwargs) (line 57)
    orthogonal_procrustes_call_result_104849 = invoke(stypy.reporting.localization.Localization(__file__, 57, 23), orthogonal_procrustes_104841, *[result_mul_104844, result_mul_104847], **kwargs_104848)
    
    # Obtaining the member '__getitem__' of a type (line 57)
    getitem___104850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 16), orthogonal_procrustes_call_result_104849, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 57)
    subscript_call_result_104851 = invoke(stypy.reporting.localization.Localization(__file__, 57, 16), getitem___104850, int_104840)
    
    # Assigning a type to the variable 'tuple_var_assignment_104558' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'tuple_var_assignment_104558', subscript_call_result_104851)
    
    # Assigning a Subscript to a Name (line 57):
    
    # Obtaining the type of the subscript
    int_104852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 16), 'int')
    
    # Call to orthogonal_procrustes(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'A_orig' (line 57)
    A_orig_104854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 45), 'A_orig', False)
    # Getting the type of 'A_scale' (line 57)
    A_scale_104855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 54), 'A_scale', False)
    # Applying the binary operator '*' (line 57)
    result_mul_104856 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 45), '*', A_orig_104854, A_scale_104855)
    
    # Getting the type of 'B_orig' (line 57)
    B_orig_104857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 63), 'B_orig', False)
    # Getting the type of 'B_scale' (line 57)
    B_scale_104858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 72), 'B_scale', False)
    # Applying the binary operator '*' (line 57)
    result_mul_104859 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 63), '*', B_orig_104857, B_scale_104858)
    
    # Processing the call keyword arguments (line 57)
    kwargs_104860 = {}
    # Getting the type of 'orthogonal_procrustes' (line 57)
    orthogonal_procrustes_104853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 23), 'orthogonal_procrustes', False)
    # Calling orthogonal_procrustes(args, kwargs) (line 57)
    orthogonal_procrustes_call_result_104861 = invoke(stypy.reporting.localization.Localization(__file__, 57, 23), orthogonal_procrustes_104853, *[result_mul_104856, result_mul_104859], **kwargs_104860)
    
    # Obtaining the member '__getitem__' of a type (line 57)
    getitem___104862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 16), orthogonal_procrustes_call_result_104861, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 57)
    subscript_call_result_104863 = invoke(stypy.reporting.localization.Localization(__file__, 57, 16), getitem___104862, int_104852)
    
    # Assigning a type to the variable 'tuple_var_assignment_104559' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'tuple_var_assignment_104559', subscript_call_result_104863)
    
    # Assigning a Name to a Name (line 57):
    # Getting the type of 'tuple_var_assignment_104558' (line 57)
    tuple_var_assignment_104558_104864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'tuple_var_assignment_104558')
    # Assigning a type to the variable 'R' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'R', tuple_var_assignment_104558_104864)
    
    # Assigning a Name to a Name (line 57):
    # Getting the type of 'tuple_var_assignment_104559' (line 57)
    tuple_var_assignment_104559_104865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'tuple_var_assignment_104559')
    # Assigning a type to the variable 's' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 's', tuple_var_assignment_104559_104865)
    
    # Call to assert_allclose(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'R' (line 58)
    R_104867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 32), 'R', False)
    # Getting the type of 'R_orig' (line 58)
    R_orig_104868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 35), 'R_orig', False)
    # Processing the call keyword arguments (line 58)
    kwargs_104869 = {}
    # Getting the type of 'assert_allclose' (line 58)
    assert_allclose_104866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 58)
    assert_allclose_call_result_104870 = invoke(stypy.reporting.localization.Localization(__file__, 58, 16), assert_allclose_104866, *[R_104867, R_orig_104868], **kwargs_104869)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_orthogonal_procrustes_scale_invariance(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_orthogonal_procrustes_scale_invariance' in the type store
    # Getting the type of 'stypy_return_type' (line 48)
    stypy_return_type_104871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_104871)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_orthogonal_procrustes_scale_invariance'
    return stypy_return_type_104871

# Assigning a type to the variable 'test_orthogonal_procrustes_scale_invariance' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'test_orthogonal_procrustes_scale_invariance', test_orthogonal_procrustes_scale_invariance)

@norecursion
def test_orthogonal_procrustes_array_conversion(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_orthogonal_procrustes_array_conversion'
    module_type_store = module_type_store.open_function_context('test_orthogonal_procrustes_array_conversion', 61, 0, False)
    
    # Passed parameters checking function
    test_orthogonal_procrustes_array_conversion.stypy_localization = localization
    test_orthogonal_procrustes_array_conversion.stypy_type_of_self = None
    test_orthogonal_procrustes_array_conversion.stypy_type_store = module_type_store
    test_orthogonal_procrustes_array_conversion.stypy_function_name = 'test_orthogonal_procrustes_array_conversion'
    test_orthogonal_procrustes_array_conversion.stypy_param_names_list = []
    test_orthogonal_procrustes_array_conversion.stypy_varargs_param_name = None
    test_orthogonal_procrustes_array_conversion.stypy_kwargs_param_name = None
    test_orthogonal_procrustes_array_conversion.stypy_call_defaults = defaults
    test_orthogonal_procrustes_array_conversion.stypy_call_varargs = varargs
    test_orthogonal_procrustes_array_conversion.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_orthogonal_procrustes_array_conversion', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_orthogonal_procrustes_array_conversion', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_orthogonal_procrustes_array_conversion(...)' code ##################

    
    # Call to seed(...): (line 62)
    # Processing the call arguments (line 62)
    int_104875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 19), 'int')
    # Processing the call keyword arguments (line 62)
    kwargs_104876 = {}
    # Getting the type of 'np' (line 62)
    np_104872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 62)
    random_104873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 4), np_104872, 'random')
    # Obtaining the member 'seed' of a type (line 62)
    seed_104874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 4), random_104873, 'seed')
    # Calling seed(args, kwargs) (line 62)
    seed_call_result_104877 = invoke(stypy.reporting.localization.Localization(__file__, 62, 4), seed_104874, *[int_104875], **kwargs_104876)
    
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 63)
    tuple_104878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 63)
    # Adding element type (line 63)
    
    # Obtaining an instance of the builtin type 'tuple' (line 63)
    tuple_104879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 63)
    # Adding element type (line 63)
    int_104880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 18), tuple_104879, int_104880)
    # Adding element type (line 63)
    int_104881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 18), tuple_104879, int_104881)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 17), tuple_104878, tuple_104879)
    # Adding element type (line 63)
    
    # Obtaining an instance of the builtin type 'tuple' (line 63)
    tuple_104882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 63)
    # Adding element type (line 63)
    int_104883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 26), tuple_104882, int_104883)
    # Adding element type (line 63)
    int_104884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 26), tuple_104882, int_104884)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 17), tuple_104878, tuple_104882)
    # Adding element type (line 63)
    
    # Obtaining an instance of the builtin type 'tuple' (line 63)
    tuple_104885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 63)
    # Adding element type (line 63)
    int_104886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 34), tuple_104885, int_104886)
    # Adding element type (line 63)
    int_104887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 34), tuple_104885, int_104887)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 17), tuple_104878, tuple_104885)
    
    # Testing the type of a for loop iterable (line 63)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 63, 4), tuple_104878)
    # Getting the type of the for loop variable (line 63)
    for_loop_var_104888 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 63, 4), tuple_104878)
    # Assigning a type to the variable 'm' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'm', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 4), for_loop_var_104888))
    # Assigning a type to the variable 'n' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 4), for_loop_var_104888))
    # SSA begins for a for statement (line 63)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 64):
    
    # Assigning a Call to a Name (line 64):
    
    # Call to randn(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'm' (line 64)
    m_104892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 32), 'm', False)
    # Getting the type of 'n' (line 64)
    n_104893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 35), 'n', False)
    # Processing the call keyword arguments (line 64)
    kwargs_104894 = {}
    # Getting the type of 'np' (line 64)
    np_104889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 16), 'np', False)
    # Obtaining the member 'random' of a type (line 64)
    random_104890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 16), np_104889, 'random')
    # Obtaining the member 'randn' of a type (line 64)
    randn_104891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 16), random_104890, 'randn')
    # Calling randn(args, kwargs) (line 64)
    randn_call_result_104895 = invoke(stypy.reporting.localization.Localization(__file__, 64, 16), randn_104891, *[m_104892, n_104893], **kwargs_104894)
    
    # Assigning a type to the variable 'A_arr' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'A_arr', randn_call_result_104895)
    
    # Assigning a Call to a Name (line 65):
    
    # Assigning a Call to a Name (line 65):
    
    # Call to randn(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'm' (line 65)
    m_104899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 32), 'm', False)
    # Getting the type of 'n' (line 65)
    n_104900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 35), 'n', False)
    # Processing the call keyword arguments (line 65)
    kwargs_104901 = {}
    # Getting the type of 'np' (line 65)
    np_104896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'np', False)
    # Obtaining the member 'random' of a type (line 65)
    random_104897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 16), np_104896, 'random')
    # Obtaining the member 'randn' of a type (line 65)
    randn_104898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 16), random_104897, 'randn')
    # Calling randn(args, kwargs) (line 65)
    randn_call_result_104902 = invoke(stypy.reporting.localization.Localization(__file__, 65, 16), randn_104898, *[m_104899, n_104900], **kwargs_104901)
    
    # Assigning a type to the variable 'B_arr' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'B_arr', randn_call_result_104902)
    
    # Assigning a Tuple to a Name (line 66):
    
    # Assigning a Tuple to a Name (line 66):
    
    # Obtaining an instance of the builtin type 'tuple' (line 66)
    tuple_104903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 66)
    # Adding element type (line 66)
    # Getting the type of 'A_arr' (line 66)
    A_arr_104904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 14), 'A_arr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 14), tuple_104903, A_arr_104904)
    # Adding element type (line 66)
    
    # Call to tolist(...): (line 66)
    # Processing the call keyword arguments (line 66)
    kwargs_104907 = {}
    # Getting the type of 'A_arr' (line 66)
    A_arr_104905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 21), 'A_arr', False)
    # Obtaining the member 'tolist' of a type (line 66)
    tolist_104906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 21), A_arr_104905, 'tolist')
    # Calling tolist(args, kwargs) (line 66)
    tolist_call_result_104908 = invoke(stypy.reporting.localization.Localization(__file__, 66, 21), tolist_104906, *[], **kwargs_104907)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 14), tuple_104903, tolist_call_result_104908)
    # Adding element type (line 66)
    
    # Call to matrix(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'A_arr' (line 66)
    A_arr_104911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 47), 'A_arr', False)
    # Processing the call keyword arguments (line 66)
    kwargs_104912 = {}
    # Getting the type of 'np' (line 66)
    np_104909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 37), 'np', False)
    # Obtaining the member 'matrix' of a type (line 66)
    matrix_104910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 37), np_104909, 'matrix')
    # Calling matrix(args, kwargs) (line 66)
    matrix_call_result_104913 = invoke(stypy.reporting.localization.Localization(__file__, 66, 37), matrix_104910, *[A_arr_104911], **kwargs_104912)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 14), tuple_104903, matrix_call_result_104913)
    
    # Assigning a type to the variable 'As' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'As', tuple_104903)
    
    # Assigning a Tuple to a Name (line 67):
    
    # Assigning a Tuple to a Name (line 67):
    
    # Obtaining an instance of the builtin type 'tuple' (line 67)
    tuple_104914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 67)
    # Adding element type (line 67)
    # Getting the type of 'B_arr' (line 67)
    B_arr_104915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 14), 'B_arr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 14), tuple_104914, B_arr_104915)
    # Adding element type (line 67)
    
    # Call to tolist(...): (line 67)
    # Processing the call keyword arguments (line 67)
    kwargs_104918 = {}
    # Getting the type of 'B_arr' (line 67)
    B_arr_104916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 21), 'B_arr', False)
    # Obtaining the member 'tolist' of a type (line 67)
    tolist_104917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 21), B_arr_104916, 'tolist')
    # Calling tolist(args, kwargs) (line 67)
    tolist_call_result_104919 = invoke(stypy.reporting.localization.Localization(__file__, 67, 21), tolist_104917, *[], **kwargs_104918)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 14), tuple_104914, tolist_call_result_104919)
    # Adding element type (line 67)
    
    # Call to matrix(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'B_arr' (line 67)
    B_arr_104922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 47), 'B_arr', False)
    # Processing the call keyword arguments (line 67)
    kwargs_104923 = {}
    # Getting the type of 'np' (line 67)
    np_104920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 37), 'np', False)
    # Obtaining the member 'matrix' of a type (line 67)
    matrix_104921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 37), np_104920, 'matrix')
    # Calling matrix(args, kwargs) (line 67)
    matrix_call_result_104924 = invoke(stypy.reporting.localization.Localization(__file__, 67, 37), matrix_104921, *[B_arr_104922], **kwargs_104923)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 14), tuple_104914, matrix_call_result_104924)
    
    # Assigning a type to the variable 'Bs' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'Bs', tuple_104914)
    
    # Assigning a Call to a Tuple (line 68):
    
    # Assigning a Subscript to a Name (line 68):
    
    # Obtaining the type of the subscript
    int_104925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 8), 'int')
    
    # Call to orthogonal_procrustes(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'A_arr' (line 68)
    A_arr_104927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 41), 'A_arr', False)
    # Getting the type of 'B_arr' (line 68)
    B_arr_104928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 48), 'B_arr', False)
    # Processing the call keyword arguments (line 68)
    kwargs_104929 = {}
    # Getting the type of 'orthogonal_procrustes' (line 68)
    orthogonal_procrustes_104926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 19), 'orthogonal_procrustes', False)
    # Calling orthogonal_procrustes(args, kwargs) (line 68)
    orthogonal_procrustes_call_result_104930 = invoke(stypy.reporting.localization.Localization(__file__, 68, 19), orthogonal_procrustes_104926, *[A_arr_104927, B_arr_104928], **kwargs_104929)
    
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___104931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), orthogonal_procrustes_call_result_104930, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_104932 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), getitem___104931, int_104925)
    
    # Assigning a type to the variable 'tuple_var_assignment_104560' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_104560', subscript_call_result_104932)
    
    # Assigning a Subscript to a Name (line 68):
    
    # Obtaining the type of the subscript
    int_104933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 8), 'int')
    
    # Call to orthogonal_procrustes(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'A_arr' (line 68)
    A_arr_104935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 41), 'A_arr', False)
    # Getting the type of 'B_arr' (line 68)
    B_arr_104936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 48), 'B_arr', False)
    # Processing the call keyword arguments (line 68)
    kwargs_104937 = {}
    # Getting the type of 'orthogonal_procrustes' (line 68)
    orthogonal_procrustes_104934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 19), 'orthogonal_procrustes', False)
    # Calling orthogonal_procrustes(args, kwargs) (line 68)
    orthogonal_procrustes_call_result_104938 = invoke(stypy.reporting.localization.Localization(__file__, 68, 19), orthogonal_procrustes_104934, *[A_arr_104935, B_arr_104936], **kwargs_104937)
    
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___104939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), orthogonal_procrustes_call_result_104938, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_104940 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), getitem___104939, int_104933)
    
    # Assigning a type to the variable 'tuple_var_assignment_104561' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_104561', subscript_call_result_104940)
    
    # Assigning a Name to a Name (line 68):
    # Getting the type of 'tuple_var_assignment_104560' (line 68)
    tuple_var_assignment_104560_104941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_104560')
    # Assigning a type to the variable 'R_arr' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'R_arr', tuple_var_assignment_104560_104941)
    
    # Assigning a Name to a Name (line 68):
    # Getting the type of 'tuple_var_assignment_104561' (line 68)
    tuple_var_assignment_104561_104942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_104561')
    # Assigning a type to the variable 's' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 15), 's', tuple_var_assignment_104561_104942)
    
    # Assigning a Call to a Name (line 69):
    
    # Assigning a Call to a Name (line 69):
    
    # Call to dot(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'R_arr' (line 69)
    R_arr_104945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 27), 'R_arr', False)
    # Processing the call keyword arguments (line 69)
    kwargs_104946 = {}
    # Getting the type of 'A_arr' (line 69)
    A_arr_104943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 17), 'A_arr', False)
    # Obtaining the member 'dot' of a type (line 69)
    dot_104944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 17), A_arr_104943, 'dot')
    # Calling dot(args, kwargs) (line 69)
    dot_call_result_104947 = invoke(stypy.reporting.localization.Localization(__file__, 69, 17), dot_104944, *[R_arr_104945], **kwargs_104946)
    
    # Assigning a type to the variable 'AR_arr' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'AR_arr', dot_call_result_104947)
    
    
    # Call to product(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'As' (line 70)
    As_104949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 28), 'As', False)
    # Getting the type of 'Bs' (line 70)
    Bs_104950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 32), 'Bs', False)
    # Processing the call keyword arguments (line 70)
    kwargs_104951 = {}
    # Getting the type of 'product' (line 70)
    product_104948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 20), 'product', False)
    # Calling product(args, kwargs) (line 70)
    product_call_result_104952 = invoke(stypy.reporting.localization.Localization(__file__, 70, 20), product_104948, *[As_104949, Bs_104950], **kwargs_104951)
    
    # Testing the type of a for loop iterable (line 70)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 70, 8), product_call_result_104952)
    # Getting the type of the for loop variable (line 70)
    for_loop_var_104953 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 70, 8), product_call_result_104952)
    # Assigning a type to the variable 'A' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'A', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 8), for_loop_var_104953))
    # Assigning a type to the variable 'B' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'B', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 8), for_loop_var_104953))
    # SSA begins for a for statement (line 70)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Tuple (line 71):
    
    # Assigning a Subscript to a Name (line 71):
    
    # Obtaining the type of the subscript
    int_104954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 12), 'int')
    
    # Call to orthogonal_procrustes(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'A' (line 71)
    A_104956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 41), 'A', False)
    # Getting the type of 'B' (line 71)
    B_104957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 44), 'B', False)
    # Processing the call keyword arguments (line 71)
    kwargs_104958 = {}
    # Getting the type of 'orthogonal_procrustes' (line 71)
    orthogonal_procrustes_104955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'orthogonal_procrustes', False)
    # Calling orthogonal_procrustes(args, kwargs) (line 71)
    orthogonal_procrustes_call_result_104959 = invoke(stypy.reporting.localization.Localization(__file__, 71, 19), orthogonal_procrustes_104955, *[A_104956, B_104957], **kwargs_104958)
    
    # Obtaining the member '__getitem__' of a type (line 71)
    getitem___104960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 12), orthogonal_procrustes_call_result_104959, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
    subscript_call_result_104961 = invoke(stypy.reporting.localization.Localization(__file__, 71, 12), getitem___104960, int_104954)
    
    # Assigning a type to the variable 'tuple_var_assignment_104562' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'tuple_var_assignment_104562', subscript_call_result_104961)
    
    # Assigning a Subscript to a Name (line 71):
    
    # Obtaining the type of the subscript
    int_104962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 12), 'int')
    
    # Call to orthogonal_procrustes(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'A' (line 71)
    A_104964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 41), 'A', False)
    # Getting the type of 'B' (line 71)
    B_104965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 44), 'B', False)
    # Processing the call keyword arguments (line 71)
    kwargs_104966 = {}
    # Getting the type of 'orthogonal_procrustes' (line 71)
    orthogonal_procrustes_104963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'orthogonal_procrustes', False)
    # Calling orthogonal_procrustes(args, kwargs) (line 71)
    orthogonal_procrustes_call_result_104967 = invoke(stypy.reporting.localization.Localization(__file__, 71, 19), orthogonal_procrustes_104963, *[A_104964, B_104965], **kwargs_104966)
    
    # Obtaining the member '__getitem__' of a type (line 71)
    getitem___104968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 12), orthogonal_procrustes_call_result_104967, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
    subscript_call_result_104969 = invoke(stypy.reporting.localization.Localization(__file__, 71, 12), getitem___104968, int_104962)
    
    # Assigning a type to the variable 'tuple_var_assignment_104563' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'tuple_var_assignment_104563', subscript_call_result_104969)
    
    # Assigning a Name to a Name (line 71):
    # Getting the type of 'tuple_var_assignment_104562' (line 71)
    tuple_var_assignment_104562_104970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'tuple_var_assignment_104562')
    # Assigning a type to the variable 'R' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'R', tuple_var_assignment_104562_104970)
    
    # Assigning a Name to a Name (line 71):
    # Getting the type of 'tuple_var_assignment_104563' (line 71)
    tuple_var_assignment_104563_104971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'tuple_var_assignment_104563')
    # Assigning a type to the variable 's' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 15), 's', tuple_var_assignment_104563_104971)
    
    # Assigning a Call to a Name (line 72):
    
    # Assigning a Call to a Name (line 72):
    
    # Call to dot(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'R' (line 72)
    R_104974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 27), 'R', False)
    # Processing the call keyword arguments (line 72)
    kwargs_104975 = {}
    # Getting the type of 'A_arr' (line 72)
    A_arr_104972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 17), 'A_arr', False)
    # Obtaining the member 'dot' of a type (line 72)
    dot_104973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 17), A_arr_104972, 'dot')
    # Calling dot(args, kwargs) (line 72)
    dot_call_result_104976 = invoke(stypy.reporting.localization.Localization(__file__, 72, 17), dot_104973, *[R_104974], **kwargs_104975)
    
    # Assigning a type to the variable 'AR' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'AR', dot_call_result_104976)
    
    # Call to assert_allclose(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'AR' (line 73)
    AR_104978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 28), 'AR', False)
    # Getting the type of 'AR_arr' (line 73)
    AR_arr_104979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 32), 'AR_arr', False)
    # Processing the call keyword arguments (line 73)
    kwargs_104980 = {}
    # Getting the type of 'assert_allclose' (line 73)
    assert_allclose_104977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 73)
    assert_allclose_call_result_104981 = invoke(stypy.reporting.localization.Localization(__file__, 73, 12), assert_allclose_104977, *[AR_104978, AR_arr_104979], **kwargs_104980)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_orthogonal_procrustes_array_conversion(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_orthogonal_procrustes_array_conversion' in the type store
    # Getting the type of 'stypy_return_type' (line 61)
    stypy_return_type_104982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_104982)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_orthogonal_procrustes_array_conversion'
    return stypy_return_type_104982

# Assigning a type to the variable 'test_orthogonal_procrustes_array_conversion' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'test_orthogonal_procrustes_array_conversion', test_orthogonal_procrustes_array_conversion)

@norecursion
def test_orthogonal_procrustes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_orthogonal_procrustes'
    module_type_store = module_type_store.open_function_context('test_orthogonal_procrustes', 76, 0, False)
    
    # Passed parameters checking function
    test_orthogonal_procrustes.stypy_localization = localization
    test_orthogonal_procrustes.stypy_type_of_self = None
    test_orthogonal_procrustes.stypy_type_store = module_type_store
    test_orthogonal_procrustes.stypy_function_name = 'test_orthogonal_procrustes'
    test_orthogonal_procrustes.stypy_param_names_list = []
    test_orthogonal_procrustes.stypy_varargs_param_name = None
    test_orthogonal_procrustes.stypy_kwargs_param_name = None
    test_orthogonal_procrustes.stypy_call_defaults = defaults
    test_orthogonal_procrustes.stypy_call_varargs = varargs
    test_orthogonal_procrustes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_orthogonal_procrustes', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_orthogonal_procrustes', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_orthogonal_procrustes(...)' code ##################

    
    # Call to seed(...): (line 77)
    # Processing the call arguments (line 77)
    int_104986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 19), 'int')
    # Processing the call keyword arguments (line 77)
    kwargs_104987 = {}
    # Getting the type of 'np' (line 77)
    np_104983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 77)
    random_104984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 4), np_104983, 'random')
    # Obtaining the member 'seed' of a type (line 77)
    seed_104985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 4), random_104984, 'seed')
    # Calling seed(args, kwargs) (line 77)
    seed_call_result_104988 = invoke(stypy.reporting.localization.Localization(__file__, 77, 4), seed_104985, *[int_104986], **kwargs_104987)
    
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 78)
    tuple_104989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 78)
    # Adding element type (line 78)
    
    # Obtaining an instance of the builtin type 'tuple' (line 78)
    tuple_104990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 78)
    # Adding element type (line 78)
    int_104991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 18), tuple_104990, int_104991)
    # Adding element type (line 78)
    int_104992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 18), tuple_104990, int_104992)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 17), tuple_104989, tuple_104990)
    # Adding element type (line 78)
    
    # Obtaining an instance of the builtin type 'tuple' (line 78)
    tuple_104993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 78)
    # Adding element type (line 78)
    int_104994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 26), tuple_104993, int_104994)
    # Adding element type (line 78)
    int_104995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 26), tuple_104993, int_104995)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 17), tuple_104989, tuple_104993)
    # Adding element type (line 78)
    
    # Obtaining an instance of the builtin type 'tuple' (line 78)
    tuple_104996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 78)
    # Adding element type (line 78)
    int_104997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 34), tuple_104996, int_104997)
    # Adding element type (line 78)
    int_104998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 34), tuple_104996, int_104998)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 17), tuple_104989, tuple_104996)
    
    # Testing the type of a for loop iterable (line 78)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 78, 4), tuple_104989)
    # Getting the type of the for loop variable (line 78)
    for_loop_var_104999 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 78, 4), tuple_104989)
    # Assigning a type to the variable 'm' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'm', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 4), for_loop_var_104999))
    # Assigning a type to the variable 'n' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 4), for_loop_var_104999))
    # SSA begins for a for statement (line 78)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 80):
    
    # Assigning a Call to a Name (line 80):
    
    # Call to randn(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'm' (line 80)
    m_105003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 28), 'm', False)
    # Getting the type of 'n' (line 80)
    n_105004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 31), 'n', False)
    # Processing the call keyword arguments (line 80)
    kwargs_105005 = {}
    # Getting the type of 'np' (line 80)
    np_105000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'np', False)
    # Obtaining the member 'random' of a type (line 80)
    random_105001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), np_105000, 'random')
    # Obtaining the member 'randn' of a type (line 80)
    randn_105002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), random_105001, 'randn')
    # Calling randn(args, kwargs) (line 80)
    randn_call_result_105006 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), randn_105002, *[m_105003, n_105004], **kwargs_105005)
    
    # Assigning a type to the variable 'B' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'B', randn_call_result_105006)
    
    # Assigning a Call to a Name (line 83):
    
    # Assigning a Call to a Name (line 83):
    
    # Call to randn(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'n' (line 83)
    n_105010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 28), 'n', False)
    # Getting the type of 'n' (line 83)
    n_105011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 31), 'n', False)
    # Processing the call keyword arguments (line 83)
    kwargs_105012 = {}
    # Getting the type of 'np' (line 83)
    np_105007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'np', False)
    # Obtaining the member 'random' of a type (line 83)
    random_105008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 12), np_105007, 'random')
    # Obtaining the member 'randn' of a type (line 83)
    randn_105009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 12), random_105008, 'randn')
    # Calling randn(args, kwargs) (line 83)
    randn_call_result_105013 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), randn_105009, *[n_105010, n_105011], **kwargs_105012)
    
    # Assigning a type to the variable 'X' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'X', randn_call_result_105013)
    
    # Assigning a Call to a Tuple (line 84):
    
    # Assigning a Subscript to a Name (line 84):
    
    # Obtaining the type of the subscript
    int_105014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 8), 'int')
    
    # Call to eigh(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'X' (line 84)
    X_105016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 20), 'X', False)
    # Obtaining the member 'T' of a type (line 84)
    T_105017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 20), X_105016, 'T')
    # Getting the type of 'X' (line 84)
    X_105018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 26), 'X', False)
    # Applying the binary operator '+' (line 84)
    result_add_105019 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 20), '+', T_105017, X_105018)
    
    # Processing the call keyword arguments (line 84)
    kwargs_105020 = {}
    # Getting the type of 'eigh' (line 84)
    eigh_105015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 15), 'eigh', False)
    # Calling eigh(args, kwargs) (line 84)
    eigh_call_result_105021 = invoke(stypy.reporting.localization.Localization(__file__, 84, 15), eigh_105015, *[result_add_105019], **kwargs_105020)
    
    # Obtaining the member '__getitem__' of a type (line 84)
    getitem___105022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), eigh_call_result_105021, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 84)
    subscript_call_result_105023 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), getitem___105022, int_105014)
    
    # Assigning a type to the variable 'tuple_var_assignment_104564' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'tuple_var_assignment_104564', subscript_call_result_105023)
    
    # Assigning a Subscript to a Name (line 84):
    
    # Obtaining the type of the subscript
    int_105024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 8), 'int')
    
    # Call to eigh(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'X' (line 84)
    X_105026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 20), 'X', False)
    # Obtaining the member 'T' of a type (line 84)
    T_105027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 20), X_105026, 'T')
    # Getting the type of 'X' (line 84)
    X_105028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 26), 'X', False)
    # Applying the binary operator '+' (line 84)
    result_add_105029 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 20), '+', T_105027, X_105028)
    
    # Processing the call keyword arguments (line 84)
    kwargs_105030 = {}
    # Getting the type of 'eigh' (line 84)
    eigh_105025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 15), 'eigh', False)
    # Calling eigh(args, kwargs) (line 84)
    eigh_call_result_105031 = invoke(stypy.reporting.localization.Localization(__file__, 84, 15), eigh_105025, *[result_add_105029], **kwargs_105030)
    
    # Obtaining the member '__getitem__' of a type (line 84)
    getitem___105032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), eigh_call_result_105031, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 84)
    subscript_call_result_105033 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), getitem___105032, int_105024)
    
    # Assigning a type to the variable 'tuple_var_assignment_104565' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'tuple_var_assignment_104565', subscript_call_result_105033)
    
    # Assigning a Name to a Name (line 84):
    # Getting the type of 'tuple_var_assignment_104564' (line 84)
    tuple_var_assignment_104564_105034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'tuple_var_assignment_104564')
    # Assigning a type to the variable 'w' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'w', tuple_var_assignment_104564_105034)
    
    # Assigning a Name to a Name (line 84):
    # Getting the type of 'tuple_var_assignment_104565' (line 84)
    tuple_var_assignment_104565_105035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'tuple_var_assignment_104565')
    # Assigning a type to the variable 'V' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 11), 'V', tuple_var_assignment_104565_105035)
    
    # Call to assert_allclose(...): (line 85)
    # Processing the call arguments (line 85)
    
    # Call to inv(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'V' (line 85)
    V_105038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 28), 'V', False)
    # Processing the call keyword arguments (line 85)
    kwargs_105039 = {}
    # Getting the type of 'inv' (line 85)
    inv_105037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 24), 'inv', False)
    # Calling inv(args, kwargs) (line 85)
    inv_call_result_105040 = invoke(stypy.reporting.localization.Localization(__file__, 85, 24), inv_105037, *[V_105038], **kwargs_105039)
    
    # Getting the type of 'V' (line 85)
    V_105041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 32), 'V', False)
    # Obtaining the member 'T' of a type (line 85)
    T_105042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 32), V_105041, 'T')
    # Processing the call keyword arguments (line 85)
    kwargs_105043 = {}
    # Getting the type of 'assert_allclose' (line 85)
    assert_allclose_105036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 85)
    assert_allclose_call_result_105044 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), assert_allclose_105036, *[inv_call_result_105040, T_105042], **kwargs_105043)
    
    
    # Assigning a Call to a Name (line 87):
    
    # Assigning a Call to a Name (line 87):
    
    # Call to dot(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'B' (line 87)
    B_105047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 19), 'B', False)
    # Getting the type of 'V' (line 87)
    V_105048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 22), 'V', False)
    # Obtaining the member 'T' of a type (line 87)
    T_105049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 22), V_105048, 'T')
    # Processing the call keyword arguments (line 87)
    kwargs_105050 = {}
    # Getting the type of 'np' (line 87)
    np_105045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'np', False)
    # Obtaining the member 'dot' of a type (line 87)
    dot_105046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 12), np_105045, 'dot')
    # Calling dot(args, kwargs) (line 87)
    dot_call_result_105051 = invoke(stypy.reporting.localization.Localization(__file__, 87, 12), dot_105046, *[B_105047, T_105049], **kwargs_105050)
    
    # Assigning a type to the variable 'A' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'A', dot_call_result_105051)
    
    # Assigning a Call to a Tuple (line 89):
    
    # Assigning a Subscript to a Name (line 89):
    
    # Obtaining the type of the subscript
    int_105052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 8), 'int')
    
    # Call to orthogonal_procrustes(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'A' (line 89)
    A_105054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 37), 'A', False)
    # Getting the type of 'B' (line 89)
    B_105055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 40), 'B', False)
    # Processing the call keyword arguments (line 89)
    kwargs_105056 = {}
    # Getting the type of 'orthogonal_procrustes' (line 89)
    orthogonal_procrustes_105053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), 'orthogonal_procrustes', False)
    # Calling orthogonal_procrustes(args, kwargs) (line 89)
    orthogonal_procrustes_call_result_105057 = invoke(stypy.reporting.localization.Localization(__file__, 89, 15), orthogonal_procrustes_105053, *[A_105054, B_105055], **kwargs_105056)
    
    # Obtaining the member '__getitem__' of a type (line 89)
    getitem___105058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), orthogonal_procrustes_call_result_105057, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
    subscript_call_result_105059 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), getitem___105058, int_105052)
    
    # Assigning a type to the variable 'tuple_var_assignment_104566' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'tuple_var_assignment_104566', subscript_call_result_105059)
    
    # Assigning a Subscript to a Name (line 89):
    
    # Obtaining the type of the subscript
    int_105060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 8), 'int')
    
    # Call to orthogonal_procrustes(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'A' (line 89)
    A_105062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 37), 'A', False)
    # Getting the type of 'B' (line 89)
    B_105063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 40), 'B', False)
    # Processing the call keyword arguments (line 89)
    kwargs_105064 = {}
    # Getting the type of 'orthogonal_procrustes' (line 89)
    orthogonal_procrustes_105061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), 'orthogonal_procrustes', False)
    # Calling orthogonal_procrustes(args, kwargs) (line 89)
    orthogonal_procrustes_call_result_105065 = invoke(stypy.reporting.localization.Localization(__file__, 89, 15), orthogonal_procrustes_105061, *[A_105062, B_105063], **kwargs_105064)
    
    # Obtaining the member '__getitem__' of a type (line 89)
    getitem___105066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), orthogonal_procrustes_call_result_105065, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
    subscript_call_result_105067 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), getitem___105066, int_105060)
    
    # Assigning a type to the variable 'tuple_var_assignment_104567' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'tuple_var_assignment_104567', subscript_call_result_105067)
    
    # Assigning a Name to a Name (line 89):
    # Getting the type of 'tuple_var_assignment_104566' (line 89)
    tuple_var_assignment_104566_105068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'tuple_var_assignment_104566')
    # Assigning a type to the variable 'R' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'R', tuple_var_assignment_104566_105068)
    
    # Assigning a Name to a Name (line 89):
    # Getting the type of 'tuple_var_assignment_104567' (line 89)
    tuple_var_assignment_104567_105069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'tuple_var_assignment_104567')
    # Assigning a type to the variable 's' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 's', tuple_var_assignment_104567_105069)
    
    # Call to assert_allclose(...): (line 90)
    # Processing the call arguments (line 90)
    
    # Call to inv(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'R' (line 90)
    R_105072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 28), 'R', False)
    # Processing the call keyword arguments (line 90)
    kwargs_105073 = {}
    # Getting the type of 'inv' (line 90)
    inv_105071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 24), 'inv', False)
    # Calling inv(args, kwargs) (line 90)
    inv_call_result_105074 = invoke(stypy.reporting.localization.Localization(__file__, 90, 24), inv_105071, *[R_105072], **kwargs_105073)
    
    # Getting the type of 'R' (line 90)
    R_105075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 32), 'R', False)
    # Obtaining the member 'T' of a type (line 90)
    T_105076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 32), R_105075, 'T')
    # Processing the call keyword arguments (line 90)
    kwargs_105077 = {}
    # Getting the type of 'assert_allclose' (line 90)
    assert_allclose_105070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 90)
    assert_allclose_call_result_105078 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), assert_allclose_105070, *[inv_call_result_105074, T_105076], **kwargs_105077)
    
    
    # Call to assert_allclose(...): (line 91)
    # Processing the call arguments (line 91)
    
    # Call to dot(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'R' (line 91)
    R_105082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 30), 'R', False)
    # Processing the call keyword arguments (line 91)
    kwargs_105083 = {}
    # Getting the type of 'A' (line 91)
    A_105080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 24), 'A', False)
    # Obtaining the member 'dot' of a type (line 91)
    dot_105081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 24), A_105080, 'dot')
    # Calling dot(args, kwargs) (line 91)
    dot_call_result_105084 = invoke(stypy.reporting.localization.Localization(__file__, 91, 24), dot_105081, *[R_105082], **kwargs_105083)
    
    # Getting the type of 'B' (line 91)
    B_105085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 34), 'B', False)
    # Processing the call keyword arguments (line 91)
    kwargs_105086 = {}
    # Getting the type of 'assert_allclose' (line 91)
    assert_allclose_105079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 91)
    assert_allclose_call_result_105087 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), assert_allclose_105079, *[dot_call_result_105084, B_105085], **kwargs_105086)
    
    
    # Assigning a BinOp to a Name (line 93):
    
    # Assigning a BinOp to a Name (line 93):
    # Getting the type of 'A' (line 93)
    A_105088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 22), 'A')
    float_105089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 26), 'float')
    
    # Call to randn(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'm' (line 93)
    m_105093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 49), 'm', False)
    # Getting the type of 'n' (line 93)
    n_105094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 52), 'n', False)
    # Processing the call keyword arguments (line 93)
    kwargs_105095 = {}
    # Getting the type of 'np' (line 93)
    np_105090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 33), 'np', False)
    # Obtaining the member 'random' of a type (line 93)
    random_105091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 33), np_105090, 'random')
    # Obtaining the member 'randn' of a type (line 93)
    randn_105092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 33), random_105091, 'randn')
    # Calling randn(args, kwargs) (line 93)
    randn_call_result_105096 = invoke(stypy.reporting.localization.Localization(__file__, 93, 33), randn_105092, *[m_105093, n_105094], **kwargs_105095)
    
    # Applying the binary operator '*' (line 93)
    result_mul_105097 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 26), '*', float_105089, randn_call_result_105096)
    
    # Applying the binary operator '+' (line 93)
    result_add_105098 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 22), '+', A_105088, result_mul_105097)
    
    # Assigning a type to the variable 'A_perturbed' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'A_perturbed', result_add_105098)
    
    # Assigning a Call to a Tuple (line 97):
    
    # Assigning a Subscript to a Name (line 97):
    
    # Obtaining the type of the subscript
    int_105099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 8), 'int')
    
    # Call to orthogonal_procrustes(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'A_perturbed' (line 97)
    A_perturbed_105101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 43), 'A_perturbed', False)
    # Getting the type of 'B' (line 97)
    B_105102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 56), 'B', False)
    # Processing the call keyword arguments (line 97)
    kwargs_105103 = {}
    # Getting the type of 'orthogonal_procrustes' (line 97)
    orthogonal_procrustes_105100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 21), 'orthogonal_procrustes', False)
    # Calling orthogonal_procrustes(args, kwargs) (line 97)
    orthogonal_procrustes_call_result_105104 = invoke(stypy.reporting.localization.Localization(__file__, 97, 21), orthogonal_procrustes_105100, *[A_perturbed_105101, B_105102], **kwargs_105103)
    
    # Obtaining the member '__getitem__' of a type (line 97)
    getitem___105105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), orthogonal_procrustes_call_result_105104, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
    subscript_call_result_105106 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), getitem___105105, int_105099)
    
    # Assigning a type to the variable 'tuple_var_assignment_104568' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'tuple_var_assignment_104568', subscript_call_result_105106)
    
    # Assigning a Subscript to a Name (line 97):
    
    # Obtaining the type of the subscript
    int_105107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 8), 'int')
    
    # Call to orthogonal_procrustes(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'A_perturbed' (line 97)
    A_perturbed_105109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 43), 'A_perturbed', False)
    # Getting the type of 'B' (line 97)
    B_105110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 56), 'B', False)
    # Processing the call keyword arguments (line 97)
    kwargs_105111 = {}
    # Getting the type of 'orthogonal_procrustes' (line 97)
    orthogonal_procrustes_105108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 21), 'orthogonal_procrustes', False)
    # Calling orthogonal_procrustes(args, kwargs) (line 97)
    orthogonal_procrustes_call_result_105112 = invoke(stypy.reporting.localization.Localization(__file__, 97, 21), orthogonal_procrustes_105108, *[A_perturbed_105109, B_105110], **kwargs_105111)
    
    # Obtaining the member '__getitem__' of a type (line 97)
    getitem___105113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), orthogonal_procrustes_call_result_105112, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
    subscript_call_result_105114 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), getitem___105113, int_105107)
    
    # Assigning a type to the variable 'tuple_var_assignment_104569' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'tuple_var_assignment_104569', subscript_call_result_105114)
    
    # Assigning a Name to a Name (line 97):
    # Getting the type of 'tuple_var_assignment_104568' (line 97)
    tuple_var_assignment_104568_105115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'tuple_var_assignment_104568')
    # Assigning a type to the variable 'R_prime' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'R_prime', tuple_var_assignment_104568_105115)
    
    # Assigning a Name to a Name (line 97):
    # Getting the type of 'tuple_var_assignment_104569' (line 97)
    tuple_var_assignment_104569_105116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'tuple_var_assignment_104569')
    # Assigning a type to the variable 's' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 17), 's', tuple_var_assignment_104569_105116)
    
    # Call to assert_allclose(...): (line 98)
    # Processing the call arguments (line 98)
    
    # Call to inv(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'R_prime' (line 98)
    R_prime_105119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 28), 'R_prime', False)
    # Processing the call keyword arguments (line 98)
    kwargs_105120 = {}
    # Getting the type of 'inv' (line 98)
    inv_105118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 24), 'inv', False)
    # Calling inv(args, kwargs) (line 98)
    inv_call_result_105121 = invoke(stypy.reporting.localization.Localization(__file__, 98, 24), inv_105118, *[R_prime_105119], **kwargs_105120)
    
    # Getting the type of 'R_prime' (line 98)
    R_prime_105122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 38), 'R_prime', False)
    # Obtaining the member 'T' of a type (line 98)
    T_105123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 38), R_prime_105122, 'T')
    # Processing the call keyword arguments (line 98)
    kwargs_105124 = {}
    # Getting the type of 'assert_allclose' (line 98)
    assert_allclose_105117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 98)
    assert_allclose_call_result_105125 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), assert_allclose_105117, *[inv_call_result_105121, T_105123], **kwargs_105124)
    
    
    # Assigning a Call to a Name (line 100):
    
    # Assigning a Call to a Name (line 100):
    
    # Call to dot(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'R' (line 100)
    R_105128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 39), 'R', False)
    # Processing the call keyword arguments (line 100)
    kwargs_105129 = {}
    # Getting the type of 'A_perturbed' (line 100)
    A_perturbed_105126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 23), 'A_perturbed', False)
    # Obtaining the member 'dot' of a type (line 100)
    dot_105127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 23), A_perturbed_105126, 'dot')
    # Calling dot(args, kwargs) (line 100)
    dot_call_result_105130 = invoke(stypy.reporting.localization.Localization(__file__, 100, 23), dot_105127, *[R_105128], **kwargs_105129)
    
    # Assigning a type to the variable 'naive_approx' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'naive_approx', dot_call_result_105130)
    
    # Assigning a Call to a Name (line 101):
    
    # Assigning a Call to a Name (line 101):
    
    # Call to dot(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'R_prime' (line 101)
    R_prime_105133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 39), 'R_prime', False)
    # Processing the call keyword arguments (line 101)
    kwargs_105134 = {}
    # Getting the type of 'A_perturbed' (line 101)
    A_perturbed_105131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 23), 'A_perturbed', False)
    # Obtaining the member 'dot' of a type (line 101)
    dot_105132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 23), A_perturbed_105131, 'dot')
    # Calling dot(args, kwargs) (line 101)
    dot_call_result_105135 = invoke(stypy.reporting.localization.Localization(__file__, 101, 23), dot_105132, *[R_prime_105133], **kwargs_105134)
    
    # Assigning a type to the variable 'optim_approx' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'optim_approx', dot_call_result_105135)
    
    # Assigning a Call to a Name (line 103):
    
    # Assigning a Call to a Name (line 103):
    
    # Call to norm(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'naive_approx' (line 103)
    naive_approx_105137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 34), 'naive_approx', False)
    # Getting the type of 'B' (line 103)
    B_105138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 49), 'B', False)
    # Applying the binary operator '-' (line 103)
    result_sub_105139 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 34), '-', naive_approx_105137, B_105138)
    
    # Processing the call keyword arguments (line 103)
    str_105140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 56), 'str', 'fro')
    keyword_105141 = str_105140
    kwargs_105142 = {'ord': keyword_105141}
    # Getting the type of 'norm' (line 103)
    norm_105136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 29), 'norm', False)
    # Calling norm(args, kwargs) (line 103)
    norm_call_result_105143 = invoke(stypy.reporting.localization.Localization(__file__, 103, 29), norm_105136, *[result_sub_105139], **kwargs_105142)
    
    # Assigning a type to the variable 'naive_approx_error' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'naive_approx_error', norm_call_result_105143)
    
    # Assigning a Call to a Name (line 104):
    
    # Assigning a Call to a Name (line 104):
    
    # Call to norm(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'optim_approx' (line 104)
    optim_approx_105145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 34), 'optim_approx', False)
    # Getting the type of 'B' (line 104)
    B_105146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 49), 'B', False)
    # Applying the binary operator '-' (line 104)
    result_sub_105147 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 34), '-', optim_approx_105145, B_105146)
    
    # Processing the call keyword arguments (line 104)
    str_105148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 56), 'str', 'fro')
    keyword_105149 = str_105148
    kwargs_105150 = {'ord': keyword_105149}
    # Getting the type of 'norm' (line 104)
    norm_105144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 29), 'norm', False)
    # Calling norm(args, kwargs) (line 104)
    norm_call_result_105151 = invoke(stypy.reporting.localization.Localization(__file__, 104, 29), norm_105144, *[result_sub_105147], **kwargs_105150)
    
    # Assigning a type to the variable 'optim_approx_error' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'optim_approx_error', norm_call_result_105151)
    
    # Call to assert_array_less(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'optim_approx_error' (line 106)
    optim_approx_error_105153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 26), 'optim_approx_error', False)
    # Getting the type of 'naive_approx_error' (line 106)
    naive_approx_error_105154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 46), 'naive_approx_error', False)
    # Processing the call keyword arguments (line 106)
    kwargs_105155 = {}
    # Getting the type of 'assert_array_less' (line 106)
    assert_array_less_105152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'assert_array_less', False)
    # Calling assert_array_less(args, kwargs) (line 106)
    assert_array_less_call_result_105156 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), assert_array_less_105152, *[optim_approx_error_105153, naive_approx_error_105154], **kwargs_105155)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_orthogonal_procrustes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_orthogonal_procrustes' in the type store
    # Getting the type of 'stypy_return_type' (line 76)
    stypy_return_type_105157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_105157)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_orthogonal_procrustes'
    return stypy_return_type_105157

# Assigning a type to the variable 'test_orthogonal_procrustes' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'test_orthogonal_procrustes', test_orthogonal_procrustes)

@norecursion
def _centered(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_centered'
    module_type_store = module_type_store.open_function_context('_centered', 109, 0, False)
    
    # Passed parameters checking function
    _centered.stypy_localization = localization
    _centered.stypy_type_of_self = None
    _centered.stypy_type_store = module_type_store
    _centered.stypy_function_name = '_centered'
    _centered.stypy_param_names_list = ['A']
    _centered.stypy_varargs_param_name = None
    _centered.stypy_kwargs_param_name = None
    _centered.stypy_call_defaults = defaults
    _centered.stypy_call_varargs = varargs
    _centered.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_centered', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_centered', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_centered(...)' code ##################

    
    # Assigning a Call to a Name (line 110):
    
    # Assigning a Call to a Name (line 110):
    
    # Call to mean(...): (line 110)
    # Processing the call keyword arguments (line 110)
    int_105160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 21), 'int')
    keyword_105161 = int_105160
    kwargs_105162 = {'axis': keyword_105161}
    # Getting the type of 'A' (line 110)
    A_105158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 9), 'A', False)
    # Obtaining the member 'mean' of a type (line 110)
    mean_105159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 9), A_105158, 'mean')
    # Calling mean(args, kwargs) (line 110)
    mean_call_result_105163 = invoke(stypy.reporting.localization.Localization(__file__, 110, 9), mean_105159, *[], **kwargs_105162)
    
    # Assigning a type to the variable 'mu' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'mu', mean_call_result_105163)
    
    # Obtaining an instance of the builtin type 'tuple' (line 111)
    tuple_105164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 111)
    # Adding element type (line 111)
    # Getting the type of 'A' (line 111)
    A_105165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 11), 'A')
    # Getting the type of 'mu' (line 111)
    mu_105166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'mu')
    # Applying the binary operator '-' (line 111)
    result_sub_105167 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 11), '-', A_105165, mu_105166)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 11), tuple_105164, result_sub_105167)
    # Adding element type (line 111)
    # Getting the type of 'mu' (line 111)
    mu_105168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'mu')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 11), tuple_105164, mu_105168)
    
    # Assigning a type to the variable 'stypy_return_type' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type', tuple_105164)
    
    # ################# End of '_centered(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_centered' in the type store
    # Getting the type of 'stypy_return_type' (line 109)
    stypy_return_type_105169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_105169)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_centered'
    return stypy_return_type_105169

# Assigning a type to the variable '_centered' (line 109)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), '_centered', _centered)

@norecursion
def test_orthogonal_procrustes_exact_example(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_orthogonal_procrustes_exact_example'
    module_type_store = module_type_store.open_function_context('test_orthogonal_procrustes_exact_example', 114, 0, False)
    
    # Passed parameters checking function
    test_orthogonal_procrustes_exact_example.stypy_localization = localization
    test_orthogonal_procrustes_exact_example.stypy_type_of_self = None
    test_orthogonal_procrustes_exact_example.stypy_type_store = module_type_store
    test_orthogonal_procrustes_exact_example.stypy_function_name = 'test_orthogonal_procrustes_exact_example'
    test_orthogonal_procrustes_exact_example.stypy_param_names_list = []
    test_orthogonal_procrustes_exact_example.stypy_varargs_param_name = None
    test_orthogonal_procrustes_exact_example.stypy_kwargs_param_name = None
    test_orthogonal_procrustes_exact_example.stypy_call_defaults = defaults
    test_orthogonal_procrustes_exact_example.stypy_call_varargs = varargs
    test_orthogonal_procrustes_exact_example.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_orthogonal_procrustes_exact_example', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_orthogonal_procrustes_exact_example', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_orthogonal_procrustes_exact_example(...)' code ##################

    
    # Assigning a Call to a Name (line 128):
    
    # Assigning a Call to a Name (line 128):
    
    # Call to array(...): (line 128)
    # Processing the call arguments (line 128)
    
    # Obtaining an instance of the builtin type 'list' (line 128)
    list_105172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 128)
    # Adding element type (line 128)
    
    # Obtaining an instance of the builtin type 'list' (line 128)
    list_105173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 128)
    # Adding element type (line 128)
    int_105174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 23), list_105173, int_105174)
    # Adding element type (line 128)
    int_105175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 23), list_105173, int_105175)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 22), list_105172, list_105173)
    # Adding element type (line 128)
    
    # Obtaining an instance of the builtin type 'list' (line 128)
    list_105176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 128)
    # Adding element type (line 128)
    int_105177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 32), list_105176, int_105177)
    # Adding element type (line 128)
    int_105178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 32), list_105176, int_105178)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 22), list_105172, list_105176)
    # Adding element type (line 128)
    
    # Obtaining an instance of the builtin type 'list' (line 128)
    list_105179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 128)
    # Adding element type (line 128)
    int_105180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 41), list_105179, int_105180)
    # Adding element type (line 128)
    int_105181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 41), list_105179, int_105181)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 22), list_105172, list_105179)
    # Adding element type (line 128)
    
    # Obtaining an instance of the builtin type 'list' (line 128)
    list_105182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 50), 'list')
    # Adding type elements to the builtin type 'list' instance (line 128)
    # Adding element type (line 128)
    int_105183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 50), list_105182, int_105183)
    # Adding element type (line 128)
    int_105184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 55), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 50), list_105182, int_105184)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 22), list_105172, list_105182)
    
    # Processing the call keyword arguments (line 128)
    # Getting the type of 'float' (line 128)
    float_105185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 66), 'float', False)
    keyword_105186 = float_105185
    kwargs_105187 = {'dtype': keyword_105186}
    # Getting the type of 'np' (line 128)
    np_105170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 13), 'np', False)
    # Obtaining the member 'array' of a type (line 128)
    array_105171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 13), np_105170, 'array')
    # Calling array(args, kwargs) (line 128)
    array_call_result_105188 = invoke(stypy.reporting.localization.Localization(__file__, 128, 13), array_105171, *[list_105172], **kwargs_105187)
    
    # Assigning a type to the variable 'A_orig' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'A_orig', array_call_result_105188)
    
    # Assigning a Call to a Name (line 129):
    
    # Assigning a Call to a Name (line 129):
    
    # Call to array(...): (line 129)
    # Processing the call arguments (line 129)
    
    # Obtaining an instance of the builtin type 'list' (line 129)
    list_105191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 129)
    # Adding element type (line 129)
    
    # Obtaining an instance of the builtin type 'list' (line 129)
    list_105192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 129)
    # Adding element type (line 129)
    int_105193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 23), list_105192, int_105193)
    # Adding element type (line 129)
    int_105194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 23), list_105192, int_105194)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 22), list_105191, list_105192)
    # Adding element type (line 129)
    
    # Obtaining an instance of the builtin type 'list' (line 129)
    list_105195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 129)
    # Adding element type (line 129)
    int_105196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 31), list_105195, int_105196)
    # Adding element type (line 129)
    int_105197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 31), list_105195, int_105197)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 22), list_105191, list_105195)
    # Adding element type (line 129)
    
    # Obtaining an instance of the builtin type 'list' (line 129)
    list_105198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 129)
    # Adding element type (line 129)
    int_105199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 39), list_105198, int_105199)
    # Adding element type (line 129)
    int_105200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 39), list_105198, int_105200)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 22), list_105191, list_105198)
    # Adding element type (line 129)
    
    # Obtaining an instance of the builtin type 'list' (line 129)
    list_105201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 48), 'list')
    # Adding type elements to the builtin type 'list' instance (line 129)
    # Adding element type (line 129)
    int_105202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 48), list_105201, int_105202)
    # Adding element type (line 129)
    int_105203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 52), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 48), list_105201, int_105203)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 22), list_105191, list_105201)
    
    # Processing the call keyword arguments (line 129)
    # Getting the type of 'float' (line 129)
    float_105204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 63), 'float', False)
    keyword_105205 = float_105204
    kwargs_105206 = {'dtype': keyword_105205}
    # Getting the type of 'np' (line 129)
    np_105189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 13), 'np', False)
    # Obtaining the member 'array' of a type (line 129)
    array_105190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 13), np_105189, 'array')
    # Calling array(args, kwargs) (line 129)
    array_call_result_105207 = invoke(stypy.reporting.localization.Localization(__file__, 129, 13), array_105190, *[list_105191], **kwargs_105206)
    
    # Assigning a type to the variable 'B_orig' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'B_orig', array_call_result_105207)
    
    # Assigning a Call to a Tuple (line 130):
    
    # Assigning a Subscript to a Name (line 130):
    
    # Obtaining the type of the subscript
    int_105208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 4), 'int')
    
    # Call to _centered(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'A_orig' (line 130)
    A_orig_105210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 24), 'A_orig', False)
    # Processing the call keyword arguments (line 130)
    kwargs_105211 = {}
    # Getting the type of '_centered' (line 130)
    _centered_105209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 14), '_centered', False)
    # Calling _centered(args, kwargs) (line 130)
    _centered_call_result_105212 = invoke(stypy.reporting.localization.Localization(__file__, 130, 14), _centered_105209, *[A_orig_105210], **kwargs_105211)
    
    # Obtaining the member '__getitem__' of a type (line 130)
    getitem___105213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 4), _centered_call_result_105212, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
    subscript_call_result_105214 = invoke(stypy.reporting.localization.Localization(__file__, 130, 4), getitem___105213, int_105208)
    
    # Assigning a type to the variable 'tuple_var_assignment_104570' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'tuple_var_assignment_104570', subscript_call_result_105214)
    
    # Assigning a Subscript to a Name (line 130):
    
    # Obtaining the type of the subscript
    int_105215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 4), 'int')
    
    # Call to _centered(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'A_orig' (line 130)
    A_orig_105217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 24), 'A_orig', False)
    # Processing the call keyword arguments (line 130)
    kwargs_105218 = {}
    # Getting the type of '_centered' (line 130)
    _centered_105216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 14), '_centered', False)
    # Calling _centered(args, kwargs) (line 130)
    _centered_call_result_105219 = invoke(stypy.reporting.localization.Localization(__file__, 130, 14), _centered_105216, *[A_orig_105217], **kwargs_105218)
    
    # Obtaining the member '__getitem__' of a type (line 130)
    getitem___105220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 4), _centered_call_result_105219, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
    subscript_call_result_105221 = invoke(stypy.reporting.localization.Localization(__file__, 130, 4), getitem___105220, int_105215)
    
    # Assigning a type to the variable 'tuple_var_assignment_104571' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'tuple_var_assignment_104571', subscript_call_result_105221)
    
    # Assigning a Name to a Name (line 130):
    # Getting the type of 'tuple_var_assignment_104570' (line 130)
    tuple_var_assignment_104570_105222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'tuple_var_assignment_104570')
    # Assigning a type to the variable 'A' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'A', tuple_var_assignment_104570_105222)
    
    # Assigning a Name to a Name (line 130):
    # Getting the type of 'tuple_var_assignment_104571' (line 130)
    tuple_var_assignment_104571_105223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'tuple_var_assignment_104571')
    # Assigning a type to the variable 'A_mu' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 7), 'A_mu', tuple_var_assignment_104571_105223)
    
    # Assigning a Call to a Tuple (line 131):
    
    # Assigning a Subscript to a Name (line 131):
    
    # Obtaining the type of the subscript
    int_105224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 4), 'int')
    
    # Call to _centered(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'B_orig' (line 131)
    B_orig_105226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 24), 'B_orig', False)
    # Processing the call keyword arguments (line 131)
    kwargs_105227 = {}
    # Getting the type of '_centered' (line 131)
    _centered_105225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 14), '_centered', False)
    # Calling _centered(args, kwargs) (line 131)
    _centered_call_result_105228 = invoke(stypy.reporting.localization.Localization(__file__, 131, 14), _centered_105225, *[B_orig_105226], **kwargs_105227)
    
    # Obtaining the member '__getitem__' of a type (line 131)
    getitem___105229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 4), _centered_call_result_105228, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 131)
    subscript_call_result_105230 = invoke(stypy.reporting.localization.Localization(__file__, 131, 4), getitem___105229, int_105224)
    
    # Assigning a type to the variable 'tuple_var_assignment_104572' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'tuple_var_assignment_104572', subscript_call_result_105230)
    
    # Assigning a Subscript to a Name (line 131):
    
    # Obtaining the type of the subscript
    int_105231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 4), 'int')
    
    # Call to _centered(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'B_orig' (line 131)
    B_orig_105233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 24), 'B_orig', False)
    # Processing the call keyword arguments (line 131)
    kwargs_105234 = {}
    # Getting the type of '_centered' (line 131)
    _centered_105232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 14), '_centered', False)
    # Calling _centered(args, kwargs) (line 131)
    _centered_call_result_105235 = invoke(stypy.reporting.localization.Localization(__file__, 131, 14), _centered_105232, *[B_orig_105233], **kwargs_105234)
    
    # Obtaining the member '__getitem__' of a type (line 131)
    getitem___105236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 4), _centered_call_result_105235, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 131)
    subscript_call_result_105237 = invoke(stypy.reporting.localization.Localization(__file__, 131, 4), getitem___105236, int_105231)
    
    # Assigning a type to the variable 'tuple_var_assignment_104573' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'tuple_var_assignment_104573', subscript_call_result_105237)
    
    # Assigning a Name to a Name (line 131):
    # Getting the type of 'tuple_var_assignment_104572' (line 131)
    tuple_var_assignment_104572_105238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'tuple_var_assignment_104572')
    # Assigning a type to the variable 'B' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'B', tuple_var_assignment_104572_105238)
    
    # Assigning a Name to a Name (line 131):
    # Getting the type of 'tuple_var_assignment_104573' (line 131)
    tuple_var_assignment_104573_105239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'tuple_var_assignment_104573')
    # Assigning a type to the variable 'B_mu' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 7), 'B_mu', tuple_var_assignment_104573_105239)
    
    # Assigning a Call to a Tuple (line 132):
    
    # Assigning a Subscript to a Name (line 132):
    
    # Obtaining the type of the subscript
    int_105240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 4), 'int')
    
    # Call to orthogonal_procrustes(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'A' (line 132)
    A_105242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 33), 'A', False)
    # Getting the type of 'B' (line 132)
    B_105243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 36), 'B', False)
    # Processing the call keyword arguments (line 132)
    kwargs_105244 = {}
    # Getting the type of 'orthogonal_procrustes' (line 132)
    orthogonal_procrustes_105241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 11), 'orthogonal_procrustes', False)
    # Calling orthogonal_procrustes(args, kwargs) (line 132)
    orthogonal_procrustes_call_result_105245 = invoke(stypy.reporting.localization.Localization(__file__, 132, 11), orthogonal_procrustes_105241, *[A_105242, B_105243], **kwargs_105244)
    
    # Obtaining the member '__getitem__' of a type (line 132)
    getitem___105246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 4), orthogonal_procrustes_call_result_105245, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 132)
    subscript_call_result_105247 = invoke(stypy.reporting.localization.Localization(__file__, 132, 4), getitem___105246, int_105240)
    
    # Assigning a type to the variable 'tuple_var_assignment_104574' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'tuple_var_assignment_104574', subscript_call_result_105247)
    
    # Assigning a Subscript to a Name (line 132):
    
    # Obtaining the type of the subscript
    int_105248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 4), 'int')
    
    # Call to orthogonal_procrustes(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'A' (line 132)
    A_105250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 33), 'A', False)
    # Getting the type of 'B' (line 132)
    B_105251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 36), 'B', False)
    # Processing the call keyword arguments (line 132)
    kwargs_105252 = {}
    # Getting the type of 'orthogonal_procrustes' (line 132)
    orthogonal_procrustes_105249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 11), 'orthogonal_procrustes', False)
    # Calling orthogonal_procrustes(args, kwargs) (line 132)
    orthogonal_procrustes_call_result_105253 = invoke(stypy.reporting.localization.Localization(__file__, 132, 11), orthogonal_procrustes_105249, *[A_105250, B_105251], **kwargs_105252)
    
    # Obtaining the member '__getitem__' of a type (line 132)
    getitem___105254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 4), orthogonal_procrustes_call_result_105253, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 132)
    subscript_call_result_105255 = invoke(stypy.reporting.localization.Localization(__file__, 132, 4), getitem___105254, int_105248)
    
    # Assigning a type to the variable 'tuple_var_assignment_104575' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'tuple_var_assignment_104575', subscript_call_result_105255)
    
    # Assigning a Name to a Name (line 132):
    # Getting the type of 'tuple_var_assignment_104574' (line 132)
    tuple_var_assignment_104574_105256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'tuple_var_assignment_104574')
    # Assigning a type to the variable 'R' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'R', tuple_var_assignment_104574_105256)
    
    # Assigning a Name to a Name (line 132):
    # Getting the type of 'tuple_var_assignment_104575' (line 132)
    tuple_var_assignment_104575_105257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'tuple_var_assignment_104575')
    # Assigning a type to the variable 's' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 7), 's', tuple_var_assignment_104575_105257)
    
    # Assigning a BinOp to a Name (line 133):
    
    # Assigning a BinOp to a Name (line 133):
    # Getting the type of 's' (line 133)
    s_105258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 's')
    
    # Call to square(...): (line 133)
    # Processing the call arguments (line 133)
    
    # Call to norm(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'A' (line 133)
    A_105262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 31), 'A', False)
    # Processing the call keyword arguments (line 133)
    kwargs_105263 = {}
    # Getting the type of 'norm' (line 133)
    norm_105261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 26), 'norm', False)
    # Calling norm(args, kwargs) (line 133)
    norm_call_result_105264 = invoke(stypy.reporting.localization.Localization(__file__, 133, 26), norm_105261, *[A_105262], **kwargs_105263)
    
    # Processing the call keyword arguments (line 133)
    kwargs_105265 = {}
    # Getting the type of 'np' (line 133)
    np_105259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'np', False)
    # Obtaining the member 'square' of a type (line 133)
    square_105260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 16), np_105259, 'square')
    # Calling square(args, kwargs) (line 133)
    square_call_result_105266 = invoke(stypy.reporting.localization.Localization(__file__, 133, 16), square_105260, *[norm_call_result_105264], **kwargs_105265)
    
    # Applying the binary operator 'div' (line 133)
    result_div_105267 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 12), 'div', s_105258, square_call_result_105266)
    
    # Assigning a type to the variable 'scale' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'scale', result_div_105267)
    
    # Assigning a BinOp to a Name (line 134):
    
    # Assigning a BinOp to a Name (line 134):
    # Getting the type of 'scale' (line 134)
    scale_105268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), 'scale')
    
    # Call to dot(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'A' (line 134)
    A_105271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 30), 'A', False)
    # Getting the type of 'R' (line 134)
    R_105272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 33), 'R', False)
    # Processing the call keyword arguments (line 134)
    kwargs_105273 = {}
    # Getting the type of 'np' (line 134)
    np_105269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 23), 'np', False)
    # Obtaining the member 'dot' of a type (line 134)
    dot_105270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 23), np_105269, 'dot')
    # Calling dot(args, kwargs) (line 134)
    dot_call_result_105274 = invoke(stypy.reporting.localization.Localization(__file__, 134, 23), dot_105270, *[A_105271, R_105272], **kwargs_105273)
    
    # Applying the binary operator '*' (line 134)
    result_mul_105275 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 15), '*', scale_105268, dot_call_result_105274)
    
    # Getting the type of 'B_mu' (line 134)
    B_mu_105276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 38), 'B_mu')
    # Applying the binary operator '+' (line 134)
    result_add_105277 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 15), '+', result_mul_105275, B_mu_105276)
    
    # Assigning a type to the variable 'B_approx' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'B_approx', result_add_105277)
    
    # Call to assert_allclose(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'B_approx' (line 135)
    B_approx_105279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 20), 'B_approx', False)
    # Getting the type of 'B_orig' (line 135)
    B_orig_105280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 30), 'B_orig', False)
    # Processing the call keyword arguments (line 135)
    float_105281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 43), 'float')
    keyword_105282 = float_105281
    kwargs_105283 = {'atol': keyword_105282}
    # Getting the type of 'assert_allclose' (line 135)
    assert_allclose_105278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 135)
    assert_allclose_call_result_105284 = invoke(stypy.reporting.localization.Localization(__file__, 135, 4), assert_allclose_105278, *[B_approx_105279, B_orig_105280], **kwargs_105283)
    
    
    # ################# End of 'test_orthogonal_procrustes_exact_example(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_orthogonal_procrustes_exact_example' in the type store
    # Getting the type of 'stypy_return_type' (line 114)
    stypy_return_type_105285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_105285)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_orthogonal_procrustes_exact_example'
    return stypy_return_type_105285

# Assigning a type to the variable 'test_orthogonal_procrustes_exact_example' (line 114)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), 'test_orthogonal_procrustes_exact_example', test_orthogonal_procrustes_exact_example)

@norecursion
def test_orthogonal_procrustes_stretched_example(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_orthogonal_procrustes_stretched_example'
    module_type_store = module_type_store.open_function_context('test_orthogonal_procrustes_stretched_example', 138, 0, False)
    
    # Passed parameters checking function
    test_orthogonal_procrustes_stretched_example.stypy_localization = localization
    test_orthogonal_procrustes_stretched_example.stypy_type_of_self = None
    test_orthogonal_procrustes_stretched_example.stypy_type_store = module_type_store
    test_orthogonal_procrustes_stretched_example.stypy_function_name = 'test_orthogonal_procrustes_stretched_example'
    test_orthogonal_procrustes_stretched_example.stypy_param_names_list = []
    test_orthogonal_procrustes_stretched_example.stypy_varargs_param_name = None
    test_orthogonal_procrustes_stretched_example.stypy_kwargs_param_name = None
    test_orthogonal_procrustes_stretched_example.stypy_call_defaults = defaults
    test_orthogonal_procrustes_stretched_example.stypy_call_varargs = varargs
    test_orthogonal_procrustes_stretched_example.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_orthogonal_procrustes_stretched_example', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_orthogonal_procrustes_stretched_example', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_orthogonal_procrustes_stretched_example(...)' code ##################

    
    # Assigning a Call to a Name (line 140):
    
    # Assigning a Call to a Name (line 140):
    
    # Call to array(...): (line 140)
    # Processing the call arguments (line 140)
    
    # Obtaining an instance of the builtin type 'list' (line 140)
    list_105288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 140)
    # Adding element type (line 140)
    
    # Obtaining an instance of the builtin type 'list' (line 140)
    list_105289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 140)
    # Adding element type (line 140)
    int_105290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 23), list_105289, int_105290)
    # Adding element type (line 140)
    int_105291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 23), list_105289, int_105291)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 22), list_105288, list_105289)
    # Adding element type (line 140)
    
    # Obtaining an instance of the builtin type 'list' (line 140)
    list_105292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 140)
    # Adding element type (line 140)
    int_105293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 32), list_105292, int_105293)
    # Adding element type (line 140)
    int_105294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 32), list_105292, int_105294)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 22), list_105288, list_105292)
    # Adding element type (line 140)
    
    # Obtaining an instance of the builtin type 'list' (line 140)
    list_105295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 140)
    # Adding element type (line 140)
    int_105296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 41), list_105295, int_105296)
    # Adding element type (line 140)
    int_105297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 41), list_105295, int_105297)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 22), list_105288, list_105295)
    # Adding element type (line 140)
    
    # Obtaining an instance of the builtin type 'list' (line 140)
    list_105298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 50), 'list')
    # Adding type elements to the builtin type 'list' instance (line 140)
    # Adding element type (line 140)
    int_105299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 50), list_105298, int_105299)
    # Adding element type (line 140)
    int_105300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 55), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 50), list_105298, int_105300)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 22), list_105288, list_105298)
    
    # Processing the call keyword arguments (line 140)
    # Getting the type of 'float' (line 140)
    float_105301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 66), 'float', False)
    keyword_105302 = float_105301
    kwargs_105303 = {'dtype': keyword_105302}
    # Getting the type of 'np' (line 140)
    np_105286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 13), 'np', False)
    # Obtaining the member 'array' of a type (line 140)
    array_105287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 13), np_105286, 'array')
    # Calling array(args, kwargs) (line 140)
    array_call_result_105304 = invoke(stypy.reporting.localization.Localization(__file__, 140, 13), array_105287, *[list_105288], **kwargs_105303)
    
    # Assigning a type to the variable 'A_orig' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'A_orig', array_call_result_105304)
    
    # Assigning a Call to a Name (line 141):
    
    # Assigning a Call to a Name (line 141):
    
    # Call to array(...): (line 141)
    # Processing the call arguments (line 141)
    
    # Obtaining an instance of the builtin type 'list' (line 141)
    list_105307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 141)
    # Adding element type (line 141)
    
    # Obtaining an instance of the builtin type 'list' (line 141)
    list_105308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 141)
    # Adding element type (line 141)
    int_105309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 23), list_105308, int_105309)
    # Adding element type (line 141)
    int_105310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 23), list_105308, int_105310)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 22), list_105307, list_105308)
    # Adding element type (line 141)
    
    # Obtaining an instance of the builtin type 'list' (line 141)
    list_105311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 141)
    # Adding element type (line 141)
    int_105312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 32), list_105311, int_105312)
    # Adding element type (line 141)
    int_105313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 32), list_105311, int_105313)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 22), list_105307, list_105311)
    # Adding element type (line 141)
    
    # Obtaining an instance of the builtin type 'list' (line 141)
    list_105314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 141)
    # Adding element type (line 141)
    int_105315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 40), list_105314, int_105315)
    # Adding element type (line 141)
    int_105316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 40), list_105314, int_105316)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 22), list_105307, list_105314)
    # Adding element type (line 141)
    
    # Obtaining an instance of the builtin type 'list' (line 141)
    list_105317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 50), 'list')
    # Adding type elements to the builtin type 'list' instance (line 141)
    # Adding element type (line 141)
    int_105318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 50), list_105317, int_105318)
    # Adding element type (line 141)
    int_105319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 50), list_105317, int_105319)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 22), list_105307, list_105317)
    
    # Processing the call keyword arguments (line 141)
    # Getting the type of 'float' (line 141)
    float_105320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 65), 'float', False)
    keyword_105321 = float_105320
    kwargs_105322 = {'dtype': keyword_105321}
    # Getting the type of 'np' (line 141)
    np_105305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 13), 'np', False)
    # Obtaining the member 'array' of a type (line 141)
    array_105306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 13), np_105305, 'array')
    # Calling array(args, kwargs) (line 141)
    array_call_result_105323 = invoke(stypy.reporting.localization.Localization(__file__, 141, 13), array_105306, *[list_105307], **kwargs_105322)
    
    # Assigning a type to the variable 'B_orig' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'B_orig', array_call_result_105323)
    
    # Assigning a Call to a Tuple (line 142):
    
    # Assigning a Subscript to a Name (line 142):
    
    # Obtaining the type of the subscript
    int_105324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 4), 'int')
    
    # Call to _centered(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'A_orig' (line 142)
    A_orig_105326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 24), 'A_orig', False)
    # Processing the call keyword arguments (line 142)
    kwargs_105327 = {}
    # Getting the type of '_centered' (line 142)
    _centered_105325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 14), '_centered', False)
    # Calling _centered(args, kwargs) (line 142)
    _centered_call_result_105328 = invoke(stypy.reporting.localization.Localization(__file__, 142, 14), _centered_105325, *[A_orig_105326], **kwargs_105327)
    
    # Obtaining the member '__getitem__' of a type (line 142)
    getitem___105329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 4), _centered_call_result_105328, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 142)
    subscript_call_result_105330 = invoke(stypy.reporting.localization.Localization(__file__, 142, 4), getitem___105329, int_105324)
    
    # Assigning a type to the variable 'tuple_var_assignment_104576' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'tuple_var_assignment_104576', subscript_call_result_105330)
    
    # Assigning a Subscript to a Name (line 142):
    
    # Obtaining the type of the subscript
    int_105331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 4), 'int')
    
    # Call to _centered(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'A_orig' (line 142)
    A_orig_105333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 24), 'A_orig', False)
    # Processing the call keyword arguments (line 142)
    kwargs_105334 = {}
    # Getting the type of '_centered' (line 142)
    _centered_105332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 14), '_centered', False)
    # Calling _centered(args, kwargs) (line 142)
    _centered_call_result_105335 = invoke(stypy.reporting.localization.Localization(__file__, 142, 14), _centered_105332, *[A_orig_105333], **kwargs_105334)
    
    # Obtaining the member '__getitem__' of a type (line 142)
    getitem___105336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 4), _centered_call_result_105335, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 142)
    subscript_call_result_105337 = invoke(stypy.reporting.localization.Localization(__file__, 142, 4), getitem___105336, int_105331)
    
    # Assigning a type to the variable 'tuple_var_assignment_104577' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'tuple_var_assignment_104577', subscript_call_result_105337)
    
    # Assigning a Name to a Name (line 142):
    # Getting the type of 'tuple_var_assignment_104576' (line 142)
    tuple_var_assignment_104576_105338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'tuple_var_assignment_104576')
    # Assigning a type to the variable 'A' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'A', tuple_var_assignment_104576_105338)
    
    # Assigning a Name to a Name (line 142):
    # Getting the type of 'tuple_var_assignment_104577' (line 142)
    tuple_var_assignment_104577_105339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'tuple_var_assignment_104577')
    # Assigning a type to the variable 'A_mu' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 7), 'A_mu', tuple_var_assignment_104577_105339)
    
    # Assigning a Call to a Tuple (line 143):
    
    # Assigning a Subscript to a Name (line 143):
    
    # Obtaining the type of the subscript
    int_105340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 4), 'int')
    
    # Call to _centered(...): (line 143)
    # Processing the call arguments (line 143)
    # Getting the type of 'B_orig' (line 143)
    B_orig_105342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 24), 'B_orig', False)
    # Processing the call keyword arguments (line 143)
    kwargs_105343 = {}
    # Getting the type of '_centered' (line 143)
    _centered_105341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 14), '_centered', False)
    # Calling _centered(args, kwargs) (line 143)
    _centered_call_result_105344 = invoke(stypy.reporting.localization.Localization(__file__, 143, 14), _centered_105341, *[B_orig_105342], **kwargs_105343)
    
    # Obtaining the member '__getitem__' of a type (line 143)
    getitem___105345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 4), _centered_call_result_105344, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 143)
    subscript_call_result_105346 = invoke(stypy.reporting.localization.Localization(__file__, 143, 4), getitem___105345, int_105340)
    
    # Assigning a type to the variable 'tuple_var_assignment_104578' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'tuple_var_assignment_104578', subscript_call_result_105346)
    
    # Assigning a Subscript to a Name (line 143):
    
    # Obtaining the type of the subscript
    int_105347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 4), 'int')
    
    # Call to _centered(...): (line 143)
    # Processing the call arguments (line 143)
    # Getting the type of 'B_orig' (line 143)
    B_orig_105349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 24), 'B_orig', False)
    # Processing the call keyword arguments (line 143)
    kwargs_105350 = {}
    # Getting the type of '_centered' (line 143)
    _centered_105348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 14), '_centered', False)
    # Calling _centered(args, kwargs) (line 143)
    _centered_call_result_105351 = invoke(stypy.reporting.localization.Localization(__file__, 143, 14), _centered_105348, *[B_orig_105349], **kwargs_105350)
    
    # Obtaining the member '__getitem__' of a type (line 143)
    getitem___105352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 4), _centered_call_result_105351, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 143)
    subscript_call_result_105353 = invoke(stypy.reporting.localization.Localization(__file__, 143, 4), getitem___105352, int_105347)
    
    # Assigning a type to the variable 'tuple_var_assignment_104579' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'tuple_var_assignment_104579', subscript_call_result_105353)
    
    # Assigning a Name to a Name (line 143):
    # Getting the type of 'tuple_var_assignment_104578' (line 143)
    tuple_var_assignment_104578_105354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'tuple_var_assignment_104578')
    # Assigning a type to the variable 'B' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'B', tuple_var_assignment_104578_105354)
    
    # Assigning a Name to a Name (line 143):
    # Getting the type of 'tuple_var_assignment_104579' (line 143)
    tuple_var_assignment_104579_105355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'tuple_var_assignment_104579')
    # Assigning a type to the variable 'B_mu' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 7), 'B_mu', tuple_var_assignment_104579_105355)
    
    # Assigning a Call to a Tuple (line 144):
    
    # Assigning a Subscript to a Name (line 144):
    
    # Obtaining the type of the subscript
    int_105356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 4), 'int')
    
    # Call to orthogonal_procrustes(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 'A' (line 144)
    A_105358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 33), 'A', False)
    # Getting the type of 'B' (line 144)
    B_105359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 36), 'B', False)
    # Processing the call keyword arguments (line 144)
    kwargs_105360 = {}
    # Getting the type of 'orthogonal_procrustes' (line 144)
    orthogonal_procrustes_105357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 11), 'orthogonal_procrustes', False)
    # Calling orthogonal_procrustes(args, kwargs) (line 144)
    orthogonal_procrustes_call_result_105361 = invoke(stypy.reporting.localization.Localization(__file__, 144, 11), orthogonal_procrustes_105357, *[A_105358, B_105359], **kwargs_105360)
    
    # Obtaining the member '__getitem__' of a type (line 144)
    getitem___105362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 4), orthogonal_procrustes_call_result_105361, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 144)
    subscript_call_result_105363 = invoke(stypy.reporting.localization.Localization(__file__, 144, 4), getitem___105362, int_105356)
    
    # Assigning a type to the variable 'tuple_var_assignment_104580' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'tuple_var_assignment_104580', subscript_call_result_105363)
    
    # Assigning a Subscript to a Name (line 144):
    
    # Obtaining the type of the subscript
    int_105364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 4), 'int')
    
    # Call to orthogonal_procrustes(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 'A' (line 144)
    A_105366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 33), 'A', False)
    # Getting the type of 'B' (line 144)
    B_105367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 36), 'B', False)
    # Processing the call keyword arguments (line 144)
    kwargs_105368 = {}
    # Getting the type of 'orthogonal_procrustes' (line 144)
    orthogonal_procrustes_105365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 11), 'orthogonal_procrustes', False)
    # Calling orthogonal_procrustes(args, kwargs) (line 144)
    orthogonal_procrustes_call_result_105369 = invoke(stypy.reporting.localization.Localization(__file__, 144, 11), orthogonal_procrustes_105365, *[A_105366, B_105367], **kwargs_105368)
    
    # Obtaining the member '__getitem__' of a type (line 144)
    getitem___105370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 4), orthogonal_procrustes_call_result_105369, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 144)
    subscript_call_result_105371 = invoke(stypy.reporting.localization.Localization(__file__, 144, 4), getitem___105370, int_105364)
    
    # Assigning a type to the variable 'tuple_var_assignment_104581' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'tuple_var_assignment_104581', subscript_call_result_105371)
    
    # Assigning a Name to a Name (line 144):
    # Getting the type of 'tuple_var_assignment_104580' (line 144)
    tuple_var_assignment_104580_105372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'tuple_var_assignment_104580')
    # Assigning a type to the variable 'R' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'R', tuple_var_assignment_104580_105372)
    
    # Assigning a Name to a Name (line 144):
    # Getting the type of 'tuple_var_assignment_104581' (line 144)
    tuple_var_assignment_104581_105373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'tuple_var_assignment_104581')
    # Assigning a type to the variable 's' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 7), 's', tuple_var_assignment_104581_105373)
    
    # Assigning a BinOp to a Name (line 145):
    
    # Assigning a BinOp to a Name (line 145):
    # Getting the type of 's' (line 145)
    s_105374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 's')
    
    # Call to square(...): (line 145)
    # Processing the call arguments (line 145)
    
    # Call to norm(...): (line 145)
    # Processing the call arguments (line 145)
    # Getting the type of 'A' (line 145)
    A_105378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 31), 'A', False)
    # Processing the call keyword arguments (line 145)
    kwargs_105379 = {}
    # Getting the type of 'norm' (line 145)
    norm_105377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 26), 'norm', False)
    # Calling norm(args, kwargs) (line 145)
    norm_call_result_105380 = invoke(stypy.reporting.localization.Localization(__file__, 145, 26), norm_105377, *[A_105378], **kwargs_105379)
    
    # Processing the call keyword arguments (line 145)
    kwargs_105381 = {}
    # Getting the type of 'np' (line 145)
    np_105375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'np', False)
    # Obtaining the member 'square' of a type (line 145)
    square_105376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 16), np_105375, 'square')
    # Calling square(args, kwargs) (line 145)
    square_call_result_105382 = invoke(stypy.reporting.localization.Localization(__file__, 145, 16), square_105376, *[norm_call_result_105380], **kwargs_105381)
    
    # Applying the binary operator 'div' (line 145)
    result_div_105383 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 12), 'div', s_105374, square_call_result_105382)
    
    # Assigning a type to the variable 'scale' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'scale', result_div_105383)
    
    # Assigning a BinOp to a Name (line 146):
    
    # Assigning a BinOp to a Name (line 146):
    # Getting the type of 'scale' (line 146)
    scale_105384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 15), 'scale')
    
    # Call to dot(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'A' (line 146)
    A_105387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 30), 'A', False)
    # Getting the type of 'R' (line 146)
    R_105388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 33), 'R', False)
    # Processing the call keyword arguments (line 146)
    kwargs_105389 = {}
    # Getting the type of 'np' (line 146)
    np_105385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 23), 'np', False)
    # Obtaining the member 'dot' of a type (line 146)
    dot_105386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 23), np_105385, 'dot')
    # Calling dot(args, kwargs) (line 146)
    dot_call_result_105390 = invoke(stypy.reporting.localization.Localization(__file__, 146, 23), dot_105386, *[A_105387, R_105388], **kwargs_105389)
    
    # Applying the binary operator '*' (line 146)
    result_mul_105391 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 15), '*', scale_105384, dot_call_result_105390)
    
    # Getting the type of 'B_mu' (line 146)
    B_mu_105392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 38), 'B_mu')
    # Applying the binary operator '+' (line 146)
    result_add_105393 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 15), '+', result_mul_105391, B_mu_105392)
    
    # Assigning a type to the variable 'B_approx' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'B_approx', result_add_105393)
    
    # Assigning a Call to a Name (line 147):
    
    # Assigning a Call to a Name (line 147):
    
    # Call to array(...): (line 147)
    # Processing the call arguments (line 147)
    
    # Obtaining an instance of the builtin type 'list' (line 147)
    list_105396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 147)
    # Adding element type (line 147)
    
    # Obtaining an instance of the builtin type 'list' (line 147)
    list_105397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 147)
    # Adding element type (line 147)
    int_105398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 25), list_105397, int_105398)
    # Adding element type (line 147)
    int_105399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 25), list_105397, int_105399)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 24), list_105396, list_105397)
    # Adding element type (line 147)
    
    # Obtaining an instance of the builtin type 'list' (line 147)
    list_105400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 147)
    # Adding element type (line 147)
    int_105401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 34), list_105400, int_105401)
    # Adding element type (line 147)
    int_105402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 34), list_105400, int_105402)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 24), list_105396, list_105400)
    # Adding element type (line 147)
    
    # Obtaining an instance of the builtin type 'list' (line 147)
    list_105403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 44), 'list')
    # Adding type elements to the builtin type 'list' instance (line 147)
    # Adding element type (line 147)
    int_105404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 44), list_105403, int_105404)
    # Adding element type (line 147)
    int_105405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 44), list_105403, int_105405)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 24), list_105396, list_105403)
    # Adding element type (line 147)
    
    # Obtaining an instance of the builtin type 'list' (line 147)
    list_105406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 54), 'list')
    # Adding type elements to the builtin type 'list' instance (line 147)
    # Adding element type (line 147)
    int_105407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 55), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 54), list_105406, int_105407)
    # Adding element type (line 147)
    int_105408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 59), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 54), list_105406, int_105408)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 24), list_105396, list_105406)
    
    # Processing the call keyword arguments (line 147)
    # Getting the type of 'float' (line 147)
    float_105409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 70), 'float', False)
    keyword_105410 = float_105409
    kwargs_105411 = {'dtype': keyword_105410}
    # Getting the type of 'np' (line 147)
    np_105394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 147)
    array_105395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 15), np_105394, 'array')
    # Calling array(args, kwargs) (line 147)
    array_call_result_105412 = invoke(stypy.reporting.localization.Localization(__file__, 147, 15), array_105395, *[list_105396], **kwargs_105411)
    
    # Assigning a type to the variable 'expected' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'expected', array_call_result_105412)
    
    # Call to assert_allclose(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'B_approx' (line 148)
    B_approx_105414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'B_approx', False)
    # Getting the type of 'expected' (line 148)
    expected_105415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 30), 'expected', False)
    # Processing the call keyword arguments (line 148)
    float_105416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 45), 'float')
    keyword_105417 = float_105416
    kwargs_105418 = {'atol': keyword_105417}
    # Getting the type of 'assert_allclose' (line 148)
    assert_allclose_105413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 148)
    assert_allclose_call_result_105419 = invoke(stypy.reporting.localization.Localization(__file__, 148, 4), assert_allclose_105413, *[B_approx_105414, expected_105415], **kwargs_105418)
    
    
    # Assigning a Num to a Name (line 150):
    
    # Assigning a Num to a Name (line 150):
    float_105420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 25), 'float')
    # Assigning a type to the variable 'expected_disparity' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'expected_disparity', float_105420)
    
    # Assigning a Call to a Name (line 151):
    
    # Assigning a Call to a Name (line 151):
    
    # Call to square(...): (line 151)
    # Processing the call arguments (line 151)
    
    # Call to norm(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'B_approx' (line 151)
    B_approx_105424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 34), 'B_approx', False)
    # Getting the type of 'B_orig' (line 151)
    B_orig_105425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 45), 'B_orig', False)
    # Applying the binary operator '-' (line 151)
    result_sub_105426 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 34), '-', B_approx_105424, B_orig_105425)
    
    # Processing the call keyword arguments (line 151)
    kwargs_105427 = {}
    # Getting the type of 'norm' (line 151)
    norm_105423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 29), 'norm', False)
    # Calling norm(args, kwargs) (line 151)
    norm_call_result_105428 = invoke(stypy.reporting.localization.Localization(__file__, 151, 29), norm_105423, *[result_sub_105426], **kwargs_105427)
    
    
    # Call to norm(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'B' (line 151)
    B_105430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 60), 'B', False)
    # Processing the call keyword arguments (line 151)
    kwargs_105431 = {}
    # Getting the type of 'norm' (line 151)
    norm_105429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 55), 'norm', False)
    # Calling norm(args, kwargs) (line 151)
    norm_call_result_105432 = invoke(stypy.reporting.localization.Localization(__file__, 151, 55), norm_105429, *[B_105430], **kwargs_105431)
    
    # Applying the binary operator 'div' (line 151)
    result_div_105433 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 29), 'div', norm_call_result_105428, norm_call_result_105432)
    
    # Processing the call keyword arguments (line 151)
    kwargs_105434 = {}
    # Getting the type of 'np' (line 151)
    np_105421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 19), 'np', False)
    # Obtaining the member 'square' of a type (line 151)
    square_105422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 19), np_105421, 'square')
    # Calling square(args, kwargs) (line 151)
    square_call_result_105435 = invoke(stypy.reporting.localization.Localization(__file__, 151, 19), square_105422, *[result_div_105433], **kwargs_105434)
    
    # Assigning a type to the variable 'AB_disparity' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'AB_disparity', square_call_result_105435)
    
    # Call to assert_allclose(...): (line 152)
    # Processing the call arguments (line 152)
    # Getting the type of 'AB_disparity' (line 152)
    AB_disparity_105437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 20), 'AB_disparity', False)
    # Getting the type of 'expected_disparity' (line 152)
    expected_disparity_105438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 34), 'expected_disparity', False)
    # Processing the call keyword arguments (line 152)
    kwargs_105439 = {}
    # Getting the type of 'assert_allclose' (line 152)
    assert_allclose_105436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 152)
    assert_allclose_call_result_105440 = invoke(stypy.reporting.localization.Localization(__file__, 152, 4), assert_allclose_105436, *[AB_disparity_105437, expected_disparity_105438], **kwargs_105439)
    
    
    # Assigning a Call to a Tuple (line 153):
    
    # Assigning a Subscript to a Name (line 153):
    
    # Obtaining the type of the subscript
    int_105441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 4), 'int')
    
    # Call to orthogonal_procrustes(...): (line 153)
    # Processing the call arguments (line 153)
    # Getting the type of 'B' (line 153)
    B_105443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 33), 'B', False)
    # Getting the type of 'A' (line 153)
    A_105444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 36), 'A', False)
    # Processing the call keyword arguments (line 153)
    kwargs_105445 = {}
    # Getting the type of 'orthogonal_procrustes' (line 153)
    orthogonal_procrustes_105442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 11), 'orthogonal_procrustes', False)
    # Calling orthogonal_procrustes(args, kwargs) (line 153)
    orthogonal_procrustes_call_result_105446 = invoke(stypy.reporting.localization.Localization(__file__, 153, 11), orthogonal_procrustes_105442, *[B_105443, A_105444], **kwargs_105445)
    
    # Obtaining the member '__getitem__' of a type (line 153)
    getitem___105447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 4), orthogonal_procrustes_call_result_105446, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 153)
    subscript_call_result_105448 = invoke(stypy.reporting.localization.Localization(__file__, 153, 4), getitem___105447, int_105441)
    
    # Assigning a type to the variable 'tuple_var_assignment_104582' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'tuple_var_assignment_104582', subscript_call_result_105448)
    
    # Assigning a Subscript to a Name (line 153):
    
    # Obtaining the type of the subscript
    int_105449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 4), 'int')
    
    # Call to orthogonal_procrustes(...): (line 153)
    # Processing the call arguments (line 153)
    # Getting the type of 'B' (line 153)
    B_105451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 33), 'B', False)
    # Getting the type of 'A' (line 153)
    A_105452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 36), 'A', False)
    # Processing the call keyword arguments (line 153)
    kwargs_105453 = {}
    # Getting the type of 'orthogonal_procrustes' (line 153)
    orthogonal_procrustes_105450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 11), 'orthogonal_procrustes', False)
    # Calling orthogonal_procrustes(args, kwargs) (line 153)
    orthogonal_procrustes_call_result_105454 = invoke(stypy.reporting.localization.Localization(__file__, 153, 11), orthogonal_procrustes_105450, *[B_105451, A_105452], **kwargs_105453)
    
    # Obtaining the member '__getitem__' of a type (line 153)
    getitem___105455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 4), orthogonal_procrustes_call_result_105454, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 153)
    subscript_call_result_105456 = invoke(stypy.reporting.localization.Localization(__file__, 153, 4), getitem___105455, int_105449)
    
    # Assigning a type to the variable 'tuple_var_assignment_104583' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'tuple_var_assignment_104583', subscript_call_result_105456)
    
    # Assigning a Name to a Name (line 153):
    # Getting the type of 'tuple_var_assignment_104582' (line 153)
    tuple_var_assignment_104582_105457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'tuple_var_assignment_104582')
    # Assigning a type to the variable 'R' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'R', tuple_var_assignment_104582_105457)
    
    # Assigning a Name to a Name (line 153):
    # Getting the type of 'tuple_var_assignment_104583' (line 153)
    tuple_var_assignment_104583_105458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'tuple_var_assignment_104583')
    # Assigning a type to the variable 's' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 7), 's', tuple_var_assignment_104583_105458)
    
    # Assigning a BinOp to a Name (line 154):
    
    # Assigning a BinOp to a Name (line 154):
    # Getting the type of 's' (line 154)
    s_105459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 's')
    
    # Call to square(...): (line 154)
    # Processing the call arguments (line 154)
    
    # Call to norm(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'B' (line 154)
    B_105463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 31), 'B', False)
    # Processing the call keyword arguments (line 154)
    kwargs_105464 = {}
    # Getting the type of 'norm' (line 154)
    norm_105462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 26), 'norm', False)
    # Calling norm(args, kwargs) (line 154)
    norm_call_result_105465 = invoke(stypy.reporting.localization.Localization(__file__, 154, 26), norm_105462, *[B_105463], **kwargs_105464)
    
    # Processing the call keyword arguments (line 154)
    kwargs_105466 = {}
    # Getting the type of 'np' (line 154)
    np_105460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'np', False)
    # Obtaining the member 'square' of a type (line 154)
    square_105461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 16), np_105460, 'square')
    # Calling square(args, kwargs) (line 154)
    square_call_result_105467 = invoke(stypy.reporting.localization.Localization(__file__, 154, 16), square_105461, *[norm_call_result_105465], **kwargs_105466)
    
    # Applying the binary operator 'div' (line 154)
    result_div_105468 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 12), 'div', s_105459, square_call_result_105467)
    
    # Assigning a type to the variable 'scale' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'scale', result_div_105468)
    
    # Assigning a BinOp to a Name (line 155):
    
    # Assigning a BinOp to a Name (line 155):
    # Getting the type of 'scale' (line 155)
    scale_105469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 15), 'scale')
    
    # Call to dot(...): (line 155)
    # Processing the call arguments (line 155)
    # Getting the type of 'B' (line 155)
    B_105472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 30), 'B', False)
    # Getting the type of 'R' (line 155)
    R_105473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 33), 'R', False)
    # Processing the call keyword arguments (line 155)
    kwargs_105474 = {}
    # Getting the type of 'np' (line 155)
    np_105470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 23), 'np', False)
    # Obtaining the member 'dot' of a type (line 155)
    dot_105471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 23), np_105470, 'dot')
    # Calling dot(args, kwargs) (line 155)
    dot_call_result_105475 = invoke(stypy.reporting.localization.Localization(__file__, 155, 23), dot_105471, *[B_105472, R_105473], **kwargs_105474)
    
    # Applying the binary operator '*' (line 155)
    result_mul_105476 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 15), '*', scale_105469, dot_call_result_105475)
    
    # Getting the type of 'A_mu' (line 155)
    A_mu_105477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 38), 'A_mu')
    # Applying the binary operator '+' (line 155)
    result_add_105478 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 15), '+', result_mul_105476, A_mu_105477)
    
    # Assigning a type to the variable 'A_approx' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'A_approx', result_add_105478)
    
    # Assigning a Call to a Name (line 156):
    
    # Assigning a Call to a Name (line 156):
    
    # Call to square(...): (line 156)
    # Processing the call arguments (line 156)
    
    # Call to norm(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'A_approx' (line 156)
    A_approx_105482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 34), 'A_approx', False)
    # Getting the type of 'A_orig' (line 156)
    A_orig_105483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 45), 'A_orig', False)
    # Applying the binary operator '-' (line 156)
    result_sub_105484 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 34), '-', A_approx_105482, A_orig_105483)
    
    # Processing the call keyword arguments (line 156)
    kwargs_105485 = {}
    # Getting the type of 'norm' (line 156)
    norm_105481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 29), 'norm', False)
    # Calling norm(args, kwargs) (line 156)
    norm_call_result_105486 = invoke(stypy.reporting.localization.Localization(__file__, 156, 29), norm_105481, *[result_sub_105484], **kwargs_105485)
    
    
    # Call to norm(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'A' (line 156)
    A_105488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 60), 'A', False)
    # Processing the call keyword arguments (line 156)
    kwargs_105489 = {}
    # Getting the type of 'norm' (line 156)
    norm_105487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 55), 'norm', False)
    # Calling norm(args, kwargs) (line 156)
    norm_call_result_105490 = invoke(stypy.reporting.localization.Localization(__file__, 156, 55), norm_105487, *[A_105488], **kwargs_105489)
    
    # Applying the binary operator 'div' (line 156)
    result_div_105491 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 29), 'div', norm_call_result_105486, norm_call_result_105490)
    
    # Processing the call keyword arguments (line 156)
    kwargs_105492 = {}
    # Getting the type of 'np' (line 156)
    np_105479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 19), 'np', False)
    # Obtaining the member 'square' of a type (line 156)
    square_105480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 19), np_105479, 'square')
    # Calling square(args, kwargs) (line 156)
    square_call_result_105493 = invoke(stypy.reporting.localization.Localization(__file__, 156, 19), square_105480, *[result_div_105491], **kwargs_105492)
    
    # Assigning a type to the variable 'BA_disparity' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'BA_disparity', square_call_result_105493)
    
    # Call to assert_allclose(...): (line 157)
    # Processing the call arguments (line 157)
    # Getting the type of 'BA_disparity' (line 157)
    BA_disparity_105495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'BA_disparity', False)
    # Getting the type of 'expected_disparity' (line 157)
    expected_disparity_105496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 34), 'expected_disparity', False)
    # Processing the call keyword arguments (line 157)
    kwargs_105497 = {}
    # Getting the type of 'assert_allclose' (line 157)
    assert_allclose_105494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 157)
    assert_allclose_call_result_105498 = invoke(stypy.reporting.localization.Localization(__file__, 157, 4), assert_allclose_105494, *[BA_disparity_105495, expected_disparity_105496], **kwargs_105497)
    
    
    # ################# End of 'test_orthogonal_procrustes_stretched_example(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_orthogonal_procrustes_stretched_example' in the type store
    # Getting the type of 'stypy_return_type' (line 138)
    stypy_return_type_105499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_105499)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_orthogonal_procrustes_stretched_example'
    return stypy_return_type_105499

# Assigning a type to the variable 'test_orthogonal_procrustes_stretched_example' (line 138)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), 'test_orthogonal_procrustes_stretched_example', test_orthogonal_procrustes_stretched_example)

@norecursion
def test_orthogonal_procrustes_skbio_example(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_orthogonal_procrustes_skbio_example'
    module_type_store = module_type_store.open_function_context('test_orthogonal_procrustes_skbio_example', 160, 0, False)
    
    # Passed parameters checking function
    test_orthogonal_procrustes_skbio_example.stypy_localization = localization
    test_orthogonal_procrustes_skbio_example.stypy_type_of_self = None
    test_orthogonal_procrustes_skbio_example.stypy_type_store = module_type_store
    test_orthogonal_procrustes_skbio_example.stypy_function_name = 'test_orthogonal_procrustes_skbio_example'
    test_orthogonal_procrustes_skbio_example.stypy_param_names_list = []
    test_orthogonal_procrustes_skbio_example.stypy_varargs_param_name = None
    test_orthogonal_procrustes_skbio_example.stypy_kwargs_param_name = None
    test_orthogonal_procrustes_skbio_example.stypy_call_defaults = defaults
    test_orthogonal_procrustes_skbio_example.stypy_call_varargs = varargs
    test_orthogonal_procrustes_skbio_example.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_orthogonal_procrustes_skbio_example', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_orthogonal_procrustes_skbio_example', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_orthogonal_procrustes_skbio_example(...)' code ##################

    
    # Assigning a Call to a Name (line 177):
    
    # Assigning a Call to a Name (line 177):
    
    # Call to array(...): (line 177)
    # Processing the call arguments (line 177)
    
    # Obtaining an instance of the builtin type 'list' (line 177)
    list_105502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 177)
    # Adding element type (line 177)
    
    # Obtaining an instance of the builtin type 'list' (line 177)
    list_105503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 177)
    # Adding element type (line 177)
    int_105504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 23), list_105503, int_105504)
    # Adding element type (line 177)
    int_105505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 23), list_105503, int_105505)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 22), list_105502, list_105503)
    # Adding element type (line 177)
    
    # Obtaining an instance of the builtin type 'list' (line 177)
    list_105506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 177)
    # Adding element type (line 177)
    int_105507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 32), list_105506, int_105507)
    # Adding element type (line 177)
    int_105508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 32), list_105506, int_105508)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 22), list_105502, list_105506)
    # Adding element type (line 177)
    
    # Obtaining an instance of the builtin type 'list' (line 177)
    list_105509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 177)
    # Adding element type (line 177)
    int_105510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 41), list_105509, int_105510)
    # Adding element type (line 177)
    int_105511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 41), list_105509, int_105511)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 22), list_105502, list_105509)
    # Adding element type (line 177)
    
    # Obtaining an instance of the builtin type 'list' (line 177)
    list_105512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 50), 'list')
    # Adding type elements to the builtin type 'list' instance (line 177)
    # Adding element type (line 177)
    int_105513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 50), list_105512, int_105513)
    # Adding element type (line 177)
    int_105514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 50), list_105512, int_105514)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 22), list_105502, list_105512)
    
    # Processing the call keyword arguments (line 177)
    # Getting the type of 'float' (line 177)
    float_105515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 66), 'float', False)
    keyword_105516 = float_105515
    kwargs_105517 = {'dtype': keyword_105516}
    # Getting the type of 'np' (line 177)
    np_105500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 13), 'np', False)
    # Obtaining the member 'array' of a type (line 177)
    array_105501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 13), np_105500, 'array')
    # Calling array(args, kwargs) (line 177)
    array_call_result_105518 = invoke(stypy.reporting.localization.Localization(__file__, 177, 13), array_105501, *[list_105502], **kwargs_105517)
    
    # Assigning a type to the variable 'A_orig' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'A_orig', array_call_result_105518)
    
    # Assigning a Call to a Name (line 178):
    
    # Assigning a Call to a Name (line 178):
    
    # Call to array(...): (line 178)
    # Processing the call arguments (line 178)
    
    # Obtaining an instance of the builtin type 'list' (line 178)
    list_105521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 178)
    # Adding element type (line 178)
    
    # Obtaining an instance of the builtin type 'list' (line 178)
    list_105522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 178)
    # Adding element type (line 178)
    int_105523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 23), list_105522, int_105523)
    # Adding element type (line 178)
    int_105524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 23), list_105522, int_105524)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 22), list_105521, list_105522)
    # Adding element type (line 178)
    
    # Obtaining an instance of the builtin type 'list' (line 178)
    list_105525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 178)
    # Adding element type (line 178)
    int_105526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 31), list_105525, int_105526)
    # Adding element type (line 178)
    int_105527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 31), list_105525, int_105527)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 22), list_105521, list_105525)
    # Adding element type (line 178)
    
    # Obtaining an instance of the builtin type 'list' (line 178)
    list_105528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 178)
    # Adding element type (line 178)
    int_105529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 39), list_105528, int_105529)
    # Adding element type (line 178)
    int_105530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 39), list_105528, int_105530)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 22), list_105521, list_105528)
    # Adding element type (line 178)
    
    # Obtaining an instance of the builtin type 'list' (line 178)
    list_105531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 47), 'list')
    # Adding type elements to the builtin type 'list' instance (line 178)
    # Adding element type (line 178)
    int_105532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 47), list_105531, int_105532)
    # Adding element type (line 178)
    int_105533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 47), list_105531, int_105533)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 22), list_105521, list_105531)
    
    # Processing the call keyword arguments (line 178)
    # Getting the type of 'float' (line 178)
    float_105534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 62), 'float', False)
    keyword_105535 = float_105534
    kwargs_105536 = {'dtype': keyword_105535}
    # Getting the type of 'np' (line 178)
    np_105519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 13), 'np', False)
    # Obtaining the member 'array' of a type (line 178)
    array_105520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 13), np_105519, 'array')
    # Calling array(args, kwargs) (line 178)
    array_call_result_105537 = invoke(stypy.reporting.localization.Localization(__file__, 178, 13), array_105520, *[list_105521], **kwargs_105536)
    
    # Assigning a type to the variable 'B_orig' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'B_orig', array_call_result_105537)
    
    # Assigning a Call to a Name (line 179):
    
    # Assigning a Call to a Name (line 179):
    
    # Call to array(...): (line 179)
    # Processing the call arguments (line 179)
    
    # Obtaining an instance of the builtin type 'list' (line 179)
    list_105540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 179)
    # Adding element type (line 179)
    
    # Obtaining an instance of the builtin type 'list' (line 180)
    list_105541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 180)
    # Adding element type (line 180)
    float_105542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 8), list_105541, float_105542)
    # Adding element type (line 180)
    float_105543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 22), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 8), list_105541, float_105543)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 30), list_105540, list_105541)
    # Adding element type (line 179)
    
    # Obtaining an instance of the builtin type 'list' (line 181)
    list_105544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 181)
    # Adding element type (line 181)
    float_105545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 8), list_105544, float_105545)
    # Adding element type (line 181)
    float_105546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 22), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 8), list_105544, float_105546)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 30), list_105540, list_105544)
    # Adding element type (line 179)
    
    # Obtaining an instance of the builtin type 'list' (line 182)
    list_105547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 182)
    # Adding element type (line 182)
    float_105548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 8), list_105547, float_105548)
    # Adding element type (line 182)
    float_105549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 22), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 8), list_105547, float_105549)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 30), list_105540, list_105547)
    # Adding element type (line 179)
    
    # Obtaining an instance of the builtin type 'list' (line 183)
    list_105550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 183)
    # Adding element type (line 183)
    float_105551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 8), list_105550, float_105551)
    # Adding element type (line 183)
    float_105552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 8), list_105550, float_105552)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 30), list_105540, list_105550)
    
    # Processing the call keyword arguments (line 179)
    kwargs_105553 = {}
    # Getting the type of 'np' (line 179)
    np_105538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 21), 'np', False)
    # Obtaining the member 'array' of a type (line 179)
    array_105539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 21), np_105538, 'array')
    # Calling array(args, kwargs) (line 179)
    array_call_result_105554 = invoke(stypy.reporting.localization.Localization(__file__, 179, 21), array_105539, *[list_105540], **kwargs_105553)
    
    # Assigning a type to the variable 'B_standardized' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'B_standardized', array_call_result_105554)
    
    # Assigning a Call to a Tuple (line 184):
    
    # Assigning a Subscript to a Name (line 184):
    
    # Obtaining the type of the subscript
    int_105555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 4), 'int')
    
    # Call to _centered(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'A_orig' (line 184)
    A_orig_105557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 24), 'A_orig', False)
    # Processing the call keyword arguments (line 184)
    kwargs_105558 = {}
    # Getting the type of '_centered' (line 184)
    _centered_105556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 14), '_centered', False)
    # Calling _centered(args, kwargs) (line 184)
    _centered_call_result_105559 = invoke(stypy.reporting.localization.Localization(__file__, 184, 14), _centered_105556, *[A_orig_105557], **kwargs_105558)
    
    # Obtaining the member '__getitem__' of a type (line 184)
    getitem___105560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 4), _centered_call_result_105559, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 184)
    subscript_call_result_105561 = invoke(stypy.reporting.localization.Localization(__file__, 184, 4), getitem___105560, int_105555)
    
    # Assigning a type to the variable 'tuple_var_assignment_104584' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'tuple_var_assignment_104584', subscript_call_result_105561)
    
    # Assigning a Subscript to a Name (line 184):
    
    # Obtaining the type of the subscript
    int_105562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 4), 'int')
    
    # Call to _centered(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'A_orig' (line 184)
    A_orig_105564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 24), 'A_orig', False)
    # Processing the call keyword arguments (line 184)
    kwargs_105565 = {}
    # Getting the type of '_centered' (line 184)
    _centered_105563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 14), '_centered', False)
    # Calling _centered(args, kwargs) (line 184)
    _centered_call_result_105566 = invoke(stypy.reporting.localization.Localization(__file__, 184, 14), _centered_105563, *[A_orig_105564], **kwargs_105565)
    
    # Obtaining the member '__getitem__' of a type (line 184)
    getitem___105567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 4), _centered_call_result_105566, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 184)
    subscript_call_result_105568 = invoke(stypy.reporting.localization.Localization(__file__, 184, 4), getitem___105567, int_105562)
    
    # Assigning a type to the variable 'tuple_var_assignment_104585' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'tuple_var_assignment_104585', subscript_call_result_105568)
    
    # Assigning a Name to a Name (line 184):
    # Getting the type of 'tuple_var_assignment_104584' (line 184)
    tuple_var_assignment_104584_105569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'tuple_var_assignment_104584')
    # Assigning a type to the variable 'A' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'A', tuple_var_assignment_104584_105569)
    
    # Assigning a Name to a Name (line 184):
    # Getting the type of 'tuple_var_assignment_104585' (line 184)
    tuple_var_assignment_104585_105570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'tuple_var_assignment_104585')
    # Assigning a type to the variable 'A_mu' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 7), 'A_mu', tuple_var_assignment_104585_105570)
    
    # Assigning a Call to a Tuple (line 185):
    
    # Assigning a Subscript to a Name (line 185):
    
    # Obtaining the type of the subscript
    int_105571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 4), 'int')
    
    # Call to _centered(...): (line 185)
    # Processing the call arguments (line 185)
    # Getting the type of 'B_orig' (line 185)
    B_orig_105573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 24), 'B_orig', False)
    # Processing the call keyword arguments (line 185)
    kwargs_105574 = {}
    # Getting the type of '_centered' (line 185)
    _centered_105572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 14), '_centered', False)
    # Calling _centered(args, kwargs) (line 185)
    _centered_call_result_105575 = invoke(stypy.reporting.localization.Localization(__file__, 185, 14), _centered_105572, *[B_orig_105573], **kwargs_105574)
    
    # Obtaining the member '__getitem__' of a type (line 185)
    getitem___105576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 4), _centered_call_result_105575, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 185)
    subscript_call_result_105577 = invoke(stypy.reporting.localization.Localization(__file__, 185, 4), getitem___105576, int_105571)
    
    # Assigning a type to the variable 'tuple_var_assignment_104586' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'tuple_var_assignment_104586', subscript_call_result_105577)
    
    # Assigning a Subscript to a Name (line 185):
    
    # Obtaining the type of the subscript
    int_105578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 4), 'int')
    
    # Call to _centered(...): (line 185)
    # Processing the call arguments (line 185)
    # Getting the type of 'B_orig' (line 185)
    B_orig_105580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 24), 'B_orig', False)
    # Processing the call keyword arguments (line 185)
    kwargs_105581 = {}
    # Getting the type of '_centered' (line 185)
    _centered_105579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 14), '_centered', False)
    # Calling _centered(args, kwargs) (line 185)
    _centered_call_result_105582 = invoke(stypy.reporting.localization.Localization(__file__, 185, 14), _centered_105579, *[B_orig_105580], **kwargs_105581)
    
    # Obtaining the member '__getitem__' of a type (line 185)
    getitem___105583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 4), _centered_call_result_105582, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 185)
    subscript_call_result_105584 = invoke(stypy.reporting.localization.Localization(__file__, 185, 4), getitem___105583, int_105578)
    
    # Assigning a type to the variable 'tuple_var_assignment_104587' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'tuple_var_assignment_104587', subscript_call_result_105584)
    
    # Assigning a Name to a Name (line 185):
    # Getting the type of 'tuple_var_assignment_104586' (line 185)
    tuple_var_assignment_104586_105585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'tuple_var_assignment_104586')
    # Assigning a type to the variable 'B' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'B', tuple_var_assignment_104586_105585)
    
    # Assigning a Name to a Name (line 185):
    # Getting the type of 'tuple_var_assignment_104587' (line 185)
    tuple_var_assignment_104587_105586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'tuple_var_assignment_104587')
    # Assigning a type to the variable 'B_mu' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 7), 'B_mu', tuple_var_assignment_104587_105586)
    
    # Assigning a Call to a Tuple (line 186):
    
    # Assigning a Subscript to a Name (line 186):
    
    # Obtaining the type of the subscript
    int_105587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 4), 'int')
    
    # Call to orthogonal_procrustes(...): (line 186)
    # Processing the call arguments (line 186)
    # Getting the type of 'A' (line 186)
    A_105589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 33), 'A', False)
    # Getting the type of 'B' (line 186)
    B_105590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 36), 'B', False)
    # Processing the call keyword arguments (line 186)
    kwargs_105591 = {}
    # Getting the type of 'orthogonal_procrustes' (line 186)
    orthogonal_procrustes_105588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 11), 'orthogonal_procrustes', False)
    # Calling orthogonal_procrustes(args, kwargs) (line 186)
    orthogonal_procrustes_call_result_105592 = invoke(stypy.reporting.localization.Localization(__file__, 186, 11), orthogonal_procrustes_105588, *[A_105589, B_105590], **kwargs_105591)
    
    # Obtaining the member '__getitem__' of a type (line 186)
    getitem___105593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 4), orthogonal_procrustes_call_result_105592, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 186)
    subscript_call_result_105594 = invoke(stypy.reporting.localization.Localization(__file__, 186, 4), getitem___105593, int_105587)
    
    # Assigning a type to the variable 'tuple_var_assignment_104588' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'tuple_var_assignment_104588', subscript_call_result_105594)
    
    # Assigning a Subscript to a Name (line 186):
    
    # Obtaining the type of the subscript
    int_105595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 4), 'int')
    
    # Call to orthogonal_procrustes(...): (line 186)
    # Processing the call arguments (line 186)
    # Getting the type of 'A' (line 186)
    A_105597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 33), 'A', False)
    # Getting the type of 'B' (line 186)
    B_105598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 36), 'B', False)
    # Processing the call keyword arguments (line 186)
    kwargs_105599 = {}
    # Getting the type of 'orthogonal_procrustes' (line 186)
    orthogonal_procrustes_105596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 11), 'orthogonal_procrustes', False)
    # Calling orthogonal_procrustes(args, kwargs) (line 186)
    orthogonal_procrustes_call_result_105600 = invoke(stypy.reporting.localization.Localization(__file__, 186, 11), orthogonal_procrustes_105596, *[A_105597, B_105598], **kwargs_105599)
    
    # Obtaining the member '__getitem__' of a type (line 186)
    getitem___105601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 4), orthogonal_procrustes_call_result_105600, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 186)
    subscript_call_result_105602 = invoke(stypy.reporting.localization.Localization(__file__, 186, 4), getitem___105601, int_105595)
    
    # Assigning a type to the variable 'tuple_var_assignment_104589' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'tuple_var_assignment_104589', subscript_call_result_105602)
    
    # Assigning a Name to a Name (line 186):
    # Getting the type of 'tuple_var_assignment_104588' (line 186)
    tuple_var_assignment_104588_105603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'tuple_var_assignment_104588')
    # Assigning a type to the variable 'R' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'R', tuple_var_assignment_104588_105603)
    
    # Assigning a Name to a Name (line 186):
    # Getting the type of 'tuple_var_assignment_104589' (line 186)
    tuple_var_assignment_104589_105604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'tuple_var_assignment_104589')
    # Assigning a type to the variable 's' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 7), 's', tuple_var_assignment_104589_105604)
    
    # Assigning a BinOp to a Name (line 187):
    
    # Assigning a BinOp to a Name (line 187):
    # Getting the type of 's' (line 187)
    s_105605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 's')
    
    # Call to square(...): (line 187)
    # Processing the call arguments (line 187)
    
    # Call to norm(...): (line 187)
    # Processing the call arguments (line 187)
    # Getting the type of 'A' (line 187)
    A_105609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 31), 'A', False)
    # Processing the call keyword arguments (line 187)
    kwargs_105610 = {}
    # Getting the type of 'norm' (line 187)
    norm_105608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 26), 'norm', False)
    # Calling norm(args, kwargs) (line 187)
    norm_call_result_105611 = invoke(stypy.reporting.localization.Localization(__file__, 187, 26), norm_105608, *[A_105609], **kwargs_105610)
    
    # Processing the call keyword arguments (line 187)
    kwargs_105612 = {}
    # Getting the type of 'np' (line 187)
    np_105606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 16), 'np', False)
    # Obtaining the member 'square' of a type (line 187)
    square_105607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 16), np_105606, 'square')
    # Calling square(args, kwargs) (line 187)
    square_call_result_105613 = invoke(stypy.reporting.localization.Localization(__file__, 187, 16), square_105607, *[norm_call_result_105611], **kwargs_105612)
    
    # Applying the binary operator 'div' (line 187)
    result_div_105614 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 12), 'div', s_105605, square_call_result_105613)
    
    # Assigning a type to the variable 'scale' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'scale', result_div_105614)
    
    # Assigning a BinOp to a Name (line 188):
    
    # Assigning a BinOp to a Name (line 188):
    # Getting the type of 'scale' (line 188)
    scale_105615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 15), 'scale')
    
    # Call to dot(...): (line 188)
    # Processing the call arguments (line 188)
    # Getting the type of 'A' (line 188)
    A_105618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 30), 'A', False)
    # Getting the type of 'R' (line 188)
    R_105619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 33), 'R', False)
    # Processing the call keyword arguments (line 188)
    kwargs_105620 = {}
    # Getting the type of 'np' (line 188)
    np_105616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 23), 'np', False)
    # Obtaining the member 'dot' of a type (line 188)
    dot_105617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 23), np_105616, 'dot')
    # Calling dot(args, kwargs) (line 188)
    dot_call_result_105621 = invoke(stypy.reporting.localization.Localization(__file__, 188, 23), dot_105617, *[A_105618, R_105619], **kwargs_105620)
    
    # Applying the binary operator '*' (line 188)
    result_mul_105622 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 15), '*', scale_105615, dot_call_result_105621)
    
    # Getting the type of 'B_mu' (line 188)
    B_mu_105623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 38), 'B_mu')
    # Applying the binary operator '+' (line 188)
    result_add_105624 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 15), '+', result_mul_105622, B_mu_105623)
    
    # Assigning a type to the variable 'B_approx' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'B_approx', result_add_105624)
    
    # Call to assert_allclose(...): (line 189)
    # Processing the call arguments (line 189)
    # Getting the type of 'B_approx' (line 189)
    B_approx_105626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 20), 'B_approx', False)
    # Getting the type of 'B_orig' (line 189)
    B_orig_105627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 30), 'B_orig', False)
    # Processing the call keyword arguments (line 189)
    kwargs_105628 = {}
    # Getting the type of 'assert_allclose' (line 189)
    assert_allclose_105625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 189)
    assert_allclose_call_result_105629 = invoke(stypy.reporting.localization.Localization(__file__, 189, 4), assert_allclose_105625, *[B_approx_105626, B_orig_105627], **kwargs_105628)
    
    
    # Call to assert_allclose(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'B' (line 190)
    B_105631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 20), 'B', False)
    
    # Call to norm(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'B' (line 190)
    B_105633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 29), 'B', False)
    # Processing the call keyword arguments (line 190)
    kwargs_105634 = {}
    # Getting the type of 'norm' (line 190)
    norm_105632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 24), 'norm', False)
    # Calling norm(args, kwargs) (line 190)
    norm_call_result_105635 = invoke(stypy.reporting.localization.Localization(__file__, 190, 24), norm_105632, *[B_105633], **kwargs_105634)
    
    # Applying the binary operator 'div' (line 190)
    result_div_105636 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 20), 'div', B_105631, norm_call_result_105635)
    
    # Getting the type of 'B_standardized' (line 190)
    B_standardized_105637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 33), 'B_standardized', False)
    # Processing the call keyword arguments (line 190)
    kwargs_105638 = {}
    # Getting the type of 'assert_allclose' (line 190)
    assert_allclose_105630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 190)
    assert_allclose_call_result_105639 = invoke(stypy.reporting.localization.Localization(__file__, 190, 4), assert_allclose_105630, *[result_div_105636, B_standardized_105637], **kwargs_105638)
    
    
    # ################# End of 'test_orthogonal_procrustes_skbio_example(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_orthogonal_procrustes_skbio_example' in the type store
    # Getting the type of 'stypy_return_type' (line 160)
    stypy_return_type_105640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_105640)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_orthogonal_procrustes_skbio_example'
    return stypy_return_type_105640

# Assigning a type to the variable 'test_orthogonal_procrustes_skbio_example' (line 160)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), 'test_orthogonal_procrustes_skbio_example', test_orthogonal_procrustes_skbio_example)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
