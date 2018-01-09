
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for the linalg.isolve.lgmres module
2: '''
3: 
4: from __future__ import division, print_function, absolute_import
5: 
6: from numpy.testing import assert_, assert_allclose, assert_equal
7: 
8: import numpy as np
9: from numpy import zeros, array, allclose
10: from scipy.linalg import norm
11: from scipy.sparse import csr_matrix, eye, rand
12: 
13: from scipy.sparse.linalg.interface import LinearOperator
14: from scipy.sparse.linalg import splu
15: from scipy.sparse.linalg.isolve import lgmres, gmres
16: 
17: 
18: Am = csr_matrix(array([[-2,1,0,0,0,9],
19:                        [1,-2,1,0,5,0],
20:                        [0,1,-2,1,0,0],
21:                        [0,0,1,-2,1,0],
22:                        [0,3,0,1,-2,1],
23:                        [1,0,0,0,1,-2]]))
24: b = array([1,2,3,4,5,6])
25: count = [0]
26: 
27: 
28: def matvec(v):
29:     count[0] += 1
30:     return Am*v
31: A = LinearOperator(matvec=matvec, shape=Am.shape, dtype=Am.dtype)
32: 
33: 
34: def do_solve(**kw):
35:     count[0] = 0
36:     x0, flag = lgmres(A, b, x0=zeros(A.shape[0]), inner_m=6, tol=1e-14, **kw)
37:     count_0 = count[0]
38:     assert_(allclose(A*x0, b, rtol=1e-12, atol=1e-12), norm(A*x0-b))
39:     return x0, count_0
40: 
41: 
42: class TestLGMRES(object):
43:     def test_preconditioner(self):
44:         # Check that preconditioning works
45:         pc = splu(Am.tocsc())
46:         M = LinearOperator(matvec=pc.solve, shape=A.shape, dtype=A.dtype)
47: 
48:         x0, count_0 = do_solve()
49:         x1, count_1 = do_solve(M=M)
50: 
51:         assert_(count_1 == 3)
52:         assert_(count_1 < count_0/2)
53:         assert_(allclose(x1, x0, rtol=1e-14))
54: 
55:     def test_outer_v(self):
56:         # Check that the augmentation vectors behave as expected
57: 
58:         outer_v = []
59:         x0, count_0 = do_solve(outer_k=6, outer_v=outer_v)
60:         assert_(len(outer_v) > 0)
61:         assert_(len(outer_v) <= 6)
62: 
63:         x1, count_1 = do_solve(outer_k=6, outer_v=outer_v, prepend_outer_v=True)
64:         assert_(count_1 == 2, count_1)
65:         assert_(count_1 < count_0/2)
66:         assert_(allclose(x1, x0, rtol=1e-14))
67: 
68:         # ---
69: 
70:         outer_v = []
71:         x0, count_0 = do_solve(outer_k=6, outer_v=outer_v, store_outer_Av=False)
72:         assert_(array([v[1] is None for v in outer_v]).all())
73:         assert_(len(outer_v) > 0)
74:         assert_(len(outer_v) <= 6)
75: 
76:         x1, count_1 = do_solve(outer_k=6, outer_v=outer_v, prepend_outer_v=True)
77:         assert_(count_1 == 3, count_1)
78:         assert_(count_1 < count_0/2)
79:         assert_(allclose(x1, x0, rtol=1e-14))
80: 
81:     def test_arnoldi(self):
82:         np.random.rand(1234)
83: 
84:         A = eye(10000) + rand(10000,10000,density=1e-4)
85:         b = np.random.rand(10000)
86: 
87:         # The inner arnoldi should be equivalent to gmres
88:         x0, flag0 = lgmres(A, b, x0=zeros(A.shape[0]), inner_m=15, maxiter=1)
89:         x1, flag1 = gmres(A, b, x0=zeros(A.shape[0]), restart=15, maxiter=1)
90: 
91:         assert_equal(flag0, 1)
92:         assert_equal(flag1, 1)
93:         assert_(np.linalg.norm(A.dot(x0) - b) > 1e-3)
94: 
95:         assert_allclose(x0, x1)
96: 
97:     def test_cornercase(self):
98:         np.random.seed(1234)
99: 
100:         # Rounding error may prevent convergence with tol=0 --- ensure
101:         # that the return values in this case are correct, and no
102:         # exceptions are raised
103: 
104:         for n in [3, 5, 10, 100]:
105:             A = 2*eye(n)
106: 
107:             b = np.ones(n)
108:             x, info = lgmres(A, b, maxiter=10)
109:             assert_equal(info, 0)
110:             assert_allclose(A.dot(x) - b, 0, atol=1e-14)
111: 
112:             x, info = lgmres(A, b, tol=0, maxiter=10)
113:             if info == 0:
114:                 assert_allclose(A.dot(x) - b, 0, atol=1e-14)
115: 
116:             b = np.random.rand(n)
117:             x, info = lgmres(A, b, maxiter=10)
118:             assert_equal(info, 0)
119:             assert_allclose(A.dot(x) - b, 0, atol=1e-14)
120: 
121:             x, info = lgmres(A, b, tol=0, maxiter=10)
122:             if info == 0:
123:                 assert_allclose(A.dot(x) - b, 0, atol=1e-14)
124: 
125:     def test_nans(self):
126:         A = eye(3, format='lil')
127:         A[1,1] = np.nan
128:         b = np.ones(3)
129: 
130:         x, info = lgmres(A, b, tol=0, maxiter=10)
131:         assert_equal(info, 1)
132: 
133:     def test_breakdown_with_outer_v(self):
134:         A = np.array([[1, 2], [3, 4]], dtype=float)
135:         b = np.array([1, 2])
136: 
137:         x = np.linalg.solve(A, b)
138:         v0 = np.array([1, 0])
139: 
140:         # The inner iteration should converge to the correct solution,
141:         # since it's in the outer vector list
142:         xp, info = lgmres(A, b, outer_v=[(v0, None), (x, None)], maxiter=1)
143: 
144:         assert_allclose(xp, x, atol=1e-12)
145: 
146:     def test_breakdown_underdetermined(self):
147:         # Should find LSQ solution in the Krylov span in one inner
148:         # iteration, despite solver breakdown from nilpotent A.
149:         A = np.array([[0, 1, 1, 1],
150:                       [0, 0, 1, 1],
151:                       [0, 0, 0, 1],
152:                       [0, 0, 0, 0]], dtype=float)
153: 
154:         bs = [
155:             np.array([1, 1, 1, 1]),
156:             np.array([1, 1, 1, 0]),
157:             np.array([1, 1, 0, 0]),
158:             np.array([1, 0, 0, 0]),
159:         ]
160: 
161:         for b in bs:
162:             xp, info = lgmres(A, b, maxiter=1)
163:             resp = np.linalg.norm(A.dot(xp) - b)
164: 
165:             K = np.c_[b, A.dot(b), A.dot(A.dot(b)), A.dot(A.dot(A.dot(b)))]
166:             y, _, _, _ = np.linalg.lstsq(A.dot(K), b, rcond=-1)
167:             x = K.dot(y)
168:             res = np.linalg.norm(A.dot(x) - b)
169: 
170:             assert_allclose(resp, res, err_msg=repr(b))
171: 
172:     def test_denormals(self):
173:         # Check that no warnings are emitted if the matrix contains
174:         # numbers for which 1/x has no float representation, and that
175:         # the solver behaves properly.
176:         A = np.array([[1, 2], [3, 4]], dtype=float)
177:         A *= 100 * np.nextafter(0, 1)
178: 
179:         b = np.array([1, 1])
180: 
181:         xp, info = lgmres(A, b)
182: 
183:         if info == 0:
184:             assert_allclose(A.dot(xp), b)
185: 
186: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_419573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', 'Tests for the linalg.isolve.lgmres module\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.testing import assert_, assert_allclose, assert_equal' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_419574 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing')

if (type(import_419574) is not StypyTypeError):

    if (import_419574 != 'pyd_module'):
        __import__(import_419574)
        sys_modules_419575 = sys.modules[import_419574]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', sys_modules_419575.module_type_store, module_type_store, ['assert_', 'assert_allclose', 'assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_419575, sys_modules_419575.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_allclose, assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_allclose', 'assert_equal'], [assert_, assert_allclose, assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', import_419574)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import numpy' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_419576 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy')

if (type(import_419576) is not StypyTypeError):

    if (import_419576 != 'pyd_module'):
        __import__(import_419576)
        sys_modules_419577 = sys.modules[import_419576]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', sys_modules_419577.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', import_419576)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy import zeros, array, allclose' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_419578 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy')

if (type(import_419578) is not StypyTypeError):

    if (import_419578 != 'pyd_module'):
        __import__(import_419578)
        sys_modules_419579 = sys.modules[import_419578]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', sys_modules_419579.module_type_store, module_type_store, ['zeros', 'array', 'allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_419579, sys_modules_419579.module_type_store, module_type_store)
    else:
        from numpy import zeros, array, allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', None, module_type_store, ['zeros', 'array', 'allclose'], [zeros, array, allclose])

else:
    # Assigning a type to the variable 'numpy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', import_419578)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.linalg import norm' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_419580 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.linalg')

if (type(import_419580) is not StypyTypeError):

    if (import_419580 != 'pyd_module'):
        __import__(import_419580)
        sys_modules_419581 = sys.modules[import_419580]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.linalg', sys_modules_419581.module_type_store, module_type_store, ['norm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_419581, sys_modules_419581.module_type_store, module_type_store)
    else:
        from scipy.linalg import norm

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.linalg', None, module_type_store, ['norm'], [norm])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.linalg', import_419580)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.sparse import csr_matrix, eye, rand' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_419582 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse')

if (type(import_419582) is not StypyTypeError):

    if (import_419582 != 'pyd_module'):
        __import__(import_419582)
        sys_modules_419583 = sys.modules[import_419582]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse', sys_modules_419583.module_type_store, module_type_store, ['csr_matrix', 'eye', 'rand'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_419583, sys_modules_419583.module_type_store, module_type_store)
    else:
        from scipy.sparse import csr_matrix, eye, rand

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse', None, module_type_store, ['csr_matrix', 'eye', 'rand'], [csr_matrix, eye, rand])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse', import_419582)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.sparse.linalg.interface import LinearOperator' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_419584 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.linalg.interface')

if (type(import_419584) is not StypyTypeError):

    if (import_419584 != 'pyd_module'):
        __import__(import_419584)
        sys_modules_419585 = sys.modules[import_419584]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.linalg.interface', sys_modules_419585.module_type_store, module_type_store, ['LinearOperator'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_419585, sys_modules_419585.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.interface import LinearOperator

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.linalg.interface', None, module_type_store, ['LinearOperator'], [LinearOperator])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.interface' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.linalg.interface', import_419584)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy.sparse.linalg import splu' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_419586 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse.linalg')

if (type(import_419586) is not StypyTypeError):

    if (import_419586 != 'pyd_module'):
        __import__(import_419586)
        sys_modules_419587 = sys.modules[import_419586]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse.linalg', sys_modules_419587.module_type_store, module_type_store, ['splu'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_419587, sys_modules_419587.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import splu

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse.linalg', None, module_type_store, ['splu'], [splu])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse.linalg', import_419586)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from scipy.sparse.linalg.isolve import lgmres, gmres' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_419588 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse.linalg.isolve')

if (type(import_419588) is not StypyTypeError):

    if (import_419588 != 'pyd_module'):
        __import__(import_419588)
        sys_modules_419589 = sys.modules[import_419588]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse.linalg.isolve', sys_modules_419589.module_type_store, module_type_store, ['lgmres', 'gmres'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_419589, sys_modules_419589.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.isolve import lgmres, gmres

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse.linalg.isolve', None, module_type_store, ['lgmres', 'gmres'], [lgmres, gmres])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.isolve' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse.linalg.isolve', import_419588)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')


# Assigning a Call to a Name (line 18):

# Assigning a Call to a Name (line 18):

# Call to csr_matrix(...): (line 18)
# Processing the call arguments (line 18)

# Call to array(...): (line 18)
# Processing the call arguments (line 18)

# Obtaining an instance of the builtin type 'list' (line 18)
list_419592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)

# Obtaining an instance of the builtin type 'list' (line 18)
list_419593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)
int_419594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 23), list_419593, int_419594)
# Adding element type (line 18)
int_419595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 23), list_419593, int_419595)
# Adding element type (line 18)
int_419596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 23), list_419593, int_419596)
# Adding element type (line 18)
int_419597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 23), list_419593, int_419597)
# Adding element type (line 18)
int_419598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 23), list_419593, int_419598)
# Adding element type (line 18)
int_419599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 23), list_419593, int_419599)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 22), list_419592, list_419593)
# Adding element type (line 18)

# Obtaining an instance of the builtin type 'list' (line 19)
list_419600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)
int_419601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_419600, int_419601)
# Adding element type (line 19)
int_419602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_419600, int_419602)
# Adding element type (line 19)
int_419603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_419600, int_419603)
# Adding element type (line 19)
int_419604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_419600, int_419604)
# Adding element type (line 19)
int_419605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_419600, int_419605)
# Adding element type (line 19)
int_419606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_419600, int_419606)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 22), list_419592, list_419600)
# Adding element type (line 18)

# Obtaining an instance of the builtin type 'list' (line 20)
list_419607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 20)
# Adding element type (line 20)
int_419608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), list_419607, int_419608)
# Adding element type (line 20)
int_419609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), list_419607, int_419609)
# Adding element type (line 20)
int_419610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), list_419607, int_419610)
# Adding element type (line 20)
int_419611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), list_419607, int_419611)
# Adding element type (line 20)
int_419612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), list_419607, int_419612)
# Adding element type (line 20)
int_419613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), list_419607, int_419613)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 22), list_419592, list_419607)
# Adding element type (line 18)

# Obtaining an instance of the builtin type 'list' (line 21)
list_419614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)
int_419615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_419614, int_419615)
# Adding element type (line 21)
int_419616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_419614, int_419616)
# Adding element type (line 21)
int_419617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_419614, int_419617)
# Adding element type (line 21)
int_419618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_419614, int_419618)
# Adding element type (line 21)
int_419619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_419614, int_419619)
# Adding element type (line 21)
int_419620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_419614, int_419620)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 22), list_419592, list_419614)
# Adding element type (line 18)

# Obtaining an instance of the builtin type 'list' (line 22)
list_419621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 22)
# Adding element type (line 22)
int_419622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 23), list_419621, int_419622)
# Adding element type (line 22)
int_419623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 23), list_419621, int_419623)
# Adding element type (line 22)
int_419624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 23), list_419621, int_419624)
# Adding element type (line 22)
int_419625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 23), list_419621, int_419625)
# Adding element type (line 22)
int_419626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 23), list_419621, int_419626)
# Adding element type (line 22)
int_419627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 23), list_419621, int_419627)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 22), list_419592, list_419621)
# Adding element type (line 18)

# Obtaining an instance of the builtin type 'list' (line 23)
list_419628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 23)
# Adding element type (line 23)
int_419629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 23), list_419628, int_419629)
# Adding element type (line 23)
int_419630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 23), list_419628, int_419630)
# Adding element type (line 23)
int_419631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 23), list_419628, int_419631)
# Adding element type (line 23)
int_419632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 23), list_419628, int_419632)
# Adding element type (line 23)
int_419633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 23), list_419628, int_419633)
# Adding element type (line 23)
int_419634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 23), list_419628, int_419634)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 22), list_419592, list_419628)

# Processing the call keyword arguments (line 18)
kwargs_419635 = {}
# Getting the type of 'array' (line 18)
array_419591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 16), 'array', False)
# Calling array(args, kwargs) (line 18)
array_call_result_419636 = invoke(stypy.reporting.localization.Localization(__file__, 18, 16), array_419591, *[list_419592], **kwargs_419635)

# Processing the call keyword arguments (line 18)
kwargs_419637 = {}
# Getting the type of 'csr_matrix' (line 18)
csr_matrix_419590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 5), 'csr_matrix', False)
# Calling csr_matrix(args, kwargs) (line 18)
csr_matrix_call_result_419638 = invoke(stypy.reporting.localization.Localization(__file__, 18, 5), csr_matrix_419590, *[array_call_result_419636], **kwargs_419637)

# Assigning a type to the variable 'Am' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'Am', csr_matrix_call_result_419638)

# Assigning a Call to a Name (line 24):

# Assigning a Call to a Name (line 24):

# Call to array(...): (line 24)
# Processing the call arguments (line 24)

# Obtaining an instance of the builtin type 'list' (line 24)
list_419640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 24)
# Adding element type (line 24)
int_419641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 10), list_419640, int_419641)
# Adding element type (line 24)
int_419642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 10), list_419640, int_419642)
# Adding element type (line 24)
int_419643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 10), list_419640, int_419643)
# Adding element type (line 24)
int_419644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 10), list_419640, int_419644)
# Adding element type (line 24)
int_419645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 10), list_419640, int_419645)
# Adding element type (line 24)
int_419646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 10), list_419640, int_419646)

# Processing the call keyword arguments (line 24)
kwargs_419647 = {}
# Getting the type of 'array' (line 24)
array_419639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'array', False)
# Calling array(args, kwargs) (line 24)
array_call_result_419648 = invoke(stypy.reporting.localization.Localization(__file__, 24, 4), array_419639, *[list_419640], **kwargs_419647)

# Assigning a type to the variable 'b' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'b', array_call_result_419648)

# Assigning a List to a Name (line 25):

# Assigning a List to a Name (line 25):

# Obtaining an instance of the builtin type 'list' (line 25)
list_419649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 25)
# Adding element type (line 25)
int_419650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 8), list_419649, int_419650)

# Assigning a type to the variable 'count' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'count', list_419649)

@norecursion
def matvec(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'matvec'
    module_type_store = module_type_store.open_function_context('matvec', 28, 0, False)
    
    # Passed parameters checking function
    matvec.stypy_localization = localization
    matvec.stypy_type_of_self = None
    matvec.stypy_type_store = module_type_store
    matvec.stypy_function_name = 'matvec'
    matvec.stypy_param_names_list = ['v']
    matvec.stypy_varargs_param_name = None
    matvec.stypy_kwargs_param_name = None
    matvec.stypy_call_defaults = defaults
    matvec.stypy_call_varargs = varargs
    matvec.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'matvec', ['v'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'matvec', localization, ['v'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'matvec(...)' code ##################

    
    # Getting the type of 'count' (line 29)
    count_419651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'count')
    
    # Obtaining the type of the subscript
    int_419652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 10), 'int')
    # Getting the type of 'count' (line 29)
    count_419653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'count')
    # Obtaining the member '__getitem__' of a type (line 29)
    getitem___419654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 4), count_419653, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 29)
    subscript_call_result_419655 = invoke(stypy.reporting.localization.Localization(__file__, 29, 4), getitem___419654, int_419652)
    
    int_419656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 16), 'int')
    # Applying the binary operator '+=' (line 29)
    result_iadd_419657 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 4), '+=', subscript_call_result_419655, int_419656)
    # Getting the type of 'count' (line 29)
    count_419658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'count')
    int_419659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 10), 'int')
    # Storing an element on a container (line 29)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 4), count_419658, (int_419659, result_iadd_419657))
    
    # Getting the type of 'Am' (line 30)
    Am_419660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'Am')
    # Getting the type of 'v' (line 30)
    v_419661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 14), 'v')
    # Applying the binary operator '*' (line 30)
    result_mul_419662 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 11), '*', Am_419660, v_419661)
    
    # Assigning a type to the variable 'stypy_return_type' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type', result_mul_419662)
    
    # ################# End of 'matvec(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'matvec' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_419663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_419663)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'matvec'
    return stypy_return_type_419663

# Assigning a type to the variable 'matvec' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'matvec', matvec)

# Assigning a Call to a Name (line 31):

# Assigning a Call to a Name (line 31):

# Call to LinearOperator(...): (line 31)
# Processing the call keyword arguments (line 31)
# Getting the type of 'matvec' (line 31)
matvec_419665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 26), 'matvec', False)
keyword_419666 = matvec_419665
# Getting the type of 'Am' (line 31)
Am_419667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 40), 'Am', False)
# Obtaining the member 'shape' of a type (line 31)
shape_419668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 40), Am_419667, 'shape')
keyword_419669 = shape_419668
# Getting the type of 'Am' (line 31)
Am_419670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 56), 'Am', False)
# Obtaining the member 'dtype' of a type (line 31)
dtype_419671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 56), Am_419670, 'dtype')
keyword_419672 = dtype_419671
kwargs_419673 = {'dtype': keyword_419672, 'shape': keyword_419669, 'matvec': keyword_419666}
# Getting the type of 'LinearOperator' (line 31)
LinearOperator_419664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'LinearOperator', False)
# Calling LinearOperator(args, kwargs) (line 31)
LinearOperator_call_result_419674 = invoke(stypy.reporting.localization.Localization(__file__, 31, 4), LinearOperator_419664, *[], **kwargs_419673)

# Assigning a type to the variable 'A' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'A', LinearOperator_call_result_419674)

@norecursion
def do_solve(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'do_solve'
    module_type_store = module_type_store.open_function_context('do_solve', 34, 0, False)
    
    # Passed parameters checking function
    do_solve.stypy_localization = localization
    do_solve.stypy_type_of_self = None
    do_solve.stypy_type_store = module_type_store
    do_solve.stypy_function_name = 'do_solve'
    do_solve.stypy_param_names_list = []
    do_solve.stypy_varargs_param_name = None
    do_solve.stypy_kwargs_param_name = 'kw'
    do_solve.stypy_call_defaults = defaults
    do_solve.stypy_call_varargs = varargs
    do_solve.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'do_solve', [], None, 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'do_solve', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'do_solve(...)' code ##################

    
    # Assigning a Num to a Subscript (line 35):
    
    # Assigning a Num to a Subscript (line 35):
    int_419675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 15), 'int')
    # Getting the type of 'count' (line 35)
    count_419676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'count')
    int_419677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 10), 'int')
    # Storing an element on a container (line 35)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 4), count_419676, (int_419677, int_419675))
    
    # Assigning a Call to a Tuple (line 36):
    
    # Assigning a Subscript to a Name (line 36):
    
    # Obtaining the type of the subscript
    int_419678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 4), 'int')
    
    # Call to lgmres(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'A' (line 36)
    A_419680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 22), 'A', False)
    # Getting the type of 'b' (line 36)
    b_419681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 25), 'b', False)
    # Processing the call keyword arguments (line 36)
    
    # Call to zeros(...): (line 36)
    # Processing the call arguments (line 36)
    
    # Obtaining the type of the subscript
    int_419683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 45), 'int')
    # Getting the type of 'A' (line 36)
    A_419684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 37), 'A', False)
    # Obtaining the member 'shape' of a type (line 36)
    shape_419685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 37), A_419684, 'shape')
    # Obtaining the member '__getitem__' of a type (line 36)
    getitem___419686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 37), shape_419685, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 36)
    subscript_call_result_419687 = invoke(stypy.reporting.localization.Localization(__file__, 36, 37), getitem___419686, int_419683)
    
    # Processing the call keyword arguments (line 36)
    kwargs_419688 = {}
    # Getting the type of 'zeros' (line 36)
    zeros_419682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 31), 'zeros', False)
    # Calling zeros(args, kwargs) (line 36)
    zeros_call_result_419689 = invoke(stypy.reporting.localization.Localization(__file__, 36, 31), zeros_419682, *[subscript_call_result_419687], **kwargs_419688)
    
    keyword_419690 = zeros_call_result_419689
    int_419691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 58), 'int')
    keyword_419692 = int_419691
    float_419693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 65), 'float')
    keyword_419694 = float_419693
    # Getting the type of 'kw' (line 36)
    kw_419695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 74), 'kw', False)
    kwargs_419696 = {'x0': keyword_419690, 'kw_419695': kw_419695, 'tol': keyword_419694, 'inner_m': keyword_419692}
    # Getting the type of 'lgmres' (line 36)
    lgmres_419679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 'lgmres', False)
    # Calling lgmres(args, kwargs) (line 36)
    lgmres_call_result_419697 = invoke(stypy.reporting.localization.Localization(__file__, 36, 15), lgmres_419679, *[A_419680, b_419681], **kwargs_419696)
    
    # Obtaining the member '__getitem__' of a type (line 36)
    getitem___419698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 4), lgmres_call_result_419697, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 36)
    subscript_call_result_419699 = invoke(stypy.reporting.localization.Localization(__file__, 36, 4), getitem___419698, int_419678)
    
    # Assigning a type to the variable 'tuple_var_assignment_419535' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'tuple_var_assignment_419535', subscript_call_result_419699)
    
    # Assigning a Subscript to a Name (line 36):
    
    # Obtaining the type of the subscript
    int_419700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 4), 'int')
    
    # Call to lgmres(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'A' (line 36)
    A_419702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 22), 'A', False)
    # Getting the type of 'b' (line 36)
    b_419703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 25), 'b', False)
    # Processing the call keyword arguments (line 36)
    
    # Call to zeros(...): (line 36)
    # Processing the call arguments (line 36)
    
    # Obtaining the type of the subscript
    int_419705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 45), 'int')
    # Getting the type of 'A' (line 36)
    A_419706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 37), 'A', False)
    # Obtaining the member 'shape' of a type (line 36)
    shape_419707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 37), A_419706, 'shape')
    # Obtaining the member '__getitem__' of a type (line 36)
    getitem___419708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 37), shape_419707, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 36)
    subscript_call_result_419709 = invoke(stypy.reporting.localization.Localization(__file__, 36, 37), getitem___419708, int_419705)
    
    # Processing the call keyword arguments (line 36)
    kwargs_419710 = {}
    # Getting the type of 'zeros' (line 36)
    zeros_419704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 31), 'zeros', False)
    # Calling zeros(args, kwargs) (line 36)
    zeros_call_result_419711 = invoke(stypy.reporting.localization.Localization(__file__, 36, 31), zeros_419704, *[subscript_call_result_419709], **kwargs_419710)
    
    keyword_419712 = zeros_call_result_419711
    int_419713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 58), 'int')
    keyword_419714 = int_419713
    float_419715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 65), 'float')
    keyword_419716 = float_419715
    # Getting the type of 'kw' (line 36)
    kw_419717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 74), 'kw', False)
    kwargs_419718 = {'kw_419717': kw_419717, 'x0': keyword_419712, 'tol': keyword_419716, 'inner_m': keyword_419714}
    # Getting the type of 'lgmres' (line 36)
    lgmres_419701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 'lgmres', False)
    # Calling lgmres(args, kwargs) (line 36)
    lgmres_call_result_419719 = invoke(stypy.reporting.localization.Localization(__file__, 36, 15), lgmres_419701, *[A_419702, b_419703], **kwargs_419718)
    
    # Obtaining the member '__getitem__' of a type (line 36)
    getitem___419720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 4), lgmres_call_result_419719, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 36)
    subscript_call_result_419721 = invoke(stypy.reporting.localization.Localization(__file__, 36, 4), getitem___419720, int_419700)
    
    # Assigning a type to the variable 'tuple_var_assignment_419536' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'tuple_var_assignment_419536', subscript_call_result_419721)
    
    # Assigning a Name to a Name (line 36):
    # Getting the type of 'tuple_var_assignment_419535' (line 36)
    tuple_var_assignment_419535_419722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'tuple_var_assignment_419535')
    # Assigning a type to the variable 'x0' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'x0', tuple_var_assignment_419535_419722)
    
    # Assigning a Name to a Name (line 36):
    # Getting the type of 'tuple_var_assignment_419536' (line 36)
    tuple_var_assignment_419536_419723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'tuple_var_assignment_419536')
    # Assigning a type to the variable 'flag' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'flag', tuple_var_assignment_419536_419723)
    
    # Assigning a Subscript to a Name (line 37):
    
    # Assigning a Subscript to a Name (line 37):
    
    # Obtaining the type of the subscript
    int_419724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 20), 'int')
    # Getting the type of 'count' (line 37)
    count_419725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 14), 'count')
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___419726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 14), count_419725, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_419727 = invoke(stypy.reporting.localization.Localization(__file__, 37, 14), getitem___419726, int_419724)
    
    # Assigning a type to the variable 'count_0' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'count_0', subscript_call_result_419727)
    
    # Call to assert_(...): (line 38)
    # Processing the call arguments (line 38)
    
    # Call to allclose(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'A' (line 38)
    A_419730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 21), 'A', False)
    # Getting the type of 'x0' (line 38)
    x0_419731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 23), 'x0', False)
    # Applying the binary operator '*' (line 38)
    result_mul_419732 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 21), '*', A_419730, x0_419731)
    
    # Getting the type of 'b' (line 38)
    b_419733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 27), 'b', False)
    # Processing the call keyword arguments (line 38)
    float_419734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 35), 'float')
    keyword_419735 = float_419734
    float_419736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 47), 'float')
    keyword_419737 = float_419736
    kwargs_419738 = {'rtol': keyword_419735, 'atol': keyword_419737}
    # Getting the type of 'allclose' (line 38)
    allclose_419729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'allclose', False)
    # Calling allclose(args, kwargs) (line 38)
    allclose_call_result_419739 = invoke(stypy.reporting.localization.Localization(__file__, 38, 12), allclose_419729, *[result_mul_419732, b_419733], **kwargs_419738)
    
    
    # Call to norm(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'A' (line 38)
    A_419741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 60), 'A', False)
    # Getting the type of 'x0' (line 38)
    x0_419742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 62), 'x0', False)
    # Applying the binary operator '*' (line 38)
    result_mul_419743 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 60), '*', A_419741, x0_419742)
    
    # Getting the type of 'b' (line 38)
    b_419744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 65), 'b', False)
    # Applying the binary operator '-' (line 38)
    result_sub_419745 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 60), '-', result_mul_419743, b_419744)
    
    # Processing the call keyword arguments (line 38)
    kwargs_419746 = {}
    # Getting the type of 'norm' (line 38)
    norm_419740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 55), 'norm', False)
    # Calling norm(args, kwargs) (line 38)
    norm_call_result_419747 = invoke(stypy.reporting.localization.Localization(__file__, 38, 55), norm_419740, *[result_sub_419745], **kwargs_419746)
    
    # Processing the call keyword arguments (line 38)
    kwargs_419748 = {}
    # Getting the type of 'assert_' (line 38)
    assert__419728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 38)
    assert__call_result_419749 = invoke(stypy.reporting.localization.Localization(__file__, 38, 4), assert__419728, *[allclose_call_result_419739, norm_call_result_419747], **kwargs_419748)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 39)
    tuple_419750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 39)
    # Adding element type (line 39)
    # Getting the type of 'x0' (line 39)
    x0_419751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 11), 'x0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 11), tuple_419750, x0_419751)
    # Adding element type (line 39)
    # Getting the type of 'count_0' (line 39)
    count_0_419752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'count_0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 11), tuple_419750, count_0_419752)
    
    # Assigning a type to the variable 'stypy_return_type' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type', tuple_419750)
    
    # ################# End of 'do_solve(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'do_solve' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_419753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_419753)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'do_solve'
    return stypy_return_type_419753

# Assigning a type to the variable 'do_solve' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'do_solve', do_solve)
# Declaration of the 'TestLGMRES' class

class TestLGMRES(object, ):

    @norecursion
    def test_preconditioner(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_preconditioner'
        module_type_store = module_type_store.open_function_context('test_preconditioner', 43, 4, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLGMRES.test_preconditioner.__dict__.__setitem__('stypy_localization', localization)
        TestLGMRES.test_preconditioner.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLGMRES.test_preconditioner.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLGMRES.test_preconditioner.__dict__.__setitem__('stypy_function_name', 'TestLGMRES.test_preconditioner')
        TestLGMRES.test_preconditioner.__dict__.__setitem__('stypy_param_names_list', [])
        TestLGMRES.test_preconditioner.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLGMRES.test_preconditioner.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLGMRES.test_preconditioner.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLGMRES.test_preconditioner.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLGMRES.test_preconditioner.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLGMRES.test_preconditioner.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLGMRES.test_preconditioner', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_preconditioner', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_preconditioner(...)' code ##################

        
        # Assigning a Call to a Name (line 45):
        
        # Assigning a Call to a Name (line 45):
        
        # Call to splu(...): (line 45)
        # Processing the call arguments (line 45)
        
        # Call to tocsc(...): (line 45)
        # Processing the call keyword arguments (line 45)
        kwargs_419757 = {}
        # Getting the type of 'Am' (line 45)
        Am_419755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 18), 'Am', False)
        # Obtaining the member 'tocsc' of a type (line 45)
        tocsc_419756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 18), Am_419755, 'tocsc')
        # Calling tocsc(args, kwargs) (line 45)
        tocsc_call_result_419758 = invoke(stypy.reporting.localization.Localization(__file__, 45, 18), tocsc_419756, *[], **kwargs_419757)
        
        # Processing the call keyword arguments (line 45)
        kwargs_419759 = {}
        # Getting the type of 'splu' (line 45)
        splu_419754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 13), 'splu', False)
        # Calling splu(args, kwargs) (line 45)
        splu_call_result_419760 = invoke(stypy.reporting.localization.Localization(__file__, 45, 13), splu_419754, *[tocsc_call_result_419758], **kwargs_419759)
        
        # Assigning a type to the variable 'pc' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'pc', splu_call_result_419760)
        
        # Assigning a Call to a Name (line 46):
        
        # Assigning a Call to a Name (line 46):
        
        # Call to LinearOperator(...): (line 46)
        # Processing the call keyword arguments (line 46)
        # Getting the type of 'pc' (line 46)
        pc_419762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 34), 'pc', False)
        # Obtaining the member 'solve' of a type (line 46)
        solve_419763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 34), pc_419762, 'solve')
        keyword_419764 = solve_419763
        # Getting the type of 'A' (line 46)
        A_419765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 50), 'A', False)
        # Obtaining the member 'shape' of a type (line 46)
        shape_419766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 50), A_419765, 'shape')
        keyword_419767 = shape_419766
        # Getting the type of 'A' (line 46)
        A_419768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 65), 'A', False)
        # Obtaining the member 'dtype' of a type (line 46)
        dtype_419769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 65), A_419768, 'dtype')
        keyword_419770 = dtype_419769
        kwargs_419771 = {'dtype': keyword_419770, 'shape': keyword_419767, 'matvec': keyword_419764}
        # Getting the type of 'LinearOperator' (line 46)
        LinearOperator_419761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'LinearOperator', False)
        # Calling LinearOperator(args, kwargs) (line 46)
        LinearOperator_call_result_419772 = invoke(stypy.reporting.localization.Localization(__file__, 46, 12), LinearOperator_419761, *[], **kwargs_419771)
        
        # Assigning a type to the variable 'M' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'M', LinearOperator_call_result_419772)
        
        # Assigning a Call to a Tuple (line 48):
        
        # Assigning a Subscript to a Name (line 48):
        
        # Obtaining the type of the subscript
        int_419773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 8), 'int')
        
        # Call to do_solve(...): (line 48)
        # Processing the call keyword arguments (line 48)
        kwargs_419775 = {}
        # Getting the type of 'do_solve' (line 48)
        do_solve_419774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'do_solve', False)
        # Calling do_solve(args, kwargs) (line 48)
        do_solve_call_result_419776 = invoke(stypy.reporting.localization.Localization(__file__, 48, 22), do_solve_419774, *[], **kwargs_419775)
        
        # Obtaining the member '__getitem__' of a type (line 48)
        getitem___419777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), do_solve_call_result_419776, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 48)
        subscript_call_result_419778 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), getitem___419777, int_419773)
        
        # Assigning a type to the variable 'tuple_var_assignment_419537' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'tuple_var_assignment_419537', subscript_call_result_419778)
        
        # Assigning a Subscript to a Name (line 48):
        
        # Obtaining the type of the subscript
        int_419779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 8), 'int')
        
        # Call to do_solve(...): (line 48)
        # Processing the call keyword arguments (line 48)
        kwargs_419781 = {}
        # Getting the type of 'do_solve' (line 48)
        do_solve_419780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'do_solve', False)
        # Calling do_solve(args, kwargs) (line 48)
        do_solve_call_result_419782 = invoke(stypy.reporting.localization.Localization(__file__, 48, 22), do_solve_419780, *[], **kwargs_419781)
        
        # Obtaining the member '__getitem__' of a type (line 48)
        getitem___419783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), do_solve_call_result_419782, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 48)
        subscript_call_result_419784 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), getitem___419783, int_419779)
        
        # Assigning a type to the variable 'tuple_var_assignment_419538' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'tuple_var_assignment_419538', subscript_call_result_419784)
        
        # Assigning a Name to a Name (line 48):
        # Getting the type of 'tuple_var_assignment_419537' (line 48)
        tuple_var_assignment_419537_419785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'tuple_var_assignment_419537')
        # Assigning a type to the variable 'x0' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'x0', tuple_var_assignment_419537_419785)
        
        # Assigning a Name to a Name (line 48):
        # Getting the type of 'tuple_var_assignment_419538' (line 48)
        tuple_var_assignment_419538_419786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'tuple_var_assignment_419538')
        # Assigning a type to the variable 'count_0' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'count_0', tuple_var_assignment_419538_419786)
        
        # Assigning a Call to a Tuple (line 49):
        
        # Assigning a Subscript to a Name (line 49):
        
        # Obtaining the type of the subscript
        int_419787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 8), 'int')
        
        # Call to do_solve(...): (line 49)
        # Processing the call keyword arguments (line 49)
        # Getting the type of 'M' (line 49)
        M_419789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 33), 'M', False)
        keyword_419790 = M_419789
        kwargs_419791 = {'M': keyword_419790}
        # Getting the type of 'do_solve' (line 49)
        do_solve_419788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 22), 'do_solve', False)
        # Calling do_solve(args, kwargs) (line 49)
        do_solve_call_result_419792 = invoke(stypy.reporting.localization.Localization(__file__, 49, 22), do_solve_419788, *[], **kwargs_419791)
        
        # Obtaining the member '__getitem__' of a type (line 49)
        getitem___419793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), do_solve_call_result_419792, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
        subscript_call_result_419794 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), getitem___419793, int_419787)
        
        # Assigning a type to the variable 'tuple_var_assignment_419539' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_var_assignment_419539', subscript_call_result_419794)
        
        # Assigning a Subscript to a Name (line 49):
        
        # Obtaining the type of the subscript
        int_419795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 8), 'int')
        
        # Call to do_solve(...): (line 49)
        # Processing the call keyword arguments (line 49)
        # Getting the type of 'M' (line 49)
        M_419797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 33), 'M', False)
        keyword_419798 = M_419797
        kwargs_419799 = {'M': keyword_419798}
        # Getting the type of 'do_solve' (line 49)
        do_solve_419796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 22), 'do_solve', False)
        # Calling do_solve(args, kwargs) (line 49)
        do_solve_call_result_419800 = invoke(stypy.reporting.localization.Localization(__file__, 49, 22), do_solve_419796, *[], **kwargs_419799)
        
        # Obtaining the member '__getitem__' of a type (line 49)
        getitem___419801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), do_solve_call_result_419800, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
        subscript_call_result_419802 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), getitem___419801, int_419795)
        
        # Assigning a type to the variable 'tuple_var_assignment_419540' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_var_assignment_419540', subscript_call_result_419802)
        
        # Assigning a Name to a Name (line 49):
        # Getting the type of 'tuple_var_assignment_419539' (line 49)
        tuple_var_assignment_419539_419803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_var_assignment_419539')
        # Assigning a type to the variable 'x1' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'x1', tuple_var_assignment_419539_419803)
        
        # Assigning a Name to a Name (line 49):
        # Getting the type of 'tuple_var_assignment_419540' (line 49)
        tuple_var_assignment_419540_419804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_var_assignment_419540')
        # Assigning a type to the variable 'count_1' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'count_1', tuple_var_assignment_419540_419804)
        
        # Call to assert_(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Getting the type of 'count_1' (line 51)
        count_1_419806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'count_1', False)
        int_419807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 27), 'int')
        # Applying the binary operator '==' (line 51)
        result_eq_419808 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 16), '==', count_1_419806, int_419807)
        
        # Processing the call keyword arguments (line 51)
        kwargs_419809 = {}
        # Getting the type of 'assert_' (line 51)
        assert__419805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 51)
        assert__call_result_419810 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), assert__419805, *[result_eq_419808], **kwargs_419809)
        
        
        # Call to assert_(...): (line 52)
        # Processing the call arguments (line 52)
        
        # Getting the type of 'count_1' (line 52)
        count_1_419812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'count_1', False)
        # Getting the type of 'count_0' (line 52)
        count_0_419813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 26), 'count_0', False)
        int_419814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 34), 'int')
        # Applying the binary operator 'div' (line 52)
        result_div_419815 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 26), 'div', count_0_419813, int_419814)
        
        # Applying the binary operator '<' (line 52)
        result_lt_419816 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 16), '<', count_1_419812, result_div_419815)
        
        # Processing the call keyword arguments (line 52)
        kwargs_419817 = {}
        # Getting the type of 'assert_' (line 52)
        assert__419811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 52)
        assert__call_result_419818 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), assert__419811, *[result_lt_419816], **kwargs_419817)
        
        
        # Call to assert_(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Call to allclose(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'x1' (line 53)
        x1_419821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 25), 'x1', False)
        # Getting the type of 'x0' (line 53)
        x0_419822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 29), 'x0', False)
        # Processing the call keyword arguments (line 53)
        float_419823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 38), 'float')
        keyword_419824 = float_419823
        kwargs_419825 = {'rtol': keyword_419824}
        # Getting the type of 'allclose' (line 53)
        allclose_419820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 16), 'allclose', False)
        # Calling allclose(args, kwargs) (line 53)
        allclose_call_result_419826 = invoke(stypy.reporting.localization.Localization(__file__, 53, 16), allclose_419820, *[x1_419821, x0_419822], **kwargs_419825)
        
        # Processing the call keyword arguments (line 53)
        kwargs_419827 = {}
        # Getting the type of 'assert_' (line 53)
        assert__419819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 53)
        assert__call_result_419828 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), assert__419819, *[allclose_call_result_419826], **kwargs_419827)
        
        
        # ################# End of 'test_preconditioner(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_preconditioner' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_419829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_419829)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_preconditioner'
        return stypy_return_type_419829


    @norecursion
    def test_outer_v(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_outer_v'
        module_type_store = module_type_store.open_function_context('test_outer_v', 55, 4, False)
        # Assigning a type to the variable 'self' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLGMRES.test_outer_v.__dict__.__setitem__('stypy_localization', localization)
        TestLGMRES.test_outer_v.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLGMRES.test_outer_v.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLGMRES.test_outer_v.__dict__.__setitem__('stypy_function_name', 'TestLGMRES.test_outer_v')
        TestLGMRES.test_outer_v.__dict__.__setitem__('stypy_param_names_list', [])
        TestLGMRES.test_outer_v.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLGMRES.test_outer_v.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLGMRES.test_outer_v.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLGMRES.test_outer_v.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLGMRES.test_outer_v.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLGMRES.test_outer_v.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLGMRES.test_outer_v', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_outer_v', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_outer_v(...)' code ##################

        
        # Assigning a List to a Name (line 58):
        
        # Assigning a List to a Name (line 58):
        
        # Obtaining an instance of the builtin type 'list' (line 58)
        list_419830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 58)
        
        # Assigning a type to the variable 'outer_v' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'outer_v', list_419830)
        
        # Assigning a Call to a Tuple (line 59):
        
        # Assigning a Subscript to a Name (line 59):
        
        # Obtaining the type of the subscript
        int_419831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 8), 'int')
        
        # Call to do_solve(...): (line 59)
        # Processing the call keyword arguments (line 59)
        int_419833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 39), 'int')
        keyword_419834 = int_419833
        # Getting the type of 'outer_v' (line 59)
        outer_v_419835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 50), 'outer_v', False)
        keyword_419836 = outer_v_419835
        kwargs_419837 = {'outer_k': keyword_419834, 'outer_v': keyword_419836}
        # Getting the type of 'do_solve' (line 59)
        do_solve_419832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 22), 'do_solve', False)
        # Calling do_solve(args, kwargs) (line 59)
        do_solve_call_result_419838 = invoke(stypy.reporting.localization.Localization(__file__, 59, 22), do_solve_419832, *[], **kwargs_419837)
        
        # Obtaining the member '__getitem__' of a type (line 59)
        getitem___419839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), do_solve_call_result_419838, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 59)
        subscript_call_result_419840 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), getitem___419839, int_419831)
        
        # Assigning a type to the variable 'tuple_var_assignment_419541' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'tuple_var_assignment_419541', subscript_call_result_419840)
        
        # Assigning a Subscript to a Name (line 59):
        
        # Obtaining the type of the subscript
        int_419841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 8), 'int')
        
        # Call to do_solve(...): (line 59)
        # Processing the call keyword arguments (line 59)
        int_419843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 39), 'int')
        keyword_419844 = int_419843
        # Getting the type of 'outer_v' (line 59)
        outer_v_419845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 50), 'outer_v', False)
        keyword_419846 = outer_v_419845
        kwargs_419847 = {'outer_k': keyword_419844, 'outer_v': keyword_419846}
        # Getting the type of 'do_solve' (line 59)
        do_solve_419842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 22), 'do_solve', False)
        # Calling do_solve(args, kwargs) (line 59)
        do_solve_call_result_419848 = invoke(stypy.reporting.localization.Localization(__file__, 59, 22), do_solve_419842, *[], **kwargs_419847)
        
        # Obtaining the member '__getitem__' of a type (line 59)
        getitem___419849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), do_solve_call_result_419848, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 59)
        subscript_call_result_419850 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), getitem___419849, int_419841)
        
        # Assigning a type to the variable 'tuple_var_assignment_419542' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'tuple_var_assignment_419542', subscript_call_result_419850)
        
        # Assigning a Name to a Name (line 59):
        # Getting the type of 'tuple_var_assignment_419541' (line 59)
        tuple_var_assignment_419541_419851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'tuple_var_assignment_419541')
        # Assigning a type to the variable 'x0' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'x0', tuple_var_assignment_419541_419851)
        
        # Assigning a Name to a Name (line 59):
        # Getting the type of 'tuple_var_assignment_419542' (line 59)
        tuple_var_assignment_419542_419852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'tuple_var_assignment_419542')
        # Assigning a type to the variable 'count_0' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'count_0', tuple_var_assignment_419542_419852)
        
        # Call to assert_(...): (line 60)
        # Processing the call arguments (line 60)
        
        
        # Call to len(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'outer_v' (line 60)
        outer_v_419855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'outer_v', False)
        # Processing the call keyword arguments (line 60)
        kwargs_419856 = {}
        # Getting the type of 'len' (line 60)
        len_419854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'len', False)
        # Calling len(args, kwargs) (line 60)
        len_call_result_419857 = invoke(stypy.reporting.localization.Localization(__file__, 60, 16), len_419854, *[outer_v_419855], **kwargs_419856)
        
        int_419858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 31), 'int')
        # Applying the binary operator '>' (line 60)
        result_gt_419859 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 16), '>', len_call_result_419857, int_419858)
        
        # Processing the call keyword arguments (line 60)
        kwargs_419860 = {}
        # Getting the type of 'assert_' (line 60)
        assert__419853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 60)
        assert__call_result_419861 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), assert__419853, *[result_gt_419859], **kwargs_419860)
        
        
        # Call to assert_(...): (line 61)
        # Processing the call arguments (line 61)
        
        
        # Call to len(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'outer_v' (line 61)
        outer_v_419864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 'outer_v', False)
        # Processing the call keyword arguments (line 61)
        kwargs_419865 = {}
        # Getting the type of 'len' (line 61)
        len_419863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'len', False)
        # Calling len(args, kwargs) (line 61)
        len_call_result_419866 = invoke(stypy.reporting.localization.Localization(__file__, 61, 16), len_419863, *[outer_v_419864], **kwargs_419865)
        
        int_419867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 32), 'int')
        # Applying the binary operator '<=' (line 61)
        result_le_419868 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 16), '<=', len_call_result_419866, int_419867)
        
        # Processing the call keyword arguments (line 61)
        kwargs_419869 = {}
        # Getting the type of 'assert_' (line 61)
        assert__419862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 61)
        assert__call_result_419870 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), assert__419862, *[result_le_419868], **kwargs_419869)
        
        
        # Assigning a Call to a Tuple (line 63):
        
        # Assigning a Subscript to a Name (line 63):
        
        # Obtaining the type of the subscript
        int_419871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 8), 'int')
        
        # Call to do_solve(...): (line 63)
        # Processing the call keyword arguments (line 63)
        int_419873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 39), 'int')
        keyword_419874 = int_419873
        # Getting the type of 'outer_v' (line 63)
        outer_v_419875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 50), 'outer_v', False)
        keyword_419876 = outer_v_419875
        # Getting the type of 'True' (line 63)
        True_419877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 75), 'True', False)
        keyword_419878 = True_419877
        kwargs_419879 = {'outer_k': keyword_419874, 'prepend_outer_v': keyword_419878, 'outer_v': keyword_419876}
        # Getting the type of 'do_solve' (line 63)
        do_solve_419872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 22), 'do_solve', False)
        # Calling do_solve(args, kwargs) (line 63)
        do_solve_call_result_419880 = invoke(stypy.reporting.localization.Localization(__file__, 63, 22), do_solve_419872, *[], **kwargs_419879)
        
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___419881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), do_solve_call_result_419880, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_419882 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), getitem___419881, int_419871)
        
        # Assigning a type to the variable 'tuple_var_assignment_419543' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tuple_var_assignment_419543', subscript_call_result_419882)
        
        # Assigning a Subscript to a Name (line 63):
        
        # Obtaining the type of the subscript
        int_419883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 8), 'int')
        
        # Call to do_solve(...): (line 63)
        # Processing the call keyword arguments (line 63)
        int_419885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 39), 'int')
        keyword_419886 = int_419885
        # Getting the type of 'outer_v' (line 63)
        outer_v_419887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 50), 'outer_v', False)
        keyword_419888 = outer_v_419887
        # Getting the type of 'True' (line 63)
        True_419889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 75), 'True', False)
        keyword_419890 = True_419889
        kwargs_419891 = {'outer_k': keyword_419886, 'prepend_outer_v': keyword_419890, 'outer_v': keyword_419888}
        # Getting the type of 'do_solve' (line 63)
        do_solve_419884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 22), 'do_solve', False)
        # Calling do_solve(args, kwargs) (line 63)
        do_solve_call_result_419892 = invoke(stypy.reporting.localization.Localization(__file__, 63, 22), do_solve_419884, *[], **kwargs_419891)
        
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___419893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), do_solve_call_result_419892, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_419894 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), getitem___419893, int_419883)
        
        # Assigning a type to the variable 'tuple_var_assignment_419544' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tuple_var_assignment_419544', subscript_call_result_419894)
        
        # Assigning a Name to a Name (line 63):
        # Getting the type of 'tuple_var_assignment_419543' (line 63)
        tuple_var_assignment_419543_419895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tuple_var_assignment_419543')
        # Assigning a type to the variable 'x1' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'x1', tuple_var_assignment_419543_419895)
        
        # Assigning a Name to a Name (line 63):
        # Getting the type of 'tuple_var_assignment_419544' (line 63)
        tuple_var_assignment_419544_419896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tuple_var_assignment_419544')
        # Assigning a type to the variable 'count_1' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'count_1', tuple_var_assignment_419544_419896)
        
        # Call to assert_(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Getting the type of 'count_1' (line 64)
        count_1_419898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 16), 'count_1', False)
        int_419899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 27), 'int')
        # Applying the binary operator '==' (line 64)
        result_eq_419900 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 16), '==', count_1_419898, int_419899)
        
        # Getting the type of 'count_1' (line 64)
        count_1_419901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 30), 'count_1', False)
        # Processing the call keyword arguments (line 64)
        kwargs_419902 = {}
        # Getting the type of 'assert_' (line 64)
        assert__419897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 64)
        assert__call_result_419903 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), assert__419897, *[result_eq_419900, count_1_419901], **kwargs_419902)
        
        
        # Call to assert_(...): (line 65)
        # Processing the call arguments (line 65)
        
        # Getting the type of 'count_1' (line 65)
        count_1_419905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'count_1', False)
        # Getting the type of 'count_0' (line 65)
        count_0_419906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 26), 'count_0', False)
        int_419907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 34), 'int')
        # Applying the binary operator 'div' (line 65)
        result_div_419908 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 26), 'div', count_0_419906, int_419907)
        
        # Applying the binary operator '<' (line 65)
        result_lt_419909 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 16), '<', count_1_419905, result_div_419908)
        
        # Processing the call keyword arguments (line 65)
        kwargs_419910 = {}
        # Getting the type of 'assert_' (line 65)
        assert__419904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 65)
        assert__call_result_419911 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), assert__419904, *[result_lt_419909], **kwargs_419910)
        
        
        # Call to assert_(...): (line 66)
        # Processing the call arguments (line 66)
        
        # Call to allclose(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'x1' (line 66)
        x1_419914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'x1', False)
        # Getting the type of 'x0' (line 66)
        x0_419915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 29), 'x0', False)
        # Processing the call keyword arguments (line 66)
        float_419916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 38), 'float')
        keyword_419917 = float_419916
        kwargs_419918 = {'rtol': keyword_419917}
        # Getting the type of 'allclose' (line 66)
        allclose_419913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'allclose', False)
        # Calling allclose(args, kwargs) (line 66)
        allclose_call_result_419919 = invoke(stypy.reporting.localization.Localization(__file__, 66, 16), allclose_419913, *[x1_419914, x0_419915], **kwargs_419918)
        
        # Processing the call keyword arguments (line 66)
        kwargs_419920 = {}
        # Getting the type of 'assert_' (line 66)
        assert__419912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 66)
        assert__call_result_419921 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), assert__419912, *[allclose_call_result_419919], **kwargs_419920)
        
        
        # Assigning a List to a Name (line 70):
        
        # Assigning a List to a Name (line 70):
        
        # Obtaining an instance of the builtin type 'list' (line 70)
        list_419922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 70)
        
        # Assigning a type to the variable 'outer_v' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'outer_v', list_419922)
        
        # Assigning a Call to a Tuple (line 71):
        
        # Assigning a Subscript to a Name (line 71):
        
        # Obtaining the type of the subscript
        int_419923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 8), 'int')
        
        # Call to do_solve(...): (line 71)
        # Processing the call keyword arguments (line 71)
        int_419925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 39), 'int')
        keyword_419926 = int_419925
        # Getting the type of 'outer_v' (line 71)
        outer_v_419927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 50), 'outer_v', False)
        keyword_419928 = outer_v_419927
        # Getting the type of 'False' (line 71)
        False_419929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 74), 'False', False)
        keyword_419930 = False_419929
        kwargs_419931 = {'outer_k': keyword_419926, 'store_outer_Av': keyword_419930, 'outer_v': keyword_419928}
        # Getting the type of 'do_solve' (line 71)
        do_solve_419924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), 'do_solve', False)
        # Calling do_solve(args, kwargs) (line 71)
        do_solve_call_result_419932 = invoke(stypy.reporting.localization.Localization(__file__, 71, 22), do_solve_419924, *[], **kwargs_419931)
        
        # Obtaining the member '__getitem__' of a type (line 71)
        getitem___419933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), do_solve_call_result_419932, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 71)
        subscript_call_result_419934 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), getitem___419933, int_419923)
        
        # Assigning a type to the variable 'tuple_var_assignment_419545' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'tuple_var_assignment_419545', subscript_call_result_419934)
        
        # Assigning a Subscript to a Name (line 71):
        
        # Obtaining the type of the subscript
        int_419935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 8), 'int')
        
        # Call to do_solve(...): (line 71)
        # Processing the call keyword arguments (line 71)
        int_419937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 39), 'int')
        keyword_419938 = int_419937
        # Getting the type of 'outer_v' (line 71)
        outer_v_419939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 50), 'outer_v', False)
        keyword_419940 = outer_v_419939
        # Getting the type of 'False' (line 71)
        False_419941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 74), 'False', False)
        keyword_419942 = False_419941
        kwargs_419943 = {'outer_k': keyword_419938, 'store_outer_Av': keyword_419942, 'outer_v': keyword_419940}
        # Getting the type of 'do_solve' (line 71)
        do_solve_419936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), 'do_solve', False)
        # Calling do_solve(args, kwargs) (line 71)
        do_solve_call_result_419944 = invoke(stypy.reporting.localization.Localization(__file__, 71, 22), do_solve_419936, *[], **kwargs_419943)
        
        # Obtaining the member '__getitem__' of a type (line 71)
        getitem___419945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), do_solve_call_result_419944, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 71)
        subscript_call_result_419946 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), getitem___419945, int_419935)
        
        # Assigning a type to the variable 'tuple_var_assignment_419546' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'tuple_var_assignment_419546', subscript_call_result_419946)
        
        # Assigning a Name to a Name (line 71):
        # Getting the type of 'tuple_var_assignment_419545' (line 71)
        tuple_var_assignment_419545_419947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'tuple_var_assignment_419545')
        # Assigning a type to the variable 'x0' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'x0', tuple_var_assignment_419545_419947)
        
        # Assigning a Name to a Name (line 71):
        # Getting the type of 'tuple_var_assignment_419546' (line 71)
        tuple_var_assignment_419546_419948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'tuple_var_assignment_419546')
        # Assigning a type to the variable 'count_0' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'count_0', tuple_var_assignment_419546_419948)
        
        # Call to assert_(...): (line 72)
        # Processing the call arguments (line 72)
        
        # Call to all(...): (line 72)
        # Processing the call keyword arguments (line 72)
        kwargs_419963 = {}
        
        # Call to array(...): (line 72)
        # Processing the call arguments (line 72)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'outer_v' (line 72)
        outer_v_419957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 45), 'outer_v', False)
        comprehension_419958 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 23), outer_v_419957)
        # Assigning a type to the variable 'v' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 23), 'v', comprehension_419958)
        
        
        # Obtaining the type of the subscript
        int_419951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 25), 'int')
        # Getting the type of 'v' (line 72)
        v_419952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 23), 'v', False)
        # Obtaining the member '__getitem__' of a type (line 72)
        getitem___419953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 23), v_419952, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 72)
        subscript_call_result_419954 = invoke(stypy.reporting.localization.Localization(__file__, 72, 23), getitem___419953, int_419951)
        
        # Getting the type of 'None' (line 72)
        None_419955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 31), 'None', False)
        # Applying the binary operator 'is' (line 72)
        result_is__419956 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 23), 'is', subscript_call_result_419954, None_419955)
        
        list_419959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 23), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 23), list_419959, result_is__419956)
        # Processing the call keyword arguments (line 72)
        kwargs_419960 = {}
        # Getting the type of 'array' (line 72)
        array_419950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'array', False)
        # Calling array(args, kwargs) (line 72)
        array_call_result_419961 = invoke(stypy.reporting.localization.Localization(__file__, 72, 16), array_419950, *[list_419959], **kwargs_419960)
        
        # Obtaining the member 'all' of a type (line 72)
        all_419962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 16), array_call_result_419961, 'all')
        # Calling all(args, kwargs) (line 72)
        all_call_result_419964 = invoke(stypy.reporting.localization.Localization(__file__, 72, 16), all_419962, *[], **kwargs_419963)
        
        # Processing the call keyword arguments (line 72)
        kwargs_419965 = {}
        # Getting the type of 'assert_' (line 72)
        assert__419949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 72)
        assert__call_result_419966 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), assert__419949, *[all_call_result_419964], **kwargs_419965)
        
        
        # Call to assert_(...): (line 73)
        # Processing the call arguments (line 73)
        
        
        # Call to len(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'outer_v' (line 73)
        outer_v_419969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 20), 'outer_v', False)
        # Processing the call keyword arguments (line 73)
        kwargs_419970 = {}
        # Getting the type of 'len' (line 73)
        len_419968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'len', False)
        # Calling len(args, kwargs) (line 73)
        len_call_result_419971 = invoke(stypy.reporting.localization.Localization(__file__, 73, 16), len_419968, *[outer_v_419969], **kwargs_419970)
        
        int_419972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 31), 'int')
        # Applying the binary operator '>' (line 73)
        result_gt_419973 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 16), '>', len_call_result_419971, int_419972)
        
        # Processing the call keyword arguments (line 73)
        kwargs_419974 = {}
        # Getting the type of 'assert_' (line 73)
        assert__419967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 73)
        assert__call_result_419975 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), assert__419967, *[result_gt_419973], **kwargs_419974)
        
        
        # Call to assert_(...): (line 74)
        # Processing the call arguments (line 74)
        
        
        # Call to len(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'outer_v' (line 74)
        outer_v_419978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'outer_v', False)
        # Processing the call keyword arguments (line 74)
        kwargs_419979 = {}
        # Getting the type of 'len' (line 74)
        len_419977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 16), 'len', False)
        # Calling len(args, kwargs) (line 74)
        len_call_result_419980 = invoke(stypy.reporting.localization.Localization(__file__, 74, 16), len_419977, *[outer_v_419978], **kwargs_419979)
        
        int_419981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 32), 'int')
        # Applying the binary operator '<=' (line 74)
        result_le_419982 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 16), '<=', len_call_result_419980, int_419981)
        
        # Processing the call keyword arguments (line 74)
        kwargs_419983 = {}
        # Getting the type of 'assert_' (line 74)
        assert__419976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 74)
        assert__call_result_419984 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), assert__419976, *[result_le_419982], **kwargs_419983)
        
        
        # Assigning a Call to a Tuple (line 76):
        
        # Assigning a Subscript to a Name (line 76):
        
        # Obtaining the type of the subscript
        int_419985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 8), 'int')
        
        # Call to do_solve(...): (line 76)
        # Processing the call keyword arguments (line 76)
        int_419987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 39), 'int')
        keyword_419988 = int_419987
        # Getting the type of 'outer_v' (line 76)
        outer_v_419989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 50), 'outer_v', False)
        keyword_419990 = outer_v_419989
        # Getting the type of 'True' (line 76)
        True_419991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 75), 'True', False)
        keyword_419992 = True_419991
        kwargs_419993 = {'outer_k': keyword_419988, 'prepend_outer_v': keyword_419992, 'outer_v': keyword_419990}
        # Getting the type of 'do_solve' (line 76)
        do_solve_419986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 22), 'do_solve', False)
        # Calling do_solve(args, kwargs) (line 76)
        do_solve_call_result_419994 = invoke(stypy.reporting.localization.Localization(__file__, 76, 22), do_solve_419986, *[], **kwargs_419993)
        
        # Obtaining the member '__getitem__' of a type (line 76)
        getitem___419995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), do_solve_call_result_419994, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 76)
        subscript_call_result_419996 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), getitem___419995, int_419985)
        
        # Assigning a type to the variable 'tuple_var_assignment_419547' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'tuple_var_assignment_419547', subscript_call_result_419996)
        
        # Assigning a Subscript to a Name (line 76):
        
        # Obtaining the type of the subscript
        int_419997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 8), 'int')
        
        # Call to do_solve(...): (line 76)
        # Processing the call keyword arguments (line 76)
        int_419999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 39), 'int')
        keyword_420000 = int_419999
        # Getting the type of 'outer_v' (line 76)
        outer_v_420001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 50), 'outer_v', False)
        keyword_420002 = outer_v_420001
        # Getting the type of 'True' (line 76)
        True_420003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 75), 'True', False)
        keyword_420004 = True_420003
        kwargs_420005 = {'outer_k': keyword_420000, 'prepend_outer_v': keyword_420004, 'outer_v': keyword_420002}
        # Getting the type of 'do_solve' (line 76)
        do_solve_419998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 22), 'do_solve', False)
        # Calling do_solve(args, kwargs) (line 76)
        do_solve_call_result_420006 = invoke(stypy.reporting.localization.Localization(__file__, 76, 22), do_solve_419998, *[], **kwargs_420005)
        
        # Obtaining the member '__getitem__' of a type (line 76)
        getitem___420007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), do_solve_call_result_420006, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 76)
        subscript_call_result_420008 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), getitem___420007, int_419997)
        
        # Assigning a type to the variable 'tuple_var_assignment_419548' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'tuple_var_assignment_419548', subscript_call_result_420008)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'tuple_var_assignment_419547' (line 76)
        tuple_var_assignment_419547_420009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'tuple_var_assignment_419547')
        # Assigning a type to the variable 'x1' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'x1', tuple_var_assignment_419547_420009)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'tuple_var_assignment_419548' (line 76)
        tuple_var_assignment_419548_420010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'tuple_var_assignment_419548')
        # Assigning a type to the variable 'count_1' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'count_1', tuple_var_assignment_419548_420010)
        
        # Call to assert_(...): (line 77)
        # Processing the call arguments (line 77)
        
        # Getting the type of 'count_1' (line 77)
        count_1_420012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 'count_1', False)
        int_420013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 27), 'int')
        # Applying the binary operator '==' (line 77)
        result_eq_420014 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 16), '==', count_1_420012, int_420013)
        
        # Getting the type of 'count_1' (line 77)
        count_1_420015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 30), 'count_1', False)
        # Processing the call keyword arguments (line 77)
        kwargs_420016 = {}
        # Getting the type of 'assert_' (line 77)
        assert__420011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 77)
        assert__call_result_420017 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), assert__420011, *[result_eq_420014, count_1_420015], **kwargs_420016)
        
        
        # Call to assert_(...): (line 78)
        # Processing the call arguments (line 78)
        
        # Getting the type of 'count_1' (line 78)
        count_1_420019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'count_1', False)
        # Getting the type of 'count_0' (line 78)
        count_0_420020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 26), 'count_0', False)
        int_420021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 34), 'int')
        # Applying the binary operator 'div' (line 78)
        result_div_420022 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 26), 'div', count_0_420020, int_420021)
        
        # Applying the binary operator '<' (line 78)
        result_lt_420023 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 16), '<', count_1_420019, result_div_420022)
        
        # Processing the call keyword arguments (line 78)
        kwargs_420024 = {}
        # Getting the type of 'assert_' (line 78)
        assert__420018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 78)
        assert__call_result_420025 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), assert__420018, *[result_lt_420023], **kwargs_420024)
        
        
        # Call to assert_(...): (line 79)
        # Processing the call arguments (line 79)
        
        # Call to allclose(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'x1' (line 79)
        x1_420028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 25), 'x1', False)
        # Getting the type of 'x0' (line 79)
        x0_420029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 29), 'x0', False)
        # Processing the call keyword arguments (line 79)
        float_420030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 38), 'float')
        keyword_420031 = float_420030
        kwargs_420032 = {'rtol': keyword_420031}
        # Getting the type of 'allclose' (line 79)
        allclose_420027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 16), 'allclose', False)
        # Calling allclose(args, kwargs) (line 79)
        allclose_call_result_420033 = invoke(stypy.reporting.localization.Localization(__file__, 79, 16), allclose_420027, *[x1_420028, x0_420029], **kwargs_420032)
        
        # Processing the call keyword arguments (line 79)
        kwargs_420034 = {}
        # Getting the type of 'assert_' (line 79)
        assert__420026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 79)
        assert__call_result_420035 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), assert__420026, *[allclose_call_result_420033], **kwargs_420034)
        
        
        # ################# End of 'test_outer_v(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_outer_v' in the type store
        # Getting the type of 'stypy_return_type' (line 55)
        stypy_return_type_420036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_420036)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_outer_v'
        return stypy_return_type_420036


    @norecursion
    def test_arnoldi(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_arnoldi'
        module_type_store = module_type_store.open_function_context('test_arnoldi', 81, 4, False)
        # Assigning a type to the variable 'self' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLGMRES.test_arnoldi.__dict__.__setitem__('stypy_localization', localization)
        TestLGMRES.test_arnoldi.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLGMRES.test_arnoldi.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLGMRES.test_arnoldi.__dict__.__setitem__('stypy_function_name', 'TestLGMRES.test_arnoldi')
        TestLGMRES.test_arnoldi.__dict__.__setitem__('stypy_param_names_list', [])
        TestLGMRES.test_arnoldi.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLGMRES.test_arnoldi.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLGMRES.test_arnoldi.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLGMRES.test_arnoldi.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLGMRES.test_arnoldi.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLGMRES.test_arnoldi.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLGMRES.test_arnoldi', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_arnoldi', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_arnoldi(...)' code ##################

        
        # Call to rand(...): (line 82)
        # Processing the call arguments (line 82)
        int_420040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 23), 'int')
        # Processing the call keyword arguments (line 82)
        kwargs_420041 = {}
        # Getting the type of 'np' (line 82)
        np_420037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 82)
        random_420038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), np_420037, 'random')
        # Obtaining the member 'rand' of a type (line 82)
        rand_420039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), random_420038, 'rand')
        # Calling rand(args, kwargs) (line 82)
        rand_call_result_420042 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), rand_420039, *[int_420040], **kwargs_420041)
        
        
        # Assigning a BinOp to a Name (line 84):
        
        # Assigning a BinOp to a Name (line 84):
        
        # Call to eye(...): (line 84)
        # Processing the call arguments (line 84)
        int_420044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 16), 'int')
        # Processing the call keyword arguments (line 84)
        kwargs_420045 = {}
        # Getting the type of 'eye' (line 84)
        eye_420043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'eye', False)
        # Calling eye(args, kwargs) (line 84)
        eye_call_result_420046 = invoke(stypy.reporting.localization.Localization(__file__, 84, 12), eye_420043, *[int_420044], **kwargs_420045)
        
        
        # Call to rand(...): (line 84)
        # Processing the call arguments (line 84)
        int_420048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 30), 'int')
        int_420049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 36), 'int')
        # Processing the call keyword arguments (line 84)
        float_420050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 50), 'float')
        keyword_420051 = float_420050
        kwargs_420052 = {'density': keyword_420051}
        # Getting the type of 'rand' (line 84)
        rand_420047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 25), 'rand', False)
        # Calling rand(args, kwargs) (line 84)
        rand_call_result_420053 = invoke(stypy.reporting.localization.Localization(__file__, 84, 25), rand_420047, *[int_420048, int_420049], **kwargs_420052)
        
        # Applying the binary operator '+' (line 84)
        result_add_420054 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 12), '+', eye_call_result_420046, rand_call_result_420053)
        
        # Assigning a type to the variable 'A' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'A', result_add_420054)
        
        # Assigning a Call to a Name (line 85):
        
        # Assigning a Call to a Name (line 85):
        
        # Call to rand(...): (line 85)
        # Processing the call arguments (line 85)
        int_420058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 27), 'int')
        # Processing the call keyword arguments (line 85)
        kwargs_420059 = {}
        # Getting the type of 'np' (line 85)
        np_420055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 85)
        random_420056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 12), np_420055, 'random')
        # Obtaining the member 'rand' of a type (line 85)
        rand_420057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 12), random_420056, 'rand')
        # Calling rand(args, kwargs) (line 85)
        rand_call_result_420060 = invoke(stypy.reporting.localization.Localization(__file__, 85, 12), rand_420057, *[int_420058], **kwargs_420059)
        
        # Assigning a type to the variable 'b' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'b', rand_call_result_420060)
        
        # Assigning a Call to a Tuple (line 88):
        
        # Assigning a Subscript to a Name (line 88):
        
        # Obtaining the type of the subscript
        int_420061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 8), 'int')
        
        # Call to lgmres(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'A' (line 88)
        A_420063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 27), 'A', False)
        # Getting the type of 'b' (line 88)
        b_420064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 30), 'b', False)
        # Processing the call keyword arguments (line 88)
        
        # Call to zeros(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Obtaining the type of the subscript
        int_420066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 50), 'int')
        # Getting the type of 'A' (line 88)
        A_420067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 42), 'A', False)
        # Obtaining the member 'shape' of a type (line 88)
        shape_420068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 42), A_420067, 'shape')
        # Obtaining the member '__getitem__' of a type (line 88)
        getitem___420069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 42), shape_420068, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 88)
        subscript_call_result_420070 = invoke(stypy.reporting.localization.Localization(__file__, 88, 42), getitem___420069, int_420066)
        
        # Processing the call keyword arguments (line 88)
        kwargs_420071 = {}
        # Getting the type of 'zeros' (line 88)
        zeros_420065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 36), 'zeros', False)
        # Calling zeros(args, kwargs) (line 88)
        zeros_call_result_420072 = invoke(stypy.reporting.localization.Localization(__file__, 88, 36), zeros_420065, *[subscript_call_result_420070], **kwargs_420071)
        
        keyword_420073 = zeros_call_result_420072
        int_420074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 63), 'int')
        keyword_420075 = int_420074
        int_420076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 75), 'int')
        keyword_420077 = int_420076
        kwargs_420078 = {'x0': keyword_420073, 'maxiter': keyword_420077, 'inner_m': keyword_420075}
        # Getting the type of 'lgmres' (line 88)
        lgmres_420062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 20), 'lgmres', False)
        # Calling lgmres(args, kwargs) (line 88)
        lgmres_call_result_420079 = invoke(stypy.reporting.localization.Localization(__file__, 88, 20), lgmres_420062, *[A_420063, b_420064], **kwargs_420078)
        
        # Obtaining the member '__getitem__' of a type (line 88)
        getitem___420080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), lgmres_call_result_420079, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 88)
        subscript_call_result_420081 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), getitem___420080, int_420061)
        
        # Assigning a type to the variable 'tuple_var_assignment_419549' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'tuple_var_assignment_419549', subscript_call_result_420081)
        
        # Assigning a Subscript to a Name (line 88):
        
        # Obtaining the type of the subscript
        int_420082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 8), 'int')
        
        # Call to lgmres(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'A' (line 88)
        A_420084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 27), 'A', False)
        # Getting the type of 'b' (line 88)
        b_420085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 30), 'b', False)
        # Processing the call keyword arguments (line 88)
        
        # Call to zeros(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Obtaining the type of the subscript
        int_420087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 50), 'int')
        # Getting the type of 'A' (line 88)
        A_420088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 42), 'A', False)
        # Obtaining the member 'shape' of a type (line 88)
        shape_420089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 42), A_420088, 'shape')
        # Obtaining the member '__getitem__' of a type (line 88)
        getitem___420090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 42), shape_420089, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 88)
        subscript_call_result_420091 = invoke(stypy.reporting.localization.Localization(__file__, 88, 42), getitem___420090, int_420087)
        
        # Processing the call keyword arguments (line 88)
        kwargs_420092 = {}
        # Getting the type of 'zeros' (line 88)
        zeros_420086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 36), 'zeros', False)
        # Calling zeros(args, kwargs) (line 88)
        zeros_call_result_420093 = invoke(stypy.reporting.localization.Localization(__file__, 88, 36), zeros_420086, *[subscript_call_result_420091], **kwargs_420092)
        
        keyword_420094 = zeros_call_result_420093
        int_420095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 63), 'int')
        keyword_420096 = int_420095
        int_420097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 75), 'int')
        keyword_420098 = int_420097
        kwargs_420099 = {'x0': keyword_420094, 'maxiter': keyword_420098, 'inner_m': keyword_420096}
        # Getting the type of 'lgmres' (line 88)
        lgmres_420083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 20), 'lgmres', False)
        # Calling lgmres(args, kwargs) (line 88)
        lgmres_call_result_420100 = invoke(stypy.reporting.localization.Localization(__file__, 88, 20), lgmres_420083, *[A_420084, b_420085], **kwargs_420099)
        
        # Obtaining the member '__getitem__' of a type (line 88)
        getitem___420101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), lgmres_call_result_420100, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 88)
        subscript_call_result_420102 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), getitem___420101, int_420082)
        
        # Assigning a type to the variable 'tuple_var_assignment_419550' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'tuple_var_assignment_419550', subscript_call_result_420102)
        
        # Assigning a Name to a Name (line 88):
        # Getting the type of 'tuple_var_assignment_419549' (line 88)
        tuple_var_assignment_419549_420103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'tuple_var_assignment_419549')
        # Assigning a type to the variable 'x0' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'x0', tuple_var_assignment_419549_420103)
        
        # Assigning a Name to a Name (line 88):
        # Getting the type of 'tuple_var_assignment_419550' (line 88)
        tuple_var_assignment_419550_420104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'tuple_var_assignment_419550')
        # Assigning a type to the variable 'flag0' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'flag0', tuple_var_assignment_419550_420104)
        
        # Assigning a Call to a Tuple (line 89):
        
        # Assigning a Subscript to a Name (line 89):
        
        # Obtaining the type of the subscript
        int_420105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 8), 'int')
        
        # Call to gmres(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'A' (line 89)
        A_420107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), 'A', False)
        # Getting the type of 'b' (line 89)
        b_420108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 29), 'b', False)
        # Processing the call keyword arguments (line 89)
        
        # Call to zeros(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Obtaining the type of the subscript
        int_420110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 49), 'int')
        # Getting the type of 'A' (line 89)
        A_420111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 41), 'A', False)
        # Obtaining the member 'shape' of a type (line 89)
        shape_420112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 41), A_420111, 'shape')
        # Obtaining the member '__getitem__' of a type (line 89)
        getitem___420113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 41), shape_420112, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 89)
        subscript_call_result_420114 = invoke(stypy.reporting.localization.Localization(__file__, 89, 41), getitem___420113, int_420110)
        
        # Processing the call keyword arguments (line 89)
        kwargs_420115 = {}
        # Getting the type of 'zeros' (line 89)
        zeros_420109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 35), 'zeros', False)
        # Calling zeros(args, kwargs) (line 89)
        zeros_call_result_420116 = invoke(stypy.reporting.localization.Localization(__file__, 89, 35), zeros_420109, *[subscript_call_result_420114], **kwargs_420115)
        
        keyword_420117 = zeros_call_result_420116
        int_420118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 62), 'int')
        keyword_420119 = int_420118
        int_420120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 74), 'int')
        keyword_420121 = int_420120
        kwargs_420122 = {'x0': keyword_420117, 'restart': keyword_420119, 'maxiter': keyword_420121}
        # Getting the type of 'gmres' (line 89)
        gmres_420106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 20), 'gmres', False)
        # Calling gmres(args, kwargs) (line 89)
        gmres_call_result_420123 = invoke(stypy.reporting.localization.Localization(__file__, 89, 20), gmres_420106, *[A_420107, b_420108], **kwargs_420122)
        
        # Obtaining the member '__getitem__' of a type (line 89)
        getitem___420124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), gmres_call_result_420123, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 89)
        subscript_call_result_420125 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), getitem___420124, int_420105)
        
        # Assigning a type to the variable 'tuple_var_assignment_419551' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'tuple_var_assignment_419551', subscript_call_result_420125)
        
        # Assigning a Subscript to a Name (line 89):
        
        # Obtaining the type of the subscript
        int_420126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 8), 'int')
        
        # Call to gmres(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'A' (line 89)
        A_420128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), 'A', False)
        # Getting the type of 'b' (line 89)
        b_420129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 29), 'b', False)
        # Processing the call keyword arguments (line 89)
        
        # Call to zeros(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Obtaining the type of the subscript
        int_420131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 49), 'int')
        # Getting the type of 'A' (line 89)
        A_420132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 41), 'A', False)
        # Obtaining the member 'shape' of a type (line 89)
        shape_420133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 41), A_420132, 'shape')
        # Obtaining the member '__getitem__' of a type (line 89)
        getitem___420134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 41), shape_420133, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 89)
        subscript_call_result_420135 = invoke(stypy.reporting.localization.Localization(__file__, 89, 41), getitem___420134, int_420131)
        
        # Processing the call keyword arguments (line 89)
        kwargs_420136 = {}
        # Getting the type of 'zeros' (line 89)
        zeros_420130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 35), 'zeros', False)
        # Calling zeros(args, kwargs) (line 89)
        zeros_call_result_420137 = invoke(stypy.reporting.localization.Localization(__file__, 89, 35), zeros_420130, *[subscript_call_result_420135], **kwargs_420136)
        
        keyword_420138 = zeros_call_result_420137
        int_420139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 62), 'int')
        keyword_420140 = int_420139
        int_420141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 74), 'int')
        keyword_420142 = int_420141
        kwargs_420143 = {'x0': keyword_420138, 'restart': keyword_420140, 'maxiter': keyword_420142}
        # Getting the type of 'gmres' (line 89)
        gmres_420127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 20), 'gmres', False)
        # Calling gmres(args, kwargs) (line 89)
        gmres_call_result_420144 = invoke(stypy.reporting.localization.Localization(__file__, 89, 20), gmres_420127, *[A_420128, b_420129], **kwargs_420143)
        
        # Obtaining the member '__getitem__' of a type (line 89)
        getitem___420145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), gmres_call_result_420144, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 89)
        subscript_call_result_420146 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), getitem___420145, int_420126)
        
        # Assigning a type to the variable 'tuple_var_assignment_419552' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'tuple_var_assignment_419552', subscript_call_result_420146)
        
        # Assigning a Name to a Name (line 89):
        # Getting the type of 'tuple_var_assignment_419551' (line 89)
        tuple_var_assignment_419551_420147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'tuple_var_assignment_419551')
        # Assigning a type to the variable 'x1' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'x1', tuple_var_assignment_419551_420147)
        
        # Assigning a Name to a Name (line 89):
        # Getting the type of 'tuple_var_assignment_419552' (line 89)
        tuple_var_assignment_419552_420148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'tuple_var_assignment_419552')
        # Assigning a type to the variable 'flag1' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'flag1', tuple_var_assignment_419552_420148)
        
        # Call to assert_equal(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'flag0' (line 91)
        flag0_420150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 21), 'flag0', False)
        int_420151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 28), 'int')
        # Processing the call keyword arguments (line 91)
        kwargs_420152 = {}
        # Getting the type of 'assert_equal' (line 91)
        assert_equal_420149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 91)
        assert_equal_call_result_420153 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), assert_equal_420149, *[flag0_420150, int_420151], **kwargs_420152)
        
        
        # Call to assert_equal(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'flag1' (line 92)
        flag1_420155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 21), 'flag1', False)
        int_420156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 28), 'int')
        # Processing the call keyword arguments (line 92)
        kwargs_420157 = {}
        # Getting the type of 'assert_equal' (line 92)
        assert_equal_420154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 92)
        assert_equal_call_result_420158 = invoke(stypy.reporting.localization.Localization(__file__, 92, 8), assert_equal_420154, *[flag1_420155, int_420156], **kwargs_420157)
        
        
        # Call to assert_(...): (line 93)
        # Processing the call arguments (line 93)
        
        
        # Call to norm(...): (line 93)
        # Processing the call arguments (line 93)
        
        # Call to dot(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'x0' (line 93)
        x0_420165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 37), 'x0', False)
        # Processing the call keyword arguments (line 93)
        kwargs_420166 = {}
        # Getting the type of 'A' (line 93)
        A_420163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 31), 'A', False)
        # Obtaining the member 'dot' of a type (line 93)
        dot_420164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 31), A_420163, 'dot')
        # Calling dot(args, kwargs) (line 93)
        dot_call_result_420167 = invoke(stypy.reporting.localization.Localization(__file__, 93, 31), dot_420164, *[x0_420165], **kwargs_420166)
        
        # Getting the type of 'b' (line 93)
        b_420168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 43), 'b', False)
        # Applying the binary operator '-' (line 93)
        result_sub_420169 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 31), '-', dot_call_result_420167, b_420168)
        
        # Processing the call keyword arguments (line 93)
        kwargs_420170 = {}
        # Getting the type of 'np' (line 93)
        np_420160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'np', False)
        # Obtaining the member 'linalg' of a type (line 93)
        linalg_420161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 16), np_420160, 'linalg')
        # Obtaining the member 'norm' of a type (line 93)
        norm_420162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 16), linalg_420161, 'norm')
        # Calling norm(args, kwargs) (line 93)
        norm_call_result_420171 = invoke(stypy.reporting.localization.Localization(__file__, 93, 16), norm_420162, *[result_sub_420169], **kwargs_420170)
        
        float_420172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 48), 'float')
        # Applying the binary operator '>' (line 93)
        result_gt_420173 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 16), '>', norm_call_result_420171, float_420172)
        
        # Processing the call keyword arguments (line 93)
        kwargs_420174 = {}
        # Getting the type of 'assert_' (line 93)
        assert__420159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 93)
        assert__call_result_420175 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), assert__420159, *[result_gt_420173], **kwargs_420174)
        
        
        # Call to assert_allclose(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'x0' (line 95)
        x0_420177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 24), 'x0', False)
        # Getting the type of 'x1' (line 95)
        x1_420178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 28), 'x1', False)
        # Processing the call keyword arguments (line 95)
        kwargs_420179 = {}
        # Getting the type of 'assert_allclose' (line 95)
        assert_allclose_420176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 95)
        assert_allclose_call_result_420180 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), assert_allclose_420176, *[x0_420177, x1_420178], **kwargs_420179)
        
        
        # ################# End of 'test_arnoldi(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_arnoldi' in the type store
        # Getting the type of 'stypy_return_type' (line 81)
        stypy_return_type_420181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_420181)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_arnoldi'
        return stypy_return_type_420181


    @norecursion
    def test_cornercase(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_cornercase'
        module_type_store = module_type_store.open_function_context('test_cornercase', 97, 4, False)
        # Assigning a type to the variable 'self' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLGMRES.test_cornercase.__dict__.__setitem__('stypy_localization', localization)
        TestLGMRES.test_cornercase.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLGMRES.test_cornercase.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLGMRES.test_cornercase.__dict__.__setitem__('stypy_function_name', 'TestLGMRES.test_cornercase')
        TestLGMRES.test_cornercase.__dict__.__setitem__('stypy_param_names_list', [])
        TestLGMRES.test_cornercase.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLGMRES.test_cornercase.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLGMRES.test_cornercase.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLGMRES.test_cornercase.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLGMRES.test_cornercase.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLGMRES.test_cornercase.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLGMRES.test_cornercase', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_cornercase', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_cornercase(...)' code ##################

        
        # Call to seed(...): (line 98)
        # Processing the call arguments (line 98)
        int_420185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 23), 'int')
        # Processing the call keyword arguments (line 98)
        kwargs_420186 = {}
        # Getting the type of 'np' (line 98)
        np_420182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 98)
        random_420183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), np_420182, 'random')
        # Obtaining the member 'seed' of a type (line 98)
        seed_420184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), random_420183, 'seed')
        # Calling seed(args, kwargs) (line 98)
        seed_call_result_420187 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), seed_420184, *[int_420185], **kwargs_420186)
        
        
        
        # Obtaining an instance of the builtin type 'list' (line 104)
        list_420188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 104)
        # Adding element type (line 104)
        int_420189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 17), list_420188, int_420189)
        # Adding element type (line 104)
        int_420190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 17), list_420188, int_420190)
        # Adding element type (line 104)
        int_420191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 17), list_420188, int_420191)
        # Adding element type (line 104)
        int_420192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 17), list_420188, int_420192)
        
        # Testing the type of a for loop iterable (line 104)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 104, 8), list_420188)
        # Getting the type of the for loop variable (line 104)
        for_loop_var_420193 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 104, 8), list_420188)
        # Assigning a type to the variable 'n' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'n', for_loop_var_420193)
        # SSA begins for a for statement (line 104)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 105):
        
        # Assigning a BinOp to a Name (line 105):
        int_420194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 16), 'int')
        
        # Call to eye(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'n' (line 105)
        n_420196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 22), 'n', False)
        # Processing the call keyword arguments (line 105)
        kwargs_420197 = {}
        # Getting the type of 'eye' (line 105)
        eye_420195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 18), 'eye', False)
        # Calling eye(args, kwargs) (line 105)
        eye_call_result_420198 = invoke(stypy.reporting.localization.Localization(__file__, 105, 18), eye_420195, *[n_420196], **kwargs_420197)
        
        # Applying the binary operator '*' (line 105)
        result_mul_420199 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 16), '*', int_420194, eye_call_result_420198)
        
        # Assigning a type to the variable 'A' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'A', result_mul_420199)
        
        # Assigning a Call to a Name (line 107):
        
        # Assigning a Call to a Name (line 107):
        
        # Call to ones(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'n' (line 107)
        n_420202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 24), 'n', False)
        # Processing the call keyword arguments (line 107)
        kwargs_420203 = {}
        # Getting the type of 'np' (line 107)
        np_420200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'np', False)
        # Obtaining the member 'ones' of a type (line 107)
        ones_420201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 16), np_420200, 'ones')
        # Calling ones(args, kwargs) (line 107)
        ones_call_result_420204 = invoke(stypy.reporting.localization.Localization(__file__, 107, 16), ones_420201, *[n_420202], **kwargs_420203)
        
        # Assigning a type to the variable 'b' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'b', ones_call_result_420204)
        
        # Assigning a Call to a Tuple (line 108):
        
        # Assigning a Subscript to a Name (line 108):
        
        # Obtaining the type of the subscript
        int_420205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 12), 'int')
        
        # Call to lgmres(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'A' (line 108)
        A_420207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 'A', False)
        # Getting the type of 'b' (line 108)
        b_420208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 32), 'b', False)
        # Processing the call keyword arguments (line 108)
        int_420209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 43), 'int')
        keyword_420210 = int_420209
        kwargs_420211 = {'maxiter': keyword_420210}
        # Getting the type of 'lgmres' (line 108)
        lgmres_420206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 22), 'lgmres', False)
        # Calling lgmres(args, kwargs) (line 108)
        lgmres_call_result_420212 = invoke(stypy.reporting.localization.Localization(__file__, 108, 22), lgmres_420206, *[A_420207, b_420208], **kwargs_420211)
        
        # Obtaining the member '__getitem__' of a type (line 108)
        getitem___420213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 12), lgmres_call_result_420212, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 108)
        subscript_call_result_420214 = invoke(stypy.reporting.localization.Localization(__file__, 108, 12), getitem___420213, int_420205)
        
        # Assigning a type to the variable 'tuple_var_assignment_419553' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'tuple_var_assignment_419553', subscript_call_result_420214)
        
        # Assigning a Subscript to a Name (line 108):
        
        # Obtaining the type of the subscript
        int_420215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 12), 'int')
        
        # Call to lgmres(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'A' (line 108)
        A_420217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 'A', False)
        # Getting the type of 'b' (line 108)
        b_420218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 32), 'b', False)
        # Processing the call keyword arguments (line 108)
        int_420219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 43), 'int')
        keyword_420220 = int_420219
        kwargs_420221 = {'maxiter': keyword_420220}
        # Getting the type of 'lgmres' (line 108)
        lgmres_420216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 22), 'lgmres', False)
        # Calling lgmres(args, kwargs) (line 108)
        lgmres_call_result_420222 = invoke(stypy.reporting.localization.Localization(__file__, 108, 22), lgmres_420216, *[A_420217, b_420218], **kwargs_420221)
        
        # Obtaining the member '__getitem__' of a type (line 108)
        getitem___420223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 12), lgmres_call_result_420222, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 108)
        subscript_call_result_420224 = invoke(stypy.reporting.localization.Localization(__file__, 108, 12), getitem___420223, int_420215)
        
        # Assigning a type to the variable 'tuple_var_assignment_419554' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'tuple_var_assignment_419554', subscript_call_result_420224)
        
        # Assigning a Name to a Name (line 108):
        # Getting the type of 'tuple_var_assignment_419553' (line 108)
        tuple_var_assignment_419553_420225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'tuple_var_assignment_419553')
        # Assigning a type to the variable 'x' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'x', tuple_var_assignment_419553_420225)
        
        # Assigning a Name to a Name (line 108):
        # Getting the type of 'tuple_var_assignment_419554' (line 108)
        tuple_var_assignment_419554_420226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'tuple_var_assignment_419554')
        # Assigning a type to the variable 'info' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 'info', tuple_var_assignment_419554_420226)
        
        # Call to assert_equal(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'info' (line 109)
        info_420228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 25), 'info', False)
        int_420229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 31), 'int')
        # Processing the call keyword arguments (line 109)
        kwargs_420230 = {}
        # Getting the type of 'assert_equal' (line 109)
        assert_equal_420227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 109)
        assert_equal_call_result_420231 = invoke(stypy.reporting.localization.Localization(__file__, 109, 12), assert_equal_420227, *[info_420228, int_420229], **kwargs_420230)
        
        
        # Call to assert_allclose(...): (line 110)
        # Processing the call arguments (line 110)
        
        # Call to dot(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'x' (line 110)
        x_420235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 34), 'x', False)
        # Processing the call keyword arguments (line 110)
        kwargs_420236 = {}
        # Getting the type of 'A' (line 110)
        A_420233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 28), 'A', False)
        # Obtaining the member 'dot' of a type (line 110)
        dot_420234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 28), A_420233, 'dot')
        # Calling dot(args, kwargs) (line 110)
        dot_call_result_420237 = invoke(stypy.reporting.localization.Localization(__file__, 110, 28), dot_420234, *[x_420235], **kwargs_420236)
        
        # Getting the type of 'b' (line 110)
        b_420238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 39), 'b', False)
        # Applying the binary operator '-' (line 110)
        result_sub_420239 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 28), '-', dot_call_result_420237, b_420238)
        
        int_420240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 42), 'int')
        # Processing the call keyword arguments (line 110)
        float_420241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 50), 'float')
        keyword_420242 = float_420241
        kwargs_420243 = {'atol': keyword_420242}
        # Getting the type of 'assert_allclose' (line 110)
        assert_allclose_420232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 110)
        assert_allclose_call_result_420244 = invoke(stypy.reporting.localization.Localization(__file__, 110, 12), assert_allclose_420232, *[result_sub_420239, int_420240], **kwargs_420243)
        
        
        # Assigning a Call to a Tuple (line 112):
        
        # Assigning a Subscript to a Name (line 112):
        
        # Obtaining the type of the subscript
        int_420245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 12), 'int')
        
        # Call to lgmres(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'A' (line 112)
        A_420247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 29), 'A', False)
        # Getting the type of 'b' (line 112)
        b_420248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 32), 'b', False)
        # Processing the call keyword arguments (line 112)
        int_420249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 39), 'int')
        keyword_420250 = int_420249
        int_420251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 50), 'int')
        keyword_420252 = int_420251
        kwargs_420253 = {'tol': keyword_420250, 'maxiter': keyword_420252}
        # Getting the type of 'lgmres' (line 112)
        lgmres_420246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 22), 'lgmres', False)
        # Calling lgmres(args, kwargs) (line 112)
        lgmres_call_result_420254 = invoke(stypy.reporting.localization.Localization(__file__, 112, 22), lgmres_420246, *[A_420247, b_420248], **kwargs_420253)
        
        # Obtaining the member '__getitem__' of a type (line 112)
        getitem___420255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 12), lgmres_call_result_420254, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 112)
        subscript_call_result_420256 = invoke(stypy.reporting.localization.Localization(__file__, 112, 12), getitem___420255, int_420245)
        
        # Assigning a type to the variable 'tuple_var_assignment_419555' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'tuple_var_assignment_419555', subscript_call_result_420256)
        
        # Assigning a Subscript to a Name (line 112):
        
        # Obtaining the type of the subscript
        int_420257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 12), 'int')
        
        # Call to lgmres(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'A' (line 112)
        A_420259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 29), 'A', False)
        # Getting the type of 'b' (line 112)
        b_420260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 32), 'b', False)
        # Processing the call keyword arguments (line 112)
        int_420261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 39), 'int')
        keyword_420262 = int_420261
        int_420263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 50), 'int')
        keyword_420264 = int_420263
        kwargs_420265 = {'tol': keyword_420262, 'maxiter': keyword_420264}
        # Getting the type of 'lgmres' (line 112)
        lgmres_420258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 22), 'lgmres', False)
        # Calling lgmres(args, kwargs) (line 112)
        lgmres_call_result_420266 = invoke(stypy.reporting.localization.Localization(__file__, 112, 22), lgmres_420258, *[A_420259, b_420260], **kwargs_420265)
        
        # Obtaining the member '__getitem__' of a type (line 112)
        getitem___420267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 12), lgmres_call_result_420266, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 112)
        subscript_call_result_420268 = invoke(stypy.reporting.localization.Localization(__file__, 112, 12), getitem___420267, int_420257)
        
        # Assigning a type to the variable 'tuple_var_assignment_419556' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'tuple_var_assignment_419556', subscript_call_result_420268)
        
        # Assigning a Name to a Name (line 112):
        # Getting the type of 'tuple_var_assignment_419555' (line 112)
        tuple_var_assignment_419555_420269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'tuple_var_assignment_419555')
        # Assigning a type to the variable 'x' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'x', tuple_var_assignment_419555_420269)
        
        # Assigning a Name to a Name (line 112):
        # Getting the type of 'tuple_var_assignment_419556' (line 112)
        tuple_var_assignment_419556_420270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'tuple_var_assignment_419556')
        # Assigning a type to the variable 'info' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 15), 'info', tuple_var_assignment_419556_420270)
        
        
        # Getting the type of 'info' (line 113)
        info_420271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 15), 'info')
        int_420272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 23), 'int')
        # Applying the binary operator '==' (line 113)
        result_eq_420273 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 15), '==', info_420271, int_420272)
        
        # Testing the type of an if condition (line 113)
        if_condition_420274 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 12), result_eq_420273)
        # Assigning a type to the variable 'if_condition_420274' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'if_condition_420274', if_condition_420274)
        # SSA begins for if statement (line 113)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_allclose(...): (line 114)
        # Processing the call arguments (line 114)
        
        # Call to dot(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'x' (line 114)
        x_420278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 38), 'x', False)
        # Processing the call keyword arguments (line 114)
        kwargs_420279 = {}
        # Getting the type of 'A' (line 114)
        A_420276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 32), 'A', False)
        # Obtaining the member 'dot' of a type (line 114)
        dot_420277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 32), A_420276, 'dot')
        # Calling dot(args, kwargs) (line 114)
        dot_call_result_420280 = invoke(stypy.reporting.localization.Localization(__file__, 114, 32), dot_420277, *[x_420278], **kwargs_420279)
        
        # Getting the type of 'b' (line 114)
        b_420281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 43), 'b', False)
        # Applying the binary operator '-' (line 114)
        result_sub_420282 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 32), '-', dot_call_result_420280, b_420281)
        
        int_420283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 46), 'int')
        # Processing the call keyword arguments (line 114)
        float_420284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 54), 'float')
        keyword_420285 = float_420284
        kwargs_420286 = {'atol': keyword_420285}
        # Getting the type of 'assert_allclose' (line 114)
        assert_allclose_420275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 114)
        assert_allclose_call_result_420287 = invoke(stypy.reporting.localization.Localization(__file__, 114, 16), assert_allclose_420275, *[result_sub_420282, int_420283], **kwargs_420286)
        
        # SSA join for if statement (line 113)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 116):
        
        # Assigning a Call to a Name (line 116):
        
        # Call to rand(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'n' (line 116)
        n_420291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 31), 'n', False)
        # Processing the call keyword arguments (line 116)
        kwargs_420292 = {}
        # Getting the type of 'np' (line 116)
        np_420288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'np', False)
        # Obtaining the member 'random' of a type (line 116)
        random_420289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 16), np_420288, 'random')
        # Obtaining the member 'rand' of a type (line 116)
        rand_420290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 16), random_420289, 'rand')
        # Calling rand(args, kwargs) (line 116)
        rand_call_result_420293 = invoke(stypy.reporting.localization.Localization(__file__, 116, 16), rand_420290, *[n_420291], **kwargs_420292)
        
        # Assigning a type to the variable 'b' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'b', rand_call_result_420293)
        
        # Assigning a Call to a Tuple (line 117):
        
        # Assigning a Subscript to a Name (line 117):
        
        # Obtaining the type of the subscript
        int_420294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 12), 'int')
        
        # Call to lgmres(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'A' (line 117)
        A_420296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 29), 'A', False)
        # Getting the type of 'b' (line 117)
        b_420297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 32), 'b', False)
        # Processing the call keyword arguments (line 117)
        int_420298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 43), 'int')
        keyword_420299 = int_420298
        kwargs_420300 = {'maxiter': keyword_420299}
        # Getting the type of 'lgmres' (line 117)
        lgmres_420295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 22), 'lgmres', False)
        # Calling lgmres(args, kwargs) (line 117)
        lgmres_call_result_420301 = invoke(stypy.reporting.localization.Localization(__file__, 117, 22), lgmres_420295, *[A_420296, b_420297], **kwargs_420300)
        
        # Obtaining the member '__getitem__' of a type (line 117)
        getitem___420302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), lgmres_call_result_420301, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
        subscript_call_result_420303 = invoke(stypy.reporting.localization.Localization(__file__, 117, 12), getitem___420302, int_420294)
        
        # Assigning a type to the variable 'tuple_var_assignment_419557' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'tuple_var_assignment_419557', subscript_call_result_420303)
        
        # Assigning a Subscript to a Name (line 117):
        
        # Obtaining the type of the subscript
        int_420304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 12), 'int')
        
        # Call to lgmres(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'A' (line 117)
        A_420306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 29), 'A', False)
        # Getting the type of 'b' (line 117)
        b_420307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 32), 'b', False)
        # Processing the call keyword arguments (line 117)
        int_420308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 43), 'int')
        keyword_420309 = int_420308
        kwargs_420310 = {'maxiter': keyword_420309}
        # Getting the type of 'lgmres' (line 117)
        lgmres_420305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 22), 'lgmres', False)
        # Calling lgmres(args, kwargs) (line 117)
        lgmres_call_result_420311 = invoke(stypy.reporting.localization.Localization(__file__, 117, 22), lgmres_420305, *[A_420306, b_420307], **kwargs_420310)
        
        # Obtaining the member '__getitem__' of a type (line 117)
        getitem___420312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), lgmres_call_result_420311, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
        subscript_call_result_420313 = invoke(stypy.reporting.localization.Localization(__file__, 117, 12), getitem___420312, int_420304)
        
        # Assigning a type to the variable 'tuple_var_assignment_419558' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'tuple_var_assignment_419558', subscript_call_result_420313)
        
        # Assigning a Name to a Name (line 117):
        # Getting the type of 'tuple_var_assignment_419557' (line 117)
        tuple_var_assignment_419557_420314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'tuple_var_assignment_419557')
        # Assigning a type to the variable 'x' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'x', tuple_var_assignment_419557_420314)
        
        # Assigning a Name to a Name (line 117):
        # Getting the type of 'tuple_var_assignment_419558' (line 117)
        tuple_var_assignment_419558_420315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'tuple_var_assignment_419558')
        # Assigning a type to the variable 'info' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'info', tuple_var_assignment_419558_420315)
        
        # Call to assert_equal(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'info' (line 118)
        info_420317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 25), 'info', False)
        int_420318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 31), 'int')
        # Processing the call keyword arguments (line 118)
        kwargs_420319 = {}
        # Getting the type of 'assert_equal' (line 118)
        assert_equal_420316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 118)
        assert_equal_call_result_420320 = invoke(stypy.reporting.localization.Localization(__file__, 118, 12), assert_equal_420316, *[info_420317, int_420318], **kwargs_420319)
        
        
        # Call to assert_allclose(...): (line 119)
        # Processing the call arguments (line 119)
        
        # Call to dot(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'x' (line 119)
        x_420324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 34), 'x', False)
        # Processing the call keyword arguments (line 119)
        kwargs_420325 = {}
        # Getting the type of 'A' (line 119)
        A_420322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 28), 'A', False)
        # Obtaining the member 'dot' of a type (line 119)
        dot_420323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 28), A_420322, 'dot')
        # Calling dot(args, kwargs) (line 119)
        dot_call_result_420326 = invoke(stypy.reporting.localization.Localization(__file__, 119, 28), dot_420323, *[x_420324], **kwargs_420325)
        
        # Getting the type of 'b' (line 119)
        b_420327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 39), 'b', False)
        # Applying the binary operator '-' (line 119)
        result_sub_420328 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 28), '-', dot_call_result_420326, b_420327)
        
        int_420329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 42), 'int')
        # Processing the call keyword arguments (line 119)
        float_420330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 50), 'float')
        keyword_420331 = float_420330
        kwargs_420332 = {'atol': keyword_420331}
        # Getting the type of 'assert_allclose' (line 119)
        assert_allclose_420321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 119)
        assert_allclose_call_result_420333 = invoke(stypy.reporting.localization.Localization(__file__, 119, 12), assert_allclose_420321, *[result_sub_420328, int_420329], **kwargs_420332)
        
        
        # Assigning a Call to a Tuple (line 121):
        
        # Assigning a Subscript to a Name (line 121):
        
        # Obtaining the type of the subscript
        int_420334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 12), 'int')
        
        # Call to lgmres(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'A' (line 121)
        A_420336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 29), 'A', False)
        # Getting the type of 'b' (line 121)
        b_420337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 32), 'b', False)
        # Processing the call keyword arguments (line 121)
        int_420338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 39), 'int')
        keyword_420339 = int_420338
        int_420340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 50), 'int')
        keyword_420341 = int_420340
        kwargs_420342 = {'tol': keyword_420339, 'maxiter': keyword_420341}
        # Getting the type of 'lgmres' (line 121)
        lgmres_420335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 22), 'lgmres', False)
        # Calling lgmres(args, kwargs) (line 121)
        lgmres_call_result_420343 = invoke(stypy.reporting.localization.Localization(__file__, 121, 22), lgmres_420335, *[A_420336, b_420337], **kwargs_420342)
        
        # Obtaining the member '__getitem__' of a type (line 121)
        getitem___420344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 12), lgmres_call_result_420343, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 121)
        subscript_call_result_420345 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), getitem___420344, int_420334)
        
        # Assigning a type to the variable 'tuple_var_assignment_419559' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'tuple_var_assignment_419559', subscript_call_result_420345)
        
        # Assigning a Subscript to a Name (line 121):
        
        # Obtaining the type of the subscript
        int_420346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 12), 'int')
        
        # Call to lgmres(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'A' (line 121)
        A_420348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 29), 'A', False)
        # Getting the type of 'b' (line 121)
        b_420349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 32), 'b', False)
        # Processing the call keyword arguments (line 121)
        int_420350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 39), 'int')
        keyword_420351 = int_420350
        int_420352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 50), 'int')
        keyword_420353 = int_420352
        kwargs_420354 = {'tol': keyword_420351, 'maxiter': keyword_420353}
        # Getting the type of 'lgmres' (line 121)
        lgmres_420347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 22), 'lgmres', False)
        # Calling lgmres(args, kwargs) (line 121)
        lgmres_call_result_420355 = invoke(stypy.reporting.localization.Localization(__file__, 121, 22), lgmres_420347, *[A_420348, b_420349], **kwargs_420354)
        
        # Obtaining the member '__getitem__' of a type (line 121)
        getitem___420356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 12), lgmres_call_result_420355, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 121)
        subscript_call_result_420357 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), getitem___420356, int_420346)
        
        # Assigning a type to the variable 'tuple_var_assignment_419560' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'tuple_var_assignment_419560', subscript_call_result_420357)
        
        # Assigning a Name to a Name (line 121):
        # Getting the type of 'tuple_var_assignment_419559' (line 121)
        tuple_var_assignment_419559_420358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'tuple_var_assignment_419559')
        # Assigning a type to the variable 'x' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'x', tuple_var_assignment_419559_420358)
        
        # Assigning a Name to a Name (line 121):
        # Getting the type of 'tuple_var_assignment_419560' (line 121)
        tuple_var_assignment_419560_420359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'tuple_var_assignment_419560')
        # Assigning a type to the variable 'info' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'info', tuple_var_assignment_419560_420359)
        
        
        # Getting the type of 'info' (line 122)
        info_420360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 15), 'info')
        int_420361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 23), 'int')
        # Applying the binary operator '==' (line 122)
        result_eq_420362 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 15), '==', info_420360, int_420361)
        
        # Testing the type of an if condition (line 122)
        if_condition_420363 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 12), result_eq_420362)
        # Assigning a type to the variable 'if_condition_420363' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'if_condition_420363', if_condition_420363)
        # SSA begins for if statement (line 122)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_allclose(...): (line 123)
        # Processing the call arguments (line 123)
        
        # Call to dot(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'x' (line 123)
        x_420367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 38), 'x', False)
        # Processing the call keyword arguments (line 123)
        kwargs_420368 = {}
        # Getting the type of 'A' (line 123)
        A_420365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 32), 'A', False)
        # Obtaining the member 'dot' of a type (line 123)
        dot_420366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 32), A_420365, 'dot')
        # Calling dot(args, kwargs) (line 123)
        dot_call_result_420369 = invoke(stypy.reporting.localization.Localization(__file__, 123, 32), dot_420366, *[x_420367], **kwargs_420368)
        
        # Getting the type of 'b' (line 123)
        b_420370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 43), 'b', False)
        # Applying the binary operator '-' (line 123)
        result_sub_420371 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 32), '-', dot_call_result_420369, b_420370)
        
        int_420372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 46), 'int')
        # Processing the call keyword arguments (line 123)
        float_420373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 54), 'float')
        keyword_420374 = float_420373
        kwargs_420375 = {'atol': keyword_420374}
        # Getting the type of 'assert_allclose' (line 123)
        assert_allclose_420364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 123)
        assert_allclose_call_result_420376 = invoke(stypy.reporting.localization.Localization(__file__, 123, 16), assert_allclose_420364, *[result_sub_420371, int_420372], **kwargs_420375)
        
        # SSA join for if statement (line 122)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_cornercase(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_cornercase' in the type store
        # Getting the type of 'stypy_return_type' (line 97)
        stypy_return_type_420377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_420377)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_cornercase'
        return stypy_return_type_420377


    @norecursion
    def test_nans(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_nans'
        module_type_store = module_type_store.open_function_context('test_nans', 125, 4, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLGMRES.test_nans.__dict__.__setitem__('stypy_localization', localization)
        TestLGMRES.test_nans.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLGMRES.test_nans.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLGMRES.test_nans.__dict__.__setitem__('stypy_function_name', 'TestLGMRES.test_nans')
        TestLGMRES.test_nans.__dict__.__setitem__('stypy_param_names_list', [])
        TestLGMRES.test_nans.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLGMRES.test_nans.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLGMRES.test_nans.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLGMRES.test_nans.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLGMRES.test_nans.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLGMRES.test_nans.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLGMRES.test_nans', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_nans', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_nans(...)' code ##################

        
        # Assigning a Call to a Name (line 126):
        
        # Assigning a Call to a Name (line 126):
        
        # Call to eye(...): (line 126)
        # Processing the call arguments (line 126)
        int_420379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 16), 'int')
        # Processing the call keyword arguments (line 126)
        str_420380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 26), 'str', 'lil')
        keyword_420381 = str_420380
        kwargs_420382 = {'format': keyword_420381}
        # Getting the type of 'eye' (line 126)
        eye_420378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'eye', False)
        # Calling eye(args, kwargs) (line 126)
        eye_call_result_420383 = invoke(stypy.reporting.localization.Localization(__file__, 126, 12), eye_420378, *[int_420379], **kwargs_420382)
        
        # Assigning a type to the variable 'A' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'A', eye_call_result_420383)
        
        # Assigning a Attribute to a Subscript (line 127):
        
        # Assigning a Attribute to a Subscript (line 127):
        # Getting the type of 'np' (line 127)
        np_420384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 17), 'np')
        # Obtaining the member 'nan' of a type (line 127)
        nan_420385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 17), np_420384, 'nan')
        # Getting the type of 'A' (line 127)
        A_420386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 127)
        tuple_420387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 10), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 127)
        # Adding element type (line 127)
        int_420388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 10), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 10), tuple_420387, int_420388)
        # Adding element type (line 127)
        int_420389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 12), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 10), tuple_420387, int_420389)
        
        # Storing an element on a container (line 127)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 8), A_420386, (tuple_420387, nan_420385))
        
        # Assigning a Call to a Name (line 128):
        
        # Assigning a Call to a Name (line 128):
        
        # Call to ones(...): (line 128)
        # Processing the call arguments (line 128)
        int_420392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 20), 'int')
        # Processing the call keyword arguments (line 128)
        kwargs_420393 = {}
        # Getting the type of 'np' (line 128)
        np_420390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'np', False)
        # Obtaining the member 'ones' of a type (line 128)
        ones_420391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 12), np_420390, 'ones')
        # Calling ones(args, kwargs) (line 128)
        ones_call_result_420394 = invoke(stypy.reporting.localization.Localization(__file__, 128, 12), ones_420391, *[int_420392], **kwargs_420393)
        
        # Assigning a type to the variable 'b' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'b', ones_call_result_420394)
        
        # Assigning a Call to a Tuple (line 130):
        
        # Assigning a Subscript to a Name (line 130):
        
        # Obtaining the type of the subscript
        int_420395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 8), 'int')
        
        # Call to lgmres(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'A' (line 130)
        A_420397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 25), 'A', False)
        # Getting the type of 'b' (line 130)
        b_420398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 28), 'b', False)
        # Processing the call keyword arguments (line 130)
        int_420399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 35), 'int')
        keyword_420400 = int_420399
        int_420401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 46), 'int')
        keyword_420402 = int_420401
        kwargs_420403 = {'tol': keyword_420400, 'maxiter': keyword_420402}
        # Getting the type of 'lgmres' (line 130)
        lgmres_420396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 18), 'lgmres', False)
        # Calling lgmres(args, kwargs) (line 130)
        lgmres_call_result_420404 = invoke(stypy.reporting.localization.Localization(__file__, 130, 18), lgmres_420396, *[A_420397, b_420398], **kwargs_420403)
        
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___420405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), lgmres_call_result_420404, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
        subscript_call_result_420406 = invoke(stypy.reporting.localization.Localization(__file__, 130, 8), getitem___420405, int_420395)
        
        # Assigning a type to the variable 'tuple_var_assignment_419561' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'tuple_var_assignment_419561', subscript_call_result_420406)
        
        # Assigning a Subscript to a Name (line 130):
        
        # Obtaining the type of the subscript
        int_420407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 8), 'int')
        
        # Call to lgmres(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'A' (line 130)
        A_420409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 25), 'A', False)
        # Getting the type of 'b' (line 130)
        b_420410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 28), 'b', False)
        # Processing the call keyword arguments (line 130)
        int_420411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 35), 'int')
        keyword_420412 = int_420411
        int_420413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 46), 'int')
        keyword_420414 = int_420413
        kwargs_420415 = {'tol': keyword_420412, 'maxiter': keyword_420414}
        # Getting the type of 'lgmres' (line 130)
        lgmres_420408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 18), 'lgmres', False)
        # Calling lgmres(args, kwargs) (line 130)
        lgmres_call_result_420416 = invoke(stypy.reporting.localization.Localization(__file__, 130, 18), lgmres_420408, *[A_420409, b_420410], **kwargs_420415)
        
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___420417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), lgmres_call_result_420416, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
        subscript_call_result_420418 = invoke(stypy.reporting.localization.Localization(__file__, 130, 8), getitem___420417, int_420407)
        
        # Assigning a type to the variable 'tuple_var_assignment_419562' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'tuple_var_assignment_419562', subscript_call_result_420418)
        
        # Assigning a Name to a Name (line 130):
        # Getting the type of 'tuple_var_assignment_419561' (line 130)
        tuple_var_assignment_419561_420419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'tuple_var_assignment_419561')
        # Assigning a type to the variable 'x' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'x', tuple_var_assignment_419561_420419)
        
        # Assigning a Name to a Name (line 130):
        # Getting the type of 'tuple_var_assignment_419562' (line 130)
        tuple_var_assignment_419562_420420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'tuple_var_assignment_419562')
        # Assigning a type to the variable 'info' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 11), 'info', tuple_var_assignment_419562_420420)
        
        # Call to assert_equal(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'info' (line 131)
        info_420422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 21), 'info', False)
        int_420423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 27), 'int')
        # Processing the call keyword arguments (line 131)
        kwargs_420424 = {}
        # Getting the type of 'assert_equal' (line 131)
        assert_equal_420421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 131)
        assert_equal_call_result_420425 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), assert_equal_420421, *[info_420422, int_420423], **kwargs_420424)
        
        
        # ################# End of 'test_nans(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_nans' in the type store
        # Getting the type of 'stypy_return_type' (line 125)
        stypy_return_type_420426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_420426)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_nans'
        return stypy_return_type_420426


    @norecursion
    def test_breakdown_with_outer_v(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_breakdown_with_outer_v'
        module_type_store = module_type_store.open_function_context('test_breakdown_with_outer_v', 133, 4, False)
        # Assigning a type to the variable 'self' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLGMRES.test_breakdown_with_outer_v.__dict__.__setitem__('stypy_localization', localization)
        TestLGMRES.test_breakdown_with_outer_v.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLGMRES.test_breakdown_with_outer_v.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLGMRES.test_breakdown_with_outer_v.__dict__.__setitem__('stypy_function_name', 'TestLGMRES.test_breakdown_with_outer_v')
        TestLGMRES.test_breakdown_with_outer_v.__dict__.__setitem__('stypy_param_names_list', [])
        TestLGMRES.test_breakdown_with_outer_v.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLGMRES.test_breakdown_with_outer_v.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLGMRES.test_breakdown_with_outer_v.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLGMRES.test_breakdown_with_outer_v.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLGMRES.test_breakdown_with_outer_v.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLGMRES.test_breakdown_with_outer_v.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLGMRES.test_breakdown_with_outer_v', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_breakdown_with_outer_v', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_breakdown_with_outer_v(...)' code ##################

        
        # Assigning a Call to a Name (line 134):
        
        # Assigning a Call to a Name (line 134):
        
        # Call to array(...): (line 134)
        # Processing the call arguments (line 134)
        
        # Obtaining an instance of the builtin type 'list' (line 134)
        list_420429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 134)
        # Adding element type (line 134)
        
        # Obtaining an instance of the builtin type 'list' (line 134)
        list_420430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 134)
        # Adding element type (line 134)
        int_420431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 22), list_420430, int_420431)
        # Adding element type (line 134)
        int_420432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 22), list_420430, int_420432)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 21), list_420429, list_420430)
        # Adding element type (line 134)
        
        # Obtaining an instance of the builtin type 'list' (line 134)
        list_420433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 134)
        # Adding element type (line 134)
        int_420434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 30), list_420433, int_420434)
        # Adding element type (line 134)
        int_420435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 30), list_420433, int_420435)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 21), list_420429, list_420433)
        
        # Processing the call keyword arguments (line 134)
        # Getting the type of 'float' (line 134)
        float_420436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 45), 'float', False)
        keyword_420437 = float_420436
        kwargs_420438 = {'dtype': keyword_420437}
        # Getting the type of 'np' (line 134)
        np_420427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 134)
        array_420428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 12), np_420427, 'array')
        # Calling array(args, kwargs) (line 134)
        array_call_result_420439 = invoke(stypy.reporting.localization.Localization(__file__, 134, 12), array_420428, *[list_420429], **kwargs_420438)
        
        # Assigning a type to the variable 'A' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'A', array_call_result_420439)
        
        # Assigning a Call to a Name (line 135):
        
        # Assigning a Call to a Name (line 135):
        
        # Call to array(...): (line 135)
        # Processing the call arguments (line 135)
        
        # Obtaining an instance of the builtin type 'list' (line 135)
        list_420442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 135)
        # Adding element type (line 135)
        int_420443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 21), list_420442, int_420443)
        # Adding element type (line 135)
        int_420444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 21), list_420442, int_420444)
        
        # Processing the call keyword arguments (line 135)
        kwargs_420445 = {}
        # Getting the type of 'np' (line 135)
        np_420440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 135)
        array_420441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 12), np_420440, 'array')
        # Calling array(args, kwargs) (line 135)
        array_call_result_420446 = invoke(stypy.reporting.localization.Localization(__file__, 135, 12), array_420441, *[list_420442], **kwargs_420445)
        
        # Assigning a type to the variable 'b' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'b', array_call_result_420446)
        
        # Assigning a Call to a Name (line 137):
        
        # Assigning a Call to a Name (line 137):
        
        # Call to solve(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'A' (line 137)
        A_420450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 28), 'A', False)
        # Getting the type of 'b' (line 137)
        b_420451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 31), 'b', False)
        # Processing the call keyword arguments (line 137)
        kwargs_420452 = {}
        # Getting the type of 'np' (line 137)
        np_420447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'np', False)
        # Obtaining the member 'linalg' of a type (line 137)
        linalg_420448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 12), np_420447, 'linalg')
        # Obtaining the member 'solve' of a type (line 137)
        solve_420449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 12), linalg_420448, 'solve')
        # Calling solve(args, kwargs) (line 137)
        solve_call_result_420453 = invoke(stypy.reporting.localization.Localization(__file__, 137, 12), solve_420449, *[A_420450, b_420451], **kwargs_420452)
        
        # Assigning a type to the variable 'x' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'x', solve_call_result_420453)
        
        # Assigning a Call to a Name (line 138):
        
        # Assigning a Call to a Name (line 138):
        
        # Call to array(...): (line 138)
        # Processing the call arguments (line 138)
        
        # Obtaining an instance of the builtin type 'list' (line 138)
        list_420456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 138)
        # Adding element type (line 138)
        int_420457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 22), list_420456, int_420457)
        # Adding element type (line 138)
        int_420458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 22), list_420456, int_420458)
        
        # Processing the call keyword arguments (line 138)
        kwargs_420459 = {}
        # Getting the type of 'np' (line 138)
        np_420454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 138)
        array_420455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 13), np_420454, 'array')
        # Calling array(args, kwargs) (line 138)
        array_call_result_420460 = invoke(stypy.reporting.localization.Localization(__file__, 138, 13), array_420455, *[list_420456], **kwargs_420459)
        
        # Assigning a type to the variable 'v0' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'v0', array_call_result_420460)
        
        # Assigning a Call to a Tuple (line 142):
        
        # Assigning a Subscript to a Name (line 142):
        
        # Obtaining the type of the subscript
        int_420461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 8), 'int')
        
        # Call to lgmres(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'A' (line 142)
        A_420463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 26), 'A', False)
        # Getting the type of 'b' (line 142)
        b_420464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 29), 'b', False)
        # Processing the call keyword arguments (line 142)
        
        # Obtaining an instance of the builtin type 'list' (line 142)
        list_420465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 142)
        # Adding element type (line 142)
        
        # Obtaining an instance of the builtin type 'tuple' (line 142)
        tuple_420466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 142)
        # Adding element type (line 142)
        # Getting the type of 'v0' (line 142)
        v0_420467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 42), 'v0', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 42), tuple_420466, v0_420467)
        # Adding element type (line 142)
        # Getting the type of 'None' (line 142)
        None_420468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 46), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 42), tuple_420466, None_420468)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 40), list_420465, tuple_420466)
        # Adding element type (line 142)
        
        # Obtaining an instance of the builtin type 'tuple' (line 142)
        tuple_420469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 142)
        # Adding element type (line 142)
        # Getting the type of 'x' (line 142)
        x_420470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 54), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 54), tuple_420469, x_420470)
        # Adding element type (line 142)
        # Getting the type of 'None' (line 142)
        None_420471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 57), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 54), tuple_420469, None_420471)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 40), list_420465, tuple_420469)
        
        keyword_420472 = list_420465
        int_420473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 73), 'int')
        keyword_420474 = int_420473
        kwargs_420475 = {'outer_v': keyword_420472, 'maxiter': keyword_420474}
        # Getting the type of 'lgmres' (line 142)
        lgmres_420462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), 'lgmres', False)
        # Calling lgmres(args, kwargs) (line 142)
        lgmres_call_result_420476 = invoke(stypy.reporting.localization.Localization(__file__, 142, 19), lgmres_420462, *[A_420463, b_420464], **kwargs_420475)
        
        # Obtaining the member '__getitem__' of a type (line 142)
        getitem___420477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), lgmres_call_result_420476, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 142)
        subscript_call_result_420478 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), getitem___420477, int_420461)
        
        # Assigning a type to the variable 'tuple_var_assignment_419563' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'tuple_var_assignment_419563', subscript_call_result_420478)
        
        # Assigning a Subscript to a Name (line 142):
        
        # Obtaining the type of the subscript
        int_420479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 8), 'int')
        
        # Call to lgmres(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'A' (line 142)
        A_420481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 26), 'A', False)
        # Getting the type of 'b' (line 142)
        b_420482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 29), 'b', False)
        # Processing the call keyword arguments (line 142)
        
        # Obtaining an instance of the builtin type 'list' (line 142)
        list_420483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 142)
        # Adding element type (line 142)
        
        # Obtaining an instance of the builtin type 'tuple' (line 142)
        tuple_420484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 142)
        # Adding element type (line 142)
        # Getting the type of 'v0' (line 142)
        v0_420485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 42), 'v0', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 42), tuple_420484, v0_420485)
        # Adding element type (line 142)
        # Getting the type of 'None' (line 142)
        None_420486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 46), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 42), tuple_420484, None_420486)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 40), list_420483, tuple_420484)
        # Adding element type (line 142)
        
        # Obtaining an instance of the builtin type 'tuple' (line 142)
        tuple_420487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 142)
        # Adding element type (line 142)
        # Getting the type of 'x' (line 142)
        x_420488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 54), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 54), tuple_420487, x_420488)
        # Adding element type (line 142)
        # Getting the type of 'None' (line 142)
        None_420489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 57), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 54), tuple_420487, None_420489)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 40), list_420483, tuple_420487)
        
        keyword_420490 = list_420483
        int_420491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 73), 'int')
        keyword_420492 = int_420491
        kwargs_420493 = {'outer_v': keyword_420490, 'maxiter': keyword_420492}
        # Getting the type of 'lgmres' (line 142)
        lgmres_420480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), 'lgmres', False)
        # Calling lgmres(args, kwargs) (line 142)
        lgmres_call_result_420494 = invoke(stypy.reporting.localization.Localization(__file__, 142, 19), lgmres_420480, *[A_420481, b_420482], **kwargs_420493)
        
        # Obtaining the member '__getitem__' of a type (line 142)
        getitem___420495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), lgmres_call_result_420494, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 142)
        subscript_call_result_420496 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), getitem___420495, int_420479)
        
        # Assigning a type to the variable 'tuple_var_assignment_419564' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'tuple_var_assignment_419564', subscript_call_result_420496)
        
        # Assigning a Name to a Name (line 142):
        # Getting the type of 'tuple_var_assignment_419563' (line 142)
        tuple_var_assignment_419563_420497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'tuple_var_assignment_419563')
        # Assigning a type to the variable 'xp' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'xp', tuple_var_assignment_419563_420497)
        
        # Assigning a Name to a Name (line 142):
        # Getting the type of 'tuple_var_assignment_419564' (line 142)
        tuple_var_assignment_419564_420498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'tuple_var_assignment_419564')
        # Assigning a type to the variable 'info' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'info', tuple_var_assignment_419564_420498)
        
        # Call to assert_allclose(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'xp' (line 144)
        xp_420500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 24), 'xp', False)
        # Getting the type of 'x' (line 144)
        x_420501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 28), 'x', False)
        # Processing the call keyword arguments (line 144)
        float_420502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 36), 'float')
        keyword_420503 = float_420502
        kwargs_420504 = {'atol': keyword_420503}
        # Getting the type of 'assert_allclose' (line 144)
        assert_allclose_420499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 144)
        assert_allclose_call_result_420505 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), assert_allclose_420499, *[xp_420500, x_420501], **kwargs_420504)
        
        
        # ################# End of 'test_breakdown_with_outer_v(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_breakdown_with_outer_v' in the type store
        # Getting the type of 'stypy_return_type' (line 133)
        stypy_return_type_420506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_420506)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_breakdown_with_outer_v'
        return stypy_return_type_420506


    @norecursion
    def test_breakdown_underdetermined(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_breakdown_underdetermined'
        module_type_store = module_type_store.open_function_context('test_breakdown_underdetermined', 146, 4, False)
        # Assigning a type to the variable 'self' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLGMRES.test_breakdown_underdetermined.__dict__.__setitem__('stypy_localization', localization)
        TestLGMRES.test_breakdown_underdetermined.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLGMRES.test_breakdown_underdetermined.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLGMRES.test_breakdown_underdetermined.__dict__.__setitem__('stypy_function_name', 'TestLGMRES.test_breakdown_underdetermined')
        TestLGMRES.test_breakdown_underdetermined.__dict__.__setitem__('stypy_param_names_list', [])
        TestLGMRES.test_breakdown_underdetermined.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLGMRES.test_breakdown_underdetermined.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLGMRES.test_breakdown_underdetermined.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLGMRES.test_breakdown_underdetermined.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLGMRES.test_breakdown_underdetermined.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLGMRES.test_breakdown_underdetermined.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLGMRES.test_breakdown_underdetermined', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_breakdown_underdetermined', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_breakdown_underdetermined(...)' code ##################

        
        # Assigning a Call to a Name (line 149):
        
        # Assigning a Call to a Name (line 149):
        
        # Call to array(...): (line 149)
        # Processing the call arguments (line 149)
        
        # Obtaining an instance of the builtin type 'list' (line 149)
        list_420509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 149)
        # Adding element type (line 149)
        
        # Obtaining an instance of the builtin type 'list' (line 149)
        list_420510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 149)
        # Adding element type (line 149)
        int_420511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 22), list_420510, int_420511)
        # Adding element type (line 149)
        int_420512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 22), list_420510, int_420512)
        # Adding element type (line 149)
        int_420513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 22), list_420510, int_420513)
        # Adding element type (line 149)
        int_420514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 22), list_420510, int_420514)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 21), list_420509, list_420510)
        # Adding element type (line 149)
        
        # Obtaining an instance of the builtin type 'list' (line 150)
        list_420515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 150)
        # Adding element type (line 150)
        int_420516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 22), list_420515, int_420516)
        # Adding element type (line 150)
        int_420517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 22), list_420515, int_420517)
        # Adding element type (line 150)
        int_420518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 22), list_420515, int_420518)
        # Adding element type (line 150)
        int_420519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 22), list_420515, int_420519)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 21), list_420509, list_420515)
        # Adding element type (line 149)
        
        # Obtaining an instance of the builtin type 'list' (line 151)
        list_420520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 151)
        # Adding element type (line 151)
        int_420521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 22), list_420520, int_420521)
        # Adding element type (line 151)
        int_420522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 22), list_420520, int_420522)
        # Adding element type (line 151)
        int_420523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 22), list_420520, int_420523)
        # Adding element type (line 151)
        int_420524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 22), list_420520, int_420524)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 21), list_420509, list_420520)
        # Adding element type (line 149)
        
        # Obtaining an instance of the builtin type 'list' (line 152)
        list_420525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 152)
        # Adding element type (line 152)
        int_420526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 22), list_420525, int_420526)
        # Adding element type (line 152)
        int_420527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 22), list_420525, int_420527)
        # Adding element type (line 152)
        int_420528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 22), list_420525, int_420528)
        # Adding element type (line 152)
        int_420529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 22), list_420525, int_420529)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 21), list_420509, list_420525)
        
        # Processing the call keyword arguments (line 149)
        # Getting the type of 'float' (line 152)
        float_420530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 43), 'float', False)
        keyword_420531 = float_420530
        kwargs_420532 = {'dtype': keyword_420531}
        # Getting the type of 'np' (line 149)
        np_420507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 149)
        array_420508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 12), np_420507, 'array')
        # Calling array(args, kwargs) (line 149)
        array_call_result_420533 = invoke(stypy.reporting.localization.Localization(__file__, 149, 12), array_420508, *[list_420509], **kwargs_420532)
        
        # Assigning a type to the variable 'A' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'A', array_call_result_420533)
        
        # Assigning a List to a Name (line 154):
        
        # Assigning a List to a Name (line 154):
        
        # Obtaining an instance of the builtin type 'list' (line 154)
        list_420534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 154)
        # Adding element type (line 154)
        
        # Call to array(...): (line 155)
        # Processing the call arguments (line 155)
        
        # Obtaining an instance of the builtin type 'list' (line 155)
        list_420537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 155)
        # Adding element type (line 155)
        int_420538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 21), list_420537, int_420538)
        # Adding element type (line 155)
        int_420539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 21), list_420537, int_420539)
        # Adding element type (line 155)
        int_420540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 21), list_420537, int_420540)
        # Adding element type (line 155)
        int_420541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 21), list_420537, int_420541)
        
        # Processing the call keyword arguments (line 155)
        kwargs_420542 = {}
        # Getting the type of 'np' (line 155)
        np_420535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 155)
        array_420536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 12), np_420535, 'array')
        # Calling array(args, kwargs) (line 155)
        array_call_result_420543 = invoke(stypy.reporting.localization.Localization(__file__, 155, 12), array_420536, *[list_420537], **kwargs_420542)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 13), list_420534, array_call_result_420543)
        # Adding element type (line 154)
        
        # Call to array(...): (line 156)
        # Processing the call arguments (line 156)
        
        # Obtaining an instance of the builtin type 'list' (line 156)
        list_420546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 156)
        # Adding element type (line 156)
        int_420547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 21), list_420546, int_420547)
        # Adding element type (line 156)
        int_420548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 21), list_420546, int_420548)
        # Adding element type (line 156)
        int_420549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 21), list_420546, int_420549)
        # Adding element type (line 156)
        int_420550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 21), list_420546, int_420550)
        
        # Processing the call keyword arguments (line 156)
        kwargs_420551 = {}
        # Getting the type of 'np' (line 156)
        np_420544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 156)
        array_420545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 12), np_420544, 'array')
        # Calling array(args, kwargs) (line 156)
        array_call_result_420552 = invoke(stypy.reporting.localization.Localization(__file__, 156, 12), array_420545, *[list_420546], **kwargs_420551)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 13), list_420534, array_call_result_420552)
        # Adding element type (line 154)
        
        # Call to array(...): (line 157)
        # Processing the call arguments (line 157)
        
        # Obtaining an instance of the builtin type 'list' (line 157)
        list_420555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 157)
        # Adding element type (line 157)
        int_420556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 21), list_420555, int_420556)
        # Adding element type (line 157)
        int_420557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 21), list_420555, int_420557)
        # Adding element type (line 157)
        int_420558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 21), list_420555, int_420558)
        # Adding element type (line 157)
        int_420559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 21), list_420555, int_420559)
        
        # Processing the call keyword arguments (line 157)
        kwargs_420560 = {}
        # Getting the type of 'np' (line 157)
        np_420553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 157)
        array_420554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 12), np_420553, 'array')
        # Calling array(args, kwargs) (line 157)
        array_call_result_420561 = invoke(stypy.reporting.localization.Localization(__file__, 157, 12), array_420554, *[list_420555], **kwargs_420560)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 13), list_420534, array_call_result_420561)
        # Adding element type (line 154)
        
        # Call to array(...): (line 158)
        # Processing the call arguments (line 158)
        
        # Obtaining an instance of the builtin type 'list' (line 158)
        list_420564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 158)
        # Adding element type (line 158)
        int_420565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 21), list_420564, int_420565)
        # Adding element type (line 158)
        int_420566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 21), list_420564, int_420566)
        # Adding element type (line 158)
        int_420567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 21), list_420564, int_420567)
        # Adding element type (line 158)
        int_420568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 21), list_420564, int_420568)
        
        # Processing the call keyword arguments (line 158)
        kwargs_420569 = {}
        # Getting the type of 'np' (line 158)
        np_420562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 158)
        array_420563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 12), np_420562, 'array')
        # Calling array(args, kwargs) (line 158)
        array_call_result_420570 = invoke(stypy.reporting.localization.Localization(__file__, 158, 12), array_420563, *[list_420564], **kwargs_420569)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 13), list_420534, array_call_result_420570)
        
        # Assigning a type to the variable 'bs' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'bs', list_420534)
        
        # Getting the type of 'bs' (line 161)
        bs_420571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 17), 'bs')
        # Testing the type of a for loop iterable (line 161)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 161, 8), bs_420571)
        # Getting the type of the for loop variable (line 161)
        for_loop_var_420572 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 161, 8), bs_420571)
        # Assigning a type to the variable 'b' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'b', for_loop_var_420572)
        # SSA begins for a for statement (line 161)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 162):
        
        # Assigning a Subscript to a Name (line 162):
        
        # Obtaining the type of the subscript
        int_420573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 12), 'int')
        
        # Call to lgmres(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'A' (line 162)
        A_420575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 30), 'A', False)
        # Getting the type of 'b' (line 162)
        b_420576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 33), 'b', False)
        # Processing the call keyword arguments (line 162)
        int_420577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 44), 'int')
        keyword_420578 = int_420577
        kwargs_420579 = {'maxiter': keyword_420578}
        # Getting the type of 'lgmres' (line 162)
        lgmres_420574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 23), 'lgmres', False)
        # Calling lgmres(args, kwargs) (line 162)
        lgmres_call_result_420580 = invoke(stypy.reporting.localization.Localization(__file__, 162, 23), lgmres_420574, *[A_420575, b_420576], **kwargs_420579)
        
        # Obtaining the member '__getitem__' of a type (line 162)
        getitem___420581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 12), lgmres_call_result_420580, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 162)
        subscript_call_result_420582 = invoke(stypy.reporting.localization.Localization(__file__, 162, 12), getitem___420581, int_420573)
        
        # Assigning a type to the variable 'tuple_var_assignment_419565' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'tuple_var_assignment_419565', subscript_call_result_420582)
        
        # Assigning a Subscript to a Name (line 162):
        
        # Obtaining the type of the subscript
        int_420583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 12), 'int')
        
        # Call to lgmres(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'A' (line 162)
        A_420585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 30), 'A', False)
        # Getting the type of 'b' (line 162)
        b_420586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 33), 'b', False)
        # Processing the call keyword arguments (line 162)
        int_420587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 44), 'int')
        keyword_420588 = int_420587
        kwargs_420589 = {'maxiter': keyword_420588}
        # Getting the type of 'lgmres' (line 162)
        lgmres_420584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 23), 'lgmres', False)
        # Calling lgmres(args, kwargs) (line 162)
        lgmres_call_result_420590 = invoke(stypy.reporting.localization.Localization(__file__, 162, 23), lgmres_420584, *[A_420585, b_420586], **kwargs_420589)
        
        # Obtaining the member '__getitem__' of a type (line 162)
        getitem___420591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 12), lgmres_call_result_420590, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 162)
        subscript_call_result_420592 = invoke(stypy.reporting.localization.Localization(__file__, 162, 12), getitem___420591, int_420583)
        
        # Assigning a type to the variable 'tuple_var_assignment_419566' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'tuple_var_assignment_419566', subscript_call_result_420592)
        
        # Assigning a Name to a Name (line 162):
        # Getting the type of 'tuple_var_assignment_419565' (line 162)
        tuple_var_assignment_419565_420593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'tuple_var_assignment_419565')
        # Assigning a type to the variable 'xp' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'xp', tuple_var_assignment_419565_420593)
        
        # Assigning a Name to a Name (line 162):
        # Getting the type of 'tuple_var_assignment_419566' (line 162)
        tuple_var_assignment_419566_420594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'tuple_var_assignment_419566')
        # Assigning a type to the variable 'info' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'info', tuple_var_assignment_419566_420594)
        
        # Assigning a Call to a Name (line 163):
        
        # Assigning a Call to a Name (line 163):
        
        # Call to norm(...): (line 163)
        # Processing the call arguments (line 163)
        
        # Call to dot(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'xp' (line 163)
        xp_420600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 40), 'xp', False)
        # Processing the call keyword arguments (line 163)
        kwargs_420601 = {}
        # Getting the type of 'A' (line 163)
        A_420598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 34), 'A', False)
        # Obtaining the member 'dot' of a type (line 163)
        dot_420599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 34), A_420598, 'dot')
        # Calling dot(args, kwargs) (line 163)
        dot_call_result_420602 = invoke(stypy.reporting.localization.Localization(__file__, 163, 34), dot_420599, *[xp_420600], **kwargs_420601)
        
        # Getting the type of 'b' (line 163)
        b_420603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 46), 'b', False)
        # Applying the binary operator '-' (line 163)
        result_sub_420604 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 34), '-', dot_call_result_420602, b_420603)
        
        # Processing the call keyword arguments (line 163)
        kwargs_420605 = {}
        # Getting the type of 'np' (line 163)
        np_420595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 19), 'np', False)
        # Obtaining the member 'linalg' of a type (line 163)
        linalg_420596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 19), np_420595, 'linalg')
        # Obtaining the member 'norm' of a type (line 163)
        norm_420597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 19), linalg_420596, 'norm')
        # Calling norm(args, kwargs) (line 163)
        norm_call_result_420606 = invoke(stypy.reporting.localization.Localization(__file__, 163, 19), norm_420597, *[result_sub_420604], **kwargs_420605)
        
        # Assigning a type to the variable 'resp' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'resp', norm_call_result_420606)
        
        # Assigning a Subscript to a Name (line 165):
        
        # Assigning a Subscript to a Name (line 165):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 165)
        tuple_420607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 165)
        # Adding element type (line 165)
        # Getting the type of 'b' (line 165)
        b_420608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 22), 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 22), tuple_420607, b_420608)
        # Adding element type (line 165)
        
        # Call to dot(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'b' (line 165)
        b_420611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 31), 'b', False)
        # Processing the call keyword arguments (line 165)
        kwargs_420612 = {}
        # Getting the type of 'A' (line 165)
        A_420609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 25), 'A', False)
        # Obtaining the member 'dot' of a type (line 165)
        dot_420610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 25), A_420609, 'dot')
        # Calling dot(args, kwargs) (line 165)
        dot_call_result_420613 = invoke(stypy.reporting.localization.Localization(__file__, 165, 25), dot_420610, *[b_420611], **kwargs_420612)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 22), tuple_420607, dot_call_result_420613)
        # Adding element type (line 165)
        
        # Call to dot(...): (line 165)
        # Processing the call arguments (line 165)
        
        # Call to dot(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'b' (line 165)
        b_420618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 47), 'b', False)
        # Processing the call keyword arguments (line 165)
        kwargs_420619 = {}
        # Getting the type of 'A' (line 165)
        A_420616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 41), 'A', False)
        # Obtaining the member 'dot' of a type (line 165)
        dot_420617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 41), A_420616, 'dot')
        # Calling dot(args, kwargs) (line 165)
        dot_call_result_420620 = invoke(stypy.reporting.localization.Localization(__file__, 165, 41), dot_420617, *[b_420618], **kwargs_420619)
        
        # Processing the call keyword arguments (line 165)
        kwargs_420621 = {}
        # Getting the type of 'A' (line 165)
        A_420614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 35), 'A', False)
        # Obtaining the member 'dot' of a type (line 165)
        dot_420615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 35), A_420614, 'dot')
        # Calling dot(args, kwargs) (line 165)
        dot_call_result_420622 = invoke(stypy.reporting.localization.Localization(__file__, 165, 35), dot_420615, *[dot_call_result_420620], **kwargs_420621)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 22), tuple_420607, dot_call_result_420622)
        # Adding element type (line 165)
        
        # Call to dot(...): (line 165)
        # Processing the call arguments (line 165)
        
        # Call to dot(...): (line 165)
        # Processing the call arguments (line 165)
        
        # Call to dot(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'b' (line 165)
        b_420629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 70), 'b', False)
        # Processing the call keyword arguments (line 165)
        kwargs_420630 = {}
        # Getting the type of 'A' (line 165)
        A_420627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 64), 'A', False)
        # Obtaining the member 'dot' of a type (line 165)
        dot_420628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 64), A_420627, 'dot')
        # Calling dot(args, kwargs) (line 165)
        dot_call_result_420631 = invoke(stypy.reporting.localization.Localization(__file__, 165, 64), dot_420628, *[b_420629], **kwargs_420630)
        
        # Processing the call keyword arguments (line 165)
        kwargs_420632 = {}
        # Getting the type of 'A' (line 165)
        A_420625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 58), 'A', False)
        # Obtaining the member 'dot' of a type (line 165)
        dot_420626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 58), A_420625, 'dot')
        # Calling dot(args, kwargs) (line 165)
        dot_call_result_420633 = invoke(stypy.reporting.localization.Localization(__file__, 165, 58), dot_420626, *[dot_call_result_420631], **kwargs_420632)
        
        # Processing the call keyword arguments (line 165)
        kwargs_420634 = {}
        # Getting the type of 'A' (line 165)
        A_420623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 52), 'A', False)
        # Obtaining the member 'dot' of a type (line 165)
        dot_420624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 52), A_420623, 'dot')
        # Calling dot(args, kwargs) (line 165)
        dot_call_result_420635 = invoke(stypy.reporting.localization.Localization(__file__, 165, 52), dot_420624, *[dot_call_result_420633], **kwargs_420634)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 22), tuple_420607, dot_call_result_420635)
        
        # Getting the type of 'np' (line 165)
        np_420636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'np')
        # Obtaining the member 'c_' of a type (line 165)
        c__420637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 16), np_420636, 'c_')
        # Obtaining the member '__getitem__' of a type (line 165)
        getitem___420638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 16), c__420637, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 165)
        subscript_call_result_420639 = invoke(stypy.reporting.localization.Localization(__file__, 165, 16), getitem___420638, tuple_420607)
        
        # Assigning a type to the variable 'K' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'K', subscript_call_result_420639)
        
        # Assigning a Call to a Tuple (line 166):
        
        # Assigning a Subscript to a Name (line 166):
        
        # Obtaining the type of the subscript
        int_420640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 12), 'int')
        
        # Call to lstsq(...): (line 166)
        # Processing the call arguments (line 166)
        
        # Call to dot(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'K' (line 166)
        K_420646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 47), 'K', False)
        # Processing the call keyword arguments (line 166)
        kwargs_420647 = {}
        # Getting the type of 'A' (line 166)
        A_420644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 41), 'A', False)
        # Obtaining the member 'dot' of a type (line 166)
        dot_420645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 41), A_420644, 'dot')
        # Calling dot(args, kwargs) (line 166)
        dot_call_result_420648 = invoke(stypy.reporting.localization.Localization(__file__, 166, 41), dot_420645, *[K_420646], **kwargs_420647)
        
        # Getting the type of 'b' (line 166)
        b_420649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 51), 'b', False)
        # Processing the call keyword arguments (line 166)
        int_420650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 60), 'int')
        keyword_420651 = int_420650
        kwargs_420652 = {'rcond': keyword_420651}
        # Getting the type of 'np' (line 166)
        np_420641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 25), 'np', False)
        # Obtaining the member 'linalg' of a type (line 166)
        linalg_420642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 25), np_420641, 'linalg')
        # Obtaining the member 'lstsq' of a type (line 166)
        lstsq_420643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 25), linalg_420642, 'lstsq')
        # Calling lstsq(args, kwargs) (line 166)
        lstsq_call_result_420653 = invoke(stypy.reporting.localization.Localization(__file__, 166, 25), lstsq_420643, *[dot_call_result_420648, b_420649], **kwargs_420652)
        
        # Obtaining the member '__getitem__' of a type (line 166)
        getitem___420654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), lstsq_call_result_420653, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 166)
        subscript_call_result_420655 = invoke(stypy.reporting.localization.Localization(__file__, 166, 12), getitem___420654, int_420640)
        
        # Assigning a type to the variable 'tuple_var_assignment_419567' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'tuple_var_assignment_419567', subscript_call_result_420655)
        
        # Assigning a Subscript to a Name (line 166):
        
        # Obtaining the type of the subscript
        int_420656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 12), 'int')
        
        # Call to lstsq(...): (line 166)
        # Processing the call arguments (line 166)
        
        # Call to dot(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'K' (line 166)
        K_420662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 47), 'K', False)
        # Processing the call keyword arguments (line 166)
        kwargs_420663 = {}
        # Getting the type of 'A' (line 166)
        A_420660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 41), 'A', False)
        # Obtaining the member 'dot' of a type (line 166)
        dot_420661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 41), A_420660, 'dot')
        # Calling dot(args, kwargs) (line 166)
        dot_call_result_420664 = invoke(stypy.reporting.localization.Localization(__file__, 166, 41), dot_420661, *[K_420662], **kwargs_420663)
        
        # Getting the type of 'b' (line 166)
        b_420665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 51), 'b', False)
        # Processing the call keyword arguments (line 166)
        int_420666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 60), 'int')
        keyword_420667 = int_420666
        kwargs_420668 = {'rcond': keyword_420667}
        # Getting the type of 'np' (line 166)
        np_420657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 25), 'np', False)
        # Obtaining the member 'linalg' of a type (line 166)
        linalg_420658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 25), np_420657, 'linalg')
        # Obtaining the member 'lstsq' of a type (line 166)
        lstsq_420659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 25), linalg_420658, 'lstsq')
        # Calling lstsq(args, kwargs) (line 166)
        lstsq_call_result_420669 = invoke(stypy.reporting.localization.Localization(__file__, 166, 25), lstsq_420659, *[dot_call_result_420664, b_420665], **kwargs_420668)
        
        # Obtaining the member '__getitem__' of a type (line 166)
        getitem___420670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), lstsq_call_result_420669, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 166)
        subscript_call_result_420671 = invoke(stypy.reporting.localization.Localization(__file__, 166, 12), getitem___420670, int_420656)
        
        # Assigning a type to the variable 'tuple_var_assignment_419568' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'tuple_var_assignment_419568', subscript_call_result_420671)
        
        # Assigning a Subscript to a Name (line 166):
        
        # Obtaining the type of the subscript
        int_420672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 12), 'int')
        
        # Call to lstsq(...): (line 166)
        # Processing the call arguments (line 166)
        
        # Call to dot(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'K' (line 166)
        K_420678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 47), 'K', False)
        # Processing the call keyword arguments (line 166)
        kwargs_420679 = {}
        # Getting the type of 'A' (line 166)
        A_420676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 41), 'A', False)
        # Obtaining the member 'dot' of a type (line 166)
        dot_420677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 41), A_420676, 'dot')
        # Calling dot(args, kwargs) (line 166)
        dot_call_result_420680 = invoke(stypy.reporting.localization.Localization(__file__, 166, 41), dot_420677, *[K_420678], **kwargs_420679)
        
        # Getting the type of 'b' (line 166)
        b_420681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 51), 'b', False)
        # Processing the call keyword arguments (line 166)
        int_420682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 60), 'int')
        keyword_420683 = int_420682
        kwargs_420684 = {'rcond': keyword_420683}
        # Getting the type of 'np' (line 166)
        np_420673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 25), 'np', False)
        # Obtaining the member 'linalg' of a type (line 166)
        linalg_420674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 25), np_420673, 'linalg')
        # Obtaining the member 'lstsq' of a type (line 166)
        lstsq_420675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 25), linalg_420674, 'lstsq')
        # Calling lstsq(args, kwargs) (line 166)
        lstsq_call_result_420685 = invoke(stypy.reporting.localization.Localization(__file__, 166, 25), lstsq_420675, *[dot_call_result_420680, b_420681], **kwargs_420684)
        
        # Obtaining the member '__getitem__' of a type (line 166)
        getitem___420686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), lstsq_call_result_420685, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 166)
        subscript_call_result_420687 = invoke(stypy.reporting.localization.Localization(__file__, 166, 12), getitem___420686, int_420672)
        
        # Assigning a type to the variable 'tuple_var_assignment_419569' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'tuple_var_assignment_419569', subscript_call_result_420687)
        
        # Assigning a Subscript to a Name (line 166):
        
        # Obtaining the type of the subscript
        int_420688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 12), 'int')
        
        # Call to lstsq(...): (line 166)
        # Processing the call arguments (line 166)
        
        # Call to dot(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'K' (line 166)
        K_420694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 47), 'K', False)
        # Processing the call keyword arguments (line 166)
        kwargs_420695 = {}
        # Getting the type of 'A' (line 166)
        A_420692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 41), 'A', False)
        # Obtaining the member 'dot' of a type (line 166)
        dot_420693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 41), A_420692, 'dot')
        # Calling dot(args, kwargs) (line 166)
        dot_call_result_420696 = invoke(stypy.reporting.localization.Localization(__file__, 166, 41), dot_420693, *[K_420694], **kwargs_420695)
        
        # Getting the type of 'b' (line 166)
        b_420697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 51), 'b', False)
        # Processing the call keyword arguments (line 166)
        int_420698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 60), 'int')
        keyword_420699 = int_420698
        kwargs_420700 = {'rcond': keyword_420699}
        # Getting the type of 'np' (line 166)
        np_420689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 25), 'np', False)
        # Obtaining the member 'linalg' of a type (line 166)
        linalg_420690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 25), np_420689, 'linalg')
        # Obtaining the member 'lstsq' of a type (line 166)
        lstsq_420691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 25), linalg_420690, 'lstsq')
        # Calling lstsq(args, kwargs) (line 166)
        lstsq_call_result_420701 = invoke(stypy.reporting.localization.Localization(__file__, 166, 25), lstsq_420691, *[dot_call_result_420696, b_420697], **kwargs_420700)
        
        # Obtaining the member '__getitem__' of a type (line 166)
        getitem___420702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), lstsq_call_result_420701, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 166)
        subscript_call_result_420703 = invoke(stypy.reporting.localization.Localization(__file__, 166, 12), getitem___420702, int_420688)
        
        # Assigning a type to the variable 'tuple_var_assignment_419570' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'tuple_var_assignment_419570', subscript_call_result_420703)
        
        # Assigning a Name to a Name (line 166):
        # Getting the type of 'tuple_var_assignment_419567' (line 166)
        tuple_var_assignment_419567_420704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'tuple_var_assignment_419567')
        # Assigning a type to the variable 'y' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'y', tuple_var_assignment_419567_420704)
        
        # Assigning a Name to a Name (line 166):
        # Getting the type of 'tuple_var_assignment_419568' (line 166)
        tuple_var_assignment_419568_420705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'tuple_var_assignment_419568')
        # Assigning a type to the variable '_' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 15), '_', tuple_var_assignment_419568_420705)
        
        # Assigning a Name to a Name (line 166):
        # Getting the type of 'tuple_var_assignment_419569' (line 166)
        tuple_var_assignment_419569_420706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'tuple_var_assignment_419569')
        # Assigning a type to the variable '_' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 18), '_', tuple_var_assignment_419569_420706)
        
        # Assigning a Name to a Name (line 166):
        # Getting the type of 'tuple_var_assignment_419570' (line 166)
        tuple_var_assignment_419570_420707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'tuple_var_assignment_419570')
        # Assigning a type to the variable '_' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 21), '_', tuple_var_assignment_419570_420707)
        
        # Assigning a Call to a Name (line 167):
        
        # Assigning a Call to a Name (line 167):
        
        # Call to dot(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'y' (line 167)
        y_420710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 22), 'y', False)
        # Processing the call keyword arguments (line 167)
        kwargs_420711 = {}
        # Getting the type of 'K' (line 167)
        K_420708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'K', False)
        # Obtaining the member 'dot' of a type (line 167)
        dot_420709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 16), K_420708, 'dot')
        # Calling dot(args, kwargs) (line 167)
        dot_call_result_420712 = invoke(stypy.reporting.localization.Localization(__file__, 167, 16), dot_420709, *[y_420710], **kwargs_420711)
        
        # Assigning a type to the variable 'x' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'x', dot_call_result_420712)
        
        # Assigning a Call to a Name (line 168):
        
        # Assigning a Call to a Name (line 168):
        
        # Call to norm(...): (line 168)
        # Processing the call arguments (line 168)
        
        # Call to dot(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'x' (line 168)
        x_420718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 39), 'x', False)
        # Processing the call keyword arguments (line 168)
        kwargs_420719 = {}
        # Getting the type of 'A' (line 168)
        A_420716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 33), 'A', False)
        # Obtaining the member 'dot' of a type (line 168)
        dot_420717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 33), A_420716, 'dot')
        # Calling dot(args, kwargs) (line 168)
        dot_call_result_420720 = invoke(stypy.reporting.localization.Localization(__file__, 168, 33), dot_420717, *[x_420718], **kwargs_420719)
        
        # Getting the type of 'b' (line 168)
        b_420721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 44), 'b', False)
        # Applying the binary operator '-' (line 168)
        result_sub_420722 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 33), '-', dot_call_result_420720, b_420721)
        
        # Processing the call keyword arguments (line 168)
        kwargs_420723 = {}
        # Getting the type of 'np' (line 168)
        np_420713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 18), 'np', False)
        # Obtaining the member 'linalg' of a type (line 168)
        linalg_420714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 18), np_420713, 'linalg')
        # Obtaining the member 'norm' of a type (line 168)
        norm_420715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 18), linalg_420714, 'norm')
        # Calling norm(args, kwargs) (line 168)
        norm_call_result_420724 = invoke(stypy.reporting.localization.Localization(__file__, 168, 18), norm_420715, *[result_sub_420722], **kwargs_420723)
        
        # Assigning a type to the variable 'res' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'res', norm_call_result_420724)
        
        # Call to assert_allclose(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'resp' (line 170)
        resp_420726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 28), 'resp', False)
        # Getting the type of 'res' (line 170)
        res_420727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 34), 'res', False)
        # Processing the call keyword arguments (line 170)
        
        # Call to repr(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'b' (line 170)
        b_420729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 52), 'b', False)
        # Processing the call keyword arguments (line 170)
        kwargs_420730 = {}
        # Getting the type of 'repr' (line 170)
        repr_420728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 47), 'repr', False)
        # Calling repr(args, kwargs) (line 170)
        repr_call_result_420731 = invoke(stypy.reporting.localization.Localization(__file__, 170, 47), repr_420728, *[b_420729], **kwargs_420730)
        
        keyword_420732 = repr_call_result_420731
        kwargs_420733 = {'err_msg': keyword_420732}
        # Getting the type of 'assert_allclose' (line 170)
        assert_allclose_420725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 170)
        assert_allclose_call_result_420734 = invoke(stypy.reporting.localization.Localization(__file__, 170, 12), assert_allclose_420725, *[resp_420726, res_420727], **kwargs_420733)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_breakdown_underdetermined(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_breakdown_underdetermined' in the type store
        # Getting the type of 'stypy_return_type' (line 146)
        stypy_return_type_420735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_420735)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_breakdown_underdetermined'
        return stypy_return_type_420735


    @norecursion
    def test_denormals(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_denormals'
        module_type_store = module_type_store.open_function_context('test_denormals', 172, 4, False)
        # Assigning a type to the variable 'self' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLGMRES.test_denormals.__dict__.__setitem__('stypy_localization', localization)
        TestLGMRES.test_denormals.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLGMRES.test_denormals.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLGMRES.test_denormals.__dict__.__setitem__('stypy_function_name', 'TestLGMRES.test_denormals')
        TestLGMRES.test_denormals.__dict__.__setitem__('stypy_param_names_list', [])
        TestLGMRES.test_denormals.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLGMRES.test_denormals.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLGMRES.test_denormals.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLGMRES.test_denormals.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLGMRES.test_denormals.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLGMRES.test_denormals.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLGMRES.test_denormals', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_denormals', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_denormals(...)' code ##################

        
        # Assigning a Call to a Name (line 176):
        
        # Assigning a Call to a Name (line 176):
        
        # Call to array(...): (line 176)
        # Processing the call arguments (line 176)
        
        # Obtaining an instance of the builtin type 'list' (line 176)
        list_420738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 176)
        # Adding element type (line 176)
        
        # Obtaining an instance of the builtin type 'list' (line 176)
        list_420739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 176)
        # Adding element type (line 176)
        int_420740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 22), list_420739, int_420740)
        # Adding element type (line 176)
        int_420741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 22), list_420739, int_420741)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 21), list_420738, list_420739)
        # Adding element type (line 176)
        
        # Obtaining an instance of the builtin type 'list' (line 176)
        list_420742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 176)
        # Adding element type (line 176)
        int_420743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 30), list_420742, int_420743)
        # Adding element type (line 176)
        int_420744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 30), list_420742, int_420744)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 21), list_420738, list_420742)
        
        # Processing the call keyword arguments (line 176)
        # Getting the type of 'float' (line 176)
        float_420745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 45), 'float', False)
        keyword_420746 = float_420745
        kwargs_420747 = {'dtype': keyword_420746}
        # Getting the type of 'np' (line 176)
        np_420736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 176)
        array_420737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 12), np_420736, 'array')
        # Calling array(args, kwargs) (line 176)
        array_call_result_420748 = invoke(stypy.reporting.localization.Localization(__file__, 176, 12), array_420737, *[list_420738], **kwargs_420747)
        
        # Assigning a type to the variable 'A' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'A', array_call_result_420748)
        
        # Getting the type of 'A' (line 177)
        A_420749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'A')
        int_420750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 13), 'int')
        
        # Call to nextafter(...): (line 177)
        # Processing the call arguments (line 177)
        int_420753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 32), 'int')
        int_420754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 35), 'int')
        # Processing the call keyword arguments (line 177)
        kwargs_420755 = {}
        # Getting the type of 'np' (line 177)
        np_420751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 19), 'np', False)
        # Obtaining the member 'nextafter' of a type (line 177)
        nextafter_420752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 19), np_420751, 'nextafter')
        # Calling nextafter(args, kwargs) (line 177)
        nextafter_call_result_420756 = invoke(stypy.reporting.localization.Localization(__file__, 177, 19), nextafter_420752, *[int_420753, int_420754], **kwargs_420755)
        
        # Applying the binary operator '*' (line 177)
        result_mul_420757 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 13), '*', int_420750, nextafter_call_result_420756)
        
        # Applying the binary operator '*=' (line 177)
        result_imul_420758 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 8), '*=', A_420749, result_mul_420757)
        # Assigning a type to the variable 'A' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'A', result_imul_420758)
        
        
        # Assigning a Call to a Name (line 179):
        
        # Assigning a Call to a Name (line 179):
        
        # Call to array(...): (line 179)
        # Processing the call arguments (line 179)
        
        # Obtaining an instance of the builtin type 'list' (line 179)
        list_420761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 179)
        # Adding element type (line 179)
        int_420762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 21), list_420761, int_420762)
        # Adding element type (line 179)
        int_420763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 21), list_420761, int_420763)
        
        # Processing the call keyword arguments (line 179)
        kwargs_420764 = {}
        # Getting the type of 'np' (line 179)
        np_420759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 179)
        array_420760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 12), np_420759, 'array')
        # Calling array(args, kwargs) (line 179)
        array_call_result_420765 = invoke(stypy.reporting.localization.Localization(__file__, 179, 12), array_420760, *[list_420761], **kwargs_420764)
        
        # Assigning a type to the variable 'b' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'b', array_call_result_420765)
        
        # Assigning a Call to a Tuple (line 181):
        
        # Assigning a Subscript to a Name (line 181):
        
        # Obtaining the type of the subscript
        int_420766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 8), 'int')
        
        # Call to lgmres(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'A' (line 181)
        A_420768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 26), 'A', False)
        # Getting the type of 'b' (line 181)
        b_420769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 29), 'b', False)
        # Processing the call keyword arguments (line 181)
        kwargs_420770 = {}
        # Getting the type of 'lgmres' (line 181)
        lgmres_420767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 19), 'lgmres', False)
        # Calling lgmres(args, kwargs) (line 181)
        lgmres_call_result_420771 = invoke(stypy.reporting.localization.Localization(__file__, 181, 19), lgmres_420767, *[A_420768, b_420769], **kwargs_420770)
        
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___420772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), lgmres_call_result_420771, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_420773 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), getitem___420772, int_420766)
        
        # Assigning a type to the variable 'tuple_var_assignment_419571' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'tuple_var_assignment_419571', subscript_call_result_420773)
        
        # Assigning a Subscript to a Name (line 181):
        
        # Obtaining the type of the subscript
        int_420774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 8), 'int')
        
        # Call to lgmres(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'A' (line 181)
        A_420776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 26), 'A', False)
        # Getting the type of 'b' (line 181)
        b_420777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 29), 'b', False)
        # Processing the call keyword arguments (line 181)
        kwargs_420778 = {}
        # Getting the type of 'lgmres' (line 181)
        lgmres_420775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 19), 'lgmres', False)
        # Calling lgmres(args, kwargs) (line 181)
        lgmres_call_result_420779 = invoke(stypy.reporting.localization.Localization(__file__, 181, 19), lgmres_420775, *[A_420776, b_420777], **kwargs_420778)
        
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___420780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), lgmres_call_result_420779, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_420781 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), getitem___420780, int_420774)
        
        # Assigning a type to the variable 'tuple_var_assignment_419572' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'tuple_var_assignment_419572', subscript_call_result_420781)
        
        # Assigning a Name to a Name (line 181):
        # Getting the type of 'tuple_var_assignment_419571' (line 181)
        tuple_var_assignment_419571_420782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'tuple_var_assignment_419571')
        # Assigning a type to the variable 'xp' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'xp', tuple_var_assignment_419571_420782)
        
        # Assigning a Name to a Name (line 181):
        # Getting the type of 'tuple_var_assignment_419572' (line 181)
        tuple_var_assignment_419572_420783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'tuple_var_assignment_419572')
        # Assigning a type to the variable 'info' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'info', tuple_var_assignment_419572_420783)
        
        
        # Getting the type of 'info' (line 183)
        info_420784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 11), 'info')
        int_420785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 19), 'int')
        # Applying the binary operator '==' (line 183)
        result_eq_420786 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 11), '==', info_420784, int_420785)
        
        # Testing the type of an if condition (line 183)
        if_condition_420787 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 183, 8), result_eq_420786)
        # Assigning a type to the variable 'if_condition_420787' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'if_condition_420787', if_condition_420787)
        # SSA begins for if statement (line 183)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_allclose(...): (line 184)
        # Processing the call arguments (line 184)
        
        # Call to dot(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'xp' (line 184)
        xp_420791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 34), 'xp', False)
        # Processing the call keyword arguments (line 184)
        kwargs_420792 = {}
        # Getting the type of 'A' (line 184)
        A_420789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 28), 'A', False)
        # Obtaining the member 'dot' of a type (line 184)
        dot_420790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 28), A_420789, 'dot')
        # Calling dot(args, kwargs) (line 184)
        dot_call_result_420793 = invoke(stypy.reporting.localization.Localization(__file__, 184, 28), dot_420790, *[xp_420791], **kwargs_420792)
        
        # Getting the type of 'b' (line 184)
        b_420794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 39), 'b', False)
        # Processing the call keyword arguments (line 184)
        kwargs_420795 = {}
        # Getting the type of 'assert_allclose' (line 184)
        assert_allclose_420788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 184)
        assert_allclose_call_result_420796 = invoke(stypy.reporting.localization.Localization(__file__, 184, 12), assert_allclose_420788, *[dot_call_result_420793, b_420794], **kwargs_420795)
        
        # SSA join for if statement (line 183)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_denormals(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_denormals' in the type store
        # Getting the type of 'stypy_return_type' (line 172)
        stypy_return_type_420797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_420797)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_denormals'
        return stypy_return_type_420797


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 42, 0, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLGMRES.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'TestLGMRES' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'TestLGMRES', TestLGMRES)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
