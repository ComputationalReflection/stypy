
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: '''Tests for the linalg.isolve.gcrotmk module
3: '''
4: 
5: from __future__ import division, print_function, absolute_import
6: 
7: from numpy.testing import assert_, assert_allclose, assert_equal
8: 
9: import numpy as np
10: from numpy import zeros, array, allclose
11: from scipy.linalg import norm
12: from scipy.sparse import csr_matrix, eye, rand
13: 
14: from scipy.sparse.linalg.interface import LinearOperator
15: from scipy.sparse.linalg import splu
16: from scipy.sparse.linalg.isolve import gcrotmk, gmres
17: 
18: 
19: Am = csr_matrix(array([[-2,1,0,0,0,9],
20:                        [1,-2,1,0,5,0],
21:                        [0,1,-2,1,0,0],
22:                        [0,0,1,-2,1,0],
23:                        [0,3,0,1,-2,1],
24:                        [1,0,0,0,1,-2]]))
25: b = array([1,2,3,4,5,6])
26: count = [0]
27: 
28: 
29: def matvec(v):
30:     count[0] += 1
31:     return Am*v
32: A = LinearOperator(matvec=matvec, shape=Am.shape, dtype=Am.dtype)
33: 
34: 
35: def do_solve(**kw):
36:     count[0] = 0
37:     x0, flag = gcrotmk(A, b, x0=zeros(A.shape[0]), tol=1e-14, **kw)
38:     count_0 = count[0]
39:     assert_(allclose(A*x0, b, rtol=1e-12, atol=1e-12), norm(A*x0-b))
40:     return x0, count_0
41: 
42: 
43: class TestGCROTMK(object):
44:     def test_preconditioner(self):
45:         # Check that preconditioning works
46:         pc = splu(Am.tocsc())
47:         M = LinearOperator(matvec=pc.solve, shape=A.shape, dtype=A.dtype)
48: 
49:         x0, count_0 = do_solve()
50:         x1, count_1 = do_solve(M=M)
51: 
52:         assert_equal(count_1, 2)
53:         assert_(count_1 < count_0/2)
54:         assert_(allclose(x1, x0, rtol=1e-14))
55: 
56:     def test_arnoldi(self):
57:         np.random.rand(1234)
58: 
59:         A = eye(10000) + rand(10000,10000,density=1e-4)
60:         b = np.random.rand(10000)
61: 
62:         # The inner arnoldi should be equivalent to gmres
63:         x0, flag0 = gcrotmk(A, b, x0=zeros(A.shape[0]), m=15, k=0, maxiter=1)
64:         x1, flag1 = gmres(A, b, x0=zeros(A.shape[0]), restart=15, maxiter=1)
65: 
66:         assert_equal(flag0, 1)
67:         assert_equal(flag1, 1)
68:         assert_(np.linalg.norm(A.dot(x0) - b) > 1e-3)
69: 
70:         assert_allclose(x0, x1)
71: 
72:     def test_cornercase(self):
73:         np.random.seed(1234)
74: 
75:         # Rounding error may prevent convergence with tol=0 --- ensure
76:         # that the return values in this case are correct, and no
77:         # exceptions are raised
78: 
79:         for n in [3, 5, 10, 100]:
80:             A = 2*eye(n)
81: 
82:             b = np.ones(n)
83:             x, info = gcrotmk(A, b, maxiter=10)
84:             assert_equal(info, 0)
85:             assert_allclose(A.dot(x) - b, 0, atol=1e-14)
86: 
87:             x, info = gcrotmk(A, b, tol=0, maxiter=10)
88:             if info == 0:
89:                 assert_allclose(A.dot(x) - b, 0, atol=1e-14)
90: 
91:             b = np.random.rand(n)
92:             x, info = gcrotmk(A, b, maxiter=10)
93:             assert_equal(info, 0)
94:             assert_allclose(A.dot(x) - b, 0, atol=1e-14)
95: 
96:             x, info = gcrotmk(A, b, tol=0, maxiter=10)
97:             if info == 0:
98:                 assert_allclose(A.dot(x) - b, 0, atol=1e-14)
99: 
100:     def test_nans(self):
101:         A = eye(3, format='lil')
102:         A[1,1] = np.nan
103:         b = np.ones(3)
104: 
105:         x, info = gcrotmk(A, b, tol=0, maxiter=10)
106:         assert_equal(info, 1)
107: 
108:     def test_truncate(self):
109:         np.random.seed(1234)
110:         A = np.random.rand(30, 30) + np.eye(30)
111:         b = np.random.rand(30)
112: 
113:         for truncate in ['oldest', 'smallest']:
114:             x, info = gcrotmk(A, b, m=10, k=10, truncate=truncate, tol=1e-4,
115:                               maxiter=200)
116:             assert_equal(info, 0)
117:             assert_allclose(A.dot(x) - b, 0, atol=1e-3)
118: 
119:     def test_CU(self):
120:         for discard_C in (True, False):
121:             # Check that C,U behave as expected
122:             CU = []
123:             x0, count_0 = do_solve(CU=CU, discard_C=discard_C)
124:             assert_(len(CU) > 0)
125:             assert_(len(CU) <= 6)
126: 
127:             if discard_C:
128:                 for c, u in CU:
129:                     assert_(c is None)
130: 
131:             # should converge immediately
132:             x1, count_1 = do_solve(CU=CU, discard_C=discard_C)
133:             if discard_C:
134:                 assert_equal(count_1, 1 + len(CU))
135:             else:
136:                 assert_equal(count_1, 2)
137:             assert_(count_1 < count_0/2)
138:             assert_allclose(x1, x0, atol=1e-14)
139: 
140:     def test_denormals(self):
141:         # Check that no warnings are emitted if the matrix contains
142:         # numbers for which 1/x has no float representation, and that
143:         # the solver behaves properly.
144:         A = np.array([[1, 2], [3, 4]], dtype=float)
145:         A *= 100 * np.nextafter(0, 1)
146: 
147:         b = np.array([1, 1])
148: 
149:         xp, info = gcrotmk(A, b)
150: 
151:         if info == 0:
152:             assert_allclose(A.dot(xp), b)
153: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_416769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', 'Tests for the linalg.isolve.gcrotmk module\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.testing import assert_, assert_allclose, assert_equal' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_416770 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing')

if (type(import_416770) is not StypyTypeError):

    if (import_416770 != 'pyd_module'):
        __import__(import_416770)
        sys_modules_416771 = sys.modules[import_416770]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', sys_modules_416771.module_type_store, module_type_store, ['assert_', 'assert_allclose', 'assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_416771, sys_modules_416771.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_allclose, assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_allclose', 'assert_equal'], [assert_, assert_allclose, assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', import_416770)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import numpy' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_416772 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy')

if (type(import_416772) is not StypyTypeError):

    if (import_416772 != 'pyd_module'):
        __import__(import_416772)
        sys_modules_416773 = sys.modules[import_416772]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', sys_modules_416773.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', import_416772)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from numpy import zeros, array, allclose' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_416774 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy')

if (type(import_416774) is not StypyTypeError):

    if (import_416774 != 'pyd_module'):
        __import__(import_416774)
        sys_modules_416775 = sys.modules[import_416774]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', sys_modules_416775.module_type_store, module_type_store, ['zeros', 'array', 'allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_416775, sys_modules_416775.module_type_store, module_type_store)
    else:
        from numpy import zeros, array, allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', None, module_type_store, ['zeros', 'array', 'allclose'], [zeros, array, allclose])

else:
    # Assigning a type to the variable 'numpy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', import_416774)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.linalg import norm' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_416776 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg')

if (type(import_416776) is not StypyTypeError):

    if (import_416776 != 'pyd_module'):
        __import__(import_416776)
        sys_modules_416777 = sys.modules[import_416776]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg', sys_modules_416777.module_type_store, module_type_store, ['norm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_416777, sys_modules_416777.module_type_store, module_type_store)
    else:
        from scipy.linalg import norm

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg', None, module_type_store, ['norm'], [norm])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg', import_416776)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.sparse import csr_matrix, eye, rand' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_416778 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse')

if (type(import_416778) is not StypyTypeError):

    if (import_416778 != 'pyd_module'):
        __import__(import_416778)
        sys_modules_416779 = sys.modules[import_416778]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse', sys_modules_416779.module_type_store, module_type_store, ['csr_matrix', 'eye', 'rand'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_416779, sys_modules_416779.module_type_store, module_type_store)
    else:
        from scipy.sparse import csr_matrix, eye, rand

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse', None, module_type_store, ['csr_matrix', 'eye', 'rand'], [csr_matrix, eye, rand])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse', import_416778)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy.sparse.linalg.interface import LinearOperator' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_416780 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse.linalg.interface')

if (type(import_416780) is not StypyTypeError):

    if (import_416780 != 'pyd_module'):
        __import__(import_416780)
        sys_modules_416781 = sys.modules[import_416780]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse.linalg.interface', sys_modules_416781.module_type_store, module_type_store, ['LinearOperator'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_416781, sys_modules_416781.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.interface import LinearOperator

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse.linalg.interface', None, module_type_store, ['LinearOperator'], [LinearOperator])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.interface' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse.linalg.interface', import_416780)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from scipy.sparse.linalg import splu' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_416782 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse.linalg')

if (type(import_416782) is not StypyTypeError):

    if (import_416782 != 'pyd_module'):
        __import__(import_416782)
        sys_modules_416783 = sys.modules[import_416782]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse.linalg', sys_modules_416783.module_type_store, module_type_store, ['splu'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_416783, sys_modules_416783.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import splu

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse.linalg', None, module_type_store, ['splu'], [splu])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse.linalg', import_416782)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy.sparse.linalg.isolve import gcrotmk, gmres' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_416784 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse.linalg.isolve')

if (type(import_416784) is not StypyTypeError):

    if (import_416784 != 'pyd_module'):
        __import__(import_416784)
        sys_modules_416785 = sys.modules[import_416784]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse.linalg.isolve', sys_modules_416785.module_type_store, module_type_store, ['gcrotmk', 'gmres'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_416785, sys_modules_416785.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.isolve import gcrotmk, gmres

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse.linalg.isolve', None, module_type_store, ['gcrotmk', 'gmres'], [gcrotmk, gmres])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.isolve' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse.linalg.isolve', import_416784)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')


# Assigning a Call to a Name (line 19):

# Assigning a Call to a Name (line 19):

# Call to csr_matrix(...): (line 19)
# Processing the call arguments (line 19)

# Call to array(...): (line 19)
# Processing the call arguments (line 19)

# Obtaining an instance of the builtin type 'list' (line 19)
list_416788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)

# Obtaining an instance of the builtin type 'list' (line 19)
list_416789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)
int_416790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_416789, int_416790)
# Adding element type (line 19)
int_416791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_416789, int_416791)
# Adding element type (line 19)
int_416792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_416789, int_416792)
# Adding element type (line 19)
int_416793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_416789, int_416793)
# Adding element type (line 19)
int_416794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_416789, int_416794)
# Adding element type (line 19)
int_416795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_416789, int_416795)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 22), list_416788, list_416789)
# Adding element type (line 19)

# Obtaining an instance of the builtin type 'list' (line 20)
list_416796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 20)
# Adding element type (line 20)
int_416797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), list_416796, int_416797)
# Adding element type (line 20)
int_416798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), list_416796, int_416798)
# Adding element type (line 20)
int_416799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), list_416796, int_416799)
# Adding element type (line 20)
int_416800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), list_416796, int_416800)
# Adding element type (line 20)
int_416801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), list_416796, int_416801)
# Adding element type (line 20)
int_416802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), list_416796, int_416802)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 22), list_416788, list_416796)
# Adding element type (line 19)

# Obtaining an instance of the builtin type 'list' (line 21)
list_416803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)
int_416804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_416803, int_416804)
# Adding element type (line 21)
int_416805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_416803, int_416805)
# Adding element type (line 21)
int_416806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_416803, int_416806)
# Adding element type (line 21)
int_416807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_416803, int_416807)
# Adding element type (line 21)
int_416808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_416803, int_416808)
# Adding element type (line 21)
int_416809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_416803, int_416809)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 22), list_416788, list_416803)
# Adding element type (line 19)

# Obtaining an instance of the builtin type 'list' (line 22)
list_416810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 22)
# Adding element type (line 22)
int_416811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 23), list_416810, int_416811)
# Adding element type (line 22)
int_416812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 23), list_416810, int_416812)
# Adding element type (line 22)
int_416813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 23), list_416810, int_416813)
# Adding element type (line 22)
int_416814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 23), list_416810, int_416814)
# Adding element type (line 22)
int_416815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 23), list_416810, int_416815)
# Adding element type (line 22)
int_416816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 23), list_416810, int_416816)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 22), list_416788, list_416810)
# Adding element type (line 19)

# Obtaining an instance of the builtin type 'list' (line 23)
list_416817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 23)
# Adding element type (line 23)
int_416818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 23), list_416817, int_416818)
# Adding element type (line 23)
int_416819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 23), list_416817, int_416819)
# Adding element type (line 23)
int_416820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 23), list_416817, int_416820)
# Adding element type (line 23)
int_416821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 23), list_416817, int_416821)
# Adding element type (line 23)
int_416822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 23), list_416817, int_416822)
# Adding element type (line 23)
int_416823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 23), list_416817, int_416823)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 22), list_416788, list_416817)
# Adding element type (line 19)

# Obtaining an instance of the builtin type 'list' (line 24)
list_416824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 24)
# Adding element type (line 24)
int_416825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 23), list_416824, int_416825)
# Adding element type (line 24)
int_416826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 23), list_416824, int_416826)
# Adding element type (line 24)
int_416827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 23), list_416824, int_416827)
# Adding element type (line 24)
int_416828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 23), list_416824, int_416828)
# Adding element type (line 24)
int_416829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 23), list_416824, int_416829)
# Adding element type (line 24)
int_416830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 23), list_416824, int_416830)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 22), list_416788, list_416824)

# Processing the call keyword arguments (line 19)
kwargs_416831 = {}
# Getting the type of 'array' (line 19)
array_416787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 16), 'array', False)
# Calling array(args, kwargs) (line 19)
array_call_result_416832 = invoke(stypy.reporting.localization.Localization(__file__, 19, 16), array_416787, *[list_416788], **kwargs_416831)

# Processing the call keyword arguments (line 19)
kwargs_416833 = {}
# Getting the type of 'csr_matrix' (line 19)
csr_matrix_416786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'csr_matrix', False)
# Calling csr_matrix(args, kwargs) (line 19)
csr_matrix_call_result_416834 = invoke(stypy.reporting.localization.Localization(__file__, 19, 5), csr_matrix_416786, *[array_call_result_416832], **kwargs_416833)

# Assigning a type to the variable 'Am' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'Am', csr_matrix_call_result_416834)

# Assigning a Call to a Name (line 25):

# Assigning a Call to a Name (line 25):

# Call to array(...): (line 25)
# Processing the call arguments (line 25)

# Obtaining an instance of the builtin type 'list' (line 25)
list_416836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 25)
# Adding element type (line 25)
int_416837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 10), list_416836, int_416837)
# Adding element type (line 25)
int_416838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 10), list_416836, int_416838)
# Adding element type (line 25)
int_416839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 10), list_416836, int_416839)
# Adding element type (line 25)
int_416840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 10), list_416836, int_416840)
# Adding element type (line 25)
int_416841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 10), list_416836, int_416841)
# Adding element type (line 25)
int_416842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 10), list_416836, int_416842)

# Processing the call keyword arguments (line 25)
kwargs_416843 = {}
# Getting the type of 'array' (line 25)
array_416835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'array', False)
# Calling array(args, kwargs) (line 25)
array_call_result_416844 = invoke(stypy.reporting.localization.Localization(__file__, 25, 4), array_416835, *[list_416836], **kwargs_416843)

# Assigning a type to the variable 'b' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'b', array_call_result_416844)

# Assigning a List to a Name (line 26):

# Assigning a List to a Name (line 26):

# Obtaining an instance of the builtin type 'list' (line 26)
list_416845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 26)
# Adding element type (line 26)
int_416846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 8), list_416845, int_416846)

# Assigning a type to the variable 'count' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'count', list_416845)

@norecursion
def matvec(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'matvec'
    module_type_store = module_type_store.open_function_context('matvec', 29, 0, False)
    
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

    
    # Getting the type of 'count' (line 30)
    count_416847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'count')
    
    # Obtaining the type of the subscript
    int_416848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 10), 'int')
    # Getting the type of 'count' (line 30)
    count_416849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'count')
    # Obtaining the member '__getitem__' of a type (line 30)
    getitem___416850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 4), count_416849, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 30)
    subscript_call_result_416851 = invoke(stypy.reporting.localization.Localization(__file__, 30, 4), getitem___416850, int_416848)
    
    int_416852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 16), 'int')
    # Applying the binary operator '+=' (line 30)
    result_iadd_416853 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 4), '+=', subscript_call_result_416851, int_416852)
    # Getting the type of 'count' (line 30)
    count_416854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'count')
    int_416855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 10), 'int')
    # Storing an element on a container (line 30)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 4), count_416854, (int_416855, result_iadd_416853))
    
    # Getting the type of 'Am' (line 31)
    Am_416856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), 'Am')
    # Getting the type of 'v' (line 31)
    v_416857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'v')
    # Applying the binary operator '*' (line 31)
    result_mul_416858 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 11), '*', Am_416856, v_416857)
    
    # Assigning a type to the variable 'stypy_return_type' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type', result_mul_416858)
    
    # ################# End of 'matvec(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'matvec' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_416859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_416859)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'matvec'
    return stypy_return_type_416859

# Assigning a type to the variable 'matvec' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'matvec', matvec)

# Assigning a Call to a Name (line 32):

# Assigning a Call to a Name (line 32):

# Call to LinearOperator(...): (line 32)
# Processing the call keyword arguments (line 32)
# Getting the type of 'matvec' (line 32)
matvec_416861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 26), 'matvec', False)
keyword_416862 = matvec_416861
# Getting the type of 'Am' (line 32)
Am_416863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 40), 'Am', False)
# Obtaining the member 'shape' of a type (line 32)
shape_416864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 40), Am_416863, 'shape')
keyword_416865 = shape_416864
# Getting the type of 'Am' (line 32)
Am_416866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 56), 'Am', False)
# Obtaining the member 'dtype' of a type (line 32)
dtype_416867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 56), Am_416866, 'dtype')
keyword_416868 = dtype_416867
kwargs_416869 = {'dtype': keyword_416868, 'shape': keyword_416865, 'matvec': keyword_416862}
# Getting the type of 'LinearOperator' (line 32)
LinearOperator_416860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'LinearOperator', False)
# Calling LinearOperator(args, kwargs) (line 32)
LinearOperator_call_result_416870 = invoke(stypy.reporting.localization.Localization(__file__, 32, 4), LinearOperator_416860, *[], **kwargs_416869)

# Assigning a type to the variable 'A' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'A', LinearOperator_call_result_416870)

@norecursion
def do_solve(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'do_solve'
    module_type_store = module_type_store.open_function_context('do_solve', 35, 0, False)
    
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

    
    # Assigning a Num to a Subscript (line 36):
    
    # Assigning a Num to a Subscript (line 36):
    int_416871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 15), 'int')
    # Getting the type of 'count' (line 36)
    count_416872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'count')
    int_416873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 10), 'int')
    # Storing an element on a container (line 36)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 4), count_416872, (int_416873, int_416871))
    
    # Assigning a Call to a Tuple (line 37):
    
    # Assigning a Subscript to a Name (line 37):
    
    # Obtaining the type of the subscript
    int_416874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 4), 'int')
    
    # Call to gcrotmk(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'A' (line 37)
    A_416876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 23), 'A', False)
    # Getting the type of 'b' (line 37)
    b_416877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 26), 'b', False)
    # Processing the call keyword arguments (line 37)
    
    # Call to zeros(...): (line 37)
    # Processing the call arguments (line 37)
    
    # Obtaining the type of the subscript
    int_416879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 46), 'int')
    # Getting the type of 'A' (line 37)
    A_416880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 38), 'A', False)
    # Obtaining the member 'shape' of a type (line 37)
    shape_416881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 38), A_416880, 'shape')
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___416882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 38), shape_416881, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_416883 = invoke(stypy.reporting.localization.Localization(__file__, 37, 38), getitem___416882, int_416879)
    
    # Processing the call keyword arguments (line 37)
    kwargs_416884 = {}
    # Getting the type of 'zeros' (line 37)
    zeros_416878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 32), 'zeros', False)
    # Calling zeros(args, kwargs) (line 37)
    zeros_call_result_416885 = invoke(stypy.reporting.localization.Localization(__file__, 37, 32), zeros_416878, *[subscript_call_result_416883], **kwargs_416884)
    
    keyword_416886 = zeros_call_result_416885
    float_416887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 55), 'float')
    keyword_416888 = float_416887
    # Getting the type of 'kw' (line 37)
    kw_416889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 64), 'kw', False)
    kwargs_416890 = {'x0': keyword_416886, 'kw_416889': kw_416889, 'tol': keyword_416888}
    # Getting the type of 'gcrotmk' (line 37)
    gcrotmk_416875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'gcrotmk', False)
    # Calling gcrotmk(args, kwargs) (line 37)
    gcrotmk_call_result_416891 = invoke(stypy.reporting.localization.Localization(__file__, 37, 15), gcrotmk_416875, *[A_416876, b_416877], **kwargs_416890)
    
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___416892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 4), gcrotmk_call_result_416891, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_416893 = invoke(stypy.reporting.localization.Localization(__file__, 37, 4), getitem___416892, int_416874)
    
    # Assigning a type to the variable 'tuple_var_assignment_416741' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'tuple_var_assignment_416741', subscript_call_result_416893)
    
    # Assigning a Subscript to a Name (line 37):
    
    # Obtaining the type of the subscript
    int_416894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 4), 'int')
    
    # Call to gcrotmk(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'A' (line 37)
    A_416896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 23), 'A', False)
    # Getting the type of 'b' (line 37)
    b_416897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 26), 'b', False)
    # Processing the call keyword arguments (line 37)
    
    # Call to zeros(...): (line 37)
    # Processing the call arguments (line 37)
    
    # Obtaining the type of the subscript
    int_416899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 46), 'int')
    # Getting the type of 'A' (line 37)
    A_416900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 38), 'A', False)
    # Obtaining the member 'shape' of a type (line 37)
    shape_416901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 38), A_416900, 'shape')
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___416902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 38), shape_416901, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_416903 = invoke(stypy.reporting.localization.Localization(__file__, 37, 38), getitem___416902, int_416899)
    
    # Processing the call keyword arguments (line 37)
    kwargs_416904 = {}
    # Getting the type of 'zeros' (line 37)
    zeros_416898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 32), 'zeros', False)
    # Calling zeros(args, kwargs) (line 37)
    zeros_call_result_416905 = invoke(stypy.reporting.localization.Localization(__file__, 37, 32), zeros_416898, *[subscript_call_result_416903], **kwargs_416904)
    
    keyword_416906 = zeros_call_result_416905
    float_416907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 55), 'float')
    keyword_416908 = float_416907
    # Getting the type of 'kw' (line 37)
    kw_416909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 64), 'kw', False)
    kwargs_416910 = {'x0': keyword_416906, 'kw_416909': kw_416909, 'tol': keyword_416908}
    # Getting the type of 'gcrotmk' (line 37)
    gcrotmk_416895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'gcrotmk', False)
    # Calling gcrotmk(args, kwargs) (line 37)
    gcrotmk_call_result_416911 = invoke(stypy.reporting.localization.Localization(__file__, 37, 15), gcrotmk_416895, *[A_416896, b_416897], **kwargs_416910)
    
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___416912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 4), gcrotmk_call_result_416911, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_416913 = invoke(stypy.reporting.localization.Localization(__file__, 37, 4), getitem___416912, int_416894)
    
    # Assigning a type to the variable 'tuple_var_assignment_416742' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'tuple_var_assignment_416742', subscript_call_result_416913)
    
    # Assigning a Name to a Name (line 37):
    # Getting the type of 'tuple_var_assignment_416741' (line 37)
    tuple_var_assignment_416741_416914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'tuple_var_assignment_416741')
    # Assigning a type to the variable 'x0' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'x0', tuple_var_assignment_416741_416914)
    
    # Assigning a Name to a Name (line 37):
    # Getting the type of 'tuple_var_assignment_416742' (line 37)
    tuple_var_assignment_416742_416915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'tuple_var_assignment_416742')
    # Assigning a type to the variable 'flag' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'flag', tuple_var_assignment_416742_416915)
    
    # Assigning a Subscript to a Name (line 38):
    
    # Assigning a Subscript to a Name (line 38):
    
    # Obtaining the type of the subscript
    int_416916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 20), 'int')
    # Getting the type of 'count' (line 38)
    count_416917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 14), 'count')
    # Obtaining the member '__getitem__' of a type (line 38)
    getitem___416918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 14), count_416917, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 38)
    subscript_call_result_416919 = invoke(stypy.reporting.localization.Localization(__file__, 38, 14), getitem___416918, int_416916)
    
    # Assigning a type to the variable 'count_0' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'count_0', subscript_call_result_416919)
    
    # Call to assert_(...): (line 39)
    # Processing the call arguments (line 39)
    
    # Call to allclose(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'A' (line 39)
    A_416922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 21), 'A', False)
    # Getting the type of 'x0' (line 39)
    x0_416923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 23), 'x0', False)
    # Applying the binary operator '*' (line 39)
    result_mul_416924 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 21), '*', A_416922, x0_416923)
    
    # Getting the type of 'b' (line 39)
    b_416925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 27), 'b', False)
    # Processing the call keyword arguments (line 39)
    float_416926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 35), 'float')
    keyword_416927 = float_416926
    float_416928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 47), 'float')
    keyword_416929 = float_416928
    kwargs_416930 = {'rtol': keyword_416927, 'atol': keyword_416929}
    # Getting the type of 'allclose' (line 39)
    allclose_416921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'allclose', False)
    # Calling allclose(args, kwargs) (line 39)
    allclose_call_result_416931 = invoke(stypy.reporting.localization.Localization(__file__, 39, 12), allclose_416921, *[result_mul_416924, b_416925], **kwargs_416930)
    
    
    # Call to norm(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'A' (line 39)
    A_416933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 60), 'A', False)
    # Getting the type of 'x0' (line 39)
    x0_416934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 62), 'x0', False)
    # Applying the binary operator '*' (line 39)
    result_mul_416935 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 60), '*', A_416933, x0_416934)
    
    # Getting the type of 'b' (line 39)
    b_416936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 65), 'b', False)
    # Applying the binary operator '-' (line 39)
    result_sub_416937 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 60), '-', result_mul_416935, b_416936)
    
    # Processing the call keyword arguments (line 39)
    kwargs_416938 = {}
    # Getting the type of 'norm' (line 39)
    norm_416932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 55), 'norm', False)
    # Calling norm(args, kwargs) (line 39)
    norm_call_result_416939 = invoke(stypy.reporting.localization.Localization(__file__, 39, 55), norm_416932, *[result_sub_416937], **kwargs_416938)
    
    # Processing the call keyword arguments (line 39)
    kwargs_416940 = {}
    # Getting the type of 'assert_' (line 39)
    assert__416920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 39)
    assert__call_result_416941 = invoke(stypy.reporting.localization.Localization(__file__, 39, 4), assert__416920, *[allclose_call_result_416931, norm_call_result_416939], **kwargs_416940)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 40)
    tuple_416942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 40)
    # Adding element type (line 40)
    # Getting the type of 'x0' (line 40)
    x0_416943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'x0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 11), tuple_416942, x0_416943)
    # Adding element type (line 40)
    # Getting the type of 'count_0' (line 40)
    count_0_416944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'count_0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 11), tuple_416942, count_0_416944)
    
    # Assigning a type to the variable 'stypy_return_type' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'stypy_return_type', tuple_416942)
    
    # ################# End of 'do_solve(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'do_solve' in the type store
    # Getting the type of 'stypy_return_type' (line 35)
    stypy_return_type_416945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_416945)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'do_solve'
    return stypy_return_type_416945

# Assigning a type to the variable 'do_solve' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'do_solve', do_solve)
# Declaration of the 'TestGCROTMK' class

class TestGCROTMK(object, ):

    @norecursion
    def test_preconditioner(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_preconditioner'
        module_type_store = module_type_store.open_function_context('test_preconditioner', 44, 4, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGCROTMK.test_preconditioner.__dict__.__setitem__('stypy_localization', localization)
        TestGCROTMK.test_preconditioner.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGCROTMK.test_preconditioner.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGCROTMK.test_preconditioner.__dict__.__setitem__('stypy_function_name', 'TestGCROTMK.test_preconditioner')
        TestGCROTMK.test_preconditioner.__dict__.__setitem__('stypy_param_names_list', [])
        TestGCROTMK.test_preconditioner.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGCROTMK.test_preconditioner.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGCROTMK.test_preconditioner.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGCROTMK.test_preconditioner.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGCROTMK.test_preconditioner.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGCROTMK.test_preconditioner.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGCROTMK.test_preconditioner', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 46):
        
        # Assigning a Call to a Name (line 46):
        
        # Call to splu(...): (line 46)
        # Processing the call arguments (line 46)
        
        # Call to tocsc(...): (line 46)
        # Processing the call keyword arguments (line 46)
        kwargs_416949 = {}
        # Getting the type of 'Am' (line 46)
        Am_416947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 18), 'Am', False)
        # Obtaining the member 'tocsc' of a type (line 46)
        tocsc_416948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 18), Am_416947, 'tocsc')
        # Calling tocsc(args, kwargs) (line 46)
        tocsc_call_result_416950 = invoke(stypy.reporting.localization.Localization(__file__, 46, 18), tocsc_416948, *[], **kwargs_416949)
        
        # Processing the call keyword arguments (line 46)
        kwargs_416951 = {}
        # Getting the type of 'splu' (line 46)
        splu_416946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 13), 'splu', False)
        # Calling splu(args, kwargs) (line 46)
        splu_call_result_416952 = invoke(stypy.reporting.localization.Localization(__file__, 46, 13), splu_416946, *[tocsc_call_result_416950], **kwargs_416951)
        
        # Assigning a type to the variable 'pc' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'pc', splu_call_result_416952)
        
        # Assigning a Call to a Name (line 47):
        
        # Assigning a Call to a Name (line 47):
        
        # Call to LinearOperator(...): (line 47)
        # Processing the call keyword arguments (line 47)
        # Getting the type of 'pc' (line 47)
        pc_416954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 34), 'pc', False)
        # Obtaining the member 'solve' of a type (line 47)
        solve_416955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 34), pc_416954, 'solve')
        keyword_416956 = solve_416955
        # Getting the type of 'A' (line 47)
        A_416957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 50), 'A', False)
        # Obtaining the member 'shape' of a type (line 47)
        shape_416958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 50), A_416957, 'shape')
        keyword_416959 = shape_416958
        # Getting the type of 'A' (line 47)
        A_416960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 65), 'A', False)
        # Obtaining the member 'dtype' of a type (line 47)
        dtype_416961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 65), A_416960, 'dtype')
        keyword_416962 = dtype_416961
        kwargs_416963 = {'dtype': keyword_416962, 'shape': keyword_416959, 'matvec': keyword_416956}
        # Getting the type of 'LinearOperator' (line 47)
        LinearOperator_416953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'LinearOperator', False)
        # Calling LinearOperator(args, kwargs) (line 47)
        LinearOperator_call_result_416964 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), LinearOperator_416953, *[], **kwargs_416963)
        
        # Assigning a type to the variable 'M' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'M', LinearOperator_call_result_416964)
        
        # Assigning a Call to a Tuple (line 49):
        
        # Assigning a Subscript to a Name (line 49):
        
        # Obtaining the type of the subscript
        int_416965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 8), 'int')
        
        # Call to do_solve(...): (line 49)
        # Processing the call keyword arguments (line 49)
        kwargs_416967 = {}
        # Getting the type of 'do_solve' (line 49)
        do_solve_416966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 22), 'do_solve', False)
        # Calling do_solve(args, kwargs) (line 49)
        do_solve_call_result_416968 = invoke(stypy.reporting.localization.Localization(__file__, 49, 22), do_solve_416966, *[], **kwargs_416967)
        
        # Obtaining the member '__getitem__' of a type (line 49)
        getitem___416969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), do_solve_call_result_416968, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
        subscript_call_result_416970 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), getitem___416969, int_416965)
        
        # Assigning a type to the variable 'tuple_var_assignment_416743' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_var_assignment_416743', subscript_call_result_416970)
        
        # Assigning a Subscript to a Name (line 49):
        
        # Obtaining the type of the subscript
        int_416971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 8), 'int')
        
        # Call to do_solve(...): (line 49)
        # Processing the call keyword arguments (line 49)
        kwargs_416973 = {}
        # Getting the type of 'do_solve' (line 49)
        do_solve_416972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 22), 'do_solve', False)
        # Calling do_solve(args, kwargs) (line 49)
        do_solve_call_result_416974 = invoke(stypy.reporting.localization.Localization(__file__, 49, 22), do_solve_416972, *[], **kwargs_416973)
        
        # Obtaining the member '__getitem__' of a type (line 49)
        getitem___416975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), do_solve_call_result_416974, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
        subscript_call_result_416976 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), getitem___416975, int_416971)
        
        # Assigning a type to the variable 'tuple_var_assignment_416744' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_var_assignment_416744', subscript_call_result_416976)
        
        # Assigning a Name to a Name (line 49):
        # Getting the type of 'tuple_var_assignment_416743' (line 49)
        tuple_var_assignment_416743_416977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_var_assignment_416743')
        # Assigning a type to the variable 'x0' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'x0', tuple_var_assignment_416743_416977)
        
        # Assigning a Name to a Name (line 49):
        # Getting the type of 'tuple_var_assignment_416744' (line 49)
        tuple_var_assignment_416744_416978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_var_assignment_416744')
        # Assigning a type to the variable 'count_0' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'count_0', tuple_var_assignment_416744_416978)
        
        # Assigning a Call to a Tuple (line 50):
        
        # Assigning a Subscript to a Name (line 50):
        
        # Obtaining the type of the subscript
        int_416979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 8), 'int')
        
        # Call to do_solve(...): (line 50)
        # Processing the call keyword arguments (line 50)
        # Getting the type of 'M' (line 50)
        M_416981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 33), 'M', False)
        keyword_416982 = M_416981
        kwargs_416983 = {'M': keyword_416982}
        # Getting the type of 'do_solve' (line 50)
        do_solve_416980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 22), 'do_solve', False)
        # Calling do_solve(args, kwargs) (line 50)
        do_solve_call_result_416984 = invoke(stypy.reporting.localization.Localization(__file__, 50, 22), do_solve_416980, *[], **kwargs_416983)
        
        # Obtaining the member '__getitem__' of a type (line 50)
        getitem___416985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), do_solve_call_result_416984, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 50)
        subscript_call_result_416986 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), getitem___416985, int_416979)
        
        # Assigning a type to the variable 'tuple_var_assignment_416745' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'tuple_var_assignment_416745', subscript_call_result_416986)
        
        # Assigning a Subscript to a Name (line 50):
        
        # Obtaining the type of the subscript
        int_416987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 8), 'int')
        
        # Call to do_solve(...): (line 50)
        # Processing the call keyword arguments (line 50)
        # Getting the type of 'M' (line 50)
        M_416989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 33), 'M', False)
        keyword_416990 = M_416989
        kwargs_416991 = {'M': keyword_416990}
        # Getting the type of 'do_solve' (line 50)
        do_solve_416988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 22), 'do_solve', False)
        # Calling do_solve(args, kwargs) (line 50)
        do_solve_call_result_416992 = invoke(stypy.reporting.localization.Localization(__file__, 50, 22), do_solve_416988, *[], **kwargs_416991)
        
        # Obtaining the member '__getitem__' of a type (line 50)
        getitem___416993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), do_solve_call_result_416992, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 50)
        subscript_call_result_416994 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), getitem___416993, int_416987)
        
        # Assigning a type to the variable 'tuple_var_assignment_416746' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'tuple_var_assignment_416746', subscript_call_result_416994)
        
        # Assigning a Name to a Name (line 50):
        # Getting the type of 'tuple_var_assignment_416745' (line 50)
        tuple_var_assignment_416745_416995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'tuple_var_assignment_416745')
        # Assigning a type to the variable 'x1' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'x1', tuple_var_assignment_416745_416995)
        
        # Assigning a Name to a Name (line 50):
        # Getting the type of 'tuple_var_assignment_416746' (line 50)
        tuple_var_assignment_416746_416996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'tuple_var_assignment_416746')
        # Assigning a type to the variable 'count_1' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'count_1', tuple_var_assignment_416746_416996)
        
        # Call to assert_equal(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'count_1' (line 52)
        count_1_416998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 21), 'count_1', False)
        int_416999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 30), 'int')
        # Processing the call keyword arguments (line 52)
        kwargs_417000 = {}
        # Getting the type of 'assert_equal' (line 52)
        assert_equal_416997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 52)
        assert_equal_call_result_417001 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), assert_equal_416997, *[count_1_416998, int_416999], **kwargs_417000)
        
        
        # Call to assert_(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Getting the type of 'count_1' (line 53)
        count_1_417003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 16), 'count_1', False)
        # Getting the type of 'count_0' (line 53)
        count_0_417004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 26), 'count_0', False)
        int_417005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 34), 'int')
        # Applying the binary operator 'div' (line 53)
        result_div_417006 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 26), 'div', count_0_417004, int_417005)
        
        # Applying the binary operator '<' (line 53)
        result_lt_417007 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 16), '<', count_1_417003, result_div_417006)
        
        # Processing the call keyword arguments (line 53)
        kwargs_417008 = {}
        # Getting the type of 'assert_' (line 53)
        assert__417002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 53)
        assert__call_result_417009 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), assert__417002, *[result_lt_417007], **kwargs_417008)
        
        
        # Call to assert_(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Call to allclose(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'x1' (line 54)
        x1_417012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 25), 'x1', False)
        # Getting the type of 'x0' (line 54)
        x0_417013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 29), 'x0', False)
        # Processing the call keyword arguments (line 54)
        float_417014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 38), 'float')
        keyword_417015 = float_417014
        kwargs_417016 = {'rtol': keyword_417015}
        # Getting the type of 'allclose' (line 54)
        allclose_417011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'allclose', False)
        # Calling allclose(args, kwargs) (line 54)
        allclose_call_result_417017 = invoke(stypy.reporting.localization.Localization(__file__, 54, 16), allclose_417011, *[x1_417012, x0_417013], **kwargs_417016)
        
        # Processing the call keyword arguments (line 54)
        kwargs_417018 = {}
        # Getting the type of 'assert_' (line 54)
        assert__417010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 54)
        assert__call_result_417019 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), assert__417010, *[allclose_call_result_417017], **kwargs_417018)
        
        
        # ################# End of 'test_preconditioner(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_preconditioner' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_417020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_417020)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_preconditioner'
        return stypy_return_type_417020


    @norecursion
    def test_arnoldi(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_arnoldi'
        module_type_store = module_type_store.open_function_context('test_arnoldi', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGCROTMK.test_arnoldi.__dict__.__setitem__('stypy_localization', localization)
        TestGCROTMK.test_arnoldi.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGCROTMK.test_arnoldi.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGCROTMK.test_arnoldi.__dict__.__setitem__('stypy_function_name', 'TestGCROTMK.test_arnoldi')
        TestGCROTMK.test_arnoldi.__dict__.__setitem__('stypy_param_names_list', [])
        TestGCROTMK.test_arnoldi.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGCROTMK.test_arnoldi.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGCROTMK.test_arnoldi.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGCROTMK.test_arnoldi.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGCROTMK.test_arnoldi.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGCROTMK.test_arnoldi.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGCROTMK.test_arnoldi', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to rand(...): (line 57)
        # Processing the call arguments (line 57)
        int_417024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 23), 'int')
        # Processing the call keyword arguments (line 57)
        kwargs_417025 = {}
        # Getting the type of 'np' (line 57)
        np_417021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 57)
        random_417022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), np_417021, 'random')
        # Obtaining the member 'rand' of a type (line 57)
        rand_417023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), random_417022, 'rand')
        # Calling rand(args, kwargs) (line 57)
        rand_call_result_417026 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), rand_417023, *[int_417024], **kwargs_417025)
        
        
        # Assigning a BinOp to a Name (line 59):
        
        # Assigning a BinOp to a Name (line 59):
        
        # Call to eye(...): (line 59)
        # Processing the call arguments (line 59)
        int_417028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 16), 'int')
        # Processing the call keyword arguments (line 59)
        kwargs_417029 = {}
        # Getting the type of 'eye' (line 59)
        eye_417027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'eye', False)
        # Calling eye(args, kwargs) (line 59)
        eye_call_result_417030 = invoke(stypy.reporting.localization.Localization(__file__, 59, 12), eye_417027, *[int_417028], **kwargs_417029)
        
        
        # Call to rand(...): (line 59)
        # Processing the call arguments (line 59)
        int_417032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 30), 'int')
        int_417033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 36), 'int')
        # Processing the call keyword arguments (line 59)
        float_417034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 50), 'float')
        keyword_417035 = float_417034
        kwargs_417036 = {'density': keyword_417035}
        # Getting the type of 'rand' (line 59)
        rand_417031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 25), 'rand', False)
        # Calling rand(args, kwargs) (line 59)
        rand_call_result_417037 = invoke(stypy.reporting.localization.Localization(__file__, 59, 25), rand_417031, *[int_417032, int_417033], **kwargs_417036)
        
        # Applying the binary operator '+' (line 59)
        result_add_417038 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 12), '+', eye_call_result_417030, rand_call_result_417037)
        
        # Assigning a type to the variable 'A' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'A', result_add_417038)
        
        # Assigning a Call to a Name (line 60):
        
        # Assigning a Call to a Name (line 60):
        
        # Call to rand(...): (line 60)
        # Processing the call arguments (line 60)
        int_417042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 27), 'int')
        # Processing the call keyword arguments (line 60)
        kwargs_417043 = {}
        # Getting the type of 'np' (line 60)
        np_417039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 60)
        random_417040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 12), np_417039, 'random')
        # Obtaining the member 'rand' of a type (line 60)
        rand_417041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 12), random_417040, 'rand')
        # Calling rand(args, kwargs) (line 60)
        rand_call_result_417044 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), rand_417041, *[int_417042], **kwargs_417043)
        
        # Assigning a type to the variable 'b' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'b', rand_call_result_417044)
        
        # Assigning a Call to a Tuple (line 63):
        
        # Assigning a Subscript to a Name (line 63):
        
        # Obtaining the type of the subscript
        int_417045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 8), 'int')
        
        # Call to gcrotmk(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'A' (line 63)
        A_417047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 28), 'A', False)
        # Getting the type of 'b' (line 63)
        b_417048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 31), 'b', False)
        # Processing the call keyword arguments (line 63)
        
        # Call to zeros(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Obtaining the type of the subscript
        int_417050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 51), 'int')
        # Getting the type of 'A' (line 63)
        A_417051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 43), 'A', False)
        # Obtaining the member 'shape' of a type (line 63)
        shape_417052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 43), A_417051, 'shape')
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___417053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 43), shape_417052, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_417054 = invoke(stypy.reporting.localization.Localization(__file__, 63, 43), getitem___417053, int_417050)
        
        # Processing the call keyword arguments (line 63)
        kwargs_417055 = {}
        # Getting the type of 'zeros' (line 63)
        zeros_417049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 37), 'zeros', False)
        # Calling zeros(args, kwargs) (line 63)
        zeros_call_result_417056 = invoke(stypy.reporting.localization.Localization(__file__, 63, 37), zeros_417049, *[subscript_call_result_417054], **kwargs_417055)
        
        keyword_417057 = zeros_call_result_417056
        int_417058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 58), 'int')
        keyword_417059 = int_417058
        int_417060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 64), 'int')
        keyword_417061 = int_417060
        int_417062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 75), 'int')
        keyword_417063 = int_417062
        kwargs_417064 = {'x0': keyword_417057, 'm': keyword_417059, 'k': keyword_417061, 'maxiter': keyword_417063}
        # Getting the type of 'gcrotmk' (line 63)
        gcrotmk_417046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'gcrotmk', False)
        # Calling gcrotmk(args, kwargs) (line 63)
        gcrotmk_call_result_417065 = invoke(stypy.reporting.localization.Localization(__file__, 63, 20), gcrotmk_417046, *[A_417047, b_417048], **kwargs_417064)
        
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___417066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), gcrotmk_call_result_417065, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_417067 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), getitem___417066, int_417045)
        
        # Assigning a type to the variable 'tuple_var_assignment_416747' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tuple_var_assignment_416747', subscript_call_result_417067)
        
        # Assigning a Subscript to a Name (line 63):
        
        # Obtaining the type of the subscript
        int_417068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 8), 'int')
        
        # Call to gcrotmk(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'A' (line 63)
        A_417070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 28), 'A', False)
        # Getting the type of 'b' (line 63)
        b_417071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 31), 'b', False)
        # Processing the call keyword arguments (line 63)
        
        # Call to zeros(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Obtaining the type of the subscript
        int_417073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 51), 'int')
        # Getting the type of 'A' (line 63)
        A_417074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 43), 'A', False)
        # Obtaining the member 'shape' of a type (line 63)
        shape_417075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 43), A_417074, 'shape')
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___417076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 43), shape_417075, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_417077 = invoke(stypy.reporting.localization.Localization(__file__, 63, 43), getitem___417076, int_417073)
        
        # Processing the call keyword arguments (line 63)
        kwargs_417078 = {}
        # Getting the type of 'zeros' (line 63)
        zeros_417072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 37), 'zeros', False)
        # Calling zeros(args, kwargs) (line 63)
        zeros_call_result_417079 = invoke(stypy.reporting.localization.Localization(__file__, 63, 37), zeros_417072, *[subscript_call_result_417077], **kwargs_417078)
        
        keyword_417080 = zeros_call_result_417079
        int_417081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 58), 'int')
        keyword_417082 = int_417081
        int_417083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 64), 'int')
        keyword_417084 = int_417083
        int_417085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 75), 'int')
        keyword_417086 = int_417085
        kwargs_417087 = {'x0': keyword_417080, 'm': keyword_417082, 'k': keyword_417084, 'maxiter': keyword_417086}
        # Getting the type of 'gcrotmk' (line 63)
        gcrotmk_417069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'gcrotmk', False)
        # Calling gcrotmk(args, kwargs) (line 63)
        gcrotmk_call_result_417088 = invoke(stypy.reporting.localization.Localization(__file__, 63, 20), gcrotmk_417069, *[A_417070, b_417071], **kwargs_417087)
        
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___417089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), gcrotmk_call_result_417088, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_417090 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), getitem___417089, int_417068)
        
        # Assigning a type to the variable 'tuple_var_assignment_416748' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tuple_var_assignment_416748', subscript_call_result_417090)
        
        # Assigning a Name to a Name (line 63):
        # Getting the type of 'tuple_var_assignment_416747' (line 63)
        tuple_var_assignment_416747_417091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tuple_var_assignment_416747')
        # Assigning a type to the variable 'x0' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'x0', tuple_var_assignment_416747_417091)
        
        # Assigning a Name to a Name (line 63):
        # Getting the type of 'tuple_var_assignment_416748' (line 63)
        tuple_var_assignment_416748_417092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tuple_var_assignment_416748')
        # Assigning a type to the variable 'flag0' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'flag0', tuple_var_assignment_416748_417092)
        
        # Assigning a Call to a Tuple (line 64):
        
        # Assigning a Subscript to a Name (line 64):
        
        # Obtaining the type of the subscript
        int_417093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'int')
        
        # Call to gmres(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'A' (line 64)
        A_417095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 26), 'A', False)
        # Getting the type of 'b' (line 64)
        b_417096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 29), 'b', False)
        # Processing the call keyword arguments (line 64)
        
        # Call to zeros(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Obtaining the type of the subscript
        int_417098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 49), 'int')
        # Getting the type of 'A' (line 64)
        A_417099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 41), 'A', False)
        # Obtaining the member 'shape' of a type (line 64)
        shape_417100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 41), A_417099, 'shape')
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___417101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 41), shape_417100, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_417102 = invoke(stypy.reporting.localization.Localization(__file__, 64, 41), getitem___417101, int_417098)
        
        # Processing the call keyword arguments (line 64)
        kwargs_417103 = {}
        # Getting the type of 'zeros' (line 64)
        zeros_417097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 35), 'zeros', False)
        # Calling zeros(args, kwargs) (line 64)
        zeros_call_result_417104 = invoke(stypy.reporting.localization.Localization(__file__, 64, 35), zeros_417097, *[subscript_call_result_417102], **kwargs_417103)
        
        keyword_417105 = zeros_call_result_417104
        int_417106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 62), 'int')
        keyword_417107 = int_417106
        int_417108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 74), 'int')
        keyword_417109 = int_417108
        kwargs_417110 = {'x0': keyword_417105, 'restart': keyword_417107, 'maxiter': keyword_417109}
        # Getting the type of 'gmres' (line 64)
        gmres_417094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 20), 'gmres', False)
        # Calling gmres(args, kwargs) (line 64)
        gmres_call_result_417111 = invoke(stypy.reporting.localization.Localization(__file__, 64, 20), gmres_417094, *[A_417095, b_417096], **kwargs_417110)
        
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___417112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), gmres_call_result_417111, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_417113 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), getitem___417112, int_417093)
        
        # Assigning a type to the variable 'tuple_var_assignment_416749' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_416749', subscript_call_result_417113)
        
        # Assigning a Subscript to a Name (line 64):
        
        # Obtaining the type of the subscript
        int_417114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'int')
        
        # Call to gmres(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'A' (line 64)
        A_417116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 26), 'A', False)
        # Getting the type of 'b' (line 64)
        b_417117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 29), 'b', False)
        # Processing the call keyword arguments (line 64)
        
        # Call to zeros(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Obtaining the type of the subscript
        int_417119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 49), 'int')
        # Getting the type of 'A' (line 64)
        A_417120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 41), 'A', False)
        # Obtaining the member 'shape' of a type (line 64)
        shape_417121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 41), A_417120, 'shape')
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___417122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 41), shape_417121, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_417123 = invoke(stypy.reporting.localization.Localization(__file__, 64, 41), getitem___417122, int_417119)
        
        # Processing the call keyword arguments (line 64)
        kwargs_417124 = {}
        # Getting the type of 'zeros' (line 64)
        zeros_417118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 35), 'zeros', False)
        # Calling zeros(args, kwargs) (line 64)
        zeros_call_result_417125 = invoke(stypy.reporting.localization.Localization(__file__, 64, 35), zeros_417118, *[subscript_call_result_417123], **kwargs_417124)
        
        keyword_417126 = zeros_call_result_417125
        int_417127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 62), 'int')
        keyword_417128 = int_417127
        int_417129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 74), 'int')
        keyword_417130 = int_417129
        kwargs_417131 = {'x0': keyword_417126, 'restart': keyword_417128, 'maxiter': keyword_417130}
        # Getting the type of 'gmres' (line 64)
        gmres_417115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 20), 'gmres', False)
        # Calling gmres(args, kwargs) (line 64)
        gmres_call_result_417132 = invoke(stypy.reporting.localization.Localization(__file__, 64, 20), gmres_417115, *[A_417116, b_417117], **kwargs_417131)
        
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___417133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), gmres_call_result_417132, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_417134 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), getitem___417133, int_417114)
        
        # Assigning a type to the variable 'tuple_var_assignment_416750' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_416750', subscript_call_result_417134)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'tuple_var_assignment_416749' (line 64)
        tuple_var_assignment_416749_417135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_416749')
        # Assigning a type to the variable 'x1' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'x1', tuple_var_assignment_416749_417135)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'tuple_var_assignment_416750' (line 64)
        tuple_var_assignment_416750_417136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_416750')
        # Assigning a type to the variable 'flag1' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'flag1', tuple_var_assignment_416750_417136)
        
        # Call to assert_equal(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'flag0' (line 66)
        flag0_417138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 21), 'flag0', False)
        int_417139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 28), 'int')
        # Processing the call keyword arguments (line 66)
        kwargs_417140 = {}
        # Getting the type of 'assert_equal' (line 66)
        assert_equal_417137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 66)
        assert_equal_call_result_417141 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), assert_equal_417137, *[flag0_417138, int_417139], **kwargs_417140)
        
        
        # Call to assert_equal(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'flag1' (line 67)
        flag1_417143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 21), 'flag1', False)
        int_417144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 28), 'int')
        # Processing the call keyword arguments (line 67)
        kwargs_417145 = {}
        # Getting the type of 'assert_equal' (line 67)
        assert_equal_417142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 67)
        assert_equal_call_result_417146 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), assert_equal_417142, *[flag1_417143, int_417144], **kwargs_417145)
        
        
        # Call to assert_(...): (line 68)
        # Processing the call arguments (line 68)
        
        
        # Call to norm(...): (line 68)
        # Processing the call arguments (line 68)
        
        # Call to dot(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'x0' (line 68)
        x0_417153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 37), 'x0', False)
        # Processing the call keyword arguments (line 68)
        kwargs_417154 = {}
        # Getting the type of 'A' (line 68)
        A_417151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 31), 'A', False)
        # Obtaining the member 'dot' of a type (line 68)
        dot_417152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 31), A_417151, 'dot')
        # Calling dot(args, kwargs) (line 68)
        dot_call_result_417155 = invoke(stypy.reporting.localization.Localization(__file__, 68, 31), dot_417152, *[x0_417153], **kwargs_417154)
        
        # Getting the type of 'b' (line 68)
        b_417156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 43), 'b', False)
        # Applying the binary operator '-' (line 68)
        result_sub_417157 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 31), '-', dot_call_result_417155, b_417156)
        
        # Processing the call keyword arguments (line 68)
        kwargs_417158 = {}
        # Getting the type of 'np' (line 68)
        np_417148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'np', False)
        # Obtaining the member 'linalg' of a type (line 68)
        linalg_417149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 16), np_417148, 'linalg')
        # Obtaining the member 'norm' of a type (line 68)
        norm_417150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 16), linalg_417149, 'norm')
        # Calling norm(args, kwargs) (line 68)
        norm_call_result_417159 = invoke(stypy.reporting.localization.Localization(__file__, 68, 16), norm_417150, *[result_sub_417157], **kwargs_417158)
        
        float_417160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 48), 'float')
        # Applying the binary operator '>' (line 68)
        result_gt_417161 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 16), '>', norm_call_result_417159, float_417160)
        
        # Processing the call keyword arguments (line 68)
        kwargs_417162 = {}
        # Getting the type of 'assert_' (line 68)
        assert__417147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 68)
        assert__call_result_417163 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), assert__417147, *[result_gt_417161], **kwargs_417162)
        
        
        # Call to assert_allclose(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'x0' (line 70)
        x0_417165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 24), 'x0', False)
        # Getting the type of 'x1' (line 70)
        x1_417166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 28), 'x1', False)
        # Processing the call keyword arguments (line 70)
        kwargs_417167 = {}
        # Getting the type of 'assert_allclose' (line 70)
        assert_allclose_417164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 70)
        assert_allclose_call_result_417168 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), assert_allclose_417164, *[x0_417165, x1_417166], **kwargs_417167)
        
        
        # ################# End of 'test_arnoldi(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_arnoldi' in the type store
        # Getting the type of 'stypy_return_type' (line 56)
        stypy_return_type_417169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_417169)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_arnoldi'
        return stypy_return_type_417169


    @norecursion
    def test_cornercase(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_cornercase'
        module_type_store = module_type_store.open_function_context('test_cornercase', 72, 4, False)
        # Assigning a type to the variable 'self' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGCROTMK.test_cornercase.__dict__.__setitem__('stypy_localization', localization)
        TestGCROTMK.test_cornercase.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGCROTMK.test_cornercase.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGCROTMK.test_cornercase.__dict__.__setitem__('stypy_function_name', 'TestGCROTMK.test_cornercase')
        TestGCROTMK.test_cornercase.__dict__.__setitem__('stypy_param_names_list', [])
        TestGCROTMK.test_cornercase.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGCROTMK.test_cornercase.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGCROTMK.test_cornercase.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGCROTMK.test_cornercase.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGCROTMK.test_cornercase.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGCROTMK.test_cornercase.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGCROTMK.test_cornercase', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to seed(...): (line 73)
        # Processing the call arguments (line 73)
        int_417173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 23), 'int')
        # Processing the call keyword arguments (line 73)
        kwargs_417174 = {}
        # Getting the type of 'np' (line 73)
        np_417170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 73)
        random_417171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), np_417170, 'random')
        # Obtaining the member 'seed' of a type (line 73)
        seed_417172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), random_417171, 'seed')
        # Calling seed(args, kwargs) (line 73)
        seed_call_result_417175 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), seed_417172, *[int_417173], **kwargs_417174)
        
        
        
        # Obtaining an instance of the builtin type 'list' (line 79)
        list_417176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 79)
        # Adding element type (line 79)
        int_417177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 17), list_417176, int_417177)
        # Adding element type (line 79)
        int_417178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 17), list_417176, int_417178)
        # Adding element type (line 79)
        int_417179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 17), list_417176, int_417179)
        # Adding element type (line 79)
        int_417180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 17), list_417176, int_417180)
        
        # Testing the type of a for loop iterable (line 79)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 79, 8), list_417176)
        # Getting the type of the for loop variable (line 79)
        for_loop_var_417181 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 79, 8), list_417176)
        # Assigning a type to the variable 'n' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'n', for_loop_var_417181)
        # SSA begins for a for statement (line 79)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 80):
        
        # Assigning a BinOp to a Name (line 80):
        int_417182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 16), 'int')
        
        # Call to eye(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'n' (line 80)
        n_417184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 22), 'n', False)
        # Processing the call keyword arguments (line 80)
        kwargs_417185 = {}
        # Getting the type of 'eye' (line 80)
        eye_417183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 18), 'eye', False)
        # Calling eye(args, kwargs) (line 80)
        eye_call_result_417186 = invoke(stypy.reporting.localization.Localization(__file__, 80, 18), eye_417183, *[n_417184], **kwargs_417185)
        
        # Applying the binary operator '*' (line 80)
        result_mul_417187 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 16), '*', int_417182, eye_call_result_417186)
        
        # Assigning a type to the variable 'A' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'A', result_mul_417187)
        
        # Assigning a Call to a Name (line 82):
        
        # Assigning a Call to a Name (line 82):
        
        # Call to ones(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'n' (line 82)
        n_417190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 24), 'n', False)
        # Processing the call keyword arguments (line 82)
        kwargs_417191 = {}
        # Getting the type of 'np' (line 82)
        np_417188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'np', False)
        # Obtaining the member 'ones' of a type (line 82)
        ones_417189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 16), np_417188, 'ones')
        # Calling ones(args, kwargs) (line 82)
        ones_call_result_417192 = invoke(stypy.reporting.localization.Localization(__file__, 82, 16), ones_417189, *[n_417190], **kwargs_417191)
        
        # Assigning a type to the variable 'b' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'b', ones_call_result_417192)
        
        # Assigning a Call to a Tuple (line 83):
        
        # Assigning a Subscript to a Name (line 83):
        
        # Obtaining the type of the subscript
        int_417193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 12), 'int')
        
        # Call to gcrotmk(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'A' (line 83)
        A_417195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 30), 'A', False)
        # Getting the type of 'b' (line 83)
        b_417196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 33), 'b', False)
        # Processing the call keyword arguments (line 83)
        int_417197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 44), 'int')
        keyword_417198 = int_417197
        kwargs_417199 = {'maxiter': keyword_417198}
        # Getting the type of 'gcrotmk' (line 83)
        gcrotmk_417194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 22), 'gcrotmk', False)
        # Calling gcrotmk(args, kwargs) (line 83)
        gcrotmk_call_result_417200 = invoke(stypy.reporting.localization.Localization(__file__, 83, 22), gcrotmk_417194, *[A_417195, b_417196], **kwargs_417199)
        
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___417201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 12), gcrotmk_call_result_417200, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_417202 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), getitem___417201, int_417193)
        
        # Assigning a type to the variable 'tuple_var_assignment_416751' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'tuple_var_assignment_416751', subscript_call_result_417202)
        
        # Assigning a Subscript to a Name (line 83):
        
        # Obtaining the type of the subscript
        int_417203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 12), 'int')
        
        # Call to gcrotmk(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'A' (line 83)
        A_417205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 30), 'A', False)
        # Getting the type of 'b' (line 83)
        b_417206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 33), 'b', False)
        # Processing the call keyword arguments (line 83)
        int_417207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 44), 'int')
        keyword_417208 = int_417207
        kwargs_417209 = {'maxiter': keyword_417208}
        # Getting the type of 'gcrotmk' (line 83)
        gcrotmk_417204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 22), 'gcrotmk', False)
        # Calling gcrotmk(args, kwargs) (line 83)
        gcrotmk_call_result_417210 = invoke(stypy.reporting.localization.Localization(__file__, 83, 22), gcrotmk_417204, *[A_417205, b_417206], **kwargs_417209)
        
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___417211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 12), gcrotmk_call_result_417210, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_417212 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), getitem___417211, int_417203)
        
        # Assigning a type to the variable 'tuple_var_assignment_416752' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'tuple_var_assignment_416752', subscript_call_result_417212)
        
        # Assigning a Name to a Name (line 83):
        # Getting the type of 'tuple_var_assignment_416751' (line 83)
        tuple_var_assignment_416751_417213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'tuple_var_assignment_416751')
        # Assigning a type to the variable 'x' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'x', tuple_var_assignment_416751_417213)
        
        # Assigning a Name to a Name (line 83):
        # Getting the type of 'tuple_var_assignment_416752' (line 83)
        tuple_var_assignment_416752_417214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'tuple_var_assignment_416752')
        # Assigning a type to the variable 'info' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 15), 'info', tuple_var_assignment_416752_417214)
        
        # Call to assert_equal(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'info' (line 84)
        info_417216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 25), 'info', False)
        int_417217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 31), 'int')
        # Processing the call keyword arguments (line 84)
        kwargs_417218 = {}
        # Getting the type of 'assert_equal' (line 84)
        assert_equal_417215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 84)
        assert_equal_call_result_417219 = invoke(stypy.reporting.localization.Localization(__file__, 84, 12), assert_equal_417215, *[info_417216, int_417217], **kwargs_417218)
        
        
        # Call to assert_allclose(...): (line 85)
        # Processing the call arguments (line 85)
        
        # Call to dot(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'x' (line 85)
        x_417223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 34), 'x', False)
        # Processing the call keyword arguments (line 85)
        kwargs_417224 = {}
        # Getting the type of 'A' (line 85)
        A_417221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 28), 'A', False)
        # Obtaining the member 'dot' of a type (line 85)
        dot_417222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 28), A_417221, 'dot')
        # Calling dot(args, kwargs) (line 85)
        dot_call_result_417225 = invoke(stypy.reporting.localization.Localization(__file__, 85, 28), dot_417222, *[x_417223], **kwargs_417224)
        
        # Getting the type of 'b' (line 85)
        b_417226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 39), 'b', False)
        # Applying the binary operator '-' (line 85)
        result_sub_417227 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 28), '-', dot_call_result_417225, b_417226)
        
        int_417228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 42), 'int')
        # Processing the call keyword arguments (line 85)
        float_417229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 50), 'float')
        keyword_417230 = float_417229
        kwargs_417231 = {'atol': keyword_417230}
        # Getting the type of 'assert_allclose' (line 85)
        assert_allclose_417220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 85)
        assert_allclose_call_result_417232 = invoke(stypy.reporting.localization.Localization(__file__, 85, 12), assert_allclose_417220, *[result_sub_417227, int_417228], **kwargs_417231)
        
        
        # Assigning a Call to a Tuple (line 87):
        
        # Assigning a Subscript to a Name (line 87):
        
        # Obtaining the type of the subscript
        int_417233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 12), 'int')
        
        # Call to gcrotmk(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'A' (line 87)
        A_417235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 30), 'A', False)
        # Getting the type of 'b' (line 87)
        b_417236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 33), 'b', False)
        # Processing the call keyword arguments (line 87)
        int_417237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 40), 'int')
        keyword_417238 = int_417237
        int_417239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 51), 'int')
        keyword_417240 = int_417239
        kwargs_417241 = {'tol': keyword_417238, 'maxiter': keyword_417240}
        # Getting the type of 'gcrotmk' (line 87)
        gcrotmk_417234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 22), 'gcrotmk', False)
        # Calling gcrotmk(args, kwargs) (line 87)
        gcrotmk_call_result_417242 = invoke(stypy.reporting.localization.Localization(__file__, 87, 22), gcrotmk_417234, *[A_417235, b_417236], **kwargs_417241)
        
        # Obtaining the member '__getitem__' of a type (line 87)
        getitem___417243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 12), gcrotmk_call_result_417242, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 87)
        subscript_call_result_417244 = invoke(stypy.reporting.localization.Localization(__file__, 87, 12), getitem___417243, int_417233)
        
        # Assigning a type to the variable 'tuple_var_assignment_416753' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'tuple_var_assignment_416753', subscript_call_result_417244)
        
        # Assigning a Subscript to a Name (line 87):
        
        # Obtaining the type of the subscript
        int_417245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 12), 'int')
        
        # Call to gcrotmk(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'A' (line 87)
        A_417247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 30), 'A', False)
        # Getting the type of 'b' (line 87)
        b_417248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 33), 'b', False)
        # Processing the call keyword arguments (line 87)
        int_417249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 40), 'int')
        keyword_417250 = int_417249
        int_417251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 51), 'int')
        keyword_417252 = int_417251
        kwargs_417253 = {'tol': keyword_417250, 'maxiter': keyword_417252}
        # Getting the type of 'gcrotmk' (line 87)
        gcrotmk_417246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 22), 'gcrotmk', False)
        # Calling gcrotmk(args, kwargs) (line 87)
        gcrotmk_call_result_417254 = invoke(stypy.reporting.localization.Localization(__file__, 87, 22), gcrotmk_417246, *[A_417247, b_417248], **kwargs_417253)
        
        # Obtaining the member '__getitem__' of a type (line 87)
        getitem___417255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 12), gcrotmk_call_result_417254, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 87)
        subscript_call_result_417256 = invoke(stypy.reporting.localization.Localization(__file__, 87, 12), getitem___417255, int_417245)
        
        # Assigning a type to the variable 'tuple_var_assignment_416754' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'tuple_var_assignment_416754', subscript_call_result_417256)
        
        # Assigning a Name to a Name (line 87):
        # Getting the type of 'tuple_var_assignment_416753' (line 87)
        tuple_var_assignment_416753_417257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'tuple_var_assignment_416753')
        # Assigning a type to the variable 'x' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'x', tuple_var_assignment_416753_417257)
        
        # Assigning a Name to a Name (line 87):
        # Getting the type of 'tuple_var_assignment_416754' (line 87)
        tuple_var_assignment_416754_417258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'tuple_var_assignment_416754')
        # Assigning a type to the variable 'info' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'info', tuple_var_assignment_416754_417258)
        
        
        # Getting the type of 'info' (line 88)
        info_417259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'info')
        int_417260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 23), 'int')
        # Applying the binary operator '==' (line 88)
        result_eq_417261 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 15), '==', info_417259, int_417260)
        
        # Testing the type of an if condition (line 88)
        if_condition_417262 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 12), result_eq_417261)
        # Assigning a type to the variable 'if_condition_417262' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'if_condition_417262', if_condition_417262)
        # SSA begins for if statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_allclose(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Call to dot(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'x' (line 89)
        x_417266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 38), 'x', False)
        # Processing the call keyword arguments (line 89)
        kwargs_417267 = {}
        # Getting the type of 'A' (line 89)
        A_417264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 32), 'A', False)
        # Obtaining the member 'dot' of a type (line 89)
        dot_417265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 32), A_417264, 'dot')
        # Calling dot(args, kwargs) (line 89)
        dot_call_result_417268 = invoke(stypy.reporting.localization.Localization(__file__, 89, 32), dot_417265, *[x_417266], **kwargs_417267)
        
        # Getting the type of 'b' (line 89)
        b_417269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 43), 'b', False)
        # Applying the binary operator '-' (line 89)
        result_sub_417270 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 32), '-', dot_call_result_417268, b_417269)
        
        int_417271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 46), 'int')
        # Processing the call keyword arguments (line 89)
        float_417272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 54), 'float')
        keyword_417273 = float_417272
        kwargs_417274 = {'atol': keyword_417273}
        # Getting the type of 'assert_allclose' (line 89)
        assert_allclose_417263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 89)
        assert_allclose_call_result_417275 = invoke(stypy.reporting.localization.Localization(__file__, 89, 16), assert_allclose_417263, *[result_sub_417270, int_417271], **kwargs_417274)
        
        # SSA join for if statement (line 88)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 91):
        
        # Assigning a Call to a Name (line 91):
        
        # Call to rand(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'n' (line 91)
        n_417279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 31), 'n', False)
        # Processing the call keyword arguments (line 91)
        kwargs_417280 = {}
        # Getting the type of 'np' (line 91)
        np_417276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'np', False)
        # Obtaining the member 'random' of a type (line 91)
        random_417277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 16), np_417276, 'random')
        # Obtaining the member 'rand' of a type (line 91)
        rand_417278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 16), random_417277, 'rand')
        # Calling rand(args, kwargs) (line 91)
        rand_call_result_417281 = invoke(stypy.reporting.localization.Localization(__file__, 91, 16), rand_417278, *[n_417279], **kwargs_417280)
        
        # Assigning a type to the variable 'b' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'b', rand_call_result_417281)
        
        # Assigning a Call to a Tuple (line 92):
        
        # Assigning a Subscript to a Name (line 92):
        
        # Obtaining the type of the subscript
        int_417282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 12), 'int')
        
        # Call to gcrotmk(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'A' (line 92)
        A_417284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 30), 'A', False)
        # Getting the type of 'b' (line 92)
        b_417285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 33), 'b', False)
        # Processing the call keyword arguments (line 92)
        int_417286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 44), 'int')
        keyword_417287 = int_417286
        kwargs_417288 = {'maxiter': keyword_417287}
        # Getting the type of 'gcrotmk' (line 92)
        gcrotmk_417283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 22), 'gcrotmk', False)
        # Calling gcrotmk(args, kwargs) (line 92)
        gcrotmk_call_result_417289 = invoke(stypy.reporting.localization.Localization(__file__, 92, 22), gcrotmk_417283, *[A_417284, b_417285], **kwargs_417288)
        
        # Obtaining the member '__getitem__' of a type (line 92)
        getitem___417290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), gcrotmk_call_result_417289, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 92)
        subscript_call_result_417291 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), getitem___417290, int_417282)
        
        # Assigning a type to the variable 'tuple_var_assignment_416755' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'tuple_var_assignment_416755', subscript_call_result_417291)
        
        # Assigning a Subscript to a Name (line 92):
        
        # Obtaining the type of the subscript
        int_417292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 12), 'int')
        
        # Call to gcrotmk(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'A' (line 92)
        A_417294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 30), 'A', False)
        # Getting the type of 'b' (line 92)
        b_417295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 33), 'b', False)
        # Processing the call keyword arguments (line 92)
        int_417296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 44), 'int')
        keyword_417297 = int_417296
        kwargs_417298 = {'maxiter': keyword_417297}
        # Getting the type of 'gcrotmk' (line 92)
        gcrotmk_417293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 22), 'gcrotmk', False)
        # Calling gcrotmk(args, kwargs) (line 92)
        gcrotmk_call_result_417299 = invoke(stypy.reporting.localization.Localization(__file__, 92, 22), gcrotmk_417293, *[A_417294, b_417295], **kwargs_417298)
        
        # Obtaining the member '__getitem__' of a type (line 92)
        getitem___417300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), gcrotmk_call_result_417299, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 92)
        subscript_call_result_417301 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), getitem___417300, int_417292)
        
        # Assigning a type to the variable 'tuple_var_assignment_416756' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'tuple_var_assignment_416756', subscript_call_result_417301)
        
        # Assigning a Name to a Name (line 92):
        # Getting the type of 'tuple_var_assignment_416755' (line 92)
        tuple_var_assignment_416755_417302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'tuple_var_assignment_416755')
        # Assigning a type to the variable 'x' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'x', tuple_var_assignment_416755_417302)
        
        # Assigning a Name to a Name (line 92):
        # Getting the type of 'tuple_var_assignment_416756' (line 92)
        tuple_var_assignment_416756_417303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'tuple_var_assignment_416756')
        # Assigning a type to the variable 'info' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 15), 'info', tuple_var_assignment_416756_417303)
        
        # Call to assert_equal(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'info' (line 93)
        info_417305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 25), 'info', False)
        int_417306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 31), 'int')
        # Processing the call keyword arguments (line 93)
        kwargs_417307 = {}
        # Getting the type of 'assert_equal' (line 93)
        assert_equal_417304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 93)
        assert_equal_call_result_417308 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), assert_equal_417304, *[info_417305, int_417306], **kwargs_417307)
        
        
        # Call to assert_allclose(...): (line 94)
        # Processing the call arguments (line 94)
        
        # Call to dot(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'x' (line 94)
        x_417312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 34), 'x', False)
        # Processing the call keyword arguments (line 94)
        kwargs_417313 = {}
        # Getting the type of 'A' (line 94)
        A_417310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 28), 'A', False)
        # Obtaining the member 'dot' of a type (line 94)
        dot_417311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 28), A_417310, 'dot')
        # Calling dot(args, kwargs) (line 94)
        dot_call_result_417314 = invoke(stypy.reporting.localization.Localization(__file__, 94, 28), dot_417311, *[x_417312], **kwargs_417313)
        
        # Getting the type of 'b' (line 94)
        b_417315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 39), 'b', False)
        # Applying the binary operator '-' (line 94)
        result_sub_417316 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 28), '-', dot_call_result_417314, b_417315)
        
        int_417317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 42), 'int')
        # Processing the call keyword arguments (line 94)
        float_417318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 50), 'float')
        keyword_417319 = float_417318
        kwargs_417320 = {'atol': keyword_417319}
        # Getting the type of 'assert_allclose' (line 94)
        assert_allclose_417309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 94)
        assert_allclose_call_result_417321 = invoke(stypy.reporting.localization.Localization(__file__, 94, 12), assert_allclose_417309, *[result_sub_417316, int_417317], **kwargs_417320)
        
        
        # Assigning a Call to a Tuple (line 96):
        
        # Assigning a Subscript to a Name (line 96):
        
        # Obtaining the type of the subscript
        int_417322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 12), 'int')
        
        # Call to gcrotmk(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'A' (line 96)
        A_417324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 30), 'A', False)
        # Getting the type of 'b' (line 96)
        b_417325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 33), 'b', False)
        # Processing the call keyword arguments (line 96)
        int_417326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 40), 'int')
        keyword_417327 = int_417326
        int_417328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 51), 'int')
        keyword_417329 = int_417328
        kwargs_417330 = {'tol': keyword_417327, 'maxiter': keyword_417329}
        # Getting the type of 'gcrotmk' (line 96)
        gcrotmk_417323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 22), 'gcrotmk', False)
        # Calling gcrotmk(args, kwargs) (line 96)
        gcrotmk_call_result_417331 = invoke(stypy.reporting.localization.Localization(__file__, 96, 22), gcrotmk_417323, *[A_417324, b_417325], **kwargs_417330)
        
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___417332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 12), gcrotmk_call_result_417331, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 96)
        subscript_call_result_417333 = invoke(stypy.reporting.localization.Localization(__file__, 96, 12), getitem___417332, int_417322)
        
        # Assigning a type to the variable 'tuple_var_assignment_416757' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'tuple_var_assignment_416757', subscript_call_result_417333)
        
        # Assigning a Subscript to a Name (line 96):
        
        # Obtaining the type of the subscript
        int_417334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 12), 'int')
        
        # Call to gcrotmk(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'A' (line 96)
        A_417336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 30), 'A', False)
        # Getting the type of 'b' (line 96)
        b_417337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 33), 'b', False)
        # Processing the call keyword arguments (line 96)
        int_417338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 40), 'int')
        keyword_417339 = int_417338
        int_417340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 51), 'int')
        keyword_417341 = int_417340
        kwargs_417342 = {'tol': keyword_417339, 'maxiter': keyword_417341}
        # Getting the type of 'gcrotmk' (line 96)
        gcrotmk_417335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 22), 'gcrotmk', False)
        # Calling gcrotmk(args, kwargs) (line 96)
        gcrotmk_call_result_417343 = invoke(stypy.reporting.localization.Localization(__file__, 96, 22), gcrotmk_417335, *[A_417336, b_417337], **kwargs_417342)
        
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___417344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 12), gcrotmk_call_result_417343, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 96)
        subscript_call_result_417345 = invoke(stypy.reporting.localization.Localization(__file__, 96, 12), getitem___417344, int_417334)
        
        # Assigning a type to the variable 'tuple_var_assignment_416758' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'tuple_var_assignment_416758', subscript_call_result_417345)
        
        # Assigning a Name to a Name (line 96):
        # Getting the type of 'tuple_var_assignment_416757' (line 96)
        tuple_var_assignment_416757_417346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'tuple_var_assignment_416757')
        # Assigning a type to the variable 'x' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'x', tuple_var_assignment_416757_417346)
        
        # Assigning a Name to a Name (line 96):
        # Getting the type of 'tuple_var_assignment_416758' (line 96)
        tuple_var_assignment_416758_417347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'tuple_var_assignment_416758')
        # Assigning a type to the variable 'info' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 15), 'info', tuple_var_assignment_416758_417347)
        
        
        # Getting the type of 'info' (line 97)
        info_417348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 'info')
        int_417349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 23), 'int')
        # Applying the binary operator '==' (line 97)
        result_eq_417350 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 15), '==', info_417348, int_417349)
        
        # Testing the type of an if condition (line 97)
        if_condition_417351 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 12), result_eq_417350)
        # Assigning a type to the variable 'if_condition_417351' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'if_condition_417351', if_condition_417351)
        # SSA begins for if statement (line 97)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_allclose(...): (line 98)
        # Processing the call arguments (line 98)
        
        # Call to dot(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'x' (line 98)
        x_417355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 38), 'x', False)
        # Processing the call keyword arguments (line 98)
        kwargs_417356 = {}
        # Getting the type of 'A' (line 98)
        A_417353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 32), 'A', False)
        # Obtaining the member 'dot' of a type (line 98)
        dot_417354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 32), A_417353, 'dot')
        # Calling dot(args, kwargs) (line 98)
        dot_call_result_417357 = invoke(stypy.reporting.localization.Localization(__file__, 98, 32), dot_417354, *[x_417355], **kwargs_417356)
        
        # Getting the type of 'b' (line 98)
        b_417358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 43), 'b', False)
        # Applying the binary operator '-' (line 98)
        result_sub_417359 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 32), '-', dot_call_result_417357, b_417358)
        
        int_417360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 46), 'int')
        # Processing the call keyword arguments (line 98)
        float_417361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 54), 'float')
        keyword_417362 = float_417361
        kwargs_417363 = {'atol': keyword_417362}
        # Getting the type of 'assert_allclose' (line 98)
        assert_allclose_417352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 98)
        assert_allclose_call_result_417364 = invoke(stypy.reporting.localization.Localization(__file__, 98, 16), assert_allclose_417352, *[result_sub_417359, int_417360], **kwargs_417363)
        
        # SSA join for if statement (line 97)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_cornercase(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_cornercase' in the type store
        # Getting the type of 'stypy_return_type' (line 72)
        stypy_return_type_417365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_417365)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_cornercase'
        return stypy_return_type_417365


    @norecursion
    def test_nans(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_nans'
        module_type_store = module_type_store.open_function_context('test_nans', 100, 4, False)
        # Assigning a type to the variable 'self' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGCROTMK.test_nans.__dict__.__setitem__('stypy_localization', localization)
        TestGCROTMK.test_nans.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGCROTMK.test_nans.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGCROTMK.test_nans.__dict__.__setitem__('stypy_function_name', 'TestGCROTMK.test_nans')
        TestGCROTMK.test_nans.__dict__.__setitem__('stypy_param_names_list', [])
        TestGCROTMK.test_nans.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGCROTMK.test_nans.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGCROTMK.test_nans.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGCROTMK.test_nans.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGCROTMK.test_nans.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGCROTMK.test_nans.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGCROTMK.test_nans', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 101):
        
        # Assigning a Call to a Name (line 101):
        
        # Call to eye(...): (line 101)
        # Processing the call arguments (line 101)
        int_417367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 16), 'int')
        # Processing the call keyword arguments (line 101)
        str_417368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 26), 'str', 'lil')
        keyword_417369 = str_417368
        kwargs_417370 = {'format': keyword_417369}
        # Getting the type of 'eye' (line 101)
        eye_417366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'eye', False)
        # Calling eye(args, kwargs) (line 101)
        eye_call_result_417371 = invoke(stypy.reporting.localization.Localization(__file__, 101, 12), eye_417366, *[int_417367], **kwargs_417370)
        
        # Assigning a type to the variable 'A' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'A', eye_call_result_417371)
        
        # Assigning a Attribute to a Subscript (line 102):
        
        # Assigning a Attribute to a Subscript (line 102):
        # Getting the type of 'np' (line 102)
        np_417372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 17), 'np')
        # Obtaining the member 'nan' of a type (line 102)
        nan_417373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 17), np_417372, 'nan')
        # Getting the type of 'A' (line 102)
        A_417374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 102)
        tuple_417375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 10), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 102)
        # Adding element type (line 102)
        int_417376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 10), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 10), tuple_417375, int_417376)
        # Adding element type (line 102)
        int_417377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 12), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 10), tuple_417375, int_417377)
        
        # Storing an element on a container (line 102)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 8), A_417374, (tuple_417375, nan_417373))
        
        # Assigning a Call to a Name (line 103):
        
        # Assigning a Call to a Name (line 103):
        
        # Call to ones(...): (line 103)
        # Processing the call arguments (line 103)
        int_417380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 20), 'int')
        # Processing the call keyword arguments (line 103)
        kwargs_417381 = {}
        # Getting the type of 'np' (line 103)
        np_417378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'np', False)
        # Obtaining the member 'ones' of a type (line 103)
        ones_417379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), np_417378, 'ones')
        # Calling ones(args, kwargs) (line 103)
        ones_call_result_417382 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), ones_417379, *[int_417380], **kwargs_417381)
        
        # Assigning a type to the variable 'b' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'b', ones_call_result_417382)
        
        # Assigning a Call to a Tuple (line 105):
        
        # Assigning a Subscript to a Name (line 105):
        
        # Obtaining the type of the subscript
        int_417383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 8), 'int')
        
        # Call to gcrotmk(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'A' (line 105)
        A_417385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 26), 'A', False)
        # Getting the type of 'b' (line 105)
        b_417386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 29), 'b', False)
        # Processing the call keyword arguments (line 105)
        int_417387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 36), 'int')
        keyword_417388 = int_417387
        int_417389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 47), 'int')
        keyword_417390 = int_417389
        kwargs_417391 = {'tol': keyword_417388, 'maxiter': keyword_417390}
        # Getting the type of 'gcrotmk' (line 105)
        gcrotmk_417384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 18), 'gcrotmk', False)
        # Calling gcrotmk(args, kwargs) (line 105)
        gcrotmk_call_result_417392 = invoke(stypy.reporting.localization.Localization(__file__, 105, 18), gcrotmk_417384, *[A_417385, b_417386], **kwargs_417391)
        
        # Obtaining the member '__getitem__' of a type (line 105)
        getitem___417393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 8), gcrotmk_call_result_417392, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 105)
        subscript_call_result_417394 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), getitem___417393, int_417383)
        
        # Assigning a type to the variable 'tuple_var_assignment_416759' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'tuple_var_assignment_416759', subscript_call_result_417394)
        
        # Assigning a Subscript to a Name (line 105):
        
        # Obtaining the type of the subscript
        int_417395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 8), 'int')
        
        # Call to gcrotmk(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'A' (line 105)
        A_417397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 26), 'A', False)
        # Getting the type of 'b' (line 105)
        b_417398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 29), 'b', False)
        # Processing the call keyword arguments (line 105)
        int_417399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 36), 'int')
        keyword_417400 = int_417399
        int_417401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 47), 'int')
        keyword_417402 = int_417401
        kwargs_417403 = {'tol': keyword_417400, 'maxiter': keyword_417402}
        # Getting the type of 'gcrotmk' (line 105)
        gcrotmk_417396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 18), 'gcrotmk', False)
        # Calling gcrotmk(args, kwargs) (line 105)
        gcrotmk_call_result_417404 = invoke(stypy.reporting.localization.Localization(__file__, 105, 18), gcrotmk_417396, *[A_417397, b_417398], **kwargs_417403)
        
        # Obtaining the member '__getitem__' of a type (line 105)
        getitem___417405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 8), gcrotmk_call_result_417404, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 105)
        subscript_call_result_417406 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), getitem___417405, int_417395)
        
        # Assigning a type to the variable 'tuple_var_assignment_416760' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'tuple_var_assignment_416760', subscript_call_result_417406)
        
        # Assigning a Name to a Name (line 105):
        # Getting the type of 'tuple_var_assignment_416759' (line 105)
        tuple_var_assignment_416759_417407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'tuple_var_assignment_416759')
        # Assigning a type to the variable 'x' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'x', tuple_var_assignment_416759_417407)
        
        # Assigning a Name to a Name (line 105):
        # Getting the type of 'tuple_var_assignment_416760' (line 105)
        tuple_var_assignment_416760_417408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'tuple_var_assignment_416760')
        # Assigning a type to the variable 'info' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 'info', tuple_var_assignment_416760_417408)
        
        # Call to assert_equal(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'info' (line 106)
        info_417410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 21), 'info', False)
        int_417411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 27), 'int')
        # Processing the call keyword arguments (line 106)
        kwargs_417412 = {}
        # Getting the type of 'assert_equal' (line 106)
        assert_equal_417409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 106)
        assert_equal_call_result_417413 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), assert_equal_417409, *[info_417410, int_417411], **kwargs_417412)
        
        
        # ################# End of 'test_nans(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_nans' in the type store
        # Getting the type of 'stypy_return_type' (line 100)
        stypy_return_type_417414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_417414)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_nans'
        return stypy_return_type_417414


    @norecursion
    def test_truncate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_truncate'
        module_type_store = module_type_store.open_function_context('test_truncate', 108, 4, False)
        # Assigning a type to the variable 'self' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGCROTMK.test_truncate.__dict__.__setitem__('stypy_localization', localization)
        TestGCROTMK.test_truncate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGCROTMK.test_truncate.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGCROTMK.test_truncate.__dict__.__setitem__('stypy_function_name', 'TestGCROTMK.test_truncate')
        TestGCROTMK.test_truncate.__dict__.__setitem__('stypy_param_names_list', [])
        TestGCROTMK.test_truncate.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGCROTMK.test_truncate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGCROTMK.test_truncate.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGCROTMK.test_truncate.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGCROTMK.test_truncate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGCROTMK.test_truncate.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGCROTMK.test_truncate', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_truncate', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_truncate(...)' code ##################

        
        # Call to seed(...): (line 109)
        # Processing the call arguments (line 109)
        int_417418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 23), 'int')
        # Processing the call keyword arguments (line 109)
        kwargs_417419 = {}
        # Getting the type of 'np' (line 109)
        np_417415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 109)
        random_417416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), np_417415, 'random')
        # Obtaining the member 'seed' of a type (line 109)
        seed_417417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), random_417416, 'seed')
        # Calling seed(args, kwargs) (line 109)
        seed_call_result_417420 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), seed_417417, *[int_417418], **kwargs_417419)
        
        
        # Assigning a BinOp to a Name (line 110):
        
        # Assigning a BinOp to a Name (line 110):
        
        # Call to rand(...): (line 110)
        # Processing the call arguments (line 110)
        int_417424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 27), 'int')
        int_417425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 31), 'int')
        # Processing the call keyword arguments (line 110)
        kwargs_417426 = {}
        # Getting the type of 'np' (line 110)
        np_417421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 110)
        random_417422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), np_417421, 'random')
        # Obtaining the member 'rand' of a type (line 110)
        rand_417423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), random_417422, 'rand')
        # Calling rand(args, kwargs) (line 110)
        rand_call_result_417427 = invoke(stypy.reporting.localization.Localization(__file__, 110, 12), rand_417423, *[int_417424, int_417425], **kwargs_417426)
        
        
        # Call to eye(...): (line 110)
        # Processing the call arguments (line 110)
        int_417430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 44), 'int')
        # Processing the call keyword arguments (line 110)
        kwargs_417431 = {}
        # Getting the type of 'np' (line 110)
        np_417428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 37), 'np', False)
        # Obtaining the member 'eye' of a type (line 110)
        eye_417429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 37), np_417428, 'eye')
        # Calling eye(args, kwargs) (line 110)
        eye_call_result_417432 = invoke(stypy.reporting.localization.Localization(__file__, 110, 37), eye_417429, *[int_417430], **kwargs_417431)
        
        # Applying the binary operator '+' (line 110)
        result_add_417433 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 12), '+', rand_call_result_417427, eye_call_result_417432)
        
        # Assigning a type to the variable 'A' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'A', result_add_417433)
        
        # Assigning a Call to a Name (line 111):
        
        # Assigning a Call to a Name (line 111):
        
        # Call to rand(...): (line 111)
        # Processing the call arguments (line 111)
        int_417437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 27), 'int')
        # Processing the call keyword arguments (line 111)
        kwargs_417438 = {}
        # Getting the type of 'np' (line 111)
        np_417434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 111)
        random_417435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 12), np_417434, 'random')
        # Obtaining the member 'rand' of a type (line 111)
        rand_417436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 12), random_417435, 'rand')
        # Calling rand(args, kwargs) (line 111)
        rand_call_result_417439 = invoke(stypy.reporting.localization.Localization(__file__, 111, 12), rand_417436, *[int_417437], **kwargs_417438)
        
        # Assigning a type to the variable 'b' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'b', rand_call_result_417439)
        
        
        # Obtaining an instance of the builtin type 'list' (line 113)
        list_417440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 113)
        # Adding element type (line 113)
        str_417441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 25), 'str', 'oldest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 24), list_417440, str_417441)
        # Adding element type (line 113)
        str_417442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 35), 'str', 'smallest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 24), list_417440, str_417442)
        
        # Testing the type of a for loop iterable (line 113)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 113, 8), list_417440)
        # Getting the type of the for loop variable (line 113)
        for_loop_var_417443 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 113, 8), list_417440)
        # Assigning a type to the variable 'truncate' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'truncate', for_loop_var_417443)
        # SSA begins for a for statement (line 113)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 114):
        
        # Assigning a Subscript to a Name (line 114):
        
        # Obtaining the type of the subscript
        int_417444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 12), 'int')
        
        # Call to gcrotmk(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'A' (line 114)
        A_417446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 30), 'A', False)
        # Getting the type of 'b' (line 114)
        b_417447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 33), 'b', False)
        # Processing the call keyword arguments (line 114)
        int_417448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 38), 'int')
        keyword_417449 = int_417448
        int_417450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 44), 'int')
        keyword_417451 = int_417450
        # Getting the type of 'truncate' (line 114)
        truncate_417452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 57), 'truncate', False)
        keyword_417453 = truncate_417452
        float_417454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 71), 'float')
        keyword_417455 = float_417454
        int_417456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 38), 'int')
        keyword_417457 = int_417456
        kwargs_417458 = {'k': keyword_417451, 'm': keyword_417449, 'tol': keyword_417455, 'truncate': keyword_417453, 'maxiter': keyword_417457}
        # Getting the type of 'gcrotmk' (line 114)
        gcrotmk_417445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 22), 'gcrotmk', False)
        # Calling gcrotmk(args, kwargs) (line 114)
        gcrotmk_call_result_417459 = invoke(stypy.reporting.localization.Localization(__file__, 114, 22), gcrotmk_417445, *[A_417446, b_417447], **kwargs_417458)
        
        # Obtaining the member '__getitem__' of a type (line 114)
        getitem___417460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), gcrotmk_call_result_417459, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 114)
        subscript_call_result_417461 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), getitem___417460, int_417444)
        
        # Assigning a type to the variable 'tuple_var_assignment_416761' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'tuple_var_assignment_416761', subscript_call_result_417461)
        
        # Assigning a Subscript to a Name (line 114):
        
        # Obtaining the type of the subscript
        int_417462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 12), 'int')
        
        # Call to gcrotmk(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'A' (line 114)
        A_417464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 30), 'A', False)
        # Getting the type of 'b' (line 114)
        b_417465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 33), 'b', False)
        # Processing the call keyword arguments (line 114)
        int_417466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 38), 'int')
        keyword_417467 = int_417466
        int_417468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 44), 'int')
        keyword_417469 = int_417468
        # Getting the type of 'truncate' (line 114)
        truncate_417470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 57), 'truncate', False)
        keyword_417471 = truncate_417470
        float_417472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 71), 'float')
        keyword_417473 = float_417472
        int_417474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 38), 'int')
        keyword_417475 = int_417474
        kwargs_417476 = {'k': keyword_417469, 'm': keyword_417467, 'tol': keyword_417473, 'truncate': keyword_417471, 'maxiter': keyword_417475}
        # Getting the type of 'gcrotmk' (line 114)
        gcrotmk_417463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 22), 'gcrotmk', False)
        # Calling gcrotmk(args, kwargs) (line 114)
        gcrotmk_call_result_417477 = invoke(stypy.reporting.localization.Localization(__file__, 114, 22), gcrotmk_417463, *[A_417464, b_417465], **kwargs_417476)
        
        # Obtaining the member '__getitem__' of a type (line 114)
        getitem___417478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), gcrotmk_call_result_417477, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 114)
        subscript_call_result_417479 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), getitem___417478, int_417462)
        
        # Assigning a type to the variable 'tuple_var_assignment_416762' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'tuple_var_assignment_416762', subscript_call_result_417479)
        
        # Assigning a Name to a Name (line 114):
        # Getting the type of 'tuple_var_assignment_416761' (line 114)
        tuple_var_assignment_416761_417480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'tuple_var_assignment_416761')
        # Assigning a type to the variable 'x' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'x', tuple_var_assignment_416761_417480)
        
        # Assigning a Name to a Name (line 114):
        # Getting the type of 'tuple_var_assignment_416762' (line 114)
        tuple_var_assignment_416762_417481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'tuple_var_assignment_416762')
        # Assigning a type to the variable 'info' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 15), 'info', tuple_var_assignment_416762_417481)
        
        # Call to assert_equal(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'info' (line 116)
        info_417483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 25), 'info', False)
        int_417484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 31), 'int')
        # Processing the call keyword arguments (line 116)
        kwargs_417485 = {}
        # Getting the type of 'assert_equal' (line 116)
        assert_equal_417482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 116)
        assert_equal_call_result_417486 = invoke(stypy.reporting.localization.Localization(__file__, 116, 12), assert_equal_417482, *[info_417483, int_417484], **kwargs_417485)
        
        
        # Call to assert_allclose(...): (line 117)
        # Processing the call arguments (line 117)
        
        # Call to dot(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'x' (line 117)
        x_417490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 34), 'x', False)
        # Processing the call keyword arguments (line 117)
        kwargs_417491 = {}
        # Getting the type of 'A' (line 117)
        A_417488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 28), 'A', False)
        # Obtaining the member 'dot' of a type (line 117)
        dot_417489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 28), A_417488, 'dot')
        # Calling dot(args, kwargs) (line 117)
        dot_call_result_417492 = invoke(stypy.reporting.localization.Localization(__file__, 117, 28), dot_417489, *[x_417490], **kwargs_417491)
        
        # Getting the type of 'b' (line 117)
        b_417493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 39), 'b', False)
        # Applying the binary operator '-' (line 117)
        result_sub_417494 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 28), '-', dot_call_result_417492, b_417493)
        
        int_417495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 42), 'int')
        # Processing the call keyword arguments (line 117)
        float_417496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 50), 'float')
        keyword_417497 = float_417496
        kwargs_417498 = {'atol': keyword_417497}
        # Getting the type of 'assert_allclose' (line 117)
        assert_allclose_417487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 117)
        assert_allclose_call_result_417499 = invoke(stypy.reporting.localization.Localization(__file__, 117, 12), assert_allclose_417487, *[result_sub_417494, int_417495], **kwargs_417498)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_truncate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_truncate' in the type store
        # Getting the type of 'stypy_return_type' (line 108)
        stypy_return_type_417500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_417500)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_truncate'
        return stypy_return_type_417500


    @norecursion
    def test_CU(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_CU'
        module_type_store = module_type_store.open_function_context('test_CU', 119, 4, False)
        # Assigning a type to the variable 'self' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGCROTMK.test_CU.__dict__.__setitem__('stypy_localization', localization)
        TestGCROTMK.test_CU.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGCROTMK.test_CU.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGCROTMK.test_CU.__dict__.__setitem__('stypy_function_name', 'TestGCROTMK.test_CU')
        TestGCROTMK.test_CU.__dict__.__setitem__('stypy_param_names_list', [])
        TestGCROTMK.test_CU.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGCROTMK.test_CU.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGCROTMK.test_CU.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGCROTMK.test_CU.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGCROTMK.test_CU.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGCROTMK.test_CU.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGCROTMK.test_CU', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_CU', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_CU(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'tuple' (line 120)
        tuple_417501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 120)
        # Adding element type (line 120)
        # Getting the type of 'True' (line 120)
        True_417502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 26), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 26), tuple_417501, True_417502)
        # Adding element type (line 120)
        # Getting the type of 'False' (line 120)
        False_417503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 32), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 26), tuple_417501, False_417503)
        
        # Testing the type of a for loop iterable (line 120)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 120, 8), tuple_417501)
        # Getting the type of the for loop variable (line 120)
        for_loop_var_417504 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 120, 8), tuple_417501)
        # Assigning a type to the variable 'discard_C' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'discard_C', for_loop_var_417504)
        # SSA begins for a for statement (line 120)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a List to a Name (line 122):
        
        # Assigning a List to a Name (line 122):
        
        # Obtaining an instance of the builtin type 'list' (line 122)
        list_417505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 122)
        
        # Assigning a type to the variable 'CU' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'CU', list_417505)
        
        # Assigning a Call to a Tuple (line 123):
        
        # Assigning a Subscript to a Name (line 123):
        
        # Obtaining the type of the subscript
        int_417506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 12), 'int')
        
        # Call to do_solve(...): (line 123)
        # Processing the call keyword arguments (line 123)
        # Getting the type of 'CU' (line 123)
        CU_417508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 38), 'CU', False)
        keyword_417509 = CU_417508
        # Getting the type of 'discard_C' (line 123)
        discard_C_417510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 52), 'discard_C', False)
        keyword_417511 = discard_C_417510
        kwargs_417512 = {'discard_C': keyword_417511, 'CU': keyword_417509}
        # Getting the type of 'do_solve' (line 123)
        do_solve_417507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 26), 'do_solve', False)
        # Calling do_solve(args, kwargs) (line 123)
        do_solve_call_result_417513 = invoke(stypy.reporting.localization.Localization(__file__, 123, 26), do_solve_417507, *[], **kwargs_417512)
        
        # Obtaining the member '__getitem__' of a type (line 123)
        getitem___417514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 12), do_solve_call_result_417513, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 123)
        subscript_call_result_417515 = invoke(stypy.reporting.localization.Localization(__file__, 123, 12), getitem___417514, int_417506)
        
        # Assigning a type to the variable 'tuple_var_assignment_416763' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'tuple_var_assignment_416763', subscript_call_result_417515)
        
        # Assigning a Subscript to a Name (line 123):
        
        # Obtaining the type of the subscript
        int_417516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 12), 'int')
        
        # Call to do_solve(...): (line 123)
        # Processing the call keyword arguments (line 123)
        # Getting the type of 'CU' (line 123)
        CU_417518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 38), 'CU', False)
        keyword_417519 = CU_417518
        # Getting the type of 'discard_C' (line 123)
        discard_C_417520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 52), 'discard_C', False)
        keyword_417521 = discard_C_417520
        kwargs_417522 = {'discard_C': keyword_417521, 'CU': keyword_417519}
        # Getting the type of 'do_solve' (line 123)
        do_solve_417517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 26), 'do_solve', False)
        # Calling do_solve(args, kwargs) (line 123)
        do_solve_call_result_417523 = invoke(stypy.reporting.localization.Localization(__file__, 123, 26), do_solve_417517, *[], **kwargs_417522)
        
        # Obtaining the member '__getitem__' of a type (line 123)
        getitem___417524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 12), do_solve_call_result_417523, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 123)
        subscript_call_result_417525 = invoke(stypy.reporting.localization.Localization(__file__, 123, 12), getitem___417524, int_417516)
        
        # Assigning a type to the variable 'tuple_var_assignment_416764' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'tuple_var_assignment_416764', subscript_call_result_417525)
        
        # Assigning a Name to a Name (line 123):
        # Getting the type of 'tuple_var_assignment_416763' (line 123)
        tuple_var_assignment_416763_417526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'tuple_var_assignment_416763')
        # Assigning a type to the variable 'x0' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'x0', tuple_var_assignment_416763_417526)
        
        # Assigning a Name to a Name (line 123):
        # Getting the type of 'tuple_var_assignment_416764' (line 123)
        tuple_var_assignment_416764_417527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'tuple_var_assignment_416764')
        # Assigning a type to the variable 'count_0' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'count_0', tuple_var_assignment_416764_417527)
        
        # Call to assert_(...): (line 124)
        # Processing the call arguments (line 124)
        
        
        # Call to len(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'CU' (line 124)
        CU_417530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 24), 'CU', False)
        # Processing the call keyword arguments (line 124)
        kwargs_417531 = {}
        # Getting the type of 'len' (line 124)
        len_417529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'len', False)
        # Calling len(args, kwargs) (line 124)
        len_call_result_417532 = invoke(stypy.reporting.localization.Localization(__file__, 124, 20), len_417529, *[CU_417530], **kwargs_417531)
        
        int_417533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 30), 'int')
        # Applying the binary operator '>' (line 124)
        result_gt_417534 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 20), '>', len_call_result_417532, int_417533)
        
        # Processing the call keyword arguments (line 124)
        kwargs_417535 = {}
        # Getting the type of 'assert_' (line 124)
        assert__417528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 124)
        assert__call_result_417536 = invoke(stypy.reporting.localization.Localization(__file__, 124, 12), assert__417528, *[result_gt_417534], **kwargs_417535)
        
        
        # Call to assert_(...): (line 125)
        # Processing the call arguments (line 125)
        
        
        # Call to len(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'CU' (line 125)
        CU_417539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 24), 'CU', False)
        # Processing the call keyword arguments (line 125)
        kwargs_417540 = {}
        # Getting the type of 'len' (line 125)
        len_417538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 20), 'len', False)
        # Calling len(args, kwargs) (line 125)
        len_call_result_417541 = invoke(stypy.reporting.localization.Localization(__file__, 125, 20), len_417538, *[CU_417539], **kwargs_417540)
        
        int_417542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 31), 'int')
        # Applying the binary operator '<=' (line 125)
        result_le_417543 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 20), '<=', len_call_result_417541, int_417542)
        
        # Processing the call keyword arguments (line 125)
        kwargs_417544 = {}
        # Getting the type of 'assert_' (line 125)
        assert__417537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 125)
        assert__call_result_417545 = invoke(stypy.reporting.localization.Localization(__file__, 125, 12), assert__417537, *[result_le_417543], **kwargs_417544)
        
        
        # Getting the type of 'discard_C' (line 127)
        discard_C_417546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'discard_C')
        # Testing the type of an if condition (line 127)
        if_condition_417547 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 12), discard_C_417546)
        # Assigning a type to the variable 'if_condition_417547' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'if_condition_417547', if_condition_417547)
        # SSA begins for if statement (line 127)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'CU' (line 128)
        CU_417548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 28), 'CU')
        # Testing the type of a for loop iterable (line 128)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 128, 16), CU_417548)
        # Getting the type of the for loop variable (line 128)
        for_loop_var_417549 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 128, 16), CU_417548)
        # Assigning a type to the variable 'c' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'c', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 16), for_loop_var_417549))
        # Assigning a type to the variable 'u' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'u', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 16), for_loop_var_417549))
        # SSA begins for a for statement (line 128)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_(...): (line 129)
        # Processing the call arguments (line 129)
        
        # Getting the type of 'c' (line 129)
        c_417551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 28), 'c', False)
        # Getting the type of 'None' (line 129)
        None_417552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 33), 'None', False)
        # Applying the binary operator 'is' (line 129)
        result_is__417553 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 28), 'is', c_417551, None_417552)
        
        # Processing the call keyword arguments (line 129)
        kwargs_417554 = {}
        # Getting the type of 'assert_' (line 129)
        assert__417550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 20), 'assert_', False)
        # Calling assert_(args, kwargs) (line 129)
        assert__call_result_417555 = invoke(stypy.reporting.localization.Localization(__file__, 129, 20), assert__417550, *[result_is__417553], **kwargs_417554)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 127)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 132):
        
        # Assigning a Subscript to a Name (line 132):
        
        # Obtaining the type of the subscript
        int_417556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 12), 'int')
        
        # Call to do_solve(...): (line 132)
        # Processing the call keyword arguments (line 132)
        # Getting the type of 'CU' (line 132)
        CU_417558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 38), 'CU', False)
        keyword_417559 = CU_417558
        # Getting the type of 'discard_C' (line 132)
        discard_C_417560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 52), 'discard_C', False)
        keyword_417561 = discard_C_417560
        kwargs_417562 = {'discard_C': keyword_417561, 'CU': keyword_417559}
        # Getting the type of 'do_solve' (line 132)
        do_solve_417557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 26), 'do_solve', False)
        # Calling do_solve(args, kwargs) (line 132)
        do_solve_call_result_417563 = invoke(stypy.reporting.localization.Localization(__file__, 132, 26), do_solve_417557, *[], **kwargs_417562)
        
        # Obtaining the member '__getitem__' of a type (line 132)
        getitem___417564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 12), do_solve_call_result_417563, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
        subscript_call_result_417565 = invoke(stypy.reporting.localization.Localization(__file__, 132, 12), getitem___417564, int_417556)
        
        # Assigning a type to the variable 'tuple_var_assignment_416765' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'tuple_var_assignment_416765', subscript_call_result_417565)
        
        # Assigning a Subscript to a Name (line 132):
        
        # Obtaining the type of the subscript
        int_417566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 12), 'int')
        
        # Call to do_solve(...): (line 132)
        # Processing the call keyword arguments (line 132)
        # Getting the type of 'CU' (line 132)
        CU_417568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 38), 'CU', False)
        keyword_417569 = CU_417568
        # Getting the type of 'discard_C' (line 132)
        discard_C_417570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 52), 'discard_C', False)
        keyword_417571 = discard_C_417570
        kwargs_417572 = {'discard_C': keyword_417571, 'CU': keyword_417569}
        # Getting the type of 'do_solve' (line 132)
        do_solve_417567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 26), 'do_solve', False)
        # Calling do_solve(args, kwargs) (line 132)
        do_solve_call_result_417573 = invoke(stypy.reporting.localization.Localization(__file__, 132, 26), do_solve_417567, *[], **kwargs_417572)
        
        # Obtaining the member '__getitem__' of a type (line 132)
        getitem___417574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 12), do_solve_call_result_417573, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
        subscript_call_result_417575 = invoke(stypy.reporting.localization.Localization(__file__, 132, 12), getitem___417574, int_417566)
        
        # Assigning a type to the variable 'tuple_var_assignment_416766' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'tuple_var_assignment_416766', subscript_call_result_417575)
        
        # Assigning a Name to a Name (line 132):
        # Getting the type of 'tuple_var_assignment_416765' (line 132)
        tuple_var_assignment_416765_417576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'tuple_var_assignment_416765')
        # Assigning a type to the variable 'x1' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'x1', tuple_var_assignment_416765_417576)
        
        # Assigning a Name to a Name (line 132):
        # Getting the type of 'tuple_var_assignment_416766' (line 132)
        tuple_var_assignment_416766_417577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'tuple_var_assignment_416766')
        # Assigning a type to the variable 'count_1' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'count_1', tuple_var_assignment_416766_417577)
        
        # Getting the type of 'discard_C' (line 133)
        discard_C_417578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), 'discard_C')
        # Testing the type of an if condition (line 133)
        if_condition_417579 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 12), discard_C_417578)
        # Assigning a type to the variable 'if_condition_417579' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'if_condition_417579', if_condition_417579)
        # SSA begins for if statement (line 133)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_equal(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'count_1' (line 134)
        count_1_417581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 29), 'count_1', False)
        int_417582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 38), 'int')
        
        # Call to len(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'CU' (line 134)
        CU_417584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 46), 'CU', False)
        # Processing the call keyword arguments (line 134)
        kwargs_417585 = {}
        # Getting the type of 'len' (line 134)
        len_417583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 42), 'len', False)
        # Calling len(args, kwargs) (line 134)
        len_call_result_417586 = invoke(stypy.reporting.localization.Localization(__file__, 134, 42), len_417583, *[CU_417584], **kwargs_417585)
        
        # Applying the binary operator '+' (line 134)
        result_add_417587 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 38), '+', int_417582, len_call_result_417586)
        
        # Processing the call keyword arguments (line 134)
        kwargs_417588 = {}
        # Getting the type of 'assert_equal' (line 134)
        assert_equal_417580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 134)
        assert_equal_call_result_417589 = invoke(stypy.reporting.localization.Localization(__file__, 134, 16), assert_equal_417580, *[count_1_417581, result_add_417587], **kwargs_417588)
        
        # SSA branch for the else part of an if statement (line 133)
        module_type_store.open_ssa_branch('else')
        
        # Call to assert_equal(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'count_1' (line 136)
        count_1_417591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 29), 'count_1', False)
        int_417592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 38), 'int')
        # Processing the call keyword arguments (line 136)
        kwargs_417593 = {}
        # Getting the type of 'assert_equal' (line 136)
        assert_equal_417590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 136)
        assert_equal_call_result_417594 = invoke(stypy.reporting.localization.Localization(__file__, 136, 16), assert_equal_417590, *[count_1_417591, int_417592], **kwargs_417593)
        
        # SSA join for if statement (line 133)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_(...): (line 137)
        # Processing the call arguments (line 137)
        
        # Getting the type of 'count_1' (line 137)
        count_1_417596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 20), 'count_1', False)
        # Getting the type of 'count_0' (line 137)
        count_0_417597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 30), 'count_0', False)
        int_417598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 38), 'int')
        # Applying the binary operator 'div' (line 137)
        result_div_417599 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 30), 'div', count_0_417597, int_417598)
        
        # Applying the binary operator '<' (line 137)
        result_lt_417600 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 20), '<', count_1_417596, result_div_417599)
        
        # Processing the call keyword arguments (line 137)
        kwargs_417601 = {}
        # Getting the type of 'assert_' (line 137)
        assert__417595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 137)
        assert__call_result_417602 = invoke(stypy.reporting.localization.Localization(__file__, 137, 12), assert__417595, *[result_lt_417600], **kwargs_417601)
        
        
        # Call to assert_allclose(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'x1' (line 138)
        x1_417604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 28), 'x1', False)
        # Getting the type of 'x0' (line 138)
        x0_417605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 32), 'x0', False)
        # Processing the call keyword arguments (line 138)
        float_417606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 41), 'float')
        keyword_417607 = float_417606
        kwargs_417608 = {'atol': keyword_417607}
        # Getting the type of 'assert_allclose' (line 138)
        assert_allclose_417603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 138)
        assert_allclose_call_result_417609 = invoke(stypy.reporting.localization.Localization(__file__, 138, 12), assert_allclose_417603, *[x1_417604, x0_417605], **kwargs_417608)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_CU(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_CU' in the type store
        # Getting the type of 'stypy_return_type' (line 119)
        stypy_return_type_417610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_417610)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_CU'
        return stypy_return_type_417610


    @norecursion
    def test_denormals(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_denormals'
        module_type_store = module_type_store.open_function_context('test_denormals', 140, 4, False)
        # Assigning a type to the variable 'self' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGCROTMK.test_denormals.__dict__.__setitem__('stypy_localization', localization)
        TestGCROTMK.test_denormals.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGCROTMK.test_denormals.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGCROTMK.test_denormals.__dict__.__setitem__('stypy_function_name', 'TestGCROTMK.test_denormals')
        TestGCROTMK.test_denormals.__dict__.__setitem__('stypy_param_names_list', [])
        TestGCROTMK.test_denormals.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGCROTMK.test_denormals.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGCROTMK.test_denormals.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGCROTMK.test_denormals.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGCROTMK.test_denormals.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGCROTMK.test_denormals.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGCROTMK.test_denormals', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 144):
        
        # Assigning a Call to a Name (line 144):
        
        # Call to array(...): (line 144)
        # Processing the call arguments (line 144)
        
        # Obtaining an instance of the builtin type 'list' (line 144)
        list_417613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 144)
        # Adding element type (line 144)
        
        # Obtaining an instance of the builtin type 'list' (line 144)
        list_417614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 144)
        # Adding element type (line 144)
        int_417615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 22), list_417614, int_417615)
        # Adding element type (line 144)
        int_417616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 22), list_417614, int_417616)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 21), list_417613, list_417614)
        # Adding element type (line 144)
        
        # Obtaining an instance of the builtin type 'list' (line 144)
        list_417617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 144)
        # Adding element type (line 144)
        int_417618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 30), list_417617, int_417618)
        # Adding element type (line 144)
        int_417619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 30), list_417617, int_417619)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 21), list_417613, list_417617)
        
        # Processing the call keyword arguments (line 144)
        # Getting the type of 'float' (line 144)
        float_417620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 45), 'float', False)
        keyword_417621 = float_417620
        kwargs_417622 = {'dtype': keyword_417621}
        # Getting the type of 'np' (line 144)
        np_417611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 144)
        array_417612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 12), np_417611, 'array')
        # Calling array(args, kwargs) (line 144)
        array_call_result_417623 = invoke(stypy.reporting.localization.Localization(__file__, 144, 12), array_417612, *[list_417613], **kwargs_417622)
        
        # Assigning a type to the variable 'A' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'A', array_call_result_417623)
        
        # Getting the type of 'A' (line 145)
        A_417624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'A')
        int_417625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 13), 'int')
        
        # Call to nextafter(...): (line 145)
        # Processing the call arguments (line 145)
        int_417628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 32), 'int')
        int_417629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 35), 'int')
        # Processing the call keyword arguments (line 145)
        kwargs_417630 = {}
        # Getting the type of 'np' (line 145)
        np_417626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 19), 'np', False)
        # Obtaining the member 'nextafter' of a type (line 145)
        nextafter_417627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 19), np_417626, 'nextafter')
        # Calling nextafter(args, kwargs) (line 145)
        nextafter_call_result_417631 = invoke(stypy.reporting.localization.Localization(__file__, 145, 19), nextafter_417627, *[int_417628, int_417629], **kwargs_417630)
        
        # Applying the binary operator '*' (line 145)
        result_mul_417632 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 13), '*', int_417625, nextafter_call_result_417631)
        
        # Applying the binary operator '*=' (line 145)
        result_imul_417633 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 8), '*=', A_417624, result_mul_417632)
        # Assigning a type to the variable 'A' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'A', result_imul_417633)
        
        
        # Assigning a Call to a Name (line 147):
        
        # Assigning a Call to a Name (line 147):
        
        # Call to array(...): (line 147)
        # Processing the call arguments (line 147)
        
        # Obtaining an instance of the builtin type 'list' (line 147)
        list_417636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 147)
        # Adding element type (line 147)
        int_417637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 21), list_417636, int_417637)
        # Adding element type (line 147)
        int_417638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 21), list_417636, int_417638)
        
        # Processing the call keyword arguments (line 147)
        kwargs_417639 = {}
        # Getting the type of 'np' (line 147)
        np_417634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 147)
        array_417635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 12), np_417634, 'array')
        # Calling array(args, kwargs) (line 147)
        array_call_result_417640 = invoke(stypy.reporting.localization.Localization(__file__, 147, 12), array_417635, *[list_417636], **kwargs_417639)
        
        # Assigning a type to the variable 'b' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'b', array_call_result_417640)
        
        # Assigning a Call to a Tuple (line 149):
        
        # Assigning a Subscript to a Name (line 149):
        
        # Obtaining the type of the subscript
        int_417641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 8), 'int')
        
        # Call to gcrotmk(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'A' (line 149)
        A_417643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 27), 'A', False)
        # Getting the type of 'b' (line 149)
        b_417644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 30), 'b', False)
        # Processing the call keyword arguments (line 149)
        kwargs_417645 = {}
        # Getting the type of 'gcrotmk' (line 149)
        gcrotmk_417642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 19), 'gcrotmk', False)
        # Calling gcrotmk(args, kwargs) (line 149)
        gcrotmk_call_result_417646 = invoke(stypy.reporting.localization.Localization(__file__, 149, 19), gcrotmk_417642, *[A_417643, b_417644], **kwargs_417645)
        
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___417647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), gcrotmk_call_result_417646, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_417648 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), getitem___417647, int_417641)
        
        # Assigning a type to the variable 'tuple_var_assignment_416767' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'tuple_var_assignment_416767', subscript_call_result_417648)
        
        # Assigning a Subscript to a Name (line 149):
        
        # Obtaining the type of the subscript
        int_417649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 8), 'int')
        
        # Call to gcrotmk(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'A' (line 149)
        A_417651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 27), 'A', False)
        # Getting the type of 'b' (line 149)
        b_417652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 30), 'b', False)
        # Processing the call keyword arguments (line 149)
        kwargs_417653 = {}
        # Getting the type of 'gcrotmk' (line 149)
        gcrotmk_417650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 19), 'gcrotmk', False)
        # Calling gcrotmk(args, kwargs) (line 149)
        gcrotmk_call_result_417654 = invoke(stypy.reporting.localization.Localization(__file__, 149, 19), gcrotmk_417650, *[A_417651, b_417652], **kwargs_417653)
        
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___417655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), gcrotmk_call_result_417654, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_417656 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), getitem___417655, int_417649)
        
        # Assigning a type to the variable 'tuple_var_assignment_416768' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'tuple_var_assignment_416768', subscript_call_result_417656)
        
        # Assigning a Name to a Name (line 149):
        # Getting the type of 'tuple_var_assignment_416767' (line 149)
        tuple_var_assignment_416767_417657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'tuple_var_assignment_416767')
        # Assigning a type to the variable 'xp' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'xp', tuple_var_assignment_416767_417657)
        
        # Assigning a Name to a Name (line 149):
        # Getting the type of 'tuple_var_assignment_416768' (line 149)
        tuple_var_assignment_416768_417658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'tuple_var_assignment_416768')
        # Assigning a type to the variable 'info' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'info', tuple_var_assignment_416768_417658)
        
        
        # Getting the type of 'info' (line 151)
        info_417659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 11), 'info')
        int_417660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 19), 'int')
        # Applying the binary operator '==' (line 151)
        result_eq_417661 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 11), '==', info_417659, int_417660)
        
        # Testing the type of an if condition (line 151)
        if_condition_417662 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 8), result_eq_417661)
        # Assigning a type to the variable 'if_condition_417662' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'if_condition_417662', if_condition_417662)
        # SSA begins for if statement (line 151)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_allclose(...): (line 152)
        # Processing the call arguments (line 152)
        
        # Call to dot(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'xp' (line 152)
        xp_417666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 34), 'xp', False)
        # Processing the call keyword arguments (line 152)
        kwargs_417667 = {}
        # Getting the type of 'A' (line 152)
        A_417664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 28), 'A', False)
        # Obtaining the member 'dot' of a type (line 152)
        dot_417665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 28), A_417664, 'dot')
        # Calling dot(args, kwargs) (line 152)
        dot_call_result_417668 = invoke(stypy.reporting.localization.Localization(__file__, 152, 28), dot_417665, *[xp_417666], **kwargs_417667)
        
        # Getting the type of 'b' (line 152)
        b_417669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 39), 'b', False)
        # Processing the call keyword arguments (line 152)
        kwargs_417670 = {}
        # Getting the type of 'assert_allclose' (line 152)
        assert_allclose_417663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 152)
        assert_allclose_call_result_417671 = invoke(stypy.reporting.localization.Localization(__file__, 152, 12), assert_allclose_417663, *[dot_call_result_417668, b_417669], **kwargs_417670)
        
        # SSA join for if statement (line 151)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_denormals(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_denormals' in the type store
        # Getting the type of 'stypy_return_type' (line 140)
        stypy_return_type_417672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_417672)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_denormals'
        return stypy_return_type_417672


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 43, 0, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGCROTMK.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestGCROTMK' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'TestGCROTMK', TestGCROTMK)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
