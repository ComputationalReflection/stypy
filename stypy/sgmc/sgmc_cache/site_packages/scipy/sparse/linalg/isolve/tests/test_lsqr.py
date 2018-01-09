
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import (assert_, assert_equal, assert_almost_equal,
5:                            assert_array_almost_equal)
6: from scipy._lib.six import xrange
7: 
8: import scipy.sparse
9: import scipy.sparse.linalg
10: from scipy.sparse.linalg import lsqr
11: from time import time
12: 
13: # Set up a test problem
14: n = 35
15: G = np.eye(n)
16: normal = np.random.normal
17: norm = np.linalg.norm
18: 
19: for jj in xrange(5):
20:     gg = normal(size=n)
21:     hh = gg * gg.T
22:     G += (hh + hh.T) * 0.5
23:     G += normal(size=n) * normal(size=n)
24: 
25: b = normal(size=n)
26: 
27: tol = 1e-10
28: show = False
29: maxit = None
30: 
31: 
32: def test_basic():
33:     b_copy = b.copy()
34:     X = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=maxit)
35:     assert_(np.all(b_copy == b))
36: 
37:     svx = np.linalg.solve(G, b)
38:     xo = X[0]
39:     assert_(norm(svx - xo) < 1e-5)
40: 
41: def test_gh_2466():
42:     row = np.array([0, 0])
43:     col = np.array([0, 1])
44:     val = np.array([1, -1])
45:     A = scipy.sparse.coo_matrix((val, (row, col)), shape=(1, 2))
46:     b = np.asarray([4])
47:     lsqr(A, b)
48: 
49: 
50: def test_well_conditioned_problems():
51:     # Test that sparse the lsqr solver returns the right solution
52:     # on various problems with different random seeds.
53:     # This is a non-regression test for a potential ZeroDivisionError
54:     # raised when computing the `test2` & `test3` convergence conditions.
55:     n = 10
56:     A_sparse = scipy.sparse.eye(n, n)
57:     A_dense = A_sparse.toarray()
58: 
59:     with np.errstate(invalid='raise'):
60:         for seed in range(30):
61:             rng = np.random.RandomState(seed + 10)
62:             beta = rng.rand(n)
63:             beta[beta == 0] = 0.00001  # ensure that all the betas are not null
64:             b = A_sparse * beta[:, np.newaxis]
65:             output = lsqr(A_sparse, b, show=show)
66: 
67:             # Check that the termination condition corresponds to an approximate
68:             # solution to Ax = b
69:             assert_equal(output[1], 1)
70:             solution = output[0]
71: 
72:             # Check that we recover the ground truth solution
73:             assert_array_almost_equal(solution, beta)
74: 
75:             # Sanity check: compare to the dense array solver
76:             reference_solution = np.linalg.solve(A_dense, b).ravel()
77:             assert_array_almost_equal(solution, reference_solution)
78: 
79: 
80: def test_b_shapes():
81:     # Test b being a scalar.
82:     A = np.array([[1.0, 2.0]])
83:     b = 3.0
84:     x = lsqr(A, b)[0]
85:     assert_almost_equal(norm(A.dot(x) - b), 0)
86: 
87:     # Test b being a column vector.
88:     A = np.eye(10)
89:     b = np.ones((10, 1))
90:     x = lsqr(A, b)[0]
91:     assert_almost_equal(norm(A.dot(x) - b.ravel()), 0)
92: 
93: 
94: def test_initialization():
95:     # Test the default setting is the same as zeros
96:     b_copy = b.copy()
97:     x_ref = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=maxit)
98:     x0 = np.zeros(x_ref[0].shape)
99:     x = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=maxit, x0=x0)
100:     assert_(np.all(b_copy == b))
101:     assert_array_almost_equal(x_ref[0], x[0])
102: 
103:     # Test warm-start with single iteration
104:     x0 = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=1)[0]
105:     x = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=maxit, x0=x0)
106:     assert_array_almost_equal(x_ref[0], x[0])
107:     assert_(np.all(b_copy == b))
108: 
109: 
110: if __name__ == "__main__":
111:     svx = np.linalg.solve(G, b)
112: 
113:     tic = time()
114:     X = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=maxit)
115:     xo = X[0]
116:     phio = X[3]
117:     psio = X[7]
118:     k = X[2]
119:     chio = X[8]
120:     mg = np.amax(G - G.T)
121:     if mg > 1e-14:
122:         sym = 'No'
123:     else:
124:         sym = 'Yes'
125: 
126:     print('LSQR')
127:     print("Is linear operator symmetric? " + sym)
128:     print("n: %3g  iterations:   %3g" % (n, k))
129:     print("Norms computed in %.2fs by LSQR" % (time() - tic))
130:     print(" ||x||  %9.4e  ||r|| %9.4e  ||Ar||  %9.4e " % (chio, phio, psio))
131:     print("Residual norms computed directly:")
132:     print(" ||x||  %9.4e  ||r|| %9.4e  ||Ar||  %9.4e" % (norm(xo),
133:                                                           norm(G*xo - b),
134:                                                           norm(G.T*(G*xo-b))))
135:     print("Direct solution norms:")
136:     print(" ||x||  %9.4e  ||r|| %9.4e " % (norm(svx), norm(G*svx - b)))
137:     print("")
138:     print(" || x_{direct} - x_{LSQR}|| %9.4e " % norm(svx-xo))
139:     print("")
140: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_421733 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_421733) is not StypyTypeError):

    if (import_421733 != 'pyd_module'):
        __import__(import_421733)
        sys_modules_421734 = sys.modules[import_421733]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_421734.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_421733)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_, assert_equal, assert_almost_equal, assert_array_almost_equal' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_421735 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_421735) is not StypyTypeError):

    if (import_421735 != 'pyd_module'):
        __import__(import_421735)
        sys_modules_421736 = sys.modules[import_421735]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_421736.module_type_store, module_type_store, ['assert_', 'assert_equal', 'assert_almost_equal', 'assert_array_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_421736, sys_modules_421736.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_equal, assert_almost_equal, assert_array_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_equal', 'assert_almost_equal', 'assert_array_almost_equal'], [assert_, assert_equal, assert_almost_equal, assert_array_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_421735)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy._lib.six import xrange' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_421737 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy._lib.six')

if (type(import_421737) is not StypyTypeError):

    if (import_421737 != 'pyd_module'):
        __import__(import_421737)
        sys_modules_421738 = sys.modules[import_421737]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy._lib.six', sys_modules_421738.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_421738, sys_modules_421738.module_type_store, module_type_store)
    else:
        from scipy._lib.six import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy._lib.six', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy._lib.six', import_421737)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import scipy.sparse' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_421739 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse')

if (type(import_421739) is not StypyTypeError):

    if (import_421739 != 'pyd_module'):
        __import__(import_421739)
        sys_modules_421740 = sys.modules[import_421739]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse', sys_modules_421740.module_type_store, module_type_store)
    else:
        import scipy.sparse

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse', scipy.sparse, module_type_store)

else:
    # Assigning a type to the variable 'scipy.sparse' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse', import_421739)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import scipy.sparse.linalg' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_421741 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse.linalg')

if (type(import_421741) is not StypyTypeError):

    if (import_421741 != 'pyd_module'):
        __import__(import_421741)
        sys_modules_421742 = sys.modules[import_421741]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse.linalg', sys_modules_421742.module_type_store, module_type_store)
    else:
        import scipy.sparse.linalg

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse.linalg', scipy.sparse.linalg, module_type_store)

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse.linalg', import_421741)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.sparse.linalg import lsqr' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_421743 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg')

if (type(import_421743) is not StypyTypeError):

    if (import_421743 != 'pyd_module'):
        __import__(import_421743)
        sys_modules_421744 = sys.modules[import_421743]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg', sys_modules_421744.module_type_store, module_type_store, ['lsqr'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_421744, sys_modules_421744.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import lsqr

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg', None, module_type_store, ['lsqr'], [lsqr])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg', import_421743)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from time import time' statement (line 11)
try:
    from time import time

except:
    time = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'time', None, module_type_store, ['time'], [time])


# Assigning a Num to a Name (line 14):
int_421745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'int')
# Assigning a type to the variable 'n' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'n', int_421745)

# Assigning a Call to a Name (line 15):

# Call to eye(...): (line 15)
# Processing the call arguments (line 15)
# Getting the type of 'n' (line 15)
n_421748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 11), 'n', False)
# Processing the call keyword arguments (line 15)
kwargs_421749 = {}
# Getting the type of 'np' (line 15)
np_421746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'np', False)
# Obtaining the member 'eye' of a type (line 15)
eye_421747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 4), np_421746, 'eye')
# Calling eye(args, kwargs) (line 15)
eye_call_result_421750 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), eye_421747, *[n_421748], **kwargs_421749)

# Assigning a type to the variable 'G' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'G', eye_call_result_421750)

# Assigning a Attribute to a Name (line 16):
# Getting the type of 'np' (line 16)
np_421751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 9), 'np')
# Obtaining the member 'random' of a type (line 16)
random_421752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 9), np_421751, 'random')
# Obtaining the member 'normal' of a type (line 16)
normal_421753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 9), random_421752, 'normal')
# Assigning a type to the variable 'normal' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'normal', normal_421753)

# Assigning a Attribute to a Name (line 17):
# Getting the type of 'np' (line 17)
np_421754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 7), 'np')
# Obtaining the member 'linalg' of a type (line 17)
linalg_421755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 7), np_421754, 'linalg')
# Obtaining the member 'norm' of a type (line 17)
norm_421756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 7), linalg_421755, 'norm')
# Assigning a type to the variable 'norm' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'norm', norm_421756)


# Call to xrange(...): (line 19)
# Processing the call arguments (line 19)
int_421758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 17), 'int')
# Processing the call keyword arguments (line 19)
kwargs_421759 = {}
# Getting the type of 'xrange' (line 19)
xrange_421757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'xrange', False)
# Calling xrange(args, kwargs) (line 19)
xrange_call_result_421760 = invoke(stypy.reporting.localization.Localization(__file__, 19, 10), xrange_421757, *[int_421758], **kwargs_421759)

# Testing the type of a for loop iterable (line 19)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 19, 0), xrange_call_result_421760)
# Getting the type of the for loop variable (line 19)
for_loop_var_421761 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 19, 0), xrange_call_result_421760)
# Assigning a type to the variable 'jj' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'jj', for_loop_var_421761)
# SSA begins for a for statement (line 19)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a Call to a Name (line 20):

# Call to normal(...): (line 20)
# Processing the call keyword arguments (line 20)
# Getting the type of 'n' (line 20)
n_421763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 21), 'n', False)
keyword_421764 = n_421763
kwargs_421765 = {'size': keyword_421764}
# Getting the type of 'normal' (line 20)
normal_421762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 9), 'normal', False)
# Calling normal(args, kwargs) (line 20)
normal_call_result_421766 = invoke(stypy.reporting.localization.Localization(__file__, 20, 9), normal_421762, *[], **kwargs_421765)

# Assigning a type to the variable 'gg' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'gg', normal_call_result_421766)

# Assigning a BinOp to a Name (line 21):
# Getting the type of 'gg' (line 21)
gg_421767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 9), 'gg')
# Getting the type of 'gg' (line 21)
gg_421768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 14), 'gg')
# Obtaining the member 'T' of a type (line 21)
T_421769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 14), gg_421768, 'T')
# Applying the binary operator '*' (line 21)
result_mul_421770 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 9), '*', gg_421767, T_421769)

# Assigning a type to the variable 'hh' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'hh', result_mul_421770)

# Getting the type of 'G' (line 22)
G_421771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'G')
# Getting the type of 'hh' (line 22)
hh_421772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 10), 'hh')
# Getting the type of 'hh' (line 22)
hh_421773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 15), 'hh')
# Obtaining the member 'T' of a type (line 22)
T_421774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 15), hh_421773, 'T')
# Applying the binary operator '+' (line 22)
result_add_421775 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 10), '+', hh_421772, T_421774)

float_421776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'float')
# Applying the binary operator '*' (line 22)
result_mul_421777 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 9), '*', result_add_421775, float_421776)

# Applying the binary operator '+=' (line 22)
result_iadd_421778 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 4), '+=', G_421771, result_mul_421777)
# Assigning a type to the variable 'G' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'G', result_iadd_421778)


# Getting the type of 'G' (line 23)
G_421779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'G')

# Call to normal(...): (line 23)
# Processing the call keyword arguments (line 23)
# Getting the type of 'n' (line 23)
n_421781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 21), 'n', False)
keyword_421782 = n_421781
kwargs_421783 = {'size': keyword_421782}
# Getting the type of 'normal' (line 23)
normal_421780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 9), 'normal', False)
# Calling normal(args, kwargs) (line 23)
normal_call_result_421784 = invoke(stypy.reporting.localization.Localization(__file__, 23, 9), normal_421780, *[], **kwargs_421783)


# Call to normal(...): (line 23)
# Processing the call keyword arguments (line 23)
# Getting the type of 'n' (line 23)
n_421786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 38), 'n', False)
keyword_421787 = n_421786
kwargs_421788 = {'size': keyword_421787}
# Getting the type of 'normal' (line 23)
normal_421785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 26), 'normal', False)
# Calling normal(args, kwargs) (line 23)
normal_call_result_421789 = invoke(stypy.reporting.localization.Localization(__file__, 23, 26), normal_421785, *[], **kwargs_421788)

# Applying the binary operator '*' (line 23)
result_mul_421790 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 9), '*', normal_call_result_421784, normal_call_result_421789)

# Applying the binary operator '+=' (line 23)
result_iadd_421791 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 4), '+=', G_421779, result_mul_421790)
# Assigning a type to the variable 'G' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'G', result_iadd_421791)

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# Assigning a Call to a Name (line 25):

# Call to normal(...): (line 25)
# Processing the call keyword arguments (line 25)
# Getting the type of 'n' (line 25)
n_421793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'n', False)
keyword_421794 = n_421793
kwargs_421795 = {'size': keyword_421794}
# Getting the type of 'normal' (line 25)
normal_421792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'normal', False)
# Calling normal(args, kwargs) (line 25)
normal_call_result_421796 = invoke(stypy.reporting.localization.Localization(__file__, 25, 4), normal_421792, *[], **kwargs_421795)

# Assigning a type to the variable 'b' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'b', normal_call_result_421796)

# Assigning a Num to a Name (line 27):
float_421797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 6), 'float')
# Assigning a type to the variable 'tol' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'tol', float_421797)

# Assigning a Name to a Name (line 28):
# Getting the type of 'False' (line 28)
False_421798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 7), 'False')
# Assigning a type to the variable 'show' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'show', False_421798)

# Assigning a Name to a Name (line 29):
# Getting the type of 'None' (line 29)
None_421799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'None')
# Assigning a type to the variable 'maxit' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'maxit', None_421799)

@norecursion
def test_basic(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_basic'
    module_type_store = module_type_store.open_function_context('test_basic', 32, 0, False)
    
    # Passed parameters checking function
    test_basic.stypy_localization = localization
    test_basic.stypy_type_of_self = None
    test_basic.stypy_type_store = module_type_store
    test_basic.stypy_function_name = 'test_basic'
    test_basic.stypy_param_names_list = []
    test_basic.stypy_varargs_param_name = None
    test_basic.stypy_kwargs_param_name = None
    test_basic.stypy_call_defaults = defaults
    test_basic.stypy_call_varargs = varargs
    test_basic.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_basic', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_basic', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_basic(...)' code ##################

    
    # Assigning a Call to a Name (line 33):
    
    # Call to copy(...): (line 33)
    # Processing the call keyword arguments (line 33)
    kwargs_421802 = {}
    # Getting the type of 'b' (line 33)
    b_421800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 13), 'b', False)
    # Obtaining the member 'copy' of a type (line 33)
    copy_421801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 13), b_421800, 'copy')
    # Calling copy(args, kwargs) (line 33)
    copy_call_result_421803 = invoke(stypy.reporting.localization.Localization(__file__, 33, 13), copy_421801, *[], **kwargs_421802)
    
    # Assigning a type to the variable 'b_copy' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'b_copy', copy_call_result_421803)
    
    # Assigning a Call to a Name (line 34):
    
    # Call to lsqr(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'G' (line 34)
    G_421805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 13), 'G', False)
    # Getting the type of 'b' (line 34)
    b_421806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'b', False)
    # Processing the call keyword arguments (line 34)
    # Getting the type of 'show' (line 34)
    show_421807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 24), 'show', False)
    keyword_421808 = show_421807
    # Getting the type of 'tol' (line 34)
    tol_421809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 35), 'tol', False)
    keyword_421810 = tol_421809
    # Getting the type of 'tol' (line 34)
    tol_421811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 45), 'tol', False)
    keyword_421812 = tol_421811
    # Getting the type of 'maxit' (line 34)
    maxit_421813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 59), 'maxit', False)
    keyword_421814 = maxit_421813
    kwargs_421815 = {'iter_lim': keyword_421814, 'btol': keyword_421812, 'atol': keyword_421810, 'show': keyword_421808}
    # Getting the type of 'lsqr' (line 34)
    lsqr_421804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'lsqr', False)
    # Calling lsqr(args, kwargs) (line 34)
    lsqr_call_result_421816 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), lsqr_421804, *[G_421805, b_421806], **kwargs_421815)
    
    # Assigning a type to the variable 'X' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'X', lsqr_call_result_421816)
    
    # Call to assert_(...): (line 35)
    # Processing the call arguments (line 35)
    
    # Call to all(...): (line 35)
    # Processing the call arguments (line 35)
    
    # Getting the type of 'b_copy' (line 35)
    b_copy_421820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 19), 'b_copy', False)
    # Getting the type of 'b' (line 35)
    b_421821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 29), 'b', False)
    # Applying the binary operator '==' (line 35)
    result_eq_421822 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 19), '==', b_copy_421820, b_421821)
    
    # Processing the call keyword arguments (line 35)
    kwargs_421823 = {}
    # Getting the type of 'np' (line 35)
    np_421818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'np', False)
    # Obtaining the member 'all' of a type (line 35)
    all_421819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), np_421818, 'all')
    # Calling all(args, kwargs) (line 35)
    all_call_result_421824 = invoke(stypy.reporting.localization.Localization(__file__, 35, 12), all_421819, *[result_eq_421822], **kwargs_421823)
    
    # Processing the call keyword arguments (line 35)
    kwargs_421825 = {}
    # Getting the type of 'assert_' (line 35)
    assert__421817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 35)
    assert__call_result_421826 = invoke(stypy.reporting.localization.Localization(__file__, 35, 4), assert__421817, *[all_call_result_421824], **kwargs_421825)
    
    
    # Assigning a Call to a Name (line 37):
    
    # Call to solve(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'G' (line 37)
    G_421830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 26), 'G', False)
    # Getting the type of 'b' (line 37)
    b_421831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 29), 'b', False)
    # Processing the call keyword arguments (line 37)
    kwargs_421832 = {}
    # Getting the type of 'np' (line 37)
    np_421827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 10), 'np', False)
    # Obtaining the member 'linalg' of a type (line 37)
    linalg_421828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 10), np_421827, 'linalg')
    # Obtaining the member 'solve' of a type (line 37)
    solve_421829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 10), linalg_421828, 'solve')
    # Calling solve(args, kwargs) (line 37)
    solve_call_result_421833 = invoke(stypy.reporting.localization.Localization(__file__, 37, 10), solve_421829, *[G_421830, b_421831], **kwargs_421832)
    
    # Assigning a type to the variable 'svx' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'svx', solve_call_result_421833)
    
    # Assigning a Subscript to a Name (line 38):
    
    # Obtaining the type of the subscript
    int_421834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 11), 'int')
    # Getting the type of 'X' (line 38)
    X_421835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 9), 'X')
    # Obtaining the member '__getitem__' of a type (line 38)
    getitem___421836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 9), X_421835, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 38)
    subscript_call_result_421837 = invoke(stypy.reporting.localization.Localization(__file__, 38, 9), getitem___421836, int_421834)
    
    # Assigning a type to the variable 'xo' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'xo', subscript_call_result_421837)
    
    # Call to assert_(...): (line 39)
    # Processing the call arguments (line 39)
    
    
    # Call to norm(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'svx' (line 39)
    svx_421840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 17), 'svx', False)
    # Getting the type of 'xo' (line 39)
    xo_421841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 23), 'xo', False)
    # Applying the binary operator '-' (line 39)
    result_sub_421842 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 17), '-', svx_421840, xo_421841)
    
    # Processing the call keyword arguments (line 39)
    kwargs_421843 = {}
    # Getting the type of 'norm' (line 39)
    norm_421839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'norm', False)
    # Calling norm(args, kwargs) (line 39)
    norm_call_result_421844 = invoke(stypy.reporting.localization.Localization(__file__, 39, 12), norm_421839, *[result_sub_421842], **kwargs_421843)
    
    float_421845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 29), 'float')
    # Applying the binary operator '<' (line 39)
    result_lt_421846 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 12), '<', norm_call_result_421844, float_421845)
    
    # Processing the call keyword arguments (line 39)
    kwargs_421847 = {}
    # Getting the type of 'assert_' (line 39)
    assert__421838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 39)
    assert__call_result_421848 = invoke(stypy.reporting.localization.Localization(__file__, 39, 4), assert__421838, *[result_lt_421846], **kwargs_421847)
    
    
    # ################# End of 'test_basic(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_basic' in the type store
    # Getting the type of 'stypy_return_type' (line 32)
    stypy_return_type_421849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_421849)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_basic'
    return stypy_return_type_421849

# Assigning a type to the variable 'test_basic' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'test_basic', test_basic)

@norecursion
def test_gh_2466(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_gh_2466'
    module_type_store = module_type_store.open_function_context('test_gh_2466', 41, 0, False)
    
    # Passed parameters checking function
    test_gh_2466.stypy_localization = localization
    test_gh_2466.stypy_type_of_self = None
    test_gh_2466.stypy_type_store = module_type_store
    test_gh_2466.stypy_function_name = 'test_gh_2466'
    test_gh_2466.stypy_param_names_list = []
    test_gh_2466.stypy_varargs_param_name = None
    test_gh_2466.stypy_kwargs_param_name = None
    test_gh_2466.stypy_call_defaults = defaults
    test_gh_2466.stypy_call_varargs = varargs
    test_gh_2466.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_gh_2466', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_gh_2466', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_gh_2466(...)' code ##################

    
    # Assigning a Call to a Name (line 42):
    
    # Call to array(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Obtaining an instance of the builtin type 'list' (line 42)
    list_421852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 42)
    # Adding element type (line 42)
    int_421853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 19), list_421852, int_421853)
    # Adding element type (line 42)
    int_421854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 19), list_421852, int_421854)
    
    # Processing the call keyword arguments (line 42)
    kwargs_421855 = {}
    # Getting the type of 'np' (line 42)
    np_421850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 42)
    array_421851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 10), np_421850, 'array')
    # Calling array(args, kwargs) (line 42)
    array_call_result_421856 = invoke(stypy.reporting.localization.Localization(__file__, 42, 10), array_421851, *[list_421852], **kwargs_421855)
    
    # Assigning a type to the variable 'row' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'row', array_call_result_421856)
    
    # Assigning a Call to a Name (line 43):
    
    # Call to array(...): (line 43)
    # Processing the call arguments (line 43)
    
    # Obtaining an instance of the builtin type 'list' (line 43)
    list_421859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 43)
    # Adding element type (line 43)
    int_421860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 19), list_421859, int_421860)
    # Adding element type (line 43)
    int_421861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 19), list_421859, int_421861)
    
    # Processing the call keyword arguments (line 43)
    kwargs_421862 = {}
    # Getting the type of 'np' (line 43)
    np_421857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 43)
    array_421858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 10), np_421857, 'array')
    # Calling array(args, kwargs) (line 43)
    array_call_result_421863 = invoke(stypy.reporting.localization.Localization(__file__, 43, 10), array_421858, *[list_421859], **kwargs_421862)
    
    # Assigning a type to the variable 'col' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'col', array_call_result_421863)
    
    # Assigning a Call to a Name (line 44):
    
    # Call to array(...): (line 44)
    # Processing the call arguments (line 44)
    
    # Obtaining an instance of the builtin type 'list' (line 44)
    list_421866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 44)
    # Adding element type (line 44)
    int_421867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 19), list_421866, int_421867)
    # Adding element type (line 44)
    int_421868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 19), list_421866, int_421868)
    
    # Processing the call keyword arguments (line 44)
    kwargs_421869 = {}
    # Getting the type of 'np' (line 44)
    np_421864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 44)
    array_421865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 10), np_421864, 'array')
    # Calling array(args, kwargs) (line 44)
    array_call_result_421870 = invoke(stypy.reporting.localization.Localization(__file__, 44, 10), array_421865, *[list_421866], **kwargs_421869)
    
    # Assigning a type to the variable 'val' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'val', array_call_result_421870)
    
    # Assigning a Call to a Name (line 45):
    
    # Call to coo_matrix(...): (line 45)
    # Processing the call arguments (line 45)
    
    # Obtaining an instance of the builtin type 'tuple' (line 45)
    tuple_421874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 45)
    # Adding element type (line 45)
    # Getting the type of 'val' (line 45)
    val_421875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 33), 'val', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 33), tuple_421874, val_421875)
    # Adding element type (line 45)
    
    # Obtaining an instance of the builtin type 'tuple' (line 45)
    tuple_421876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 45)
    # Adding element type (line 45)
    # Getting the type of 'row' (line 45)
    row_421877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 39), 'row', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 39), tuple_421876, row_421877)
    # Adding element type (line 45)
    # Getting the type of 'col' (line 45)
    col_421878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 44), 'col', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 39), tuple_421876, col_421878)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 33), tuple_421874, tuple_421876)
    
    # Processing the call keyword arguments (line 45)
    
    # Obtaining an instance of the builtin type 'tuple' (line 45)
    tuple_421879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 58), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 45)
    # Adding element type (line 45)
    int_421880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 58), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 58), tuple_421879, int_421880)
    # Adding element type (line 45)
    int_421881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 61), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 58), tuple_421879, int_421881)
    
    keyword_421882 = tuple_421879
    kwargs_421883 = {'shape': keyword_421882}
    # Getting the type of 'scipy' (line 45)
    scipy_421871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'scipy', False)
    # Obtaining the member 'sparse' of a type (line 45)
    sparse_421872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), scipy_421871, 'sparse')
    # Obtaining the member 'coo_matrix' of a type (line 45)
    coo_matrix_421873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), sparse_421872, 'coo_matrix')
    # Calling coo_matrix(args, kwargs) (line 45)
    coo_matrix_call_result_421884 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), coo_matrix_421873, *[tuple_421874], **kwargs_421883)
    
    # Assigning a type to the variable 'A' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'A', coo_matrix_call_result_421884)
    
    # Assigning a Call to a Name (line 46):
    
    # Call to asarray(...): (line 46)
    # Processing the call arguments (line 46)
    
    # Obtaining an instance of the builtin type 'list' (line 46)
    list_421887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 46)
    # Adding element type (line 46)
    int_421888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 19), list_421887, int_421888)
    
    # Processing the call keyword arguments (line 46)
    kwargs_421889 = {}
    # Getting the type of 'np' (line 46)
    np_421885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 46)
    asarray_421886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), np_421885, 'asarray')
    # Calling asarray(args, kwargs) (line 46)
    asarray_call_result_421890 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), asarray_421886, *[list_421887], **kwargs_421889)
    
    # Assigning a type to the variable 'b' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'b', asarray_call_result_421890)
    
    # Call to lsqr(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'A' (line 47)
    A_421892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 9), 'A', False)
    # Getting the type of 'b' (line 47)
    b_421893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'b', False)
    # Processing the call keyword arguments (line 47)
    kwargs_421894 = {}
    # Getting the type of 'lsqr' (line 47)
    lsqr_421891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'lsqr', False)
    # Calling lsqr(args, kwargs) (line 47)
    lsqr_call_result_421895 = invoke(stypy.reporting.localization.Localization(__file__, 47, 4), lsqr_421891, *[A_421892, b_421893], **kwargs_421894)
    
    
    # ################# End of 'test_gh_2466(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_gh_2466' in the type store
    # Getting the type of 'stypy_return_type' (line 41)
    stypy_return_type_421896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_421896)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_gh_2466'
    return stypy_return_type_421896

# Assigning a type to the variable 'test_gh_2466' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'test_gh_2466', test_gh_2466)

@norecursion
def test_well_conditioned_problems(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_well_conditioned_problems'
    module_type_store = module_type_store.open_function_context('test_well_conditioned_problems', 50, 0, False)
    
    # Passed parameters checking function
    test_well_conditioned_problems.stypy_localization = localization
    test_well_conditioned_problems.stypy_type_of_self = None
    test_well_conditioned_problems.stypy_type_store = module_type_store
    test_well_conditioned_problems.stypy_function_name = 'test_well_conditioned_problems'
    test_well_conditioned_problems.stypy_param_names_list = []
    test_well_conditioned_problems.stypy_varargs_param_name = None
    test_well_conditioned_problems.stypy_kwargs_param_name = None
    test_well_conditioned_problems.stypy_call_defaults = defaults
    test_well_conditioned_problems.stypy_call_varargs = varargs
    test_well_conditioned_problems.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_well_conditioned_problems', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_well_conditioned_problems', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_well_conditioned_problems(...)' code ##################

    
    # Assigning a Num to a Name (line 55):
    int_421897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 8), 'int')
    # Assigning a type to the variable 'n' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'n', int_421897)
    
    # Assigning a Call to a Name (line 56):
    
    # Call to eye(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'n' (line 56)
    n_421901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 32), 'n', False)
    # Getting the type of 'n' (line 56)
    n_421902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 35), 'n', False)
    # Processing the call keyword arguments (line 56)
    kwargs_421903 = {}
    # Getting the type of 'scipy' (line 56)
    scipy_421898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 15), 'scipy', False)
    # Obtaining the member 'sparse' of a type (line 56)
    sparse_421899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 15), scipy_421898, 'sparse')
    # Obtaining the member 'eye' of a type (line 56)
    eye_421900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 15), sparse_421899, 'eye')
    # Calling eye(args, kwargs) (line 56)
    eye_call_result_421904 = invoke(stypy.reporting.localization.Localization(__file__, 56, 15), eye_421900, *[n_421901, n_421902], **kwargs_421903)
    
    # Assigning a type to the variable 'A_sparse' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'A_sparse', eye_call_result_421904)
    
    # Assigning a Call to a Name (line 57):
    
    # Call to toarray(...): (line 57)
    # Processing the call keyword arguments (line 57)
    kwargs_421907 = {}
    # Getting the type of 'A_sparse' (line 57)
    A_sparse_421905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 14), 'A_sparse', False)
    # Obtaining the member 'toarray' of a type (line 57)
    toarray_421906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 14), A_sparse_421905, 'toarray')
    # Calling toarray(args, kwargs) (line 57)
    toarray_call_result_421908 = invoke(stypy.reporting.localization.Localization(__file__, 57, 14), toarray_421906, *[], **kwargs_421907)
    
    # Assigning a type to the variable 'A_dense' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'A_dense', toarray_call_result_421908)
    
    # Call to errstate(...): (line 59)
    # Processing the call keyword arguments (line 59)
    str_421911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 29), 'str', 'raise')
    keyword_421912 = str_421911
    kwargs_421913 = {'invalid': keyword_421912}
    # Getting the type of 'np' (line 59)
    np_421909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 9), 'np', False)
    # Obtaining the member 'errstate' of a type (line 59)
    errstate_421910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 9), np_421909, 'errstate')
    # Calling errstate(args, kwargs) (line 59)
    errstate_call_result_421914 = invoke(stypy.reporting.localization.Localization(__file__, 59, 9), errstate_421910, *[], **kwargs_421913)
    
    with_421915 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 59, 9), errstate_call_result_421914, 'with parameter', '__enter__', '__exit__')

    if with_421915:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 59)
        enter___421916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 9), errstate_call_result_421914, '__enter__')
        with_enter_421917 = invoke(stypy.reporting.localization.Localization(__file__, 59, 9), enter___421916)
        
        
        # Call to range(...): (line 60)
        # Processing the call arguments (line 60)
        int_421919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 26), 'int')
        # Processing the call keyword arguments (line 60)
        kwargs_421920 = {}
        # Getting the type of 'range' (line 60)
        range_421918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'range', False)
        # Calling range(args, kwargs) (line 60)
        range_call_result_421921 = invoke(stypy.reporting.localization.Localization(__file__, 60, 20), range_421918, *[int_421919], **kwargs_421920)
        
        # Testing the type of a for loop iterable (line 60)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 60, 8), range_call_result_421921)
        # Getting the type of the for loop variable (line 60)
        for_loop_var_421922 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 60, 8), range_call_result_421921)
        # Assigning a type to the variable 'seed' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'seed', for_loop_var_421922)
        # SSA begins for a for statement (line 60)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 61):
        
        # Call to RandomState(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'seed' (line 61)
        seed_421926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 40), 'seed', False)
        int_421927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 47), 'int')
        # Applying the binary operator '+' (line 61)
        result_add_421928 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 40), '+', seed_421926, int_421927)
        
        # Processing the call keyword arguments (line 61)
        kwargs_421929 = {}
        # Getting the type of 'np' (line 61)
        np_421923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 18), 'np', False)
        # Obtaining the member 'random' of a type (line 61)
        random_421924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 18), np_421923, 'random')
        # Obtaining the member 'RandomState' of a type (line 61)
        RandomState_421925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 18), random_421924, 'RandomState')
        # Calling RandomState(args, kwargs) (line 61)
        RandomState_call_result_421930 = invoke(stypy.reporting.localization.Localization(__file__, 61, 18), RandomState_421925, *[result_add_421928], **kwargs_421929)
        
        # Assigning a type to the variable 'rng' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'rng', RandomState_call_result_421930)
        
        # Assigning a Call to a Name (line 62):
        
        # Call to rand(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'n' (line 62)
        n_421933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 28), 'n', False)
        # Processing the call keyword arguments (line 62)
        kwargs_421934 = {}
        # Getting the type of 'rng' (line 62)
        rng_421931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 19), 'rng', False)
        # Obtaining the member 'rand' of a type (line 62)
        rand_421932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 19), rng_421931, 'rand')
        # Calling rand(args, kwargs) (line 62)
        rand_call_result_421935 = invoke(stypy.reporting.localization.Localization(__file__, 62, 19), rand_421932, *[n_421933], **kwargs_421934)
        
        # Assigning a type to the variable 'beta' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'beta', rand_call_result_421935)
        
        # Assigning a Num to a Subscript (line 63):
        float_421936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 30), 'float')
        # Getting the type of 'beta' (line 63)
        beta_421937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'beta')
        
        # Getting the type of 'beta' (line 63)
        beta_421938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 17), 'beta')
        int_421939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 25), 'int')
        # Applying the binary operator '==' (line 63)
        result_eq_421940 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 17), '==', beta_421938, int_421939)
        
        # Storing an element on a container (line 63)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 12), beta_421937, (result_eq_421940, float_421936))
        
        # Assigning a BinOp to a Name (line 64):
        # Getting the type of 'A_sparse' (line 64)
        A_sparse_421941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 16), 'A_sparse')
        
        # Obtaining the type of the subscript
        slice_421942 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 64, 27), None, None, None)
        # Getting the type of 'np' (line 64)
        np_421943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 35), 'np')
        # Obtaining the member 'newaxis' of a type (line 64)
        newaxis_421944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 35), np_421943, 'newaxis')
        # Getting the type of 'beta' (line 64)
        beta_421945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 27), 'beta')
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___421946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 27), beta_421945, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_421947 = invoke(stypy.reporting.localization.Localization(__file__, 64, 27), getitem___421946, (slice_421942, newaxis_421944))
        
        # Applying the binary operator '*' (line 64)
        result_mul_421948 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 16), '*', A_sparse_421941, subscript_call_result_421947)
        
        # Assigning a type to the variable 'b' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'b', result_mul_421948)
        
        # Assigning a Call to a Name (line 65):
        
        # Call to lsqr(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'A_sparse' (line 65)
        A_sparse_421950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 26), 'A_sparse', False)
        # Getting the type of 'b' (line 65)
        b_421951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 36), 'b', False)
        # Processing the call keyword arguments (line 65)
        # Getting the type of 'show' (line 65)
        show_421952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 44), 'show', False)
        keyword_421953 = show_421952
        kwargs_421954 = {'show': keyword_421953}
        # Getting the type of 'lsqr' (line 65)
        lsqr_421949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 21), 'lsqr', False)
        # Calling lsqr(args, kwargs) (line 65)
        lsqr_call_result_421955 = invoke(stypy.reporting.localization.Localization(__file__, 65, 21), lsqr_421949, *[A_sparse_421950, b_421951], **kwargs_421954)
        
        # Assigning a type to the variable 'output' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'output', lsqr_call_result_421955)
        
        # Call to assert_equal(...): (line 69)
        # Processing the call arguments (line 69)
        
        # Obtaining the type of the subscript
        int_421957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 32), 'int')
        # Getting the type of 'output' (line 69)
        output_421958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 25), 'output', False)
        # Obtaining the member '__getitem__' of a type (line 69)
        getitem___421959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 25), output_421958, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 69)
        subscript_call_result_421960 = invoke(stypy.reporting.localization.Localization(__file__, 69, 25), getitem___421959, int_421957)
        
        int_421961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 36), 'int')
        # Processing the call keyword arguments (line 69)
        kwargs_421962 = {}
        # Getting the type of 'assert_equal' (line 69)
        assert_equal_421956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 69)
        assert_equal_call_result_421963 = invoke(stypy.reporting.localization.Localization(__file__, 69, 12), assert_equal_421956, *[subscript_call_result_421960, int_421961], **kwargs_421962)
        
        
        # Assigning a Subscript to a Name (line 70):
        
        # Obtaining the type of the subscript
        int_421964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 30), 'int')
        # Getting the type of 'output' (line 70)
        output_421965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 23), 'output')
        # Obtaining the member '__getitem__' of a type (line 70)
        getitem___421966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 23), output_421965, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 70)
        subscript_call_result_421967 = invoke(stypy.reporting.localization.Localization(__file__, 70, 23), getitem___421966, int_421964)
        
        # Assigning a type to the variable 'solution' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'solution', subscript_call_result_421967)
        
        # Call to assert_array_almost_equal(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'solution' (line 73)
        solution_421969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 38), 'solution', False)
        # Getting the type of 'beta' (line 73)
        beta_421970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 48), 'beta', False)
        # Processing the call keyword arguments (line 73)
        kwargs_421971 = {}
        # Getting the type of 'assert_array_almost_equal' (line 73)
        assert_array_almost_equal_421968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 73)
        assert_array_almost_equal_call_result_421972 = invoke(stypy.reporting.localization.Localization(__file__, 73, 12), assert_array_almost_equal_421968, *[solution_421969, beta_421970], **kwargs_421971)
        
        
        # Assigning a Call to a Name (line 76):
        
        # Call to ravel(...): (line 76)
        # Processing the call keyword arguments (line 76)
        kwargs_421981 = {}
        
        # Call to solve(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'A_dense' (line 76)
        A_dense_421976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 49), 'A_dense', False)
        # Getting the type of 'b' (line 76)
        b_421977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 58), 'b', False)
        # Processing the call keyword arguments (line 76)
        kwargs_421978 = {}
        # Getting the type of 'np' (line 76)
        np_421973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 33), 'np', False)
        # Obtaining the member 'linalg' of a type (line 76)
        linalg_421974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 33), np_421973, 'linalg')
        # Obtaining the member 'solve' of a type (line 76)
        solve_421975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 33), linalg_421974, 'solve')
        # Calling solve(args, kwargs) (line 76)
        solve_call_result_421979 = invoke(stypy.reporting.localization.Localization(__file__, 76, 33), solve_421975, *[A_dense_421976, b_421977], **kwargs_421978)
        
        # Obtaining the member 'ravel' of a type (line 76)
        ravel_421980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 33), solve_call_result_421979, 'ravel')
        # Calling ravel(args, kwargs) (line 76)
        ravel_call_result_421982 = invoke(stypy.reporting.localization.Localization(__file__, 76, 33), ravel_421980, *[], **kwargs_421981)
        
        # Assigning a type to the variable 'reference_solution' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'reference_solution', ravel_call_result_421982)
        
        # Call to assert_array_almost_equal(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'solution' (line 77)
        solution_421984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 38), 'solution', False)
        # Getting the type of 'reference_solution' (line 77)
        reference_solution_421985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 48), 'reference_solution', False)
        # Processing the call keyword arguments (line 77)
        kwargs_421986 = {}
        # Getting the type of 'assert_array_almost_equal' (line 77)
        assert_array_almost_equal_421983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 77)
        assert_array_almost_equal_call_result_421987 = invoke(stypy.reporting.localization.Localization(__file__, 77, 12), assert_array_almost_equal_421983, *[solution_421984, reference_solution_421985], **kwargs_421986)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 59)
        exit___421988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 9), errstate_call_result_421914, '__exit__')
        with_exit_421989 = invoke(stypy.reporting.localization.Localization(__file__, 59, 9), exit___421988, None, None, None)

    
    # ################# End of 'test_well_conditioned_problems(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_well_conditioned_problems' in the type store
    # Getting the type of 'stypy_return_type' (line 50)
    stypy_return_type_421990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_421990)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_well_conditioned_problems'
    return stypy_return_type_421990

# Assigning a type to the variable 'test_well_conditioned_problems' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'test_well_conditioned_problems', test_well_conditioned_problems)

@norecursion
def test_b_shapes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_b_shapes'
    module_type_store = module_type_store.open_function_context('test_b_shapes', 80, 0, False)
    
    # Passed parameters checking function
    test_b_shapes.stypy_localization = localization
    test_b_shapes.stypy_type_of_self = None
    test_b_shapes.stypy_type_store = module_type_store
    test_b_shapes.stypy_function_name = 'test_b_shapes'
    test_b_shapes.stypy_param_names_list = []
    test_b_shapes.stypy_varargs_param_name = None
    test_b_shapes.stypy_kwargs_param_name = None
    test_b_shapes.stypy_call_defaults = defaults
    test_b_shapes.stypy_call_varargs = varargs
    test_b_shapes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_b_shapes', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_b_shapes', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_b_shapes(...)' code ##################

    
    # Assigning a Call to a Name (line 82):
    
    # Call to array(...): (line 82)
    # Processing the call arguments (line 82)
    
    # Obtaining an instance of the builtin type 'list' (line 82)
    list_421993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 82)
    # Adding element type (line 82)
    
    # Obtaining an instance of the builtin type 'list' (line 82)
    list_421994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 82)
    # Adding element type (line 82)
    float_421995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 19), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 18), list_421994, float_421995)
    # Adding element type (line 82)
    float_421996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 24), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 18), list_421994, float_421996)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 17), list_421993, list_421994)
    
    # Processing the call keyword arguments (line 82)
    kwargs_421997 = {}
    # Getting the type of 'np' (line 82)
    np_421991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 82)
    array_421992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), np_421991, 'array')
    # Calling array(args, kwargs) (line 82)
    array_call_result_421998 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), array_421992, *[list_421993], **kwargs_421997)
    
    # Assigning a type to the variable 'A' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'A', array_call_result_421998)
    
    # Assigning a Num to a Name (line 83):
    float_421999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 8), 'float')
    # Assigning a type to the variable 'b' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'b', float_421999)
    
    # Assigning a Subscript to a Name (line 84):
    
    # Obtaining the type of the subscript
    int_422000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 19), 'int')
    
    # Call to lsqr(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'A' (line 84)
    A_422002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 13), 'A', False)
    # Getting the type of 'b' (line 84)
    b_422003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'b', False)
    # Processing the call keyword arguments (line 84)
    kwargs_422004 = {}
    # Getting the type of 'lsqr' (line 84)
    lsqr_422001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'lsqr', False)
    # Calling lsqr(args, kwargs) (line 84)
    lsqr_call_result_422005 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), lsqr_422001, *[A_422002, b_422003], **kwargs_422004)
    
    # Obtaining the member '__getitem__' of a type (line 84)
    getitem___422006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), lsqr_call_result_422005, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 84)
    subscript_call_result_422007 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), getitem___422006, int_422000)
    
    # Assigning a type to the variable 'x' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'x', subscript_call_result_422007)
    
    # Call to assert_almost_equal(...): (line 85)
    # Processing the call arguments (line 85)
    
    # Call to norm(...): (line 85)
    # Processing the call arguments (line 85)
    
    # Call to dot(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'x' (line 85)
    x_422012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 35), 'x', False)
    # Processing the call keyword arguments (line 85)
    kwargs_422013 = {}
    # Getting the type of 'A' (line 85)
    A_422010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 29), 'A', False)
    # Obtaining the member 'dot' of a type (line 85)
    dot_422011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 29), A_422010, 'dot')
    # Calling dot(args, kwargs) (line 85)
    dot_call_result_422014 = invoke(stypy.reporting.localization.Localization(__file__, 85, 29), dot_422011, *[x_422012], **kwargs_422013)
    
    # Getting the type of 'b' (line 85)
    b_422015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 40), 'b', False)
    # Applying the binary operator '-' (line 85)
    result_sub_422016 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 29), '-', dot_call_result_422014, b_422015)
    
    # Processing the call keyword arguments (line 85)
    kwargs_422017 = {}
    # Getting the type of 'norm' (line 85)
    norm_422009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 24), 'norm', False)
    # Calling norm(args, kwargs) (line 85)
    norm_call_result_422018 = invoke(stypy.reporting.localization.Localization(__file__, 85, 24), norm_422009, *[result_sub_422016], **kwargs_422017)
    
    int_422019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 44), 'int')
    # Processing the call keyword arguments (line 85)
    kwargs_422020 = {}
    # Getting the type of 'assert_almost_equal' (line 85)
    assert_almost_equal_422008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 85)
    assert_almost_equal_call_result_422021 = invoke(stypy.reporting.localization.Localization(__file__, 85, 4), assert_almost_equal_422008, *[norm_call_result_422018, int_422019], **kwargs_422020)
    
    
    # Assigning a Call to a Name (line 88):
    
    # Call to eye(...): (line 88)
    # Processing the call arguments (line 88)
    int_422024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 15), 'int')
    # Processing the call keyword arguments (line 88)
    kwargs_422025 = {}
    # Getting the type of 'np' (line 88)
    np_422022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'np', False)
    # Obtaining the member 'eye' of a type (line 88)
    eye_422023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), np_422022, 'eye')
    # Calling eye(args, kwargs) (line 88)
    eye_call_result_422026 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), eye_422023, *[int_422024], **kwargs_422025)
    
    # Assigning a type to the variable 'A' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'A', eye_call_result_422026)
    
    # Assigning a Call to a Name (line 89):
    
    # Call to ones(...): (line 89)
    # Processing the call arguments (line 89)
    
    # Obtaining an instance of the builtin type 'tuple' (line 89)
    tuple_422029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 89)
    # Adding element type (line 89)
    int_422030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 17), tuple_422029, int_422030)
    # Adding element type (line 89)
    int_422031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 17), tuple_422029, int_422031)
    
    # Processing the call keyword arguments (line 89)
    kwargs_422032 = {}
    # Getting the type of 'np' (line 89)
    np_422027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'np', False)
    # Obtaining the member 'ones' of a type (line 89)
    ones_422028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), np_422027, 'ones')
    # Calling ones(args, kwargs) (line 89)
    ones_call_result_422033 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), ones_422028, *[tuple_422029], **kwargs_422032)
    
    # Assigning a type to the variable 'b' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'b', ones_call_result_422033)
    
    # Assigning a Subscript to a Name (line 90):
    
    # Obtaining the type of the subscript
    int_422034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 19), 'int')
    
    # Call to lsqr(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'A' (line 90)
    A_422036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 13), 'A', False)
    # Getting the type of 'b' (line 90)
    b_422037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 16), 'b', False)
    # Processing the call keyword arguments (line 90)
    kwargs_422038 = {}
    # Getting the type of 'lsqr' (line 90)
    lsqr_422035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'lsqr', False)
    # Calling lsqr(args, kwargs) (line 90)
    lsqr_call_result_422039 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), lsqr_422035, *[A_422036, b_422037], **kwargs_422038)
    
    # Obtaining the member '__getitem__' of a type (line 90)
    getitem___422040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), lsqr_call_result_422039, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 90)
    subscript_call_result_422041 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), getitem___422040, int_422034)
    
    # Assigning a type to the variable 'x' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'x', subscript_call_result_422041)
    
    # Call to assert_almost_equal(...): (line 91)
    # Processing the call arguments (line 91)
    
    # Call to norm(...): (line 91)
    # Processing the call arguments (line 91)
    
    # Call to dot(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'x' (line 91)
    x_422046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 35), 'x', False)
    # Processing the call keyword arguments (line 91)
    kwargs_422047 = {}
    # Getting the type of 'A' (line 91)
    A_422044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 29), 'A', False)
    # Obtaining the member 'dot' of a type (line 91)
    dot_422045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 29), A_422044, 'dot')
    # Calling dot(args, kwargs) (line 91)
    dot_call_result_422048 = invoke(stypy.reporting.localization.Localization(__file__, 91, 29), dot_422045, *[x_422046], **kwargs_422047)
    
    
    # Call to ravel(...): (line 91)
    # Processing the call keyword arguments (line 91)
    kwargs_422051 = {}
    # Getting the type of 'b' (line 91)
    b_422049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 40), 'b', False)
    # Obtaining the member 'ravel' of a type (line 91)
    ravel_422050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 40), b_422049, 'ravel')
    # Calling ravel(args, kwargs) (line 91)
    ravel_call_result_422052 = invoke(stypy.reporting.localization.Localization(__file__, 91, 40), ravel_422050, *[], **kwargs_422051)
    
    # Applying the binary operator '-' (line 91)
    result_sub_422053 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 29), '-', dot_call_result_422048, ravel_call_result_422052)
    
    # Processing the call keyword arguments (line 91)
    kwargs_422054 = {}
    # Getting the type of 'norm' (line 91)
    norm_422043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 24), 'norm', False)
    # Calling norm(args, kwargs) (line 91)
    norm_call_result_422055 = invoke(stypy.reporting.localization.Localization(__file__, 91, 24), norm_422043, *[result_sub_422053], **kwargs_422054)
    
    int_422056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 52), 'int')
    # Processing the call keyword arguments (line 91)
    kwargs_422057 = {}
    # Getting the type of 'assert_almost_equal' (line 91)
    assert_almost_equal_422042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 91)
    assert_almost_equal_call_result_422058 = invoke(stypy.reporting.localization.Localization(__file__, 91, 4), assert_almost_equal_422042, *[norm_call_result_422055, int_422056], **kwargs_422057)
    
    
    # ################# End of 'test_b_shapes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_b_shapes' in the type store
    # Getting the type of 'stypy_return_type' (line 80)
    stypy_return_type_422059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_422059)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_b_shapes'
    return stypy_return_type_422059

# Assigning a type to the variable 'test_b_shapes' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'test_b_shapes', test_b_shapes)

@norecursion
def test_initialization(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_initialization'
    module_type_store = module_type_store.open_function_context('test_initialization', 94, 0, False)
    
    # Passed parameters checking function
    test_initialization.stypy_localization = localization
    test_initialization.stypy_type_of_self = None
    test_initialization.stypy_type_store = module_type_store
    test_initialization.stypy_function_name = 'test_initialization'
    test_initialization.stypy_param_names_list = []
    test_initialization.stypy_varargs_param_name = None
    test_initialization.stypy_kwargs_param_name = None
    test_initialization.stypy_call_defaults = defaults
    test_initialization.stypy_call_varargs = varargs
    test_initialization.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_initialization', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_initialization', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_initialization(...)' code ##################

    
    # Assigning a Call to a Name (line 96):
    
    # Call to copy(...): (line 96)
    # Processing the call keyword arguments (line 96)
    kwargs_422062 = {}
    # Getting the type of 'b' (line 96)
    b_422060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 13), 'b', False)
    # Obtaining the member 'copy' of a type (line 96)
    copy_422061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 13), b_422060, 'copy')
    # Calling copy(args, kwargs) (line 96)
    copy_call_result_422063 = invoke(stypy.reporting.localization.Localization(__file__, 96, 13), copy_422061, *[], **kwargs_422062)
    
    # Assigning a type to the variable 'b_copy' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'b_copy', copy_call_result_422063)
    
    # Assigning a Call to a Name (line 97):
    
    # Call to lsqr(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'G' (line 97)
    G_422065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 17), 'G', False)
    # Getting the type of 'b' (line 97)
    b_422066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 20), 'b', False)
    # Processing the call keyword arguments (line 97)
    # Getting the type of 'show' (line 97)
    show_422067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 28), 'show', False)
    keyword_422068 = show_422067
    # Getting the type of 'tol' (line 97)
    tol_422069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 39), 'tol', False)
    keyword_422070 = tol_422069
    # Getting the type of 'tol' (line 97)
    tol_422071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 49), 'tol', False)
    keyword_422072 = tol_422071
    # Getting the type of 'maxit' (line 97)
    maxit_422073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 63), 'maxit', False)
    keyword_422074 = maxit_422073
    kwargs_422075 = {'iter_lim': keyword_422074, 'btol': keyword_422072, 'atol': keyword_422070, 'show': keyword_422068}
    # Getting the type of 'lsqr' (line 97)
    lsqr_422064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'lsqr', False)
    # Calling lsqr(args, kwargs) (line 97)
    lsqr_call_result_422076 = invoke(stypy.reporting.localization.Localization(__file__, 97, 12), lsqr_422064, *[G_422065, b_422066], **kwargs_422075)
    
    # Assigning a type to the variable 'x_ref' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'x_ref', lsqr_call_result_422076)
    
    # Assigning a Call to a Name (line 98):
    
    # Call to zeros(...): (line 98)
    # Processing the call arguments (line 98)
    
    # Obtaining the type of the subscript
    int_422079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 24), 'int')
    # Getting the type of 'x_ref' (line 98)
    x_ref_422080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 18), 'x_ref', False)
    # Obtaining the member '__getitem__' of a type (line 98)
    getitem___422081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 18), x_ref_422080, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 98)
    subscript_call_result_422082 = invoke(stypy.reporting.localization.Localization(__file__, 98, 18), getitem___422081, int_422079)
    
    # Obtaining the member 'shape' of a type (line 98)
    shape_422083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 18), subscript_call_result_422082, 'shape')
    # Processing the call keyword arguments (line 98)
    kwargs_422084 = {}
    # Getting the type of 'np' (line 98)
    np_422077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 9), 'np', False)
    # Obtaining the member 'zeros' of a type (line 98)
    zeros_422078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 9), np_422077, 'zeros')
    # Calling zeros(args, kwargs) (line 98)
    zeros_call_result_422085 = invoke(stypy.reporting.localization.Localization(__file__, 98, 9), zeros_422078, *[shape_422083], **kwargs_422084)
    
    # Assigning a type to the variable 'x0' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'x0', zeros_call_result_422085)
    
    # Assigning a Call to a Name (line 99):
    
    # Call to lsqr(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'G' (line 99)
    G_422087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 13), 'G', False)
    # Getting the type of 'b' (line 99)
    b_422088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'b', False)
    # Processing the call keyword arguments (line 99)
    # Getting the type of 'show' (line 99)
    show_422089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 24), 'show', False)
    keyword_422090 = show_422089
    # Getting the type of 'tol' (line 99)
    tol_422091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 35), 'tol', False)
    keyword_422092 = tol_422091
    # Getting the type of 'tol' (line 99)
    tol_422093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 45), 'tol', False)
    keyword_422094 = tol_422093
    # Getting the type of 'maxit' (line 99)
    maxit_422095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 59), 'maxit', False)
    keyword_422096 = maxit_422095
    # Getting the type of 'x0' (line 99)
    x0_422097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 69), 'x0', False)
    keyword_422098 = x0_422097
    kwargs_422099 = {'iter_lim': keyword_422096, 'x0': keyword_422098, 'btol': keyword_422094, 'atol': keyword_422092, 'show': keyword_422090}
    # Getting the type of 'lsqr' (line 99)
    lsqr_422086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'lsqr', False)
    # Calling lsqr(args, kwargs) (line 99)
    lsqr_call_result_422100 = invoke(stypy.reporting.localization.Localization(__file__, 99, 8), lsqr_422086, *[G_422087, b_422088], **kwargs_422099)
    
    # Assigning a type to the variable 'x' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'x', lsqr_call_result_422100)
    
    # Call to assert_(...): (line 100)
    # Processing the call arguments (line 100)
    
    # Call to all(...): (line 100)
    # Processing the call arguments (line 100)
    
    # Getting the type of 'b_copy' (line 100)
    b_copy_422104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 19), 'b_copy', False)
    # Getting the type of 'b' (line 100)
    b_422105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 29), 'b', False)
    # Applying the binary operator '==' (line 100)
    result_eq_422106 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 19), '==', b_copy_422104, b_422105)
    
    # Processing the call keyword arguments (line 100)
    kwargs_422107 = {}
    # Getting the type of 'np' (line 100)
    np_422102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'np', False)
    # Obtaining the member 'all' of a type (line 100)
    all_422103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), np_422102, 'all')
    # Calling all(args, kwargs) (line 100)
    all_call_result_422108 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), all_422103, *[result_eq_422106], **kwargs_422107)
    
    # Processing the call keyword arguments (line 100)
    kwargs_422109 = {}
    # Getting the type of 'assert_' (line 100)
    assert__422101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 100)
    assert__call_result_422110 = invoke(stypy.reporting.localization.Localization(__file__, 100, 4), assert__422101, *[all_call_result_422108], **kwargs_422109)
    
    
    # Call to assert_array_almost_equal(...): (line 101)
    # Processing the call arguments (line 101)
    
    # Obtaining the type of the subscript
    int_422112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 36), 'int')
    # Getting the type of 'x_ref' (line 101)
    x_ref_422113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'x_ref', False)
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___422114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 30), x_ref_422113, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_422115 = invoke(stypy.reporting.localization.Localization(__file__, 101, 30), getitem___422114, int_422112)
    
    
    # Obtaining the type of the subscript
    int_422116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 42), 'int')
    # Getting the type of 'x' (line 101)
    x_422117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 40), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___422118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 40), x_422117, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_422119 = invoke(stypy.reporting.localization.Localization(__file__, 101, 40), getitem___422118, int_422116)
    
    # Processing the call keyword arguments (line 101)
    kwargs_422120 = {}
    # Getting the type of 'assert_array_almost_equal' (line 101)
    assert_array_almost_equal_422111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 101)
    assert_array_almost_equal_call_result_422121 = invoke(stypy.reporting.localization.Localization(__file__, 101, 4), assert_array_almost_equal_422111, *[subscript_call_result_422115, subscript_call_result_422119], **kwargs_422120)
    
    
    # Assigning a Subscript to a Name (line 104):
    
    # Obtaining the type of the subscript
    int_422122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 63), 'int')
    
    # Call to lsqr(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'G' (line 104)
    G_422124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 14), 'G', False)
    # Getting the type of 'b' (line 104)
    b_422125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 17), 'b', False)
    # Processing the call keyword arguments (line 104)
    # Getting the type of 'show' (line 104)
    show_422126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 25), 'show', False)
    keyword_422127 = show_422126
    # Getting the type of 'tol' (line 104)
    tol_422128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 36), 'tol', False)
    keyword_422129 = tol_422128
    # Getting the type of 'tol' (line 104)
    tol_422130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 46), 'tol', False)
    keyword_422131 = tol_422130
    int_422132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 60), 'int')
    keyword_422133 = int_422132
    kwargs_422134 = {'iter_lim': keyword_422133, 'btol': keyword_422131, 'atol': keyword_422129, 'show': keyword_422127}
    # Getting the type of 'lsqr' (line 104)
    lsqr_422123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 9), 'lsqr', False)
    # Calling lsqr(args, kwargs) (line 104)
    lsqr_call_result_422135 = invoke(stypy.reporting.localization.Localization(__file__, 104, 9), lsqr_422123, *[G_422124, b_422125], **kwargs_422134)
    
    # Obtaining the member '__getitem__' of a type (line 104)
    getitem___422136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 9), lsqr_call_result_422135, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 104)
    subscript_call_result_422137 = invoke(stypy.reporting.localization.Localization(__file__, 104, 9), getitem___422136, int_422122)
    
    # Assigning a type to the variable 'x0' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'x0', subscript_call_result_422137)
    
    # Assigning a Call to a Name (line 105):
    
    # Call to lsqr(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'G' (line 105)
    G_422139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 13), 'G', False)
    # Getting the type of 'b' (line 105)
    b_422140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'b', False)
    # Processing the call keyword arguments (line 105)
    # Getting the type of 'show' (line 105)
    show_422141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 24), 'show', False)
    keyword_422142 = show_422141
    # Getting the type of 'tol' (line 105)
    tol_422143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 35), 'tol', False)
    keyword_422144 = tol_422143
    # Getting the type of 'tol' (line 105)
    tol_422145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 45), 'tol', False)
    keyword_422146 = tol_422145
    # Getting the type of 'maxit' (line 105)
    maxit_422147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 59), 'maxit', False)
    keyword_422148 = maxit_422147
    # Getting the type of 'x0' (line 105)
    x0_422149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 69), 'x0', False)
    keyword_422150 = x0_422149
    kwargs_422151 = {'iter_lim': keyword_422148, 'x0': keyword_422150, 'btol': keyword_422146, 'atol': keyword_422144, 'show': keyword_422142}
    # Getting the type of 'lsqr' (line 105)
    lsqr_422138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'lsqr', False)
    # Calling lsqr(args, kwargs) (line 105)
    lsqr_call_result_422152 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), lsqr_422138, *[G_422139, b_422140], **kwargs_422151)
    
    # Assigning a type to the variable 'x' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'x', lsqr_call_result_422152)
    
    # Call to assert_array_almost_equal(...): (line 106)
    # Processing the call arguments (line 106)
    
    # Obtaining the type of the subscript
    int_422154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 36), 'int')
    # Getting the type of 'x_ref' (line 106)
    x_ref_422155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 30), 'x_ref', False)
    # Obtaining the member '__getitem__' of a type (line 106)
    getitem___422156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 30), x_ref_422155, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
    subscript_call_result_422157 = invoke(stypy.reporting.localization.Localization(__file__, 106, 30), getitem___422156, int_422154)
    
    
    # Obtaining the type of the subscript
    int_422158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 42), 'int')
    # Getting the type of 'x' (line 106)
    x_422159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 40), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 106)
    getitem___422160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 40), x_422159, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
    subscript_call_result_422161 = invoke(stypy.reporting.localization.Localization(__file__, 106, 40), getitem___422160, int_422158)
    
    # Processing the call keyword arguments (line 106)
    kwargs_422162 = {}
    # Getting the type of 'assert_array_almost_equal' (line 106)
    assert_array_almost_equal_422153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 106)
    assert_array_almost_equal_call_result_422163 = invoke(stypy.reporting.localization.Localization(__file__, 106, 4), assert_array_almost_equal_422153, *[subscript_call_result_422157, subscript_call_result_422161], **kwargs_422162)
    
    
    # Call to assert_(...): (line 107)
    # Processing the call arguments (line 107)
    
    # Call to all(...): (line 107)
    # Processing the call arguments (line 107)
    
    # Getting the type of 'b_copy' (line 107)
    b_copy_422167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'b_copy', False)
    # Getting the type of 'b' (line 107)
    b_422168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 29), 'b', False)
    # Applying the binary operator '==' (line 107)
    result_eq_422169 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 19), '==', b_copy_422167, b_422168)
    
    # Processing the call keyword arguments (line 107)
    kwargs_422170 = {}
    # Getting the type of 'np' (line 107)
    np_422165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'np', False)
    # Obtaining the member 'all' of a type (line 107)
    all_422166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 12), np_422165, 'all')
    # Calling all(args, kwargs) (line 107)
    all_call_result_422171 = invoke(stypy.reporting.localization.Localization(__file__, 107, 12), all_422166, *[result_eq_422169], **kwargs_422170)
    
    # Processing the call keyword arguments (line 107)
    kwargs_422172 = {}
    # Getting the type of 'assert_' (line 107)
    assert__422164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 107)
    assert__call_result_422173 = invoke(stypy.reporting.localization.Localization(__file__, 107, 4), assert__422164, *[all_call_result_422171], **kwargs_422172)
    
    
    # ################# End of 'test_initialization(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_initialization' in the type store
    # Getting the type of 'stypy_return_type' (line 94)
    stypy_return_type_422174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_422174)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_initialization'
    return stypy_return_type_422174

# Assigning a type to the variable 'test_initialization' (line 94)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'test_initialization', test_initialization)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 111):
    
    # Call to solve(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'G' (line 111)
    G_422178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 26), 'G', False)
    # Getting the type of 'b' (line 111)
    b_422179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 29), 'b', False)
    # Processing the call keyword arguments (line 111)
    kwargs_422180 = {}
    # Getting the type of 'np' (line 111)
    np_422175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 10), 'np', False)
    # Obtaining the member 'linalg' of a type (line 111)
    linalg_422176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 10), np_422175, 'linalg')
    # Obtaining the member 'solve' of a type (line 111)
    solve_422177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 10), linalg_422176, 'solve')
    # Calling solve(args, kwargs) (line 111)
    solve_call_result_422181 = invoke(stypy.reporting.localization.Localization(__file__, 111, 10), solve_422177, *[G_422178, b_422179], **kwargs_422180)
    
    # Assigning a type to the variable 'svx' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'svx', solve_call_result_422181)
    
    # Assigning a Call to a Name (line 113):
    
    # Call to time(...): (line 113)
    # Processing the call keyword arguments (line 113)
    kwargs_422183 = {}
    # Getting the type of 'time' (line 113)
    time_422182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 10), 'time', False)
    # Calling time(args, kwargs) (line 113)
    time_call_result_422184 = invoke(stypy.reporting.localization.Localization(__file__, 113, 10), time_422182, *[], **kwargs_422183)
    
    # Assigning a type to the variable 'tic' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'tic', time_call_result_422184)
    
    # Assigning a Call to a Name (line 114):
    
    # Call to lsqr(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'G' (line 114)
    G_422186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 13), 'G', False)
    # Getting the type of 'b' (line 114)
    b_422187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'b', False)
    # Processing the call keyword arguments (line 114)
    # Getting the type of 'show' (line 114)
    show_422188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 24), 'show', False)
    keyword_422189 = show_422188
    # Getting the type of 'tol' (line 114)
    tol_422190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 35), 'tol', False)
    keyword_422191 = tol_422190
    # Getting the type of 'tol' (line 114)
    tol_422192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 45), 'tol', False)
    keyword_422193 = tol_422192
    # Getting the type of 'maxit' (line 114)
    maxit_422194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 59), 'maxit', False)
    keyword_422195 = maxit_422194
    kwargs_422196 = {'iter_lim': keyword_422195, 'btol': keyword_422193, 'atol': keyword_422191, 'show': keyword_422189}
    # Getting the type of 'lsqr' (line 114)
    lsqr_422185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'lsqr', False)
    # Calling lsqr(args, kwargs) (line 114)
    lsqr_call_result_422197 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), lsqr_422185, *[G_422186, b_422187], **kwargs_422196)
    
    # Assigning a type to the variable 'X' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'X', lsqr_call_result_422197)
    
    # Assigning a Subscript to a Name (line 115):
    
    # Obtaining the type of the subscript
    int_422198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 11), 'int')
    # Getting the type of 'X' (line 115)
    X_422199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 9), 'X')
    # Obtaining the member '__getitem__' of a type (line 115)
    getitem___422200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 9), X_422199, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 115)
    subscript_call_result_422201 = invoke(stypy.reporting.localization.Localization(__file__, 115, 9), getitem___422200, int_422198)
    
    # Assigning a type to the variable 'xo' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'xo', subscript_call_result_422201)
    
    # Assigning a Subscript to a Name (line 116):
    
    # Obtaining the type of the subscript
    int_422202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 13), 'int')
    # Getting the type of 'X' (line 116)
    X_422203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 11), 'X')
    # Obtaining the member '__getitem__' of a type (line 116)
    getitem___422204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 11), X_422203, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 116)
    subscript_call_result_422205 = invoke(stypy.reporting.localization.Localization(__file__, 116, 11), getitem___422204, int_422202)
    
    # Assigning a type to the variable 'phio' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'phio', subscript_call_result_422205)
    
    # Assigning a Subscript to a Name (line 117):
    
    # Obtaining the type of the subscript
    int_422206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 13), 'int')
    # Getting the type of 'X' (line 117)
    X_422207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 11), 'X')
    # Obtaining the member '__getitem__' of a type (line 117)
    getitem___422208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 11), X_422207, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 117)
    subscript_call_result_422209 = invoke(stypy.reporting.localization.Localization(__file__, 117, 11), getitem___422208, int_422206)
    
    # Assigning a type to the variable 'psio' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'psio', subscript_call_result_422209)
    
    # Assigning a Subscript to a Name (line 118):
    
    # Obtaining the type of the subscript
    int_422210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 10), 'int')
    # Getting the type of 'X' (line 118)
    X_422211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'X')
    # Obtaining the member '__getitem__' of a type (line 118)
    getitem___422212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), X_422211, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 118)
    subscript_call_result_422213 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), getitem___422212, int_422210)
    
    # Assigning a type to the variable 'k' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'k', subscript_call_result_422213)
    
    # Assigning a Subscript to a Name (line 119):
    
    # Obtaining the type of the subscript
    int_422214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 13), 'int')
    # Getting the type of 'X' (line 119)
    X_422215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 11), 'X')
    # Obtaining the member '__getitem__' of a type (line 119)
    getitem___422216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 11), X_422215, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 119)
    subscript_call_result_422217 = invoke(stypy.reporting.localization.Localization(__file__, 119, 11), getitem___422216, int_422214)
    
    # Assigning a type to the variable 'chio' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'chio', subscript_call_result_422217)
    
    # Assigning a Call to a Name (line 120):
    
    # Call to amax(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'G' (line 120)
    G_422220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 17), 'G', False)
    # Getting the type of 'G' (line 120)
    G_422221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 21), 'G', False)
    # Obtaining the member 'T' of a type (line 120)
    T_422222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 21), G_422221, 'T')
    # Applying the binary operator '-' (line 120)
    result_sub_422223 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 17), '-', G_422220, T_422222)
    
    # Processing the call keyword arguments (line 120)
    kwargs_422224 = {}
    # Getting the type of 'np' (line 120)
    np_422218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 9), 'np', False)
    # Obtaining the member 'amax' of a type (line 120)
    amax_422219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 9), np_422218, 'amax')
    # Calling amax(args, kwargs) (line 120)
    amax_call_result_422225 = invoke(stypy.reporting.localization.Localization(__file__, 120, 9), amax_422219, *[result_sub_422223], **kwargs_422224)
    
    # Assigning a type to the variable 'mg' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'mg', amax_call_result_422225)
    
    
    # Getting the type of 'mg' (line 121)
    mg_422226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 7), 'mg')
    float_422227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 12), 'float')
    # Applying the binary operator '>' (line 121)
    result_gt_422228 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 7), '>', mg_422226, float_422227)
    
    # Testing the type of an if condition (line 121)
    if_condition_422229 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 4), result_gt_422228)
    # Assigning a type to the variable 'if_condition_422229' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'if_condition_422229', if_condition_422229)
    # SSA begins for if statement (line 121)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 122):
    str_422230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 14), 'str', 'No')
    # Assigning a type to the variable 'sym' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'sym', str_422230)
    # SSA branch for the else part of an if statement (line 121)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 124):
    str_422231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 14), 'str', 'Yes')
    # Assigning a type to the variable 'sym' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'sym', str_422231)
    # SSA join for if statement (line 121)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to print(...): (line 126)
    # Processing the call arguments (line 126)
    str_422233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 10), 'str', 'LSQR')
    # Processing the call keyword arguments (line 126)
    kwargs_422234 = {}
    # Getting the type of 'print' (line 126)
    print_422232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'print', False)
    # Calling print(args, kwargs) (line 126)
    print_call_result_422235 = invoke(stypy.reporting.localization.Localization(__file__, 126, 4), print_422232, *[str_422233], **kwargs_422234)
    
    
    # Call to print(...): (line 127)
    # Processing the call arguments (line 127)
    str_422237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 10), 'str', 'Is linear operator symmetric? ')
    # Getting the type of 'sym' (line 127)
    sym_422238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 45), 'sym', False)
    # Applying the binary operator '+' (line 127)
    result_add_422239 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 10), '+', str_422237, sym_422238)
    
    # Processing the call keyword arguments (line 127)
    kwargs_422240 = {}
    # Getting the type of 'print' (line 127)
    print_422236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'print', False)
    # Calling print(args, kwargs) (line 127)
    print_call_result_422241 = invoke(stypy.reporting.localization.Localization(__file__, 127, 4), print_422236, *[result_add_422239], **kwargs_422240)
    
    
    # Call to print(...): (line 128)
    # Processing the call arguments (line 128)
    str_422243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 10), 'str', 'n: %3g  iterations:   %3g')
    
    # Obtaining an instance of the builtin type 'tuple' (line 128)
    tuple_422244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 128)
    # Adding element type (line 128)
    # Getting the type of 'n' (line 128)
    n_422245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 41), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 41), tuple_422244, n_422245)
    # Adding element type (line 128)
    # Getting the type of 'k' (line 128)
    k_422246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 44), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 41), tuple_422244, k_422246)
    
    # Applying the binary operator '%' (line 128)
    result_mod_422247 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 10), '%', str_422243, tuple_422244)
    
    # Processing the call keyword arguments (line 128)
    kwargs_422248 = {}
    # Getting the type of 'print' (line 128)
    print_422242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'print', False)
    # Calling print(args, kwargs) (line 128)
    print_call_result_422249 = invoke(stypy.reporting.localization.Localization(__file__, 128, 4), print_422242, *[result_mod_422247], **kwargs_422248)
    
    
    # Call to print(...): (line 129)
    # Processing the call arguments (line 129)
    str_422251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 10), 'str', 'Norms computed in %.2fs by LSQR')
    
    # Call to time(...): (line 129)
    # Processing the call keyword arguments (line 129)
    kwargs_422253 = {}
    # Getting the type of 'time' (line 129)
    time_422252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 47), 'time', False)
    # Calling time(args, kwargs) (line 129)
    time_call_result_422254 = invoke(stypy.reporting.localization.Localization(__file__, 129, 47), time_422252, *[], **kwargs_422253)
    
    # Getting the type of 'tic' (line 129)
    tic_422255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 56), 'tic', False)
    # Applying the binary operator '-' (line 129)
    result_sub_422256 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 47), '-', time_call_result_422254, tic_422255)
    
    # Applying the binary operator '%' (line 129)
    result_mod_422257 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 10), '%', str_422251, result_sub_422256)
    
    # Processing the call keyword arguments (line 129)
    kwargs_422258 = {}
    # Getting the type of 'print' (line 129)
    print_422250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'print', False)
    # Calling print(args, kwargs) (line 129)
    print_call_result_422259 = invoke(stypy.reporting.localization.Localization(__file__, 129, 4), print_422250, *[result_mod_422257], **kwargs_422258)
    
    
    # Call to print(...): (line 130)
    # Processing the call arguments (line 130)
    str_422261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 10), 'str', ' ||x||  %9.4e  ||r|| %9.4e  ||Ar||  %9.4e ')
    
    # Obtaining an instance of the builtin type 'tuple' (line 130)
    tuple_422262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 58), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 130)
    # Adding element type (line 130)
    # Getting the type of 'chio' (line 130)
    chio_422263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 58), 'chio', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 58), tuple_422262, chio_422263)
    # Adding element type (line 130)
    # Getting the type of 'phio' (line 130)
    phio_422264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 64), 'phio', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 58), tuple_422262, phio_422264)
    # Adding element type (line 130)
    # Getting the type of 'psio' (line 130)
    psio_422265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 70), 'psio', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 58), tuple_422262, psio_422265)
    
    # Applying the binary operator '%' (line 130)
    result_mod_422266 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 10), '%', str_422261, tuple_422262)
    
    # Processing the call keyword arguments (line 130)
    kwargs_422267 = {}
    # Getting the type of 'print' (line 130)
    print_422260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'print', False)
    # Calling print(args, kwargs) (line 130)
    print_call_result_422268 = invoke(stypy.reporting.localization.Localization(__file__, 130, 4), print_422260, *[result_mod_422266], **kwargs_422267)
    
    
    # Call to print(...): (line 131)
    # Processing the call arguments (line 131)
    str_422270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 10), 'str', 'Residual norms computed directly:')
    # Processing the call keyword arguments (line 131)
    kwargs_422271 = {}
    # Getting the type of 'print' (line 131)
    print_422269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'print', False)
    # Calling print(args, kwargs) (line 131)
    print_call_result_422272 = invoke(stypy.reporting.localization.Localization(__file__, 131, 4), print_422269, *[str_422270], **kwargs_422271)
    
    
    # Call to print(...): (line 132)
    # Processing the call arguments (line 132)
    str_422274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 10), 'str', ' ||x||  %9.4e  ||r|| %9.4e  ||Ar||  %9.4e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 132)
    tuple_422275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 57), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 132)
    # Adding element type (line 132)
    
    # Call to norm(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'xo' (line 132)
    xo_422277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 62), 'xo', False)
    # Processing the call keyword arguments (line 132)
    kwargs_422278 = {}
    # Getting the type of 'norm' (line 132)
    norm_422276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 57), 'norm', False)
    # Calling norm(args, kwargs) (line 132)
    norm_call_result_422279 = invoke(stypy.reporting.localization.Localization(__file__, 132, 57), norm_422276, *[xo_422277], **kwargs_422278)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 57), tuple_422275, norm_call_result_422279)
    # Adding element type (line 132)
    
    # Call to norm(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'G' (line 133)
    G_422281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 63), 'G', False)
    # Getting the type of 'xo' (line 133)
    xo_422282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 65), 'xo', False)
    # Applying the binary operator '*' (line 133)
    result_mul_422283 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 63), '*', G_422281, xo_422282)
    
    # Getting the type of 'b' (line 133)
    b_422284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 70), 'b', False)
    # Applying the binary operator '-' (line 133)
    result_sub_422285 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 63), '-', result_mul_422283, b_422284)
    
    # Processing the call keyword arguments (line 133)
    kwargs_422286 = {}
    # Getting the type of 'norm' (line 133)
    norm_422280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 58), 'norm', False)
    # Calling norm(args, kwargs) (line 133)
    norm_call_result_422287 = invoke(stypy.reporting.localization.Localization(__file__, 133, 58), norm_422280, *[result_sub_422285], **kwargs_422286)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 57), tuple_422275, norm_call_result_422287)
    # Adding element type (line 132)
    
    # Call to norm(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'G' (line 134)
    G_422289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 63), 'G', False)
    # Obtaining the member 'T' of a type (line 134)
    T_422290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 63), G_422289, 'T')
    # Getting the type of 'G' (line 134)
    G_422291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 68), 'G', False)
    # Getting the type of 'xo' (line 134)
    xo_422292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 70), 'xo', False)
    # Applying the binary operator '*' (line 134)
    result_mul_422293 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 68), '*', G_422291, xo_422292)
    
    # Getting the type of 'b' (line 134)
    b_422294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 73), 'b', False)
    # Applying the binary operator '-' (line 134)
    result_sub_422295 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 68), '-', result_mul_422293, b_422294)
    
    # Applying the binary operator '*' (line 134)
    result_mul_422296 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 63), '*', T_422290, result_sub_422295)
    
    # Processing the call keyword arguments (line 134)
    kwargs_422297 = {}
    # Getting the type of 'norm' (line 134)
    norm_422288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 58), 'norm', False)
    # Calling norm(args, kwargs) (line 134)
    norm_call_result_422298 = invoke(stypy.reporting.localization.Localization(__file__, 134, 58), norm_422288, *[result_mul_422296], **kwargs_422297)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 57), tuple_422275, norm_call_result_422298)
    
    # Applying the binary operator '%' (line 132)
    result_mod_422299 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 10), '%', str_422274, tuple_422275)
    
    # Processing the call keyword arguments (line 132)
    kwargs_422300 = {}
    # Getting the type of 'print' (line 132)
    print_422273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'print', False)
    # Calling print(args, kwargs) (line 132)
    print_call_result_422301 = invoke(stypy.reporting.localization.Localization(__file__, 132, 4), print_422273, *[result_mod_422299], **kwargs_422300)
    
    
    # Call to print(...): (line 135)
    # Processing the call arguments (line 135)
    str_422303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 10), 'str', 'Direct solution norms:')
    # Processing the call keyword arguments (line 135)
    kwargs_422304 = {}
    # Getting the type of 'print' (line 135)
    print_422302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'print', False)
    # Calling print(args, kwargs) (line 135)
    print_call_result_422305 = invoke(stypy.reporting.localization.Localization(__file__, 135, 4), print_422302, *[str_422303], **kwargs_422304)
    
    
    # Call to print(...): (line 136)
    # Processing the call arguments (line 136)
    str_422307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 10), 'str', ' ||x||  %9.4e  ||r|| %9.4e ')
    
    # Obtaining an instance of the builtin type 'tuple' (line 136)
    tuple_422308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 136)
    # Adding element type (line 136)
    
    # Call to norm(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'svx' (line 136)
    svx_422310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 48), 'svx', False)
    # Processing the call keyword arguments (line 136)
    kwargs_422311 = {}
    # Getting the type of 'norm' (line 136)
    norm_422309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 43), 'norm', False)
    # Calling norm(args, kwargs) (line 136)
    norm_call_result_422312 = invoke(stypy.reporting.localization.Localization(__file__, 136, 43), norm_422309, *[svx_422310], **kwargs_422311)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 43), tuple_422308, norm_call_result_422312)
    # Adding element type (line 136)
    
    # Call to norm(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'G' (line 136)
    G_422314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 59), 'G', False)
    # Getting the type of 'svx' (line 136)
    svx_422315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 61), 'svx', False)
    # Applying the binary operator '*' (line 136)
    result_mul_422316 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 59), '*', G_422314, svx_422315)
    
    # Getting the type of 'b' (line 136)
    b_422317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 67), 'b', False)
    # Applying the binary operator '-' (line 136)
    result_sub_422318 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 59), '-', result_mul_422316, b_422317)
    
    # Processing the call keyword arguments (line 136)
    kwargs_422319 = {}
    # Getting the type of 'norm' (line 136)
    norm_422313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 54), 'norm', False)
    # Calling norm(args, kwargs) (line 136)
    norm_call_result_422320 = invoke(stypy.reporting.localization.Localization(__file__, 136, 54), norm_422313, *[result_sub_422318], **kwargs_422319)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 43), tuple_422308, norm_call_result_422320)
    
    # Applying the binary operator '%' (line 136)
    result_mod_422321 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 10), '%', str_422307, tuple_422308)
    
    # Processing the call keyword arguments (line 136)
    kwargs_422322 = {}
    # Getting the type of 'print' (line 136)
    print_422306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'print', False)
    # Calling print(args, kwargs) (line 136)
    print_call_result_422323 = invoke(stypy.reporting.localization.Localization(__file__, 136, 4), print_422306, *[result_mod_422321], **kwargs_422322)
    
    
    # Call to print(...): (line 137)
    # Processing the call arguments (line 137)
    str_422325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 10), 'str', '')
    # Processing the call keyword arguments (line 137)
    kwargs_422326 = {}
    # Getting the type of 'print' (line 137)
    print_422324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'print', False)
    # Calling print(args, kwargs) (line 137)
    print_call_result_422327 = invoke(stypy.reporting.localization.Localization(__file__, 137, 4), print_422324, *[str_422325], **kwargs_422326)
    
    
    # Call to print(...): (line 138)
    # Processing the call arguments (line 138)
    str_422329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 10), 'str', ' || x_{direct} - x_{LSQR}|| %9.4e ')
    
    # Call to norm(...): (line 138)
    # Processing the call arguments (line 138)
    # Getting the type of 'svx' (line 138)
    svx_422331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 54), 'svx', False)
    # Getting the type of 'xo' (line 138)
    xo_422332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 58), 'xo', False)
    # Applying the binary operator '-' (line 138)
    result_sub_422333 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 54), '-', svx_422331, xo_422332)
    
    # Processing the call keyword arguments (line 138)
    kwargs_422334 = {}
    # Getting the type of 'norm' (line 138)
    norm_422330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 49), 'norm', False)
    # Calling norm(args, kwargs) (line 138)
    norm_call_result_422335 = invoke(stypy.reporting.localization.Localization(__file__, 138, 49), norm_422330, *[result_sub_422333], **kwargs_422334)
    
    # Applying the binary operator '%' (line 138)
    result_mod_422336 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 10), '%', str_422329, norm_call_result_422335)
    
    # Processing the call keyword arguments (line 138)
    kwargs_422337 = {}
    # Getting the type of 'print' (line 138)
    print_422328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'print', False)
    # Calling print(args, kwargs) (line 138)
    print_call_result_422338 = invoke(stypy.reporting.localization.Localization(__file__, 138, 4), print_422328, *[result_mod_422336], **kwargs_422337)
    
    
    # Call to print(...): (line 139)
    # Processing the call arguments (line 139)
    str_422340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 10), 'str', '')
    # Processing the call keyword arguments (line 139)
    kwargs_422341 = {}
    # Getting the type of 'print' (line 139)
    print_422339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'print', False)
    # Calling print(args, kwargs) (line 139)
    print_call_result_422342 = invoke(stypy.reporting.localization.Localization(__file__, 139, 4), print_422339, *[str_422340], **kwargs_422341)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
