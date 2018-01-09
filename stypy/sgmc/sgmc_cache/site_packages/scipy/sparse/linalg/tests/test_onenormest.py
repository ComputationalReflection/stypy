
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Test functions for the sparse.linalg._onenormest module
2: '''
3: 
4: from __future__ import division, print_function, absolute_import
5: 
6: import numpy as np
7: from numpy.testing import assert_allclose, assert_equal, assert_
8: import pytest
9: import scipy.linalg
10: import scipy.sparse.linalg
11: from scipy.sparse.linalg._onenormest import _onenormest_core, _algorithm_2_2
12: 
13: 
14: class MatrixProductOperator(scipy.sparse.linalg.LinearOperator):
15:     '''
16:     This is purely for onenormest testing.
17:     '''
18: 
19:     def __init__(self, A, B):
20:         if A.ndim != 2 or B.ndim != 2:
21:             raise ValueError('expected ndarrays representing matrices')
22:         if A.shape[1] != B.shape[0]:
23:             raise ValueError('incompatible shapes')
24:         self.A = A
25:         self.B = B
26:         self.ndim = 2
27:         self.shape = (A.shape[0], B.shape[1])
28: 
29:     def _matvec(self, x):
30:         return np.dot(self.A, np.dot(self.B, x))
31: 
32:     def _rmatvec(self, x):
33:         return np.dot(np.dot(x, self.A), self.B)
34: 
35:     def _matmat(self, X):
36:         return np.dot(self.A, np.dot(self.B, X))
37: 
38:     @property
39:     def T(self):
40:         return MatrixProductOperator(self.B.T, self.A.T)
41: 
42: 
43: class TestOnenormest(object):
44: 
45:     @pytest.mark.xslow
46:     def test_onenormest_table_3_t_2(self):
47:         # This will take multiple seconds if your computer is slow like mine.
48:         # It is stochastic, so the tolerance could be too strict.
49:         np.random.seed(1234)
50:         t = 2
51:         n = 100
52:         itmax = 5
53:         nsamples = 5000
54:         observed = []
55:         expected = []
56:         nmult_list = []
57:         nresample_list = []
58:         for i in range(nsamples):
59:             A = scipy.linalg.inv(np.random.randn(n, n))
60:             est, v, w, nmults, nresamples = _onenormest_core(A, A.T, t, itmax)
61:             observed.append(est)
62:             expected.append(scipy.linalg.norm(A, 1))
63:             nmult_list.append(nmults)
64:             nresample_list.append(nresamples)
65:         observed = np.array(observed, dtype=float)
66:         expected = np.array(expected, dtype=float)
67:         relative_errors = np.abs(observed - expected) / expected
68: 
69:         # check the mean underestimation ratio
70:         underestimation_ratio = observed / expected
71:         assert_(0.99 < np.mean(underestimation_ratio) < 1.0)
72: 
73:         # check the max and mean required column resamples
74:         assert_equal(np.max(nresample_list), 2)
75:         assert_(0.05 < np.mean(nresample_list) < 0.2)
76: 
77:         # check the proportion of norms computed exactly correctly
78:         nexact = np.count_nonzero(relative_errors < 1e-14)
79:         proportion_exact = nexact / float(nsamples)
80:         assert_(0.9 < proportion_exact < 0.95)
81: 
82:         # check the average number of matrix*vector multiplications
83:         assert_(3.5 < np.mean(nmult_list) < 4.5)
84: 
85:     @pytest.mark.xslow
86:     def test_onenormest_table_4_t_7(self):
87:         # This will take multiple seconds if your computer is slow like mine.
88:         # It is stochastic, so the tolerance could be too strict.
89:         np.random.seed(1234)
90:         t = 7
91:         n = 100
92:         itmax = 5
93:         nsamples = 5000
94:         observed = []
95:         expected = []
96:         nmult_list = []
97:         nresample_list = []
98:         for i in range(nsamples):
99:             A = np.random.randint(-1, 2, size=(n, n))
100:             est, v, w, nmults, nresamples = _onenormest_core(A, A.T, t, itmax)
101:             observed.append(est)
102:             expected.append(scipy.linalg.norm(A, 1))
103:             nmult_list.append(nmults)
104:             nresample_list.append(nresamples)
105:         observed = np.array(observed, dtype=float)
106:         expected = np.array(expected, dtype=float)
107:         relative_errors = np.abs(observed - expected) / expected
108: 
109:         # check the mean underestimation ratio
110:         underestimation_ratio = observed / expected
111:         assert_(0.90 < np.mean(underestimation_ratio) < 0.99)
112: 
113:         # check the required column resamples
114:         assert_equal(np.max(nresample_list), 0)
115: 
116:         # check the proportion of norms computed exactly correctly
117:         nexact = np.count_nonzero(relative_errors < 1e-14)
118:         proportion_exact = nexact / float(nsamples)
119:         assert_(0.15 < proportion_exact < 0.25)
120: 
121:         # check the average number of matrix*vector multiplications
122:         assert_(3.5 < np.mean(nmult_list) < 4.5)
123: 
124:     def test_onenormest_table_5_t_1(self):
125:         # "note that there is no randomness and hence only one estimate for t=1"
126:         t = 1
127:         n = 100
128:         itmax = 5
129:         alpha = 1 - 1e-6
130:         A = -scipy.linalg.inv(np.identity(n) + alpha*np.eye(n, k=1))
131:         first_col = np.array([1] + [0]*(n-1))
132:         first_row = np.array([(-alpha)**i for i in range(n)])
133:         B = -scipy.linalg.toeplitz(first_col, first_row)
134:         assert_allclose(A, B)
135:         est, v, w, nmults, nresamples = _onenormest_core(B, B.T, t, itmax)
136:         exact_value = scipy.linalg.norm(B, 1)
137:         underest_ratio = est / exact_value
138:         assert_allclose(underest_ratio, 0.05, rtol=1e-4)
139:         assert_equal(nmults, 11)
140:         assert_equal(nresamples, 0)
141:         # check the non-underscored version of onenormest
142:         est_plain = scipy.sparse.linalg.onenormest(B, t=t, itmax=itmax)
143:         assert_allclose(est, est_plain)
144: 
145:     @pytest.mark.xslow
146:     def test_onenormest_table_6_t_1(self):
147:         #TODO this test seems to give estimates that match the table,
148:         #TODO even though no attempt has been made to deal with
149:         #TODO complex numbers in the one-norm estimation.
150:         # This will take multiple seconds if your computer is slow like mine.
151:         # It is stochastic, so the tolerance could be too strict.
152:         np.random.seed(1234)
153:         t = 1
154:         n = 100
155:         itmax = 5
156:         nsamples = 5000
157:         observed = []
158:         expected = []
159:         nmult_list = []
160:         nresample_list = []
161:         for i in range(nsamples):
162:             A_inv = np.random.rand(n, n) + 1j * np.random.rand(n, n)
163:             A = scipy.linalg.inv(A_inv)
164:             est, v, w, nmults, nresamples = _onenormest_core(A, A.T, t, itmax)
165:             observed.append(est)
166:             expected.append(scipy.linalg.norm(A, 1))
167:             nmult_list.append(nmults)
168:             nresample_list.append(nresamples)
169:         observed = np.array(observed, dtype=float)
170:         expected = np.array(expected, dtype=float)
171:         relative_errors = np.abs(observed - expected) / expected
172: 
173:         # check the mean underestimation ratio
174:         underestimation_ratio = observed / expected
175:         underestimation_ratio_mean = np.mean(underestimation_ratio)
176:         assert_(0.90 < underestimation_ratio_mean < 0.99)
177: 
178:         # check the required column resamples
179:         max_nresamples = np.max(nresample_list)
180:         assert_equal(max_nresamples, 0)
181: 
182:         # check the proportion of norms computed exactly correctly
183:         nexact = np.count_nonzero(relative_errors < 1e-14)
184:         proportion_exact = nexact / float(nsamples)
185:         assert_(0.7 < proportion_exact < 0.8)
186: 
187:         # check the average number of matrix*vector multiplications
188:         mean_nmult = np.mean(nmult_list)
189:         assert_(4 < mean_nmult < 5)
190: 
191:     def _help_product_norm_slow(self, A, B):
192:         # for profiling
193:         C = np.dot(A, B)
194:         return scipy.linalg.norm(C, 1)
195: 
196:     def _help_product_norm_fast(self, A, B):
197:         # for profiling
198:         t = 2
199:         itmax = 5
200:         D = MatrixProductOperator(A, B)
201:         est, v, w, nmults, nresamples = _onenormest_core(D, D.T, t, itmax)
202:         return est
203: 
204:     @pytest.mark.slow
205:     def test_onenormest_linear_operator(self):
206:         # Define a matrix through its product A B.
207:         # Depending on the shapes of A and B,
208:         # it could be easy to multiply this product by a small matrix,
209:         # but it could be annoying to look at all of
210:         # the entries of the product explicitly.
211:         np.random.seed(1234)
212:         n = 6000
213:         k = 3
214:         A = np.random.randn(n, k)
215:         B = np.random.randn(k, n)
216:         fast_estimate = self._help_product_norm_fast(A, B)
217:         exact_value = self._help_product_norm_slow(A, B)
218:         assert_(fast_estimate <= exact_value <= 3*fast_estimate,
219:                 'fast: %g\nexact:%g' % (fast_estimate, exact_value))
220: 
221:     def test_returns(self):
222:         np.random.seed(1234)
223:         A = scipy.sparse.rand(50, 50, 0.1)
224: 
225:         s0 = scipy.linalg.norm(A.todense(), 1)
226:         s1, v = scipy.sparse.linalg.onenormest(A, compute_v=True)
227:         s2, w = scipy.sparse.linalg.onenormest(A, compute_w=True)
228:         s3, v2, w2 = scipy.sparse.linalg.onenormest(A, compute_w=True, compute_v=True)
229: 
230:         assert_allclose(s1, s0, rtol=1e-9)
231:         assert_allclose(np.linalg.norm(A.dot(v), 1), s0*np.linalg.norm(v, 1), rtol=1e-9)
232:         assert_allclose(A.dot(v), w, rtol=1e-9)
233: 
234: 
235: class TestAlgorithm_2_2(object):
236: 
237:     def test_randn_inv(self):
238:         np.random.seed(1234)
239:         n = 20
240:         nsamples = 100
241:         for i in range(nsamples):
242: 
243:             # Choose integer t uniformly between 1 and 3 inclusive.
244:             t = np.random.randint(1, 4)
245: 
246:             # Choose n uniformly between 10 and 40 inclusive.
247:             n = np.random.randint(10, 41)
248: 
249:             # Sample the inverse of a matrix with random normal entries.
250:             A = scipy.linalg.inv(np.random.randn(n, n))
251: 
252:             # Compute the 1-norm bounds.
253:             g, ind = _algorithm_2_2(A, A.T, t)
254: 
255: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_428730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', 'Test functions for the sparse.linalg._onenormest module\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_428731 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_428731) is not StypyTypeError):

    if (import_428731 != 'pyd_module'):
        __import__(import_428731)
        sys_modules_428732 = sys.modules[import_428731]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_428732.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_428731)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.testing import assert_allclose, assert_equal, assert_' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_428733 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing')

if (type(import_428733) is not StypyTypeError):

    if (import_428733 != 'pyd_module'):
        __import__(import_428733)
        sys_modules_428734 = sys.modules[import_428733]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', sys_modules_428734.module_type_store, module_type_store, ['assert_allclose', 'assert_equal', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_428734, sys_modules_428734.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_allclose, assert_equal, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', None, module_type_store, ['assert_allclose', 'assert_equal', 'assert_'], [assert_allclose, assert_equal, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', import_428733)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import pytest' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_428735 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest')

if (type(import_428735) is not StypyTypeError):

    if (import_428735 != 'pyd_module'):
        __import__(import_428735)
        sys_modules_428736 = sys.modules[import_428735]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', sys_modules_428736.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', import_428735)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import scipy.linalg' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_428737 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg')

if (type(import_428737) is not StypyTypeError):

    if (import_428737 != 'pyd_module'):
        __import__(import_428737)
        sys_modules_428738 = sys.modules[import_428737]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg', sys_modules_428738.module_type_store, module_type_store)
    else:
        import scipy.linalg

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg', scipy.linalg, module_type_store)

else:
    # Assigning a type to the variable 'scipy.linalg' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg', import_428737)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import scipy.sparse.linalg' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_428739 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg')

if (type(import_428739) is not StypyTypeError):

    if (import_428739 != 'pyd_module'):
        __import__(import_428739)
        sys_modules_428740 = sys.modules[import_428739]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg', sys_modules_428740.module_type_store, module_type_store)
    else:
        import scipy.sparse.linalg

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg', scipy.sparse.linalg, module_type_store)

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg', import_428739)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.sparse.linalg._onenormest import _onenormest_core, _algorithm_2_2' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')
import_428741 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg._onenormest')

if (type(import_428741) is not StypyTypeError):

    if (import_428741 != 'pyd_module'):
        __import__(import_428741)
        sys_modules_428742 = sys.modules[import_428741]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg._onenormest', sys_modules_428742.module_type_store, module_type_store, ['_onenormest_core', '_algorithm_2_2'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_428742, sys_modules_428742.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg._onenormest import _onenormest_core, _algorithm_2_2

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg._onenormest', None, module_type_store, ['_onenormest_core', '_algorithm_2_2'], [_onenormest_core, _algorithm_2_2])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg._onenormest' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg._onenormest', import_428741)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/tests/')

# Declaration of the 'MatrixProductOperator' class
# Getting the type of 'scipy' (line 14)
scipy_428743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 28), 'scipy')
# Obtaining the member 'sparse' of a type (line 14)
sparse_428744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 28), scipy_428743, 'sparse')
# Obtaining the member 'linalg' of a type (line 14)
linalg_428745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 28), sparse_428744, 'linalg')
# Obtaining the member 'LinearOperator' of a type (line 14)
LinearOperator_428746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 28), linalg_428745, 'LinearOperator')

class MatrixProductOperator(LinearOperator_428746, ):
    str_428747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\n    This is purely for onenormest testing.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 19, 4, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatrixProductOperator.__init__', ['A', 'B'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['A', 'B'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'A' (line 20)
        A_428748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 11), 'A')
        # Obtaining the member 'ndim' of a type (line 20)
        ndim_428749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 11), A_428748, 'ndim')
        int_428750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 21), 'int')
        # Applying the binary operator '!=' (line 20)
        result_ne_428751 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 11), '!=', ndim_428749, int_428750)
        
        
        # Getting the type of 'B' (line 20)
        B_428752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 26), 'B')
        # Obtaining the member 'ndim' of a type (line 20)
        ndim_428753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 26), B_428752, 'ndim')
        int_428754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 36), 'int')
        # Applying the binary operator '!=' (line 20)
        result_ne_428755 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 26), '!=', ndim_428753, int_428754)
        
        # Applying the binary operator 'or' (line 20)
        result_or_keyword_428756 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 11), 'or', result_ne_428751, result_ne_428755)
        
        # Testing the type of an if condition (line 20)
        if_condition_428757 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 20, 8), result_or_keyword_428756)
        # Assigning a type to the variable 'if_condition_428757' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'if_condition_428757', if_condition_428757)
        # SSA begins for if statement (line 20)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 21)
        # Processing the call arguments (line 21)
        str_428759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 29), 'str', 'expected ndarrays representing matrices')
        # Processing the call keyword arguments (line 21)
        kwargs_428760 = {}
        # Getting the type of 'ValueError' (line 21)
        ValueError_428758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 21)
        ValueError_call_result_428761 = invoke(stypy.reporting.localization.Localization(__file__, 21, 18), ValueError_428758, *[str_428759], **kwargs_428760)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 21, 12), ValueError_call_result_428761, 'raise parameter', BaseException)
        # SSA join for if statement (line 20)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        int_428762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 19), 'int')
        # Getting the type of 'A' (line 22)
        A_428763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'A')
        # Obtaining the member 'shape' of a type (line 22)
        shape_428764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 11), A_428763, 'shape')
        # Obtaining the member '__getitem__' of a type (line 22)
        getitem___428765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 11), shape_428764, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 22)
        subscript_call_result_428766 = invoke(stypy.reporting.localization.Localization(__file__, 22, 11), getitem___428765, int_428762)
        
        
        # Obtaining the type of the subscript
        int_428767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 33), 'int')
        # Getting the type of 'B' (line 22)
        B_428768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 25), 'B')
        # Obtaining the member 'shape' of a type (line 22)
        shape_428769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 25), B_428768, 'shape')
        # Obtaining the member '__getitem__' of a type (line 22)
        getitem___428770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 25), shape_428769, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 22)
        subscript_call_result_428771 = invoke(stypy.reporting.localization.Localization(__file__, 22, 25), getitem___428770, int_428767)
        
        # Applying the binary operator '!=' (line 22)
        result_ne_428772 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 11), '!=', subscript_call_result_428766, subscript_call_result_428771)
        
        # Testing the type of an if condition (line 22)
        if_condition_428773 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 22, 8), result_ne_428772)
        # Assigning a type to the variable 'if_condition_428773' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'if_condition_428773', if_condition_428773)
        # SSA begins for if statement (line 22)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 23)
        # Processing the call arguments (line 23)
        str_428775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 29), 'str', 'incompatible shapes')
        # Processing the call keyword arguments (line 23)
        kwargs_428776 = {}
        # Getting the type of 'ValueError' (line 23)
        ValueError_428774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 23)
        ValueError_call_result_428777 = invoke(stypy.reporting.localization.Localization(__file__, 23, 18), ValueError_428774, *[str_428775], **kwargs_428776)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 23, 12), ValueError_call_result_428777, 'raise parameter', BaseException)
        # SSA join for if statement (line 22)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 24):
        
        # Assigning a Name to a Attribute (line 24):
        # Getting the type of 'A' (line 24)
        A_428778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 17), 'A')
        # Getting the type of 'self' (line 24)
        self_428779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'self')
        # Setting the type of the member 'A' of a type (line 24)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), self_428779, 'A', A_428778)
        
        # Assigning a Name to a Attribute (line 25):
        
        # Assigning a Name to a Attribute (line 25):
        # Getting the type of 'B' (line 25)
        B_428780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 17), 'B')
        # Getting the type of 'self' (line 25)
        self_428781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self')
        # Setting the type of the member 'B' of a type (line 25)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_428781, 'B', B_428780)
        
        # Assigning a Num to a Attribute (line 26):
        
        # Assigning a Num to a Attribute (line 26):
        int_428782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 20), 'int')
        # Getting the type of 'self' (line 26)
        self_428783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self')
        # Setting the type of the member 'ndim' of a type (line 26)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), self_428783, 'ndim', int_428782)
        
        # Assigning a Tuple to a Attribute (line 27):
        
        # Assigning a Tuple to a Attribute (line 27):
        
        # Obtaining an instance of the builtin type 'tuple' (line 27)
        tuple_428784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 27)
        # Adding element type (line 27)
        
        # Obtaining the type of the subscript
        int_428785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 30), 'int')
        # Getting the type of 'A' (line 27)
        A_428786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 22), 'A')
        # Obtaining the member 'shape' of a type (line 27)
        shape_428787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 22), A_428786, 'shape')
        # Obtaining the member '__getitem__' of a type (line 27)
        getitem___428788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 22), shape_428787, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 27)
        subscript_call_result_428789 = invoke(stypy.reporting.localization.Localization(__file__, 27, 22), getitem___428788, int_428785)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 22), tuple_428784, subscript_call_result_428789)
        # Adding element type (line 27)
        
        # Obtaining the type of the subscript
        int_428790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 42), 'int')
        # Getting the type of 'B' (line 27)
        B_428791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 34), 'B')
        # Obtaining the member 'shape' of a type (line 27)
        shape_428792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 34), B_428791, 'shape')
        # Obtaining the member '__getitem__' of a type (line 27)
        getitem___428793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 34), shape_428792, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 27)
        subscript_call_result_428794 = invoke(stypy.reporting.localization.Localization(__file__, 27, 34), getitem___428793, int_428790)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 22), tuple_428784, subscript_call_result_428794)
        
        # Getting the type of 'self' (line 27)
        self_428795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'self')
        # Setting the type of the member 'shape' of a type (line 27)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), self_428795, 'shape', tuple_428784)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _matvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_matvec'
        module_type_store = module_type_store.open_function_context('_matvec', 29, 4, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatrixProductOperator._matvec.__dict__.__setitem__('stypy_localization', localization)
        MatrixProductOperator._matvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatrixProductOperator._matvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatrixProductOperator._matvec.__dict__.__setitem__('stypy_function_name', 'MatrixProductOperator._matvec')
        MatrixProductOperator._matvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
        MatrixProductOperator._matvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatrixProductOperator._matvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatrixProductOperator._matvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatrixProductOperator._matvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatrixProductOperator._matvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatrixProductOperator._matvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatrixProductOperator._matvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_matvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_matvec(...)' code ##################

        
        # Call to dot(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'self' (line 30)
        self_428798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 22), 'self', False)
        # Obtaining the member 'A' of a type (line 30)
        A_428799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 22), self_428798, 'A')
        
        # Call to dot(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'self' (line 30)
        self_428802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 37), 'self', False)
        # Obtaining the member 'B' of a type (line 30)
        B_428803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 37), self_428802, 'B')
        # Getting the type of 'x' (line 30)
        x_428804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 45), 'x', False)
        # Processing the call keyword arguments (line 30)
        kwargs_428805 = {}
        # Getting the type of 'np' (line 30)
        np_428800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 30), 'np', False)
        # Obtaining the member 'dot' of a type (line 30)
        dot_428801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 30), np_428800, 'dot')
        # Calling dot(args, kwargs) (line 30)
        dot_call_result_428806 = invoke(stypy.reporting.localization.Localization(__file__, 30, 30), dot_428801, *[B_428803, x_428804], **kwargs_428805)
        
        # Processing the call keyword arguments (line 30)
        kwargs_428807 = {}
        # Getting the type of 'np' (line 30)
        np_428796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 15), 'np', False)
        # Obtaining the member 'dot' of a type (line 30)
        dot_428797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 15), np_428796, 'dot')
        # Calling dot(args, kwargs) (line 30)
        dot_call_result_428808 = invoke(stypy.reporting.localization.Localization(__file__, 30, 15), dot_428797, *[A_428799, dot_call_result_428806], **kwargs_428807)
        
        # Assigning a type to the variable 'stypy_return_type' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'stypy_return_type', dot_call_result_428808)
        
        # ################# End of '_matvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_matvec' in the type store
        # Getting the type of 'stypy_return_type' (line 29)
        stypy_return_type_428809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_428809)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_matvec'
        return stypy_return_type_428809


    @norecursion
    def _rmatvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_rmatvec'
        module_type_store = module_type_store.open_function_context('_rmatvec', 32, 4, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatrixProductOperator._rmatvec.__dict__.__setitem__('stypy_localization', localization)
        MatrixProductOperator._rmatvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatrixProductOperator._rmatvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatrixProductOperator._rmatvec.__dict__.__setitem__('stypy_function_name', 'MatrixProductOperator._rmatvec')
        MatrixProductOperator._rmatvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
        MatrixProductOperator._rmatvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatrixProductOperator._rmatvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatrixProductOperator._rmatvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatrixProductOperator._rmatvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatrixProductOperator._rmatvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatrixProductOperator._rmatvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatrixProductOperator._rmatvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_rmatvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_rmatvec(...)' code ##################

        
        # Call to dot(...): (line 33)
        # Processing the call arguments (line 33)
        
        # Call to dot(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'x' (line 33)
        x_428814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 29), 'x', False)
        # Getting the type of 'self' (line 33)
        self_428815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 32), 'self', False)
        # Obtaining the member 'A' of a type (line 33)
        A_428816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 32), self_428815, 'A')
        # Processing the call keyword arguments (line 33)
        kwargs_428817 = {}
        # Getting the type of 'np' (line 33)
        np_428812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 22), 'np', False)
        # Obtaining the member 'dot' of a type (line 33)
        dot_428813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 22), np_428812, 'dot')
        # Calling dot(args, kwargs) (line 33)
        dot_call_result_428818 = invoke(stypy.reporting.localization.Localization(__file__, 33, 22), dot_428813, *[x_428814, A_428816], **kwargs_428817)
        
        # Getting the type of 'self' (line 33)
        self_428819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 41), 'self', False)
        # Obtaining the member 'B' of a type (line 33)
        B_428820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 41), self_428819, 'B')
        # Processing the call keyword arguments (line 33)
        kwargs_428821 = {}
        # Getting the type of 'np' (line 33)
        np_428810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'np', False)
        # Obtaining the member 'dot' of a type (line 33)
        dot_428811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 15), np_428810, 'dot')
        # Calling dot(args, kwargs) (line 33)
        dot_call_result_428822 = invoke(stypy.reporting.localization.Localization(__file__, 33, 15), dot_428811, *[dot_call_result_428818, B_428820], **kwargs_428821)
        
        # Assigning a type to the variable 'stypy_return_type' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'stypy_return_type', dot_call_result_428822)
        
        # ################# End of '_rmatvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_rmatvec' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_428823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_428823)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_rmatvec'
        return stypy_return_type_428823


    @norecursion
    def _matmat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_matmat'
        module_type_store = module_type_store.open_function_context('_matmat', 35, 4, False)
        # Assigning a type to the variable 'self' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatrixProductOperator._matmat.__dict__.__setitem__('stypy_localization', localization)
        MatrixProductOperator._matmat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatrixProductOperator._matmat.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatrixProductOperator._matmat.__dict__.__setitem__('stypy_function_name', 'MatrixProductOperator._matmat')
        MatrixProductOperator._matmat.__dict__.__setitem__('stypy_param_names_list', ['X'])
        MatrixProductOperator._matmat.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatrixProductOperator._matmat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatrixProductOperator._matmat.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatrixProductOperator._matmat.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatrixProductOperator._matmat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatrixProductOperator._matmat.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatrixProductOperator._matmat', ['X'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_matmat', localization, ['X'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_matmat(...)' code ##################

        
        # Call to dot(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'self' (line 36)
        self_428826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 22), 'self', False)
        # Obtaining the member 'A' of a type (line 36)
        A_428827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 22), self_428826, 'A')
        
        # Call to dot(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'self' (line 36)
        self_428830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 37), 'self', False)
        # Obtaining the member 'B' of a type (line 36)
        B_428831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 37), self_428830, 'B')
        # Getting the type of 'X' (line 36)
        X_428832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 45), 'X', False)
        # Processing the call keyword arguments (line 36)
        kwargs_428833 = {}
        # Getting the type of 'np' (line 36)
        np_428828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 30), 'np', False)
        # Obtaining the member 'dot' of a type (line 36)
        dot_428829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 30), np_428828, 'dot')
        # Calling dot(args, kwargs) (line 36)
        dot_call_result_428834 = invoke(stypy.reporting.localization.Localization(__file__, 36, 30), dot_428829, *[B_428831, X_428832], **kwargs_428833)
        
        # Processing the call keyword arguments (line 36)
        kwargs_428835 = {}
        # Getting the type of 'np' (line 36)
        np_428824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 'np', False)
        # Obtaining the member 'dot' of a type (line 36)
        dot_428825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 15), np_428824, 'dot')
        # Calling dot(args, kwargs) (line 36)
        dot_call_result_428836 = invoke(stypy.reporting.localization.Localization(__file__, 36, 15), dot_428825, *[A_428827, dot_call_result_428834], **kwargs_428835)
        
        # Assigning a type to the variable 'stypy_return_type' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'stypy_return_type', dot_call_result_428836)
        
        # ################# End of '_matmat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_matmat' in the type store
        # Getting the type of 'stypy_return_type' (line 35)
        stypy_return_type_428837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_428837)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_matmat'
        return stypy_return_type_428837


    @norecursion
    def T(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'T'
        module_type_store = module_type_store.open_function_context('T', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatrixProductOperator.T.__dict__.__setitem__('stypy_localization', localization)
        MatrixProductOperator.T.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatrixProductOperator.T.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatrixProductOperator.T.__dict__.__setitem__('stypy_function_name', 'MatrixProductOperator.T')
        MatrixProductOperator.T.__dict__.__setitem__('stypy_param_names_list', [])
        MatrixProductOperator.T.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatrixProductOperator.T.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatrixProductOperator.T.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatrixProductOperator.T.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatrixProductOperator.T.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatrixProductOperator.T.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatrixProductOperator.T', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'T', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'T(...)' code ##################

        
        # Call to MatrixProductOperator(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'self' (line 40)
        self_428839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 37), 'self', False)
        # Obtaining the member 'B' of a type (line 40)
        B_428840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 37), self_428839, 'B')
        # Obtaining the member 'T' of a type (line 40)
        T_428841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 37), B_428840, 'T')
        # Getting the type of 'self' (line 40)
        self_428842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 47), 'self', False)
        # Obtaining the member 'A' of a type (line 40)
        A_428843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 47), self_428842, 'A')
        # Obtaining the member 'T' of a type (line 40)
        T_428844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 47), A_428843, 'T')
        # Processing the call keyword arguments (line 40)
        kwargs_428845 = {}
        # Getting the type of 'MatrixProductOperator' (line 40)
        MatrixProductOperator_428838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'MatrixProductOperator', False)
        # Calling MatrixProductOperator(args, kwargs) (line 40)
        MatrixProductOperator_call_result_428846 = invoke(stypy.reporting.localization.Localization(__file__, 40, 15), MatrixProductOperator_428838, *[T_428841, T_428844], **kwargs_428845)
        
        # Assigning a type to the variable 'stypy_return_type' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'stypy_return_type', MatrixProductOperator_call_result_428846)
        
        # ################# End of 'T(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'T' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_428847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_428847)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'T'
        return stypy_return_type_428847


# Assigning a type to the variable 'MatrixProductOperator' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'MatrixProductOperator', MatrixProductOperator)
# Declaration of the 'TestOnenormest' class

class TestOnenormest(object, ):

    @norecursion
    def test_onenormest_table_3_t_2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_onenormest_table_3_t_2'
        module_type_store = module_type_store.open_function_context('test_onenormest_table_3_t_2', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOnenormest.test_onenormest_table_3_t_2.__dict__.__setitem__('stypy_localization', localization)
        TestOnenormest.test_onenormest_table_3_t_2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOnenormest.test_onenormest_table_3_t_2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOnenormest.test_onenormest_table_3_t_2.__dict__.__setitem__('stypy_function_name', 'TestOnenormest.test_onenormest_table_3_t_2')
        TestOnenormest.test_onenormest_table_3_t_2.__dict__.__setitem__('stypy_param_names_list', [])
        TestOnenormest.test_onenormest_table_3_t_2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOnenormest.test_onenormest_table_3_t_2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOnenormest.test_onenormest_table_3_t_2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOnenormest.test_onenormest_table_3_t_2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOnenormest.test_onenormest_table_3_t_2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOnenormest.test_onenormest_table_3_t_2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOnenormest.test_onenormest_table_3_t_2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_onenormest_table_3_t_2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_onenormest_table_3_t_2(...)' code ##################

        
        # Call to seed(...): (line 49)
        # Processing the call arguments (line 49)
        int_428851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 23), 'int')
        # Processing the call keyword arguments (line 49)
        kwargs_428852 = {}
        # Getting the type of 'np' (line 49)
        np_428848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 49)
        random_428849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), np_428848, 'random')
        # Obtaining the member 'seed' of a type (line 49)
        seed_428850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), random_428849, 'seed')
        # Calling seed(args, kwargs) (line 49)
        seed_call_result_428853 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), seed_428850, *[int_428851], **kwargs_428852)
        
        
        # Assigning a Num to a Name (line 50):
        
        # Assigning a Num to a Name (line 50):
        int_428854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 12), 'int')
        # Assigning a type to the variable 't' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 't', int_428854)
        
        # Assigning a Num to a Name (line 51):
        
        # Assigning a Num to a Name (line 51):
        int_428855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 12), 'int')
        # Assigning a type to the variable 'n' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'n', int_428855)
        
        # Assigning a Num to a Name (line 52):
        
        # Assigning a Num to a Name (line 52):
        int_428856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 16), 'int')
        # Assigning a type to the variable 'itmax' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'itmax', int_428856)
        
        # Assigning a Num to a Name (line 53):
        
        # Assigning a Num to a Name (line 53):
        int_428857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 19), 'int')
        # Assigning a type to the variable 'nsamples' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'nsamples', int_428857)
        
        # Assigning a List to a Name (line 54):
        
        # Assigning a List to a Name (line 54):
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_428858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        
        # Assigning a type to the variable 'observed' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'observed', list_428858)
        
        # Assigning a List to a Name (line 55):
        
        # Assigning a List to a Name (line 55):
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_428859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        
        # Assigning a type to the variable 'expected' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'expected', list_428859)
        
        # Assigning a List to a Name (line 56):
        
        # Assigning a List to a Name (line 56):
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_428860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        
        # Assigning a type to the variable 'nmult_list' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'nmult_list', list_428860)
        
        # Assigning a List to a Name (line 57):
        
        # Assigning a List to a Name (line 57):
        
        # Obtaining an instance of the builtin type 'list' (line 57)
        list_428861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 57)
        
        # Assigning a type to the variable 'nresample_list' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'nresample_list', list_428861)
        
        
        # Call to range(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'nsamples' (line 58)
        nsamples_428863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 23), 'nsamples', False)
        # Processing the call keyword arguments (line 58)
        kwargs_428864 = {}
        # Getting the type of 'range' (line 58)
        range_428862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 17), 'range', False)
        # Calling range(args, kwargs) (line 58)
        range_call_result_428865 = invoke(stypy.reporting.localization.Localization(__file__, 58, 17), range_428862, *[nsamples_428863], **kwargs_428864)
        
        # Testing the type of a for loop iterable (line 58)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 58, 8), range_call_result_428865)
        # Getting the type of the for loop variable (line 58)
        for_loop_var_428866 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 58, 8), range_call_result_428865)
        # Assigning a type to the variable 'i' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'i', for_loop_var_428866)
        # SSA begins for a for statement (line 58)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 59):
        
        # Assigning a Call to a Name (line 59):
        
        # Call to inv(...): (line 59)
        # Processing the call arguments (line 59)
        
        # Call to randn(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'n' (line 59)
        n_428873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 49), 'n', False)
        # Getting the type of 'n' (line 59)
        n_428874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 52), 'n', False)
        # Processing the call keyword arguments (line 59)
        kwargs_428875 = {}
        # Getting the type of 'np' (line 59)
        np_428870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 33), 'np', False)
        # Obtaining the member 'random' of a type (line 59)
        random_428871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 33), np_428870, 'random')
        # Obtaining the member 'randn' of a type (line 59)
        randn_428872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 33), random_428871, 'randn')
        # Calling randn(args, kwargs) (line 59)
        randn_call_result_428876 = invoke(stypy.reporting.localization.Localization(__file__, 59, 33), randn_428872, *[n_428873, n_428874], **kwargs_428875)
        
        # Processing the call keyword arguments (line 59)
        kwargs_428877 = {}
        # Getting the type of 'scipy' (line 59)
        scipy_428867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 59)
        linalg_428868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 16), scipy_428867, 'linalg')
        # Obtaining the member 'inv' of a type (line 59)
        inv_428869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 16), linalg_428868, 'inv')
        # Calling inv(args, kwargs) (line 59)
        inv_call_result_428878 = invoke(stypy.reporting.localization.Localization(__file__, 59, 16), inv_428869, *[randn_call_result_428876], **kwargs_428877)
        
        # Assigning a type to the variable 'A' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'A', inv_call_result_428878)
        
        # Assigning a Call to a Tuple (line 60):
        
        # Assigning a Subscript to a Name (line 60):
        
        # Obtaining the type of the subscript
        int_428879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 12), 'int')
        
        # Call to _onenormest_core(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'A' (line 60)
        A_428881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 61), 'A', False)
        # Getting the type of 'A' (line 60)
        A_428882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 64), 'A', False)
        # Obtaining the member 'T' of a type (line 60)
        T_428883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 64), A_428882, 'T')
        # Getting the type of 't' (line 60)
        t_428884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 69), 't', False)
        # Getting the type of 'itmax' (line 60)
        itmax_428885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 72), 'itmax', False)
        # Processing the call keyword arguments (line 60)
        kwargs_428886 = {}
        # Getting the type of '_onenormest_core' (line 60)
        _onenormest_core_428880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 44), '_onenormest_core', False)
        # Calling _onenormest_core(args, kwargs) (line 60)
        _onenormest_core_call_result_428887 = invoke(stypy.reporting.localization.Localization(__file__, 60, 44), _onenormest_core_428880, *[A_428881, T_428883, t_428884, itmax_428885], **kwargs_428886)
        
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___428888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 12), _onenormest_core_call_result_428887, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 60)
        subscript_call_result_428889 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), getitem___428888, int_428879)
        
        # Assigning a type to the variable 'tuple_var_assignment_428696' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'tuple_var_assignment_428696', subscript_call_result_428889)
        
        # Assigning a Subscript to a Name (line 60):
        
        # Obtaining the type of the subscript
        int_428890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 12), 'int')
        
        # Call to _onenormest_core(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'A' (line 60)
        A_428892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 61), 'A', False)
        # Getting the type of 'A' (line 60)
        A_428893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 64), 'A', False)
        # Obtaining the member 'T' of a type (line 60)
        T_428894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 64), A_428893, 'T')
        # Getting the type of 't' (line 60)
        t_428895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 69), 't', False)
        # Getting the type of 'itmax' (line 60)
        itmax_428896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 72), 'itmax', False)
        # Processing the call keyword arguments (line 60)
        kwargs_428897 = {}
        # Getting the type of '_onenormest_core' (line 60)
        _onenormest_core_428891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 44), '_onenormest_core', False)
        # Calling _onenormest_core(args, kwargs) (line 60)
        _onenormest_core_call_result_428898 = invoke(stypy.reporting.localization.Localization(__file__, 60, 44), _onenormest_core_428891, *[A_428892, T_428894, t_428895, itmax_428896], **kwargs_428897)
        
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___428899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 12), _onenormest_core_call_result_428898, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 60)
        subscript_call_result_428900 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), getitem___428899, int_428890)
        
        # Assigning a type to the variable 'tuple_var_assignment_428697' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'tuple_var_assignment_428697', subscript_call_result_428900)
        
        # Assigning a Subscript to a Name (line 60):
        
        # Obtaining the type of the subscript
        int_428901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 12), 'int')
        
        # Call to _onenormest_core(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'A' (line 60)
        A_428903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 61), 'A', False)
        # Getting the type of 'A' (line 60)
        A_428904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 64), 'A', False)
        # Obtaining the member 'T' of a type (line 60)
        T_428905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 64), A_428904, 'T')
        # Getting the type of 't' (line 60)
        t_428906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 69), 't', False)
        # Getting the type of 'itmax' (line 60)
        itmax_428907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 72), 'itmax', False)
        # Processing the call keyword arguments (line 60)
        kwargs_428908 = {}
        # Getting the type of '_onenormest_core' (line 60)
        _onenormest_core_428902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 44), '_onenormest_core', False)
        # Calling _onenormest_core(args, kwargs) (line 60)
        _onenormest_core_call_result_428909 = invoke(stypy.reporting.localization.Localization(__file__, 60, 44), _onenormest_core_428902, *[A_428903, T_428905, t_428906, itmax_428907], **kwargs_428908)
        
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___428910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 12), _onenormest_core_call_result_428909, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 60)
        subscript_call_result_428911 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), getitem___428910, int_428901)
        
        # Assigning a type to the variable 'tuple_var_assignment_428698' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'tuple_var_assignment_428698', subscript_call_result_428911)
        
        # Assigning a Subscript to a Name (line 60):
        
        # Obtaining the type of the subscript
        int_428912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 12), 'int')
        
        # Call to _onenormest_core(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'A' (line 60)
        A_428914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 61), 'A', False)
        # Getting the type of 'A' (line 60)
        A_428915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 64), 'A', False)
        # Obtaining the member 'T' of a type (line 60)
        T_428916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 64), A_428915, 'T')
        # Getting the type of 't' (line 60)
        t_428917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 69), 't', False)
        # Getting the type of 'itmax' (line 60)
        itmax_428918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 72), 'itmax', False)
        # Processing the call keyword arguments (line 60)
        kwargs_428919 = {}
        # Getting the type of '_onenormest_core' (line 60)
        _onenormest_core_428913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 44), '_onenormest_core', False)
        # Calling _onenormest_core(args, kwargs) (line 60)
        _onenormest_core_call_result_428920 = invoke(stypy.reporting.localization.Localization(__file__, 60, 44), _onenormest_core_428913, *[A_428914, T_428916, t_428917, itmax_428918], **kwargs_428919)
        
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___428921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 12), _onenormest_core_call_result_428920, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 60)
        subscript_call_result_428922 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), getitem___428921, int_428912)
        
        # Assigning a type to the variable 'tuple_var_assignment_428699' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'tuple_var_assignment_428699', subscript_call_result_428922)
        
        # Assigning a Subscript to a Name (line 60):
        
        # Obtaining the type of the subscript
        int_428923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 12), 'int')
        
        # Call to _onenormest_core(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'A' (line 60)
        A_428925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 61), 'A', False)
        # Getting the type of 'A' (line 60)
        A_428926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 64), 'A', False)
        # Obtaining the member 'T' of a type (line 60)
        T_428927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 64), A_428926, 'T')
        # Getting the type of 't' (line 60)
        t_428928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 69), 't', False)
        # Getting the type of 'itmax' (line 60)
        itmax_428929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 72), 'itmax', False)
        # Processing the call keyword arguments (line 60)
        kwargs_428930 = {}
        # Getting the type of '_onenormest_core' (line 60)
        _onenormest_core_428924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 44), '_onenormest_core', False)
        # Calling _onenormest_core(args, kwargs) (line 60)
        _onenormest_core_call_result_428931 = invoke(stypy.reporting.localization.Localization(__file__, 60, 44), _onenormest_core_428924, *[A_428925, T_428927, t_428928, itmax_428929], **kwargs_428930)
        
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___428932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 12), _onenormest_core_call_result_428931, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 60)
        subscript_call_result_428933 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), getitem___428932, int_428923)
        
        # Assigning a type to the variable 'tuple_var_assignment_428700' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'tuple_var_assignment_428700', subscript_call_result_428933)
        
        # Assigning a Name to a Name (line 60):
        # Getting the type of 'tuple_var_assignment_428696' (line 60)
        tuple_var_assignment_428696_428934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'tuple_var_assignment_428696')
        # Assigning a type to the variable 'est' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'est', tuple_var_assignment_428696_428934)
        
        # Assigning a Name to a Name (line 60):
        # Getting the type of 'tuple_var_assignment_428697' (line 60)
        tuple_var_assignment_428697_428935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'tuple_var_assignment_428697')
        # Assigning a type to the variable 'v' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 17), 'v', tuple_var_assignment_428697_428935)
        
        # Assigning a Name to a Name (line 60):
        # Getting the type of 'tuple_var_assignment_428698' (line 60)
        tuple_var_assignment_428698_428936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'tuple_var_assignment_428698')
        # Assigning a type to the variable 'w' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'w', tuple_var_assignment_428698_428936)
        
        # Assigning a Name to a Name (line 60):
        # Getting the type of 'tuple_var_assignment_428699' (line 60)
        tuple_var_assignment_428699_428937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'tuple_var_assignment_428699')
        # Assigning a type to the variable 'nmults' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'nmults', tuple_var_assignment_428699_428937)
        
        # Assigning a Name to a Name (line 60):
        # Getting the type of 'tuple_var_assignment_428700' (line 60)
        tuple_var_assignment_428700_428938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'tuple_var_assignment_428700')
        # Assigning a type to the variable 'nresamples' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 31), 'nresamples', tuple_var_assignment_428700_428938)
        
        # Call to append(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'est' (line 61)
        est_428941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 28), 'est', False)
        # Processing the call keyword arguments (line 61)
        kwargs_428942 = {}
        # Getting the type of 'observed' (line 61)
        observed_428939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'observed', False)
        # Obtaining the member 'append' of a type (line 61)
        append_428940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 12), observed_428939, 'append')
        # Calling append(args, kwargs) (line 61)
        append_call_result_428943 = invoke(stypy.reporting.localization.Localization(__file__, 61, 12), append_428940, *[est_428941], **kwargs_428942)
        
        
        # Call to append(...): (line 62)
        # Processing the call arguments (line 62)
        
        # Call to norm(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'A' (line 62)
        A_428949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 46), 'A', False)
        int_428950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 49), 'int')
        # Processing the call keyword arguments (line 62)
        kwargs_428951 = {}
        # Getting the type of 'scipy' (line 62)
        scipy_428946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 28), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 62)
        linalg_428947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 28), scipy_428946, 'linalg')
        # Obtaining the member 'norm' of a type (line 62)
        norm_428948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 28), linalg_428947, 'norm')
        # Calling norm(args, kwargs) (line 62)
        norm_call_result_428952 = invoke(stypy.reporting.localization.Localization(__file__, 62, 28), norm_428948, *[A_428949, int_428950], **kwargs_428951)
        
        # Processing the call keyword arguments (line 62)
        kwargs_428953 = {}
        # Getting the type of 'expected' (line 62)
        expected_428944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'expected', False)
        # Obtaining the member 'append' of a type (line 62)
        append_428945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 12), expected_428944, 'append')
        # Calling append(args, kwargs) (line 62)
        append_call_result_428954 = invoke(stypy.reporting.localization.Localization(__file__, 62, 12), append_428945, *[norm_call_result_428952], **kwargs_428953)
        
        
        # Call to append(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'nmults' (line 63)
        nmults_428957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 30), 'nmults', False)
        # Processing the call keyword arguments (line 63)
        kwargs_428958 = {}
        # Getting the type of 'nmult_list' (line 63)
        nmult_list_428955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'nmult_list', False)
        # Obtaining the member 'append' of a type (line 63)
        append_428956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 12), nmult_list_428955, 'append')
        # Calling append(args, kwargs) (line 63)
        append_call_result_428959 = invoke(stypy.reporting.localization.Localization(__file__, 63, 12), append_428956, *[nmults_428957], **kwargs_428958)
        
        
        # Call to append(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'nresamples' (line 64)
        nresamples_428962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 34), 'nresamples', False)
        # Processing the call keyword arguments (line 64)
        kwargs_428963 = {}
        # Getting the type of 'nresample_list' (line 64)
        nresample_list_428960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'nresample_list', False)
        # Obtaining the member 'append' of a type (line 64)
        append_428961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 12), nresample_list_428960, 'append')
        # Calling append(args, kwargs) (line 64)
        append_call_result_428964 = invoke(stypy.reporting.localization.Localization(__file__, 64, 12), append_428961, *[nresamples_428962], **kwargs_428963)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 65):
        
        # Assigning a Call to a Name (line 65):
        
        # Call to array(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'observed' (line 65)
        observed_428967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 28), 'observed', False)
        # Processing the call keyword arguments (line 65)
        # Getting the type of 'float' (line 65)
        float_428968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 44), 'float', False)
        keyword_428969 = float_428968
        kwargs_428970 = {'dtype': keyword_428969}
        # Getting the type of 'np' (line 65)
        np_428965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 65)
        array_428966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 19), np_428965, 'array')
        # Calling array(args, kwargs) (line 65)
        array_call_result_428971 = invoke(stypy.reporting.localization.Localization(__file__, 65, 19), array_428966, *[observed_428967], **kwargs_428970)
        
        # Assigning a type to the variable 'observed' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'observed', array_call_result_428971)
        
        # Assigning a Call to a Name (line 66):
        
        # Assigning a Call to a Name (line 66):
        
        # Call to array(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'expected' (line 66)
        expected_428974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 28), 'expected', False)
        # Processing the call keyword arguments (line 66)
        # Getting the type of 'float' (line 66)
        float_428975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 44), 'float', False)
        keyword_428976 = float_428975
        kwargs_428977 = {'dtype': keyword_428976}
        # Getting the type of 'np' (line 66)
        np_428972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 66)
        array_428973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 19), np_428972, 'array')
        # Calling array(args, kwargs) (line 66)
        array_call_result_428978 = invoke(stypy.reporting.localization.Localization(__file__, 66, 19), array_428973, *[expected_428974], **kwargs_428977)
        
        # Assigning a type to the variable 'expected' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'expected', array_call_result_428978)
        
        # Assigning a BinOp to a Name (line 67):
        
        # Assigning a BinOp to a Name (line 67):
        
        # Call to abs(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'observed' (line 67)
        observed_428981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 33), 'observed', False)
        # Getting the type of 'expected' (line 67)
        expected_428982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 44), 'expected', False)
        # Applying the binary operator '-' (line 67)
        result_sub_428983 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 33), '-', observed_428981, expected_428982)
        
        # Processing the call keyword arguments (line 67)
        kwargs_428984 = {}
        # Getting the type of 'np' (line 67)
        np_428979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 26), 'np', False)
        # Obtaining the member 'abs' of a type (line 67)
        abs_428980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 26), np_428979, 'abs')
        # Calling abs(args, kwargs) (line 67)
        abs_call_result_428985 = invoke(stypy.reporting.localization.Localization(__file__, 67, 26), abs_428980, *[result_sub_428983], **kwargs_428984)
        
        # Getting the type of 'expected' (line 67)
        expected_428986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 56), 'expected')
        # Applying the binary operator 'div' (line 67)
        result_div_428987 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 26), 'div', abs_call_result_428985, expected_428986)
        
        # Assigning a type to the variable 'relative_errors' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'relative_errors', result_div_428987)
        
        # Assigning a BinOp to a Name (line 70):
        
        # Assigning a BinOp to a Name (line 70):
        # Getting the type of 'observed' (line 70)
        observed_428988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 32), 'observed')
        # Getting the type of 'expected' (line 70)
        expected_428989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 43), 'expected')
        # Applying the binary operator 'div' (line 70)
        result_div_428990 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 32), 'div', observed_428988, expected_428989)
        
        # Assigning a type to the variable 'underestimation_ratio' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'underestimation_ratio', result_div_428990)
        
        # Call to assert_(...): (line 71)
        # Processing the call arguments (line 71)
        
        float_428992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 16), 'float')
        
        # Call to mean(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'underestimation_ratio' (line 71)
        underestimation_ratio_428995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 31), 'underestimation_ratio', False)
        # Processing the call keyword arguments (line 71)
        kwargs_428996 = {}
        # Getting the type of 'np' (line 71)
        np_428993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 23), 'np', False)
        # Obtaining the member 'mean' of a type (line 71)
        mean_428994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 23), np_428993, 'mean')
        # Calling mean(args, kwargs) (line 71)
        mean_call_result_428997 = invoke(stypy.reporting.localization.Localization(__file__, 71, 23), mean_428994, *[underestimation_ratio_428995], **kwargs_428996)
        
        # Applying the binary operator '<' (line 71)
        result_lt_428998 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 16), '<', float_428992, mean_call_result_428997)
        float_428999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 56), 'float')
        # Applying the binary operator '<' (line 71)
        result_lt_429000 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 16), '<', mean_call_result_428997, float_428999)
        # Applying the binary operator '&' (line 71)
        result_and__429001 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 16), '&', result_lt_428998, result_lt_429000)
        
        # Processing the call keyword arguments (line 71)
        kwargs_429002 = {}
        # Getting the type of 'assert_' (line 71)
        assert__428991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 71)
        assert__call_result_429003 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), assert__428991, *[result_and__429001], **kwargs_429002)
        
        
        # Call to assert_equal(...): (line 74)
        # Processing the call arguments (line 74)
        
        # Call to max(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'nresample_list' (line 74)
        nresample_list_429007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'nresample_list', False)
        # Processing the call keyword arguments (line 74)
        kwargs_429008 = {}
        # Getting the type of 'np' (line 74)
        np_429005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 21), 'np', False)
        # Obtaining the member 'max' of a type (line 74)
        max_429006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 21), np_429005, 'max')
        # Calling max(args, kwargs) (line 74)
        max_call_result_429009 = invoke(stypy.reporting.localization.Localization(__file__, 74, 21), max_429006, *[nresample_list_429007], **kwargs_429008)
        
        int_429010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 45), 'int')
        # Processing the call keyword arguments (line 74)
        kwargs_429011 = {}
        # Getting the type of 'assert_equal' (line 74)
        assert_equal_429004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 74)
        assert_equal_call_result_429012 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), assert_equal_429004, *[max_call_result_429009, int_429010], **kwargs_429011)
        
        
        # Call to assert_(...): (line 75)
        # Processing the call arguments (line 75)
        
        float_429014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 16), 'float')
        
        # Call to mean(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'nresample_list' (line 75)
        nresample_list_429017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 31), 'nresample_list', False)
        # Processing the call keyword arguments (line 75)
        kwargs_429018 = {}
        # Getting the type of 'np' (line 75)
        np_429015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 23), 'np', False)
        # Obtaining the member 'mean' of a type (line 75)
        mean_429016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 23), np_429015, 'mean')
        # Calling mean(args, kwargs) (line 75)
        mean_call_result_429019 = invoke(stypy.reporting.localization.Localization(__file__, 75, 23), mean_429016, *[nresample_list_429017], **kwargs_429018)
        
        # Applying the binary operator '<' (line 75)
        result_lt_429020 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 16), '<', float_429014, mean_call_result_429019)
        float_429021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 49), 'float')
        # Applying the binary operator '<' (line 75)
        result_lt_429022 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 16), '<', mean_call_result_429019, float_429021)
        # Applying the binary operator '&' (line 75)
        result_and__429023 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 16), '&', result_lt_429020, result_lt_429022)
        
        # Processing the call keyword arguments (line 75)
        kwargs_429024 = {}
        # Getting the type of 'assert_' (line 75)
        assert__429013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 75)
        assert__call_result_429025 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), assert__429013, *[result_and__429023], **kwargs_429024)
        
        
        # Assigning a Call to a Name (line 78):
        
        # Assigning a Call to a Name (line 78):
        
        # Call to count_nonzero(...): (line 78)
        # Processing the call arguments (line 78)
        
        # Getting the type of 'relative_errors' (line 78)
        relative_errors_429028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 34), 'relative_errors', False)
        float_429029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 52), 'float')
        # Applying the binary operator '<' (line 78)
        result_lt_429030 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 34), '<', relative_errors_429028, float_429029)
        
        # Processing the call keyword arguments (line 78)
        kwargs_429031 = {}
        # Getting the type of 'np' (line 78)
        np_429026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 17), 'np', False)
        # Obtaining the member 'count_nonzero' of a type (line 78)
        count_nonzero_429027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 17), np_429026, 'count_nonzero')
        # Calling count_nonzero(args, kwargs) (line 78)
        count_nonzero_call_result_429032 = invoke(stypy.reporting.localization.Localization(__file__, 78, 17), count_nonzero_429027, *[result_lt_429030], **kwargs_429031)
        
        # Assigning a type to the variable 'nexact' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'nexact', count_nonzero_call_result_429032)
        
        # Assigning a BinOp to a Name (line 79):
        
        # Assigning a BinOp to a Name (line 79):
        # Getting the type of 'nexact' (line 79)
        nexact_429033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 27), 'nexact')
        
        # Call to float(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'nsamples' (line 79)
        nsamples_429035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 42), 'nsamples', False)
        # Processing the call keyword arguments (line 79)
        kwargs_429036 = {}
        # Getting the type of 'float' (line 79)
        float_429034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 36), 'float', False)
        # Calling float(args, kwargs) (line 79)
        float_call_result_429037 = invoke(stypy.reporting.localization.Localization(__file__, 79, 36), float_429034, *[nsamples_429035], **kwargs_429036)
        
        # Applying the binary operator 'div' (line 79)
        result_div_429038 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 27), 'div', nexact_429033, float_call_result_429037)
        
        # Assigning a type to the variable 'proportion_exact' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'proportion_exact', result_div_429038)
        
        # Call to assert_(...): (line 80)
        # Processing the call arguments (line 80)
        
        float_429040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 16), 'float')
        # Getting the type of 'proportion_exact' (line 80)
        proportion_exact_429041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 22), 'proportion_exact', False)
        # Applying the binary operator '<' (line 80)
        result_lt_429042 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 16), '<', float_429040, proportion_exact_429041)
        float_429043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 41), 'float')
        # Applying the binary operator '<' (line 80)
        result_lt_429044 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 16), '<', proportion_exact_429041, float_429043)
        # Applying the binary operator '&' (line 80)
        result_and__429045 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 16), '&', result_lt_429042, result_lt_429044)
        
        # Processing the call keyword arguments (line 80)
        kwargs_429046 = {}
        # Getting the type of 'assert_' (line 80)
        assert__429039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 80)
        assert__call_result_429047 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), assert__429039, *[result_and__429045], **kwargs_429046)
        
        
        # Call to assert_(...): (line 83)
        # Processing the call arguments (line 83)
        
        float_429049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 16), 'float')
        
        # Call to mean(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'nmult_list' (line 83)
        nmult_list_429052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 30), 'nmult_list', False)
        # Processing the call keyword arguments (line 83)
        kwargs_429053 = {}
        # Getting the type of 'np' (line 83)
        np_429050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 22), 'np', False)
        # Obtaining the member 'mean' of a type (line 83)
        mean_429051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 22), np_429050, 'mean')
        # Calling mean(args, kwargs) (line 83)
        mean_call_result_429054 = invoke(stypy.reporting.localization.Localization(__file__, 83, 22), mean_429051, *[nmult_list_429052], **kwargs_429053)
        
        # Applying the binary operator '<' (line 83)
        result_lt_429055 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 16), '<', float_429049, mean_call_result_429054)
        float_429056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 44), 'float')
        # Applying the binary operator '<' (line 83)
        result_lt_429057 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 16), '<', mean_call_result_429054, float_429056)
        # Applying the binary operator '&' (line 83)
        result_and__429058 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 16), '&', result_lt_429055, result_lt_429057)
        
        # Processing the call keyword arguments (line 83)
        kwargs_429059 = {}
        # Getting the type of 'assert_' (line 83)
        assert__429048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 83)
        assert__call_result_429060 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), assert__429048, *[result_and__429058], **kwargs_429059)
        
        
        # ################# End of 'test_onenormest_table_3_t_2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_onenormest_table_3_t_2' in the type store
        # Getting the type of 'stypy_return_type' (line 45)
        stypy_return_type_429061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_429061)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_onenormest_table_3_t_2'
        return stypy_return_type_429061


    @norecursion
    def test_onenormest_table_4_t_7(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_onenormest_table_4_t_7'
        module_type_store = module_type_store.open_function_context('test_onenormest_table_4_t_7', 85, 4, False)
        # Assigning a type to the variable 'self' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOnenormest.test_onenormest_table_4_t_7.__dict__.__setitem__('stypy_localization', localization)
        TestOnenormest.test_onenormest_table_4_t_7.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOnenormest.test_onenormest_table_4_t_7.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOnenormest.test_onenormest_table_4_t_7.__dict__.__setitem__('stypy_function_name', 'TestOnenormest.test_onenormest_table_4_t_7')
        TestOnenormest.test_onenormest_table_4_t_7.__dict__.__setitem__('stypy_param_names_list', [])
        TestOnenormest.test_onenormest_table_4_t_7.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOnenormest.test_onenormest_table_4_t_7.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOnenormest.test_onenormest_table_4_t_7.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOnenormest.test_onenormest_table_4_t_7.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOnenormest.test_onenormest_table_4_t_7.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOnenormest.test_onenormest_table_4_t_7.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOnenormest.test_onenormest_table_4_t_7', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_onenormest_table_4_t_7', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_onenormest_table_4_t_7(...)' code ##################

        
        # Call to seed(...): (line 89)
        # Processing the call arguments (line 89)
        int_429065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 23), 'int')
        # Processing the call keyword arguments (line 89)
        kwargs_429066 = {}
        # Getting the type of 'np' (line 89)
        np_429062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 89)
        random_429063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), np_429062, 'random')
        # Obtaining the member 'seed' of a type (line 89)
        seed_429064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), random_429063, 'seed')
        # Calling seed(args, kwargs) (line 89)
        seed_call_result_429067 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), seed_429064, *[int_429065], **kwargs_429066)
        
        
        # Assigning a Num to a Name (line 90):
        
        # Assigning a Num to a Name (line 90):
        int_429068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 12), 'int')
        # Assigning a type to the variable 't' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 't', int_429068)
        
        # Assigning a Num to a Name (line 91):
        
        # Assigning a Num to a Name (line 91):
        int_429069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 12), 'int')
        # Assigning a type to the variable 'n' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'n', int_429069)
        
        # Assigning a Num to a Name (line 92):
        
        # Assigning a Num to a Name (line 92):
        int_429070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 16), 'int')
        # Assigning a type to the variable 'itmax' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'itmax', int_429070)
        
        # Assigning a Num to a Name (line 93):
        
        # Assigning a Num to a Name (line 93):
        int_429071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 19), 'int')
        # Assigning a type to the variable 'nsamples' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'nsamples', int_429071)
        
        # Assigning a List to a Name (line 94):
        
        # Assigning a List to a Name (line 94):
        
        # Obtaining an instance of the builtin type 'list' (line 94)
        list_429072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 94)
        
        # Assigning a type to the variable 'observed' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'observed', list_429072)
        
        # Assigning a List to a Name (line 95):
        
        # Assigning a List to a Name (line 95):
        
        # Obtaining an instance of the builtin type 'list' (line 95)
        list_429073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 95)
        
        # Assigning a type to the variable 'expected' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'expected', list_429073)
        
        # Assigning a List to a Name (line 96):
        
        # Assigning a List to a Name (line 96):
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_429074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        
        # Assigning a type to the variable 'nmult_list' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'nmult_list', list_429074)
        
        # Assigning a List to a Name (line 97):
        
        # Assigning a List to a Name (line 97):
        
        # Obtaining an instance of the builtin type 'list' (line 97)
        list_429075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 97)
        
        # Assigning a type to the variable 'nresample_list' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'nresample_list', list_429075)
        
        
        # Call to range(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'nsamples' (line 98)
        nsamples_429077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 23), 'nsamples', False)
        # Processing the call keyword arguments (line 98)
        kwargs_429078 = {}
        # Getting the type of 'range' (line 98)
        range_429076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 17), 'range', False)
        # Calling range(args, kwargs) (line 98)
        range_call_result_429079 = invoke(stypy.reporting.localization.Localization(__file__, 98, 17), range_429076, *[nsamples_429077], **kwargs_429078)
        
        # Testing the type of a for loop iterable (line 98)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 98, 8), range_call_result_429079)
        # Getting the type of the for loop variable (line 98)
        for_loop_var_429080 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 98, 8), range_call_result_429079)
        # Assigning a type to the variable 'i' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'i', for_loop_var_429080)
        # SSA begins for a for statement (line 98)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 99):
        
        # Assigning a Call to a Name (line 99):
        
        # Call to randint(...): (line 99)
        # Processing the call arguments (line 99)
        int_429084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 34), 'int')
        int_429085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 38), 'int')
        # Processing the call keyword arguments (line 99)
        
        # Obtaining an instance of the builtin type 'tuple' (line 99)
        tuple_429086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 99)
        # Adding element type (line 99)
        # Getting the type of 'n' (line 99)
        n_429087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 47), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 47), tuple_429086, n_429087)
        # Adding element type (line 99)
        # Getting the type of 'n' (line 99)
        n_429088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 50), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 47), tuple_429086, n_429088)
        
        keyword_429089 = tuple_429086
        kwargs_429090 = {'size': keyword_429089}
        # Getting the type of 'np' (line 99)
        np_429081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'np', False)
        # Obtaining the member 'random' of a type (line 99)
        random_429082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), np_429081, 'random')
        # Obtaining the member 'randint' of a type (line 99)
        randint_429083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), random_429082, 'randint')
        # Calling randint(args, kwargs) (line 99)
        randint_call_result_429091 = invoke(stypy.reporting.localization.Localization(__file__, 99, 16), randint_429083, *[int_429084, int_429085], **kwargs_429090)
        
        # Assigning a type to the variable 'A' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'A', randint_call_result_429091)
        
        # Assigning a Call to a Tuple (line 100):
        
        # Assigning a Subscript to a Name (line 100):
        
        # Obtaining the type of the subscript
        int_429092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 12), 'int')
        
        # Call to _onenormest_core(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'A' (line 100)
        A_429094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 61), 'A', False)
        # Getting the type of 'A' (line 100)
        A_429095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 64), 'A', False)
        # Obtaining the member 'T' of a type (line 100)
        T_429096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 64), A_429095, 'T')
        # Getting the type of 't' (line 100)
        t_429097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 69), 't', False)
        # Getting the type of 'itmax' (line 100)
        itmax_429098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 72), 'itmax', False)
        # Processing the call keyword arguments (line 100)
        kwargs_429099 = {}
        # Getting the type of '_onenormest_core' (line 100)
        _onenormest_core_429093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 44), '_onenormest_core', False)
        # Calling _onenormest_core(args, kwargs) (line 100)
        _onenormest_core_call_result_429100 = invoke(stypy.reporting.localization.Localization(__file__, 100, 44), _onenormest_core_429093, *[A_429094, T_429096, t_429097, itmax_429098], **kwargs_429099)
        
        # Obtaining the member '__getitem__' of a type (line 100)
        getitem___429101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), _onenormest_core_call_result_429100, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 100)
        subscript_call_result_429102 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), getitem___429101, int_429092)
        
        # Assigning a type to the variable 'tuple_var_assignment_428701' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'tuple_var_assignment_428701', subscript_call_result_429102)
        
        # Assigning a Subscript to a Name (line 100):
        
        # Obtaining the type of the subscript
        int_429103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 12), 'int')
        
        # Call to _onenormest_core(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'A' (line 100)
        A_429105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 61), 'A', False)
        # Getting the type of 'A' (line 100)
        A_429106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 64), 'A', False)
        # Obtaining the member 'T' of a type (line 100)
        T_429107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 64), A_429106, 'T')
        # Getting the type of 't' (line 100)
        t_429108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 69), 't', False)
        # Getting the type of 'itmax' (line 100)
        itmax_429109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 72), 'itmax', False)
        # Processing the call keyword arguments (line 100)
        kwargs_429110 = {}
        # Getting the type of '_onenormest_core' (line 100)
        _onenormest_core_429104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 44), '_onenormest_core', False)
        # Calling _onenormest_core(args, kwargs) (line 100)
        _onenormest_core_call_result_429111 = invoke(stypy.reporting.localization.Localization(__file__, 100, 44), _onenormest_core_429104, *[A_429105, T_429107, t_429108, itmax_429109], **kwargs_429110)
        
        # Obtaining the member '__getitem__' of a type (line 100)
        getitem___429112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), _onenormest_core_call_result_429111, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 100)
        subscript_call_result_429113 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), getitem___429112, int_429103)
        
        # Assigning a type to the variable 'tuple_var_assignment_428702' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'tuple_var_assignment_428702', subscript_call_result_429113)
        
        # Assigning a Subscript to a Name (line 100):
        
        # Obtaining the type of the subscript
        int_429114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 12), 'int')
        
        # Call to _onenormest_core(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'A' (line 100)
        A_429116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 61), 'A', False)
        # Getting the type of 'A' (line 100)
        A_429117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 64), 'A', False)
        # Obtaining the member 'T' of a type (line 100)
        T_429118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 64), A_429117, 'T')
        # Getting the type of 't' (line 100)
        t_429119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 69), 't', False)
        # Getting the type of 'itmax' (line 100)
        itmax_429120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 72), 'itmax', False)
        # Processing the call keyword arguments (line 100)
        kwargs_429121 = {}
        # Getting the type of '_onenormest_core' (line 100)
        _onenormest_core_429115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 44), '_onenormest_core', False)
        # Calling _onenormest_core(args, kwargs) (line 100)
        _onenormest_core_call_result_429122 = invoke(stypy.reporting.localization.Localization(__file__, 100, 44), _onenormest_core_429115, *[A_429116, T_429118, t_429119, itmax_429120], **kwargs_429121)
        
        # Obtaining the member '__getitem__' of a type (line 100)
        getitem___429123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), _onenormest_core_call_result_429122, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 100)
        subscript_call_result_429124 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), getitem___429123, int_429114)
        
        # Assigning a type to the variable 'tuple_var_assignment_428703' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'tuple_var_assignment_428703', subscript_call_result_429124)
        
        # Assigning a Subscript to a Name (line 100):
        
        # Obtaining the type of the subscript
        int_429125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 12), 'int')
        
        # Call to _onenormest_core(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'A' (line 100)
        A_429127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 61), 'A', False)
        # Getting the type of 'A' (line 100)
        A_429128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 64), 'A', False)
        # Obtaining the member 'T' of a type (line 100)
        T_429129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 64), A_429128, 'T')
        # Getting the type of 't' (line 100)
        t_429130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 69), 't', False)
        # Getting the type of 'itmax' (line 100)
        itmax_429131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 72), 'itmax', False)
        # Processing the call keyword arguments (line 100)
        kwargs_429132 = {}
        # Getting the type of '_onenormest_core' (line 100)
        _onenormest_core_429126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 44), '_onenormest_core', False)
        # Calling _onenormest_core(args, kwargs) (line 100)
        _onenormest_core_call_result_429133 = invoke(stypy.reporting.localization.Localization(__file__, 100, 44), _onenormest_core_429126, *[A_429127, T_429129, t_429130, itmax_429131], **kwargs_429132)
        
        # Obtaining the member '__getitem__' of a type (line 100)
        getitem___429134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), _onenormest_core_call_result_429133, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 100)
        subscript_call_result_429135 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), getitem___429134, int_429125)
        
        # Assigning a type to the variable 'tuple_var_assignment_428704' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'tuple_var_assignment_428704', subscript_call_result_429135)
        
        # Assigning a Subscript to a Name (line 100):
        
        # Obtaining the type of the subscript
        int_429136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 12), 'int')
        
        # Call to _onenormest_core(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'A' (line 100)
        A_429138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 61), 'A', False)
        # Getting the type of 'A' (line 100)
        A_429139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 64), 'A', False)
        # Obtaining the member 'T' of a type (line 100)
        T_429140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 64), A_429139, 'T')
        # Getting the type of 't' (line 100)
        t_429141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 69), 't', False)
        # Getting the type of 'itmax' (line 100)
        itmax_429142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 72), 'itmax', False)
        # Processing the call keyword arguments (line 100)
        kwargs_429143 = {}
        # Getting the type of '_onenormest_core' (line 100)
        _onenormest_core_429137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 44), '_onenormest_core', False)
        # Calling _onenormest_core(args, kwargs) (line 100)
        _onenormest_core_call_result_429144 = invoke(stypy.reporting.localization.Localization(__file__, 100, 44), _onenormest_core_429137, *[A_429138, T_429140, t_429141, itmax_429142], **kwargs_429143)
        
        # Obtaining the member '__getitem__' of a type (line 100)
        getitem___429145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), _onenormest_core_call_result_429144, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 100)
        subscript_call_result_429146 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), getitem___429145, int_429136)
        
        # Assigning a type to the variable 'tuple_var_assignment_428705' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'tuple_var_assignment_428705', subscript_call_result_429146)
        
        # Assigning a Name to a Name (line 100):
        # Getting the type of 'tuple_var_assignment_428701' (line 100)
        tuple_var_assignment_428701_429147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'tuple_var_assignment_428701')
        # Assigning a type to the variable 'est' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'est', tuple_var_assignment_428701_429147)
        
        # Assigning a Name to a Name (line 100):
        # Getting the type of 'tuple_var_assignment_428702' (line 100)
        tuple_var_assignment_428702_429148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'tuple_var_assignment_428702')
        # Assigning a type to the variable 'v' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 17), 'v', tuple_var_assignment_428702_429148)
        
        # Assigning a Name to a Name (line 100):
        # Getting the type of 'tuple_var_assignment_428703' (line 100)
        tuple_var_assignment_428703_429149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'tuple_var_assignment_428703')
        # Assigning a type to the variable 'w' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 20), 'w', tuple_var_assignment_428703_429149)
        
        # Assigning a Name to a Name (line 100):
        # Getting the type of 'tuple_var_assignment_428704' (line 100)
        tuple_var_assignment_428704_429150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'tuple_var_assignment_428704')
        # Assigning a type to the variable 'nmults' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 23), 'nmults', tuple_var_assignment_428704_429150)
        
        # Assigning a Name to a Name (line 100):
        # Getting the type of 'tuple_var_assignment_428705' (line 100)
        tuple_var_assignment_428705_429151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'tuple_var_assignment_428705')
        # Assigning a type to the variable 'nresamples' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 31), 'nresamples', tuple_var_assignment_428705_429151)
        
        # Call to append(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'est' (line 101)
        est_429154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 28), 'est', False)
        # Processing the call keyword arguments (line 101)
        kwargs_429155 = {}
        # Getting the type of 'observed' (line 101)
        observed_429152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'observed', False)
        # Obtaining the member 'append' of a type (line 101)
        append_429153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 12), observed_429152, 'append')
        # Calling append(args, kwargs) (line 101)
        append_call_result_429156 = invoke(stypy.reporting.localization.Localization(__file__, 101, 12), append_429153, *[est_429154], **kwargs_429155)
        
        
        # Call to append(...): (line 102)
        # Processing the call arguments (line 102)
        
        # Call to norm(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'A' (line 102)
        A_429162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 46), 'A', False)
        int_429163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 49), 'int')
        # Processing the call keyword arguments (line 102)
        kwargs_429164 = {}
        # Getting the type of 'scipy' (line 102)
        scipy_429159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 28), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 102)
        linalg_429160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 28), scipy_429159, 'linalg')
        # Obtaining the member 'norm' of a type (line 102)
        norm_429161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 28), linalg_429160, 'norm')
        # Calling norm(args, kwargs) (line 102)
        norm_call_result_429165 = invoke(stypy.reporting.localization.Localization(__file__, 102, 28), norm_429161, *[A_429162, int_429163], **kwargs_429164)
        
        # Processing the call keyword arguments (line 102)
        kwargs_429166 = {}
        # Getting the type of 'expected' (line 102)
        expected_429157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'expected', False)
        # Obtaining the member 'append' of a type (line 102)
        append_429158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), expected_429157, 'append')
        # Calling append(args, kwargs) (line 102)
        append_call_result_429167 = invoke(stypy.reporting.localization.Localization(__file__, 102, 12), append_429158, *[norm_call_result_429165], **kwargs_429166)
        
        
        # Call to append(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'nmults' (line 103)
        nmults_429170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 30), 'nmults', False)
        # Processing the call keyword arguments (line 103)
        kwargs_429171 = {}
        # Getting the type of 'nmult_list' (line 103)
        nmult_list_429168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'nmult_list', False)
        # Obtaining the member 'append' of a type (line 103)
        append_429169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), nmult_list_429168, 'append')
        # Calling append(args, kwargs) (line 103)
        append_call_result_429172 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), append_429169, *[nmults_429170], **kwargs_429171)
        
        
        # Call to append(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'nresamples' (line 104)
        nresamples_429175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 34), 'nresamples', False)
        # Processing the call keyword arguments (line 104)
        kwargs_429176 = {}
        # Getting the type of 'nresample_list' (line 104)
        nresample_list_429173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'nresample_list', False)
        # Obtaining the member 'append' of a type (line 104)
        append_429174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 12), nresample_list_429173, 'append')
        # Calling append(args, kwargs) (line 104)
        append_call_result_429177 = invoke(stypy.reporting.localization.Localization(__file__, 104, 12), append_429174, *[nresamples_429175], **kwargs_429176)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 105):
        
        # Assigning a Call to a Name (line 105):
        
        # Call to array(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'observed' (line 105)
        observed_429180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 28), 'observed', False)
        # Processing the call keyword arguments (line 105)
        # Getting the type of 'float' (line 105)
        float_429181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 44), 'float', False)
        keyword_429182 = float_429181
        kwargs_429183 = {'dtype': keyword_429182}
        # Getting the type of 'np' (line 105)
        np_429178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 105)
        array_429179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 19), np_429178, 'array')
        # Calling array(args, kwargs) (line 105)
        array_call_result_429184 = invoke(stypy.reporting.localization.Localization(__file__, 105, 19), array_429179, *[observed_429180], **kwargs_429183)
        
        # Assigning a type to the variable 'observed' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'observed', array_call_result_429184)
        
        # Assigning a Call to a Name (line 106):
        
        # Assigning a Call to a Name (line 106):
        
        # Call to array(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'expected' (line 106)
        expected_429187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 28), 'expected', False)
        # Processing the call keyword arguments (line 106)
        # Getting the type of 'float' (line 106)
        float_429188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 44), 'float', False)
        keyword_429189 = float_429188
        kwargs_429190 = {'dtype': keyword_429189}
        # Getting the type of 'np' (line 106)
        np_429185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 106)
        array_429186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 19), np_429185, 'array')
        # Calling array(args, kwargs) (line 106)
        array_call_result_429191 = invoke(stypy.reporting.localization.Localization(__file__, 106, 19), array_429186, *[expected_429187], **kwargs_429190)
        
        # Assigning a type to the variable 'expected' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'expected', array_call_result_429191)
        
        # Assigning a BinOp to a Name (line 107):
        
        # Assigning a BinOp to a Name (line 107):
        
        # Call to abs(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'observed' (line 107)
        observed_429194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 33), 'observed', False)
        # Getting the type of 'expected' (line 107)
        expected_429195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 44), 'expected', False)
        # Applying the binary operator '-' (line 107)
        result_sub_429196 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 33), '-', observed_429194, expected_429195)
        
        # Processing the call keyword arguments (line 107)
        kwargs_429197 = {}
        # Getting the type of 'np' (line 107)
        np_429192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 26), 'np', False)
        # Obtaining the member 'abs' of a type (line 107)
        abs_429193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 26), np_429192, 'abs')
        # Calling abs(args, kwargs) (line 107)
        abs_call_result_429198 = invoke(stypy.reporting.localization.Localization(__file__, 107, 26), abs_429193, *[result_sub_429196], **kwargs_429197)
        
        # Getting the type of 'expected' (line 107)
        expected_429199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 56), 'expected')
        # Applying the binary operator 'div' (line 107)
        result_div_429200 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 26), 'div', abs_call_result_429198, expected_429199)
        
        # Assigning a type to the variable 'relative_errors' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'relative_errors', result_div_429200)
        
        # Assigning a BinOp to a Name (line 110):
        
        # Assigning a BinOp to a Name (line 110):
        # Getting the type of 'observed' (line 110)
        observed_429201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 32), 'observed')
        # Getting the type of 'expected' (line 110)
        expected_429202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 43), 'expected')
        # Applying the binary operator 'div' (line 110)
        result_div_429203 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 32), 'div', observed_429201, expected_429202)
        
        # Assigning a type to the variable 'underestimation_ratio' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'underestimation_ratio', result_div_429203)
        
        # Call to assert_(...): (line 111)
        # Processing the call arguments (line 111)
        
        float_429205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 16), 'float')
        
        # Call to mean(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'underestimation_ratio' (line 111)
        underestimation_ratio_429208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 31), 'underestimation_ratio', False)
        # Processing the call keyword arguments (line 111)
        kwargs_429209 = {}
        # Getting the type of 'np' (line 111)
        np_429206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 23), 'np', False)
        # Obtaining the member 'mean' of a type (line 111)
        mean_429207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 23), np_429206, 'mean')
        # Calling mean(args, kwargs) (line 111)
        mean_call_result_429210 = invoke(stypy.reporting.localization.Localization(__file__, 111, 23), mean_429207, *[underestimation_ratio_429208], **kwargs_429209)
        
        # Applying the binary operator '<' (line 111)
        result_lt_429211 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 16), '<', float_429205, mean_call_result_429210)
        float_429212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 56), 'float')
        # Applying the binary operator '<' (line 111)
        result_lt_429213 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 16), '<', mean_call_result_429210, float_429212)
        # Applying the binary operator '&' (line 111)
        result_and__429214 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 16), '&', result_lt_429211, result_lt_429213)
        
        # Processing the call keyword arguments (line 111)
        kwargs_429215 = {}
        # Getting the type of 'assert_' (line 111)
        assert__429204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 111)
        assert__call_result_429216 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), assert__429204, *[result_and__429214], **kwargs_429215)
        
        
        # Call to assert_equal(...): (line 114)
        # Processing the call arguments (line 114)
        
        # Call to max(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'nresample_list' (line 114)
        nresample_list_429220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 28), 'nresample_list', False)
        # Processing the call keyword arguments (line 114)
        kwargs_429221 = {}
        # Getting the type of 'np' (line 114)
        np_429218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 21), 'np', False)
        # Obtaining the member 'max' of a type (line 114)
        max_429219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 21), np_429218, 'max')
        # Calling max(args, kwargs) (line 114)
        max_call_result_429222 = invoke(stypy.reporting.localization.Localization(__file__, 114, 21), max_429219, *[nresample_list_429220], **kwargs_429221)
        
        int_429223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 45), 'int')
        # Processing the call keyword arguments (line 114)
        kwargs_429224 = {}
        # Getting the type of 'assert_equal' (line 114)
        assert_equal_429217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 114)
        assert_equal_call_result_429225 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), assert_equal_429217, *[max_call_result_429222, int_429223], **kwargs_429224)
        
        
        # Assigning a Call to a Name (line 117):
        
        # Assigning a Call to a Name (line 117):
        
        # Call to count_nonzero(...): (line 117)
        # Processing the call arguments (line 117)
        
        # Getting the type of 'relative_errors' (line 117)
        relative_errors_429228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 34), 'relative_errors', False)
        float_429229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 52), 'float')
        # Applying the binary operator '<' (line 117)
        result_lt_429230 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 34), '<', relative_errors_429228, float_429229)
        
        # Processing the call keyword arguments (line 117)
        kwargs_429231 = {}
        # Getting the type of 'np' (line 117)
        np_429226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 17), 'np', False)
        # Obtaining the member 'count_nonzero' of a type (line 117)
        count_nonzero_429227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 17), np_429226, 'count_nonzero')
        # Calling count_nonzero(args, kwargs) (line 117)
        count_nonzero_call_result_429232 = invoke(stypy.reporting.localization.Localization(__file__, 117, 17), count_nonzero_429227, *[result_lt_429230], **kwargs_429231)
        
        # Assigning a type to the variable 'nexact' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'nexact', count_nonzero_call_result_429232)
        
        # Assigning a BinOp to a Name (line 118):
        
        # Assigning a BinOp to a Name (line 118):
        # Getting the type of 'nexact' (line 118)
        nexact_429233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 27), 'nexact')
        
        # Call to float(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'nsamples' (line 118)
        nsamples_429235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 42), 'nsamples', False)
        # Processing the call keyword arguments (line 118)
        kwargs_429236 = {}
        # Getting the type of 'float' (line 118)
        float_429234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 36), 'float', False)
        # Calling float(args, kwargs) (line 118)
        float_call_result_429237 = invoke(stypy.reporting.localization.Localization(__file__, 118, 36), float_429234, *[nsamples_429235], **kwargs_429236)
        
        # Applying the binary operator 'div' (line 118)
        result_div_429238 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 27), 'div', nexact_429233, float_call_result_429237)
        
        # Assigning a type to the variable 'proportion_exact' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'proportion_exact', result_div_429238)
        
        # Call to assert_(...): (line 119)
        # Processing the call arguments (line 119)
        
        float_429240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 16), 'float')
        # Getting the type of 'proportion_exact' (line 119)
        proportion_exact_429241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 23), 'proportion_exact', False)
        # Applying the binary operator '<' (line 119)
        result_lt_429242 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 16), '<', float_429240, proportion_exact_429241)
        float_429243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 42), 'float')
        # Applying the binary operator '<' (line 119)
        result_lt_429244 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 16), '<', proportion_exact_429241, float_429243)
        # Applying the binary operator '&' (line 119)
        result_and__429245 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 16), '&', result_lt_429242, result_lt_429244)
        
        # Processing the call keyword arguments (line 119)
        kwargs_429246 = {}
        # Getting the type of 'assert_' (line 119)
        assert__429239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 119)
        assert__call_result_429247 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), assert__429239, *[result_and__429245], **kwargs_429246)
        
        
        # Call to assert_(...): (line 122)
        # Processing the call arguments (line 122)
        
        float_429249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 16), 'float')
        
        # Call to mean(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'nmult_list' (line 122)
        nmult_list_429252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 30), 'nmult_list', False)
        # Processing the call keyword arguments (line 122)
        kwargs_429253 = {}
        # Getting the type of 'np' (line 122)
        np_429250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 22), 'np', False)
        # Obtaining the member 'mean' of a type (line 122)
        mean_429251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 22), np_429250, 'mean')
        # Calling mean(args, kwargs) (line 122)
        mean_call_result_429254 = invoke(stypy.reporting.localization.Localization(__file__, 122, 22), mean_429251, *[nmult_list_429252], **kwargs_429253)
        
        # Applying the binary operator '<' (line 122)
        result_lt_429255 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 16), '<', float_429249, mean_call_result_429254)
        float_429256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 44), 'float')
        # Applying the binary operator '<' (line 122)
        result_lt_429257 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 16), '<', mean_call_result_429254, float_429256)
        # Applying the binary operator '&' (line 122)
        result_and__429258 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 16), '&', result_lt_429255, result_lt_429257)
        
        # Processing the call keyword arguments (line 122)
        kwargs_429259 = {}
        # Getting the type of 'assert_' (line 122)
        assert__429248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 122)
        assert__call_result_429260 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), assert__429248, *[result_and__429258], **kwargs_429259)
        
        
        # ################# End of 'test_onenormest_table_4_t_7(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_onenormest_table_4_t_7' in the type store
        # Getting the type of 'stypy_return_type' (line 85)
        stypy_return_type_429261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_429261)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_onenormest_table_4_t_7'
        return stypy_return_type_429261


    @norecursion
    def test_onenormest_table_5_t_1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_onenormest_table_5_t_1'
        module_type_store = module_type_store.open_function_context('test_onenormest_table_5_t_1', 124, 4, False)
        # Assigning a type to the variable 'self' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOnenormest.test_onenormest_table_5_t_1.__dict__.__setitem__('stypy_localization', localization)
        TestOnenormest.test_onenormest_table_5_t_1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOnenormest.test_onenormest_table_5_t_1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOnenormest.test_onenormest_table_5_t_1.__dict__.__setitem__('stypy_function_name', 'TestOnenormest.test_onenormest_table_5_t_1')
        TestOnenormest.test_onenormest_table_5_t_1.__dict__.__setitem__('stypy_param_names_list', [])
        TestOnenormest.test_onenormest_table_5_t_1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOnenormest.test_onenormest_table_5_t_1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOnenormest.test_onenormest_table_5_t_1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOnenormest.test_onenormest_table_5_t_1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOnenormest.test_onenormest_table_5_t_1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOnenormest.test_onenormest_table_5_t_1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOnenormest.test_onenormest_table_5_t_1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_onenormest_table_5_t_1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_onenormest_table_5_t_1(...)' code ##################

        
        # Assigning a Num to a Name (line 126):
        
        # Assigning a Num to a Name (line 126):
        int_429262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 12), 'int')
        # Assigning a type to the variable 't' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 't', int_429262)
        
        # Assigning a Num to a Name (line 127):
        
        # Assigning a Num to a Name (line 127):
        int_429263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 12), 'int')
        # Assigning a type to the variable 'n' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'n', int_429263)
        
        # Assigning a Num to a Name (line 128):
        
        # Assigning a Num to a Name (line 128):
        int_429264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 16), 'int')
        # Assigning a type to the variable 'itmax' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'itmax', int_429264)
        
        # Assigning a BinOp to a Name (line 129):
        
        # Assigning a BinOp to a Name (line 129):
        int_429265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 16), 'int')
        float_429266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 20), 'float')
        # Applying the binary operator '-' (line 129)
        result_sub_429267 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 16), '-', int_429265, float_429266)
        
        # Assigning a type to the variable 'alpha' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'alpha', result_sub_429267)
        
        # Assigning a UnaryOp to a Name (line 130):
        
        # Assigning a UnaryOp to a Name (line 130):
        
        
        # Call to inv(...): (line 130)
        # Processing the call arguments (line 130)
        
        # Call to identity(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'n' (line 130)
        n_429273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 42), 'n', False)
        # Processing the call keyword arguments (line 130)
        kwargs_429274 = {}
        # Getting the type of 'np' (line 130)
        np_429271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 30), 'np', False)
        # Obtaining the member 'identity' of a type (line 130)
        identity_429272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 30), np_429271, 'identity')
        # Calling identity(args, kwargs) (line 130)
        identity_call_result_429275 = invoke(stypy.reporting.localization.Localization(__file__, 130, 30), identity_429272, *[n_429273], **kwargs_429274)
        
        # Getting the type of 'alpha' (line 130)
        alpha_429276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 47), 'alpha', False)
        
        # Call to eye(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'n' (line 130)
        n_429279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 60), 'n', False)
        # Processing the call keyword arguments (line 130)
        int_429280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 65), 'int')
        keyword_429281 = int_429280
        kwargs_429282 = {'k': keyword_429281}
        # Getting the type of 'np' (line 130)
        np_429277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 53), 'np', False)
        # Obtaining the member 'eye' of a type (line 130)
        eye_429278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 53), np_429277, 'eye')
        # Calling eye(args, kwargs) (line 130)
        eye_call_result_429283 = invoke(stypy.reporting.localization.Localization(__file__, 130, 53), eye_429278, *[n_429279], **kwargs_429282)
        
        # Applying the binary operator '*' (line 130)
        result_mul_429284 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 47), '*', alpha_429276, eye_call_result_429283)
        
        # Applying the binary operator '+' (line 130)
        result_add_429285 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 30), '+', identity_call_result_429275, result_mul_429284)
        
        # Processing the call keyword arguments (line 130)
        kwargs_429286 = {}
        # Getting the type of 'scipy' (line 130)
        scipy_429268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 13), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 130)
        linalg_429269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 13), scipy_429268, 'linalg')
        # Obtaining the member 'inv' of a type (line 130)
        inv_429270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 13), linalg_429269, 'inv')
        # Calling inv(args, kwargs) (line 130)
        inv_call_result_429287 = invoke(stypy.reporting.localization.Localization(__file__, 130, 13), inv_429270, *[result_add_429285], **kwargs_429286)
        
        # Applying the 'usub' unary operator (line 130)
        result___neg___429288 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 12), 'usub', inv_call_result_429287)
        
        # Assigning a type to the variable 'A' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'A', result___neg___429288)
        
        # Assigning a Call to a Name (line 131):
        
        # Assigning a Call to a Name (line 131):
        
        # Call to array(...): (line 131)
        # Processing the call arguments (line 131)
        
        # Obtaining an instance of the builtin type 'list' (line 131)
        list_429291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 131)
        # Adding element type (line 131)
        int_429292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 29), list_429291, int_429292)
        
        
        # Obtaining an instance of the builtin type 'list' (line 131)
        list_429293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 131)
        # Adding element type (line 131)
        int_429294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 35), list_429293, int_429294)
        
        # Getting the type of 'n' (line 131)
        n_429295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 40), 'n', False)
        int_429296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 42), 'int')
        # Applying the binary operator '-' (line 131)
        result_sub_429297 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 40), '-', n_429295, int_429296)
        
        # Applying the binary operator '*' (line 131)
        result_mul_429298 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 35), '*', list_429293, result_sub_429297)
        
        # Applying the binary operator '+' (line 131)
        result_add_429299 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 29), '+', list_429291, result_mul_429298)
        
        # Processing the call keyword arguments (line 131)
        kwargs_429300 = {}
        # Getting the type of 'np' (line 131)
        np_429289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 20), 'np', False)
        # Obtaining the member 'array' of a type (line 131)
        array_429290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 20), np_429289, 'array')
        # Calling array(args, kwargs) (line 131)
        array_call_result_429301 = invoke(stypy.reporting.localization.Localization(__file__, 131, 20), array_429290, *[result_add_429299], **kwargs_429300)
        
        # Assigning a type to the variable 'first_col' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'first_col', array_call_result_429301)
        
        # Assigning a Call to a Name (line 132):
        
        # Assigning a Call to a Name (line 132):
        
        # Call to array(...): (line 132)
        # Processing the call arguments (line 132)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'n' (line 132)
        n_429309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 57), 'n', False)
        # Processing the call keyword arguments (line 132)
        kwargs_429310 = {}
        # Getting the type of 'range' (line 132)
        range_429308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 51), 'range', False)
        # Calling range(args, kwargs) (line 132)
        range_call_result_429311 = invoke(stypy.reporting.localization.Localization(__file__, 132, 51), range_429308, *[n_429309], **kwargs_429310)
        
        comprehension_429312 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 30), range_call_result_429311)
        # Assigning a type to the variable 'i' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 30), 'i', comprehension_429312)
        
        # Getting the type of 'alpha' (line 132)
        alpha_429304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 32), 'alpha', False)
        # Applying the 'usub' unary operator (line 132)
        result___neg___429305 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 31), 'usub', alpha_429304)
        
        # Getting the type of 'i' (line 132)
        i_429306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 40), 'i', False)
        # Applying the binary operator '**' (line 132)
        result_pow_429307 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 30), '**', result___neg___429305, i_429306)
        
        list_429313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 30), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 30), list_429313, result_pow_429307)
        # Processing the call keyword arguments (line 132)
        kwargs_429314 = {}
        # Getting the type of 'np' (line 132)
        np_429302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 20), 'np', False)
        # Obtaining the member 'array' of a type (line 132)
        array_429303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 20), np_429302, 'array')
        # Calling array(args, kwargs) (line 132)
        array_call_result_429315 = invoke(stypy.reporting.localization.Localization(__file__, 132, 20), array_429303, *[list_429313], **kwargs_429314)
        
        # Assigning a type to the variable 'first_row' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'first_row', array_call_result_429315)
        
        # Assigning a UnaryOp to a Name (line 133):
        
        # Assigning a UnaryOp to a Name (line 133):
        
        
        # Call to toeplitz(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'first_col' (line 133)
        first_col_429319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 35), 'first_col', False)
        # Getting the type of 'first_row' (line 133)
        first_row_429320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 46), 'first_row', False)
        # Processing the call keyword arguments (line 133)
        kwargs_429321 = {}
        # Getting the type of 'scipy' (line 133)
        scipy_429316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 13), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 133)
        linalg_429317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 13), scipy_429316, 'linalg')
        # Obtaining the member 'toeplitz' of a type (line 133)
        toeplitz_429318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 13), linalg_429317, 'toeplitz')
        # Calling toeplitz(args, kwargs) (line 133)
        toeplitz_call_result_429322 = invoke(stypy.reporting.localization.Localization(__file__, 133, 13), toeplitz_429318, *[first_col_429319, first_row_429320], **kwargs_429321)
        
        # Applying the 'usub' unary operator (line 133)
        result___neg___429323 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 12), 'usub', toeplitz_call_result_429322)
        
        # Assigning a type to the variable 'B' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'B', result___neg___429323)
        
        # Call to assert_allclose(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'A' (line 134)
        A_429325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 24), 'A', False)
        # Getting the type of 'B' (line 134)
        B_429326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 27), 'B', False)
        # Processing the call keyword arguments (line 134)
        kwargs_429327 = {}
        # Getting the type of 'assert_allclose' (line 134)
        assert_allclose_429324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 134)
        assert_allclose_call_result_429328 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), assert_allclose_429324, *[A_429325, B_429326], **kwargs_429327)
        
        
        # Assigning a Call to a Tuple (line 135):
        
        # Assigning a Subscript to a Name (line 135):
        
        # Obtaining the type of the subscript
        int_429329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 8), 'int')
        
        # Call to _onenormest_core(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'B' (line 135)
        B_429331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 57), 'B', False)
        # Getting the type of 'B' (line 135)
        B_429332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 60), 'B', False)
        # Obtaining the member 'T' of a type (line 135)
        T_429333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 60), B_429332, 'T')
        # Getting the type of 't' (line 135)
        t_429334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 65), 't', False)
        # Getting the type of 'itmax' (line 135)
        itmax_429335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 68), 'itmax', False)
        # Processing the call keyword arguments (line 135)
        kwargs_429336 = {}
        # Getting the type of '_onenormest_core' (line 135)
        _onenormest_core_429330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 40), '_onenormest_core', False)
        # Calling _onenormest_core(args, kwargs) (line 135)
        _onenormest_core_call_result_429337 = invoke(stypy.reporting.localization.Localization(__file__, 135, 40), _onenormest_core_429330, *[B_429331, T_429333, t_429334, itmax_429335], **kwargs_429336)
        
        # Obtaining the member '__getitem__' of a type (line 135)
        getitem___429338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), _onenormest_core_call_result_429337, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 135)
        subscript_call_result_429339 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), getitem___429338, int_429329)
        
        # Assigning a type to the variable 'tuple_var_assignment_428706' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'tuple_var_assignment_428706', subscript_call_result_429339)
        
        # Assigning a Subscript to a Name (line 135):
        
        # Obtaining the type of the subscript
        int_429340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 8), 'int')
        
        # Call to _onenormest_core(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'B' (line 135)
        B_429342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 57), 'B', False)
        # Getting the type of 'B' (line 135)
        B_429343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 60), 'B', False)
        # Obtaining the member 'T' of a type (line 135)
        T_429344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 60), B_429343, 'T')
        # Getting the type of 't' (line 135)
        t_429345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 65), 't', False)
        # Getting the type of 'itmax' (line 135)
        itmax_429346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 68), 'itmax', False)
        # Processing the call keyword arguments (line 135)
        kwargs_429347 = {}
        # Getting the type of '_onenormest_core' (line 135)
        _onenormest_core_429341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 40), '_onenormest_core', False)
        # Calling _onenormest_core(args, kwargs) (line 135)
        _onenormest_core_call_result_429348 = invoke(stypy.reporting.localization.Localization(__file__, 135, 40), _onenormest_core_429341, *[B_429342, T_429344, t_429345, itmax_429346], **kwargs_429347)
        
        # Obtaining the member '__getitem__' of a type (line 135)
        getitem___429349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), _onenormest_core_call_result_429348, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 135)
        subscript_call_result_429350 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), getitem___429349, int_429340)
        
        # Assigning a type to the variable 'tuple_var_assignment_428707' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'tuple_var_assignment_428707', subscript_call_result_429350)
        
        # Assigning a Subscript to a Name (line 135):
        
        # Obtaining the type of the subscript
        int_429351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 8), 'int')
        
        # Call to _onenormest_core(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'B' (line 135)
        B_429353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 57), 'B', False)
        # Getting the type of 'B' (line 135)
        B_429354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 60), 'B', False)
        # Obtaining the member 'T' of a type (line 135)
        T_429355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 60), B_429354, 'T')
        # Getting the type of 't' (line 135)
        t_429356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 65), 't', False)
        # Getting the type of 'itmax' (line 135)
        itmax_429357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 68), 'itmax', False)
        # Processing the call keyword arguments (line 135)
        kwargs_429358 = {}
        # Getting the type of '_onenormest_core' (line 135)
        _onenormest_core_429352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 40), '_onenormest_core', False)
        # Calling _onenormest_core(args, kwargs) (line 135)
        _onenormest_core_call_result_429359 = invoke(stypy.reporting.localization.Localization(__file__, 135, 40), _onenormest_core_429352, *[B_429353, T_429355, t_429356, itmax_429357], **kwargs_429358)
        
        # Obtaining the member '__getitem__' of a type (line 135)
        getitem___429360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), _onenormest_core_call_result_429359, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 135)
        subscript_call_result_429361 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), getitem___429360, int_429351)
        
        # Assigning a type to the variable 'tuple_var_assignment_428708' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'tuple_var_assignment_428708', subscript_call_result_429361)
        
        # Assigning a Subscript to a Name (line 135):
        
        # Obtaining the type of the subscript
        int_429362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 8), 'int')
        
        # Call to _onenormest_core(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'B' (line 135)
        B_429364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 57), 'B', False)
        # Getting the type of 'B' (line 135)
        B_429365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 60), 'B', False)
        # Obtaining the member 'T' of a type (line 135)
        T_429366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 60), B_429365, 'T')
        # Getting the type of 't' (line 135)
        t_429367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 65), 't', False)
        # Getting the type of 'itmax' (line 135)
        itmax_429368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 68), 'itmax', False)
        # Processing the call keyword arguments (line 135)
        kwargs_429369 = {}
        # Getting the type of '_onenormest_core' (line 135)
        _onenormest_core_429363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 40), '_onenormest_core', False)
        # Calling _onenormest_core(args, kwargs) (line 135)
        _onenormest_core_call_result_429370 = invoke(stypy.reporting.localization.Localization(__file__, 135, 40), _onenormest_core_429363, *[B_429364, T_429366, t_429367, itmax_429368], **kwargs_429369)
        
        # Obtaining the member '__getitem__' of a type (line 135)
        getitem___429371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), _onenormest_core_call_result_429370, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 135)
        subscript_call_result_429372 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), getitem___429371, int_429362)
        
        # Assigning a type to the variable 'tuple_var_assignment_428709' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'tuple_var_assignment_428709', subscript_call_result_429372)
        
        # Assigning a Subscript to a Name (line 135):
        
        # Obtaining the type of the subscript
        int_429373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 8), 'int')
        
        # Call to _onenormest_core(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'B' (line 135)
        B_429375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 57), 'B', False)
        # Getting the type of 'B' (line 135)
        B_429376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 60), 'B', False)
        # Obtaining the member 'T' of a type (line 135)
        T_429377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 60), B_429376, 'T')
        # Getting the type of 't' (line 135)
        t_429378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 65), 't', False)
        # Getting the type of 'itmax' (line 135)
        itmax_429379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 68), 'itmax', False)
        # Processing the call keyword arguments (line 135)
        kwargs_429380 = {}
        # Getting the type of '_onenormest_core' (line 135)
        _onenormest_core_429374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 40), '_onenormest_core', False)
        # Calling _onenormest_core(args, kwargs) (line 135)
        _onenormest_core_call_result_429381 = invoke(stypy.reporting.localization.Localization(__file__, 135, 40), _onenormest_core_429374, *[B_429375, T_429377, t_429378, itmax_429379], **kwargs_429380)
        
        # Obtaining the member '__getitem__' of a type (line 135)
        getitem___429382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), _onenormest_core_call_result_429381, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 135)
        subscript_call_result_429383 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), getitem___429382, int_429373)
        
        # Assigning a type to the variable 'tuple_var_assignment_428710' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'tuple_var_assignment_428710', subscript_call_result_429383)
        
        # Assigning a Name to a Name (line 135):
        # Getting the type of 'tuple_var_assignment_428706' (line 135)
        tuple_var_assignment_428706_429384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'tuple_var_assignment_428706')
        # Assigning a type to the variable 'est' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'est', tuple_var_assignment_428706_429384)
        
        # Assigning a Name to a Name (line 135):
        # Getting the type of 'tuple_var_assignment_428707' (line 135)
        tuple_var_assignment_428707_429385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'tuple_var_assignment_428707')
        # Assigning a type to the variable 'v' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 13), 'v', tuple_var_assignment_428707_429385)
        
        # Assigning a Name to a Name (line 135):
        # Getting the type of 'tuple_var_assignment_428708' (line 135)
        tuple_var_assignment_428708_429386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'tuple_var_assignment_428708')
        # Assigning a type to the variable 'w' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'w', tuple_var_assignment_428708_429386)
        
        # Assigning a Name to a Name (line 135):
        # Getting the type of 'tuple_var_assignment_428709' (line 135)
        tuple_var_assignment_428709_429387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'tuple_var_assignment_428709')
        # Assigning a type to the variable 'nmults' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 19), 'nmults', tuple_var_assignment_428709_429387)
        
        # Assigning a Name to a Name (line 135):
        # Getting the type of 'tuple_var_assignment_428710' (line 135)
        tuple_var_assignment_428710_429388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'tuple_var_assignment_428710')
        # Assigning a type to the variable 'nresamples' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 27), 'nresamples', tuple_var_assignment_428710_429388)
        
        # Assigning a Call to a Name (line 136):
        
        # Assigning a Call to a Name (line 136):
        
        # Call to norm(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'B' (line 136)
        B_429392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 40), 'B', False)
        int_429393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 43), 'int')
        # Processing the call keyword arguments (line 136)
        kwargs_429394 = {}
        # Getting the type of 'scipy' (line 136)
        scipy_429389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 22), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 136)
        linalg_429390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 22), scipy_429389, 'linalg')
        # Obtaining the member 'norm' of a type (line 136)
        norm_429391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 22), linalg_429390, 'norm')
        # Calling norm(args, kwargs) (line 136)
        norm_call_result_429395 = invoke(stypy.reporting.localization.Localization(__file__, 136, 22), norm_429391, *[B_429392, int_429393], **kwargs_429394)
        
        # Assigning a type to the variable 'exact_value' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'exact_value', norm_call_result_429395)
        
        # Assigning a BinOp to a Name (line 137):
        
        # Assigning a BinOp to a Name (line 137):
        # Getting the type of 'est' (line 137)
        est_429396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 25), 'est')
        # Getting the type of 'exact_value' (line 137)
        exact_value_429397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 31), 'exact_value')
        # Applying the binary operator 'div' (line 137)
        result_div_429398 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 25), 'div', est_429396, exact_value_429397)
        
        # Assigning a type to the variable 'underest_ratio' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'underest_ratio', result_div_429398)
        
        # Call to assert_allclose(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'underest_ratio' (line 138)
        underest_ratio_429400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 24), 'underest_ratio', False)
        float_429401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 40), 'float')
        # Processing the call keyword arguments (line 138)
        float_429402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 51), 'float')
        keyword_429403 = float_429402
        kwargs_429404 = {'rtol': keyword_429403}
        # Getting the type of 'assert_allclose' (line 138)
        assert_allclose_429399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 138)
        assert_allclose_call_result_429405 = invoke(stypy.reporting.localization.Localization(__file__, 138, 8), assert_allclose_429399, *[underest_ratio_429400, float_429401], **kwargs_429404)
        
        
        # Call to assert_equal(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'nmults' (line 139)
        nmults_429407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 21), 'nmults', False)
        int_429408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 29), 'int')
        # Processing the call keyword arguments (line 139)
        kwargs_429409 = {}
        # Getting the type of 'assert_equal' (line 139)
        assert_equal_429406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 139)
        assert_equal_call_result_429410 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), assert_equal_429406, *[nmults_429407, int_429408], **kwargs_429409)
        
        
        # Call to assert_equal(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'nresamples' (line 140)
        nresamples_429412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 21), 'nresamples', False)
        int_429413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 33), 'int')
        # Processing the call keyword arguments (line 140)
        kwargs_429414 = {}
        # Getting the type of 'assert_equal' (line 140)
        assert_equal_429411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 140)
        assert_equal_call_result_429415 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), assert_equal_429411, *[nresamples_429412, int_429413], **kwargs_429414)
        
        
        # Assigning a Call to a Name (line 142):
        
        # Assigning a Call to a Name (line 142):
        
        # Call to onenormest(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'B' (line 142)
        B_429420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 51), 'B', False)
        # Processing the call keyword arguments (line 142)
        # Getting the type of 't' (line 142)
        t_429421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 56), 't', False)
        keyword_429422 = t_429421
        # Getting the type of 'itmax' (line 142)
        itmax_429423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 65), 'itmax', False)
        keyword_429424 = itmax_429423
        kwargs_429425 = {'t': keyword_429422, 'itmax': keyword_429424}
        # Getting the type of 'scipy' (line 142)
        scipy_429416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 20), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 142)
        sparse_429417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 20), scipy_429416, 'sparse')
        # Obtaining the member 'linalg' of a type (line 142)
        linalg_429418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 20), sparse_429417, 'linalg')
        # Obtaining the member 'onenormest' of a type (line 142)
        onenormest_429419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 20), linalg_429418, 'onenormest')
        # Calling onenormest(args, kwargs) (line 142)
        onenormest_call_result_429426 = invoke(stypy.reporting.localization.Localization(__file__, 142, 20), onenormest_429419, *[B_429420], **kwargs_429425)
        
        # Assigning a type to the variable 'est_plain' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'est_plain', onenormest_call_result_429426)
        
        # Call to assert_allclose(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'est' (line 143)
        est_429428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 24), 'est', False)
        # Getting the type of 'est_plain' (line 143)
        est_plain_429429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 29), 'est_plain', False)
        # Processing the call keyword arguments (line 143)
        kwargs_429430 = {}
        # Getting the type of 'assert_allclose' (line 143)
        assert_allclose_429427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 143)
        assert_allclose_call_result_429431 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), assert_allclose_429427, *[est_429428, est_plain_429429], **kwargs_429430)
        
        
        # ################# End of 'test_onenormest_table_5_t_1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_onenormest_table_5_t_1' in the type store
        # Getting the type of 'stypy_return_type' (line 124)
        stypy_return_type_429432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_429432)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_onenormest_table_5_t_1'
        return stypy_return_type_429432


    @norecursion
    def test_onenormest_table_6_t_1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_onenormest_table_6_t_1'
        module_type_store = module_type_store.open_function_context('test_onenormest_table_6_t_1', 145, 4, False)
        # Assigning a type to the variable 'self' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOnenormest.test_onenormest_table_6_t_1.__dict__.__setitem__('stypy_localization', localization)
        TestOnenormest.test_onenormest_table_6_t_1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOnenormest.test_onenormest_table_6_t_1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOnenormest.test_onenormest_table_6_t_1.__dict__.__setitem__('stypy_function_name', 'TestOnenormest.test_onenormest_table_6_t_1')
        TestOnenormest.test_onenormest_table_6_t_1.__dict__.__setitem__('stypy_param_names_list', [])
        TestOnenormest.test_onenormest_table_6_t_1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOnenormest.test_onenormest_table_6_t_1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOnenormest.test_onenormest_table_6_t_1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOnenormest.test_onenormest_table_6_t_1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOnenormest.test_onenormest_table_6_t_1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOnenormest.test_onenormest_table_6_t_1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOnenormest.test_onenormest_table_6_t_1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_onenormest_table_6_t_1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_onenormest_table_6_t_1(...)' code ##################

        
        # Call to seed(...): (line 152)
        # Processing the call arguments (line 152)
        int_429436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 23), 'int')
        # Processing the call keyword arguments (line 152)
        kwargs_429437 = {}
        # Getting the type of 'np' (line 152)
        np_429433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 152)
        random_429434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 8), np_429433, 'random')
        # Obtaining the member 'seed' of a type (line 152)
        seed_429435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 8), random_429434, 'seed')
        # Calling seed(args, kwargs) (line 152)
        seed_call_result_429438 = invoke(stypy.reporting.localization.Localization(__file__, 152, 8), seed_429435, *[int_429436], **kwargs_429437)
        
        
        # Assigning a Num to a Name (line 153):
        
        # Assigning a Num to a Name (line 153):
        int_429439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 12), 'int')
        # Assigning a type to the variable 't' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 't', int_429439)
        
        # Assigning a Num to a Name (line 154):
        
        # Assigning a Num to a Name (line 154):
        int_429440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 12), 'int')
        # Assigning a type to the variable 'n' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'n', int_429440)
        
        # Assigning a Num to a Name (line 155):
        
        # Assigning a Num to a Name (line 155):
        int_429441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 16), 'int')
        # Assigning a type to the variable 'itmax' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'itmax', int_429441)
        
        # Assigning a Num to a Name (line 156):
        
        # Assigning a Num to a Name (line 156):
        int_429442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 19), 'int')
        # Assigning a type to the variable 'nsamples' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'nsamples', int_429442)
        
        # Assigning a List to a Name (line 157):
        
        # Assigning a List to a Name (line 157):
        
        # Obtaining an instance of the builtin type 'list' (line 157)
        list_429443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 157)
        
        # Assigning a type to the variable 'observed' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'observed', list_429443)
        
        # Assigning a List to a Name (line 158):
        
        # Assigning a List to a Name (line 158):
        
        # Obtaining an instance of the builtin type 'list' (line 158)
        list_429444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 158)
        
        # Assigning a type to the variable 'expected' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'expected', list_429444)
        
        # Assigning a List to a Name (line 159):
        
        # Assigning a List to a Name (line 159):
        
        # Obtaining an instance of the builtin type 'list' (line 159)
        list_429445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 159)
        
        # Assigning a type to the variable 'nmult_list' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'nmult_list', list_429445)
        
        # Assigning a List to a Name (line 160):
        
        # Assigning a List to a Name (line 160):
        
        # Obtaining an instance of the builtin type 'list' (line 160)
        list_429446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 160)
        
        # Assigning a type to the variable 'nresample_list' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'nresample_list', list_429446)
        
        
        # Call to range(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'nsamples' (line 161)
        nsamples_429448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 23), 'nsamples', False)
        # Processing the call keyword arguments (line 161)
        kwargs_429449 = {}
        # Getting the type of 'range' (line 161)
        range_429447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 17), 'range', False)
        # Calling range(args, kwargs) (line 161)
        range_call_result_429450 = invoke(stypy.reporting.localization.Localization(__file__, 161, 17), range_429447, *[nsamples_429448], **kwargs_429449)
        
        # Testing the type of a for loop iterable (line 161)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 161, 8), range_call_result_429450)
        # Getting the type of the for loop variable (line 161)
        for_loop_var_429451 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 161, 8), range_call_result_429450)
        # Assigning a type to the variable 'i' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'i', for_loop_var_429451)
        # SSA begins for a for statement (line 161)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 162):
        
        # Assigning a BinOp to a Name (line 162):
        
        # Call to rand(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'n' (line 162)
        n_429455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 35), 'n', False)
        # Getting the type of 'n' (line 162)
        n_429456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 38), 'n', False)
        # Processing the call keyword arguments (line 162)
        kwargs_429457 = {}
        # Getting the type of 'np' (line 162)
        np_429452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 20), 'np', False)
        # Obtaining the member 'random' of a type (line 162)
        random_429453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 20), np_429452, 'random')
        # Obtaining the member 'rand' of a type (line 162)
        rand_429454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 20), random_429453, 'rand')
        # Calling rand(args, kwargs) (line 162)
        rand_call_result_429458 = invoke(stypy.reporting.localization.Localization(__file__, 162, 20), rand_429454, *[n_429455, n_429456], **kwargs_429457)
        
        complex_429459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 43), 'complex')
        
        # Call to rand(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'n' (line 162)
        n_429463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 63), 'n', False)
        # Getting the type of 'n' (line 162)
        n_429464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 66), 'n', False)
        # Processing the call keyword arguments (line 162)
        kwargs_429465 = {}
        # Getting the type of 'np' (line 162)
        np_429460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 48), 'np', False)
        # Obtaining the member 'random' of a type (line 162)
        random_429461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 48), np_429460, 'random')
        # Obtaining the member 'rand' of a type (line 162)
        rand_429462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 48), random_429461, 'rand')
        # Calling rand(args, kwargs) (line 162)
        rand_call_result_429466 = invoke(stypy.reporting.localization.Localization(__file__, 162, 48), rand_429462, *[n_429463, n_429464], **kwargs_429465)
        
        # Applying the binary operator '*' (line 162)
        result_mul_429467 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 43), '*', complex_429459, rand_call_result_429466)
        
        # Applying the binary operator '+' (line 162)
        result_add_429468 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 20), '+', rand_call_result_429458, result_mul_429467)
        
        # Assigning a type to the variable 'A_inv' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'A_inv', result_add_429468)
        
        # Assigning a Call to a Name (line 163):
        
        # Assigning a Call to a Name (line 163):
        
        # Call to inv(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'A_inv' (line 163)
        A_inv_429472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 33), 'A_inv', False)
        # Processing the call keyword arguments (line 163)
        kwargs_429473 = {}
        # Getting the type of 'scipy' (line 163)
        scipy_429469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 16), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 163)
        linalg_429470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 16), scipy_429469, 'linalg')
        # Obtaining the member 'inv' of a type (line 163)
        inv_429471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 16), linalg_429470, 'inv')
        # Calling inv(args, kwargs) (line 163)
        inv_call_result_429474 = invoke(stypy.reporting.localization.Localization(__file__, 163, 16), inv_429471, *[A_inv_429472], **kwargs_429473)
        
        # Assigning a type to the variable 'A' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'A', inv_call_result_429474)
        
        # Assigning a Call to a Tuple (line 164):
        
        # Assigning a Subscript to a Name (line 164):
        
        # Obtaining the type of the subscript
        int_429475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 12), 'int')
        
        # Call to _onenormest_core(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'A' (line 164)
        A_429477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 61), 'A', False)
        # Getting the type of 'A' (line 164)
        A_429478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 64), 'A', False)
        # Obtaining the member 'T' of a type (line 164)
        T_429479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 64), A_429478, 'T')
        # Getting the type of 't' (line 164)
        t_429480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 69), 't', False)
        # Getting the type of 'itmax' (line 164)
        itmax_429481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 72), 'itmax', False)
        # Processing the call keyword arguments (line 164)
        kwargs_429482 = {}
        # Getting the type of '_onenormest_core' (line 164)
        _onenormest_core_429476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 44), '_onenormest_core', False)
        # Calling _onenormest_core(args, kwargs) (line 164)
        _onenormest_core_call_result_429483 = invoke(stypy.reporting.localization.Localization(__file__, 164, 44), _onenormest_core_429476, *[A_429477, T_429479, t_429480, itmax_429481], **kwargs_429482)
        
        # Obtaining the member '__getitem__' of a type (line 164)
        getitem___429484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 12), _onenormest_core_call_result_429483, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 164)
        subscript_call_result_429485 = invoke(stypy.reporting.localization.Localization(__file__, 164, 12), getitem___429484, int_429475)
        
        # Assigning a type to the variable 'tuple_var_assignment_428711' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'tuple_var_assignment_428711', subscript_call_result_429485)
        
        # Assigning a Subscript to a Name (line 164):
        
        # Obtaining the type of the subscript
        int_429486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 12), 'int')
        
        # Call to _onenormest_core(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'A' (line 164)
        A_429488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 61), 'A', False)
        # Getting the type of 'A' (line 164)
        A_429489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 64), 'A', False)
        # Obtaining the member 'T' of a type (line 164)
        T_429490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 64), A_429489, 'T')
        # Getting the type of 't' (line 164)
        t_429491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 69), 't', False)
        # Getting the type of 'itmax' (line 164)
        itmax_429492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 72), 'itmax', False)
        # Processing the call keyword arguments (line 164)
        kwargs_429493 = {}
        # Getting the type of '_onenormest_core' (line 164)
        _onenormest_core_429487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 44), '_onenormest_core', False)
        # Calling _onenormest_core(args, kwargs) (line 164)
        _onenormest_core_call_result_429494 = invoke(stypy.reporting.localization.Localization(__file__, 164, 44), _onenormest_core_429487, *[A_429488, T_429490, t_429491, itmax_429492], **kwargs_429493)
        
        # Obtaining the member '__getitem__' of a type (line 164)
        getitem___429495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 12), _onenormest_core_call_result_429494, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 164)
        subscript_call_result_429496 = invoke(stypy.reporting.localization.Localization(__file__, 164, 12), getitem___429495, int_429486)
        
        # Assigning a type to the variable 'tuple_var_assignment_428712' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'tuple_var_assignment_428712', subscript_call_result_429496)
        
        # Assigning a Subscript to a Name (line 164):
        
        # Obtaining the type of the subscript
        int_429497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 12), 'int')
        
        # Call to _onenormest_core(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'A' (line 164)
        A_429499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 61), 'A', False)
        # Getting the type of 'A' (line 164)
        A_429500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 64), 'A', False)
        # Obtaining the member 'T' of a type (line 164)
        T_429501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 64), A_429500, 'T')
        # Getting the type of 't' (line 164)
        t_429502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 69), 't', False)
        # Getting the type of 'itmax' (line 164)
        itmax_429503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 72), 'itmax', False)
        # Processing the call keyword arguments (line 164)
        kwargs_429504 = {}
        # Getting the type of '_onenormest_core' (line 164)
        _onenormest_core_429498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 44), '_onenormest_core', False)
        # Calling _onenormest_core(args, kwargs) (line 164)
        _onenormest_core_call_result_429505 = invoke(stypy.reporting.localization.Localization(__file__, 164, 44), _onenormest_core_429498, *[A_429499, T_429501, t_429502, itmax_429503], **kwargs_429504)
        
        # Obtaining the member '__getitem__' of a type (line 164)
        getitem___429506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 12), _onenormest_core_call_result_429505, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 164)
        subscript_call_result_429507 = invoke(stypy.reporting.localization.Localization(__file__, 164, 12), getitem___429506, int_429497)
        
        # Assigning a type to the variable 'tuple_var_assignment_428713' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'tuple_var_assignment_428713', subscript_call_result_429507)
        
        # Assigning a Subscript to a Name (line 164):
        
        # Obtaining the type of the subscript
        int_429508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 12), 'int')
        
        # Call to _onenormest_core(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'A' (line 164)
        A_429510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 61), 'A', False)
        # Getting the type of 'A' (line 164)
        A_429511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 64), 'A', False)
        # Obtaining the member 'T' of a type (line 164)
        T_429512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 64), A_429511, 'T')
        # Getting the type of 't' (line 164)
        t_429513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 69), 't', False)
        # Getting the type of 'itmax' (line 164)
        itmax_429514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 72), 'itmax', False)
        # Processing the call keyword arguments (line 164)
        kwargs_429515 = {}
        # Getting the type of '_onenormest_core' (line 164)
        _onenormest_core_429509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 44), '_onenormest_core', False)
        # Calling _onenormest_core(args, kwargs) (line 164)
        _onenormest_core_call_result_429516 = invoke(stypy.reporting.localization.Localization(__file__, 164, 44), _onenormest_core_429509, *[A_429510, T_429512, t_429513, itmax_429514], **kwargs_429515)
        
        # Obtaining the member '__getitem__' of a type (line 164)
        getitem___429517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 12), _onenormest_core_call_result_429516, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 164)
        subscript_call_result_429518 = invoke(stypy.reporting.localization.Localization(__file__, 164, 12), getitem___429517, int_429508)
        
        # Assigning a type to the variable 'tuple_var_assignment_428714' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'tuple_var_assignment_428714', subscript_call_result_429518)
        
        # Assigning a Subscript to a Name (line 164):
        
        # Obtaining the type of the subscript
        int_429519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 12), 'int')
        
        # Call to _onenormest_core(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'A' (line 164)
        A_429521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 61), 'A', False)
        # Getting the type of 'A' (line 164)
        A_429522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 64), 'A', False)
        # Obtaining the member 'T' of a type (line 164)
        T_429523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 64), A_429522, 'T')
        # Getting the type of 't' (line 164)
        t_429524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 69), 't', False)
        # Getting the type of 'itmax' (line 164)
        itmax_429525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 72), 'itmax', False)
        # Processing the call keyword arguments (line 164)
        kwargs_429526 = {}
        # Getting the type of '_onenormest_core' (line 164)
        _onenormest_core_429520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 44), '_onenormest_core', False)
        # Calling _onenormest_core(args, kwargs) (line 164)
        _onenormest_core_call_result_429527 = invoke(stypy.reporting.localization.Localization(__file__, 164, 44), _onenormest_core_429520, *[A_429521, T_429523, t_429524, itmax_429525], **kwargs_429526)
        
        # Obtaining the member '__getitem__' of a type (line 164)
        getitem___429528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 12), _onenormest_core_call_result_429527, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 164)
        subscript_call_result_429529 = invoke(stypy.reporting.localization.Localization(__file__, 164, 12), getitem___429528, int_429519)
        
        # Assigning a type to the variable 'tuple_var_assignment_428715' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'tuple_var_assignment_428715', subscript_call_result_429529)
        
        # Assigning a Name to a Name (line 164):
        # Getting the type of 'tuple_var_assignment_428711' (line 164)
        tuple_var_assignment_428711_429530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'tuple_var_assignment_428711')
        # Assigning a type to the variable 'est' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'est', tuple_var_assignment_428711_429530)
        
        # Assigning a Name to a Name (line 164):
        # Getting the type of 'tuple_var_assignment_428712' (line 164)
        tuple_var_assignment_428712_429531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'tuple_var_assignment_428712')
        # Assigning a type to the variable 'v' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 17), 'v', tuple_var_assignment_428712_429531)
        
        # Assigning a Name to a Name (line 164):
        # Getting the type of 'tuple_var_assignment_428713' (line 164)
        tuple_var_assignment_428713_429532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'tuple_var_assignment_428713')
        # Assigning a type to the variable 'w' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 20), 'w', tuple_var_assignment_428713_429532)
        
        # Assigning a Name to a Name (line 164):
        # Getting the type of 'tuple_var_assignment_428714' (line 164)
        tuple_var_assignment_428714_429533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'tuple_var_assignment_428714')
        # Assigning a type to the variable 'nmults' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 23), 'nmults', tuple_var_assignment_428714_429533)
        
        # Assigning a Name to a Name (line 164):
        # Getting the type of 'tuple_var_assignment_428715' (line 164)
        tuple_var_assignment_428715_429534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'tuple_var_assignment_428715')
        # Assigning a type to the variable 'nresamples' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 31), 'nresamples', tuple_var_assignment_428715_429534)
        
        # Call to append(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'est' (line 165)
        est_429537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 28), 'est', False)
        # Processing the call keyword arguments (line 165)
        kwargs_429538 = {}
        # Getting the type of 'observed' (line 165)
        observed_429535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'observed', False)
        # Obtaining the member 'append' of a type (line 165)
        append_429536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 12), observed_429535, 'append')
        # Calling append(args, kwargs) (line 165)
        append_call_result_429539 = invoke(stypy.reporting.localization.Localization(__file__, 165, 12), append_429536, *[est_429537], **kwargs_429538)
        
        
        # Call to append(...): (line 166)
        # Processing the call arguments (line 166)
        
        # Call to norm(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'A' (line 166)
        A_429545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 46), 'A', False)
        int_429546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 49), 'int')
        # Processing the call keyword arguments (line 166)
        kwargs_429547 = {}
        # Getting the type of 'scipy' (line 166)
        scipy_429542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 28), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 166)
        linalg_429543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 28), scipy_429542, 'linalg')
        # Obtaining the member 'norm' of a type (line 166)
        norm_429544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 28), linalg_429543, 'norm')
        # Calling norm(args, kwargs) (line 166)
        norm_call_result_429548 = invoke(stypy.reporting.localization.Localization(__file__, 166, 28), norm_429544, *[A_429545, int_429546], **kwargs_429547)
        
        # Processing the call keyword arguments (line 166)
        kwargs_429549 = {}
        # Getting the type of 'expected' (line 166)
        expected_429540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'expected', False)
        # Obtaining the member 'append' of a type (line 166)
        append_429541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), expected_429540, 'append')
        # Calling append(args, kwargs) (line 166)
        append_call_result_429550 = invoke(stypy.reporting.localization.Localization(__file__, 166, 12), append_429541, *[norm_call_result_429548], **kwargs_429549)
        
        
        # Call to append(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'nmults' (line 167)
        nmults_429553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 30), 'nmults', False)
        # Processing the call keyword arguments (line 167)
        kwargs_429554 = {}
        # Getting the type of 'nmult_list' (line 167)
        nmult_list_429551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'nmult_list', False)
        # Obtaining the member 'append' of a type (line 167)
        append_429552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 12), nmult_list_429551, 'append')
        # Calling append(args, kwargs) (line 167)
        append_call_result_429555 = invoke(stypy.reporting.localization.Localization(__file__, 167, 12), append_429552, *[nmults_429553], **kwargs_429554)
        
        
        # Call to append(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'nresamples' (line 168)
        nresamples_429558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 34), 'nresamples', False)
        # Processing the call keyword arguments (line 168)
        kwargs_429559 = {}
        # Getting the type of 'nresample_list' (line 168)
        nresample_list_429556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'nresample_list', False)
        # Obtaining the member 'append' of a type (line 168)
        append_429557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 12), nresample_list_429556, 'append')
        # Calling append(args, kwargs) (line 168)
        append_call_result_429560 = invoke(stypy.reporting.localization.Localization(__file__, 168, 12), append_429557, *[nresamples_429558], **kwargs_429559)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 169):
        
        # Assigning a Call to a Name (line 169):
        
        # Call to array(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'observed' (line 169)
        observed_429563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 28), 'observed', False)
        # Processing the call keyword arguments (line 169)
        # Getting the type of 'float' (line 169)
        float_429564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 44), 'float', False)
        keyword_429565 = float_429564
        kwargs_429566 = {'dtype': keyword_429565}
        # Getting the type of 'np' (line 169)
        np_429561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 169)
        array_429562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 19), np_429561, 'array')
        # Calling array(args, kwargs) (line 169)
        array_call_result_429567 = invoke(stypy.reporting.localization.Localization(__file__, 169, 19), array_429562, *[observed_429563], **kwargs_429566)
        
        # Assigning a type to the variable 'observed' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'observed', array_call_result_429567)
        
        # Assigning a Call to a Name (line 170):
        
        # Assigning a Call to a Name (line 170):
        
        # Call to array(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'expected' (line 170)
        expected_429570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 28), 'expected', False)
        # Processing the call keyword arguments (line 170)
        # Getting the type of 'float' (line 170)
        float_429571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 44), 'float', False)
        keyword_429572 = float_429571
        kwargs_429573 = {'dtype': keyword_429572}
        # Getting the type of 'np' (line 170)
        np_429568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 170)
        array_429569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 19), np_429568, 'array')
        # Calling array(args, kwargs) (line 170)
        array_call_result_429574 = invoke(stypy.reporting.localization.Localization(__file__, 170, 19), array_429569, *[expected_429570], **kwargs_429573)
        
        # Assigning a type to the variable 'expected' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'expected', array_call_result_429574)
        
        # Assigning a BinOp to a Name (line 171):
        
        # Assigning a BinOp to a Name (line 171):
        
        # Call to abs(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'observed' (line 171)
        observed_429577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 33), 'observed', False)
        # Getting the type of 'expected' (line 171)
        expected_429578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 44), 'expected', False)
        # Applying the binary operator '-' (line 171)
        result_sub_429579 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 33), '-', observed_429577, expected_429578)
        
        # Processing the call keyword arguments (line 171)
        kwargs_429580 = {}
        # Getting the type of 'np' (line 171)
        np_429575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 26), 'np', False)
        # Obtaining the member 'abs' of a type (line 171)
        abs_429576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 26), np_429575, 'abs')
        # Calling abs(args, kwargs) (line 171)
        abs_call_result_429581 = invoke(stypy.reporting.localization.Localization(__file__, 171, 26), abs_429576, *[result_sub_429579], **kwargs_429580)
        
        # Getting the type of 'expected' (line 171)
        expected_429582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 56), 'expected')
        # Applying the binary operator 'div' (line 171)
        result_div_429583 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 26), 'div', abs_call_result_429581, expected_429582)
        
        # Assigning a type to the variable 'relative_errors' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'relative_errors', result_div_429583)
        
        # Assigning a BinOp to a Name (line 174):
        
        # Assigning a BinOp to a Name (line 174):
        # Getting the type of 'observed' (line 174)
        observed_429584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 32), 'observed')
        # Getting the type of 'expected' (line 174)
        expected_429585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 43), 'expected')
        # Applying the binary operator 'div' (line 174)
        result_div_429586 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 32), 'div', observed_429584, expected_429585)
        
        # Assigning a type to the variable 'underestimation_ratio' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'underestimation_ratio', result_div_429586)
        
        # Assigning a Call to a Name (line 175):
        
        # Assigning a Call to a Name (line 175):
        
        # Call to mean(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'underestimation_ratio' (line 175)
        underestimation_ratio_429589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 45), 'underestimation_ratio', False)
        # Processing the call keyword arguments (line 175)
        kwargs_429590 = {}
        # Getting the type of 'np' (line 175)
        np_429587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 37), 'np', False)
        # Obtaining the member 'mean' of a type (line 175)
        mean_429588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 37), np_429587, 'mean')
        # Calling mean(args, kwargs) (line 175)
        mean_call_result_429591 = invoke(stypy.reporting.localization.Localization(__file__, 175, 37), mean_429588, *[underestimation_ratio_429589], **kwargs_429590)
        
        # Assigning a type to the variable 'underestimation_ratio_mean' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'underestimation_ratio_mean', mean_call_result_429591)
        
        # Call to assert_(...): (line 176)
        # Processing the call arguments (line 176)
        
        float_429593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 16), 'float')
        # Getting the type of 'underestimation_ratio_mean' (line 176)
        underestimation_ratio_mean_429594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 23), 'underestimation_ratio_mean', False)
        # Applying the binary operator '<' (line 176)
        result_lt_429595 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 16), '<', float_429593, underestimation_ratio_mean_429594)
        float_429596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 52), 'float')
        # Applying the binary operator '<' (line 176)
        result_lt_429597 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 16), '<', underestimation_ratio_mean_429594, float_429596)
        # Applying the binary operator '&' (line 176)
        result_and__429598 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 16), '&', result_lt_429595, result_lt_429597)
        
        # Processing the call keyword arguments (line 176)
        kwargs_429599 = {}
        # Getting the type of 'assert_' (line 176)
        assert__429592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 176)
        assert__call_result_429600 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), assert__429592, *[result_and__429598], **kwargs_429599)
        
        
        # Assigning a Call to a Name (line 179):
        
        # Assigning a Call to a Name (line 179):
        
        # Call to max(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'nresample_list' (line 179)
        nresample_list_429603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 32), 'nresample_list', False)
        # Processing the call keyword arguments (line 179)
        kwargs_429604 = {}
        # Getting the type of 'np' (line 179)
        np_429601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 25), 'np', False)
        # Obtaining the member 'max' of a type (line 179)
        max_429602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 25), np_429601, 'max')
        # Calling max(args, kwargs) (line 179)
        max_call_result_429605 = invoke(stypy.reporting.localization.Localization(__file__, 179, 25), max_429602, *[nresample_list_429603], **kwargs_429604)
        
        # Assigning a type to the variable 'max_nresamples' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'max_nresamples', max_call_result_429605)
        
        # Call to assert_equal(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'max_nresamples' (line 180)
        max_nresamples_429607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 21), 'max_nresamples', False)
        int_429608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 37), 'int')
        # Processing the call keyword arguments (line 180)
        kwargs_429609 = {}
        # Getting the type of 'assert_equal' (line 180)
        assert_equal_429606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 180)
        assert_equal_call_result_429610 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), assert_equal_429606, *[max_nresamples_429607, int_429608], **kwargs_429609)
        
        
        # Assigning a Call to a Name (line 183):
        
        # Assigning a Call to a Name (line 183):
        
        # Call to count_nonzero(...): (line 183)
        # Processing the call arguments (line 183)
        
        # Getting the type of 'relative_errors' (line 183)
        relative_errors_429613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 34), 'relative_errors', False)
        float_429614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 52), 'float')
        # Applying the binary operator '<' (line 183)
        result_lt_429615 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 34), '<', relative_errors_429613, float_429614)
        
        # Processing the call keyword arguments (line 183)
        kwargs_429616 = {}
        # Getting the type of 'np' (line 183)
        np_429611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 17), 'np', False)
        # Obtaining the member 'count_nonzero' of a type (line 183)
        count_nonzero_429612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 17), np_429611, 'count_nonzero')
        # Calling count_nonzero(args, kwargs) (line 183)
        count_nonzero_call_result_429617 = invoke(stypy.reporting.localization.Localization(__file__, 183, 17), count_nonzero_429612, *[result_lt_429615], **kwargs_429616)
        
        # Assigning a type to the variable 'nexact' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'nexact', count_nonzero_call_result_429617)
        
        # Assigning a BinOp to a Name (line 184):
        
        # Assigning a BinOp to a Name (line 184):
        # Getting the type of 'nexact' (line 184)
        nexact_429618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 27), 'nexact')
        
        # Call to float(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'nsamples' (line 184)
        nsamples_429620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 42), 'nsamples', False)
        # Processing the call keyword arguments (line 184)
        kwargs_429621 = {}
        # Getting the type of 'float' (line 184)
        float_429619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 36), 'float', False)
        # Calling float(args, kwargs) (line 184)
        float_call_result_429622 = invoke(stypy.reporting.localization.Localization(__file__, 184, 36), float_429619, *[nsamples_429620], **kwargs_429621)
        
        # Applying the binary operator 'div' (line 184)
        result_div_429623 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 27), 'div', nexact_429618, float_call_result_429622)
        
        # Assigning a type to the variable 'proportion_exact' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'proportion_exact', result_div_429623)
        
        # Call to assert_(...): (line 185)
        # Processing the call arguments (line 185)
        
        float_429625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 16), 'float')
        # Getting the type of 'proportion_exact' (line 185)
        proportion_exact_429626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 22), 'proportion_exact', False)
        # Applying the binary operator '<' (line 185)
        result_lt_429627 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 16), '<', float_429625, proportion_exact_429626)
        float_429628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 41), 'float')
        # Applying the binary operator '<' (line 185)
        result_lt_429629 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 16), '<', proportion_exact_429626, float_429628)
        # Applying the binary operator '&' (line 185)
        result_and__429630 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 16), '&', result_lt_429627, result_lt_429629)
        
        # Processing the call keyword arguments (line 185)
        kwargs_429631 = {}
        # Getting the type of 'assert_' (line 185)
        assert__429624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 185)
        assert__call_result_429632 = invoke(stypy.reporting.localization.Localization(__file__, 185, 8), assert__429624, *[result_and__429630], **kwargs_429631)
        
        
        # Assigning a Call to a Name (line 188):
        
        # Assigning a Call to a Name (line 188):
        
        # Call to mean(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'nmult_list' (line 188)
        nmult_list_429635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 29), 'nmult_list', False)
        # Processing the call keyword arguments (line 188)
        kwargs_429636 = {}
        # Getting the type of 'np' (line 188)
        np_429633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 21), 'np', False)
        # Obtaining the member 'mean' of a type (line 188)
        mean_429634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 21), np_429633, 'mean')
        # Calling mean(args, kwargs) (line 188)
        mean_call_result_429637 = invoke(stypy.reporting.localization.Localization(__file__, 188, 21), mean_429634, *[nmult_list_429635], **kwargs_429636)
        
        # Assigning a type to the variable 'mean_nmult' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'mean_nmult', mean_call_result_429637)
        
        # Call to assert_(...): (line 189)
        # Processing the call arguments (line 189)
        
        int_429639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 16), 'int')
        # Getting the type of 'mean_nmult' (line 189)
        mean_nmult_429640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 20), 'mean_nmult', False)
        # Applying the binary operator '<' (line 189)
        result_lt_429641 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 16), '<', int_429639, mean_nmult_429640)
        int_429642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 33), 'int')
        # Applying the binary operator '<' (line 189)
        result_lt_429643 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 16), '<', mean_nmult_429640, int_429642)
        # Applying the binary operator '&' (line 189)
        result_and__429644 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 16), '&', result_lt_429641, result_lt_429643)
        
        # Processing the call keyword arguments (line 189)
        kwargs_429645 = {}
        # Getting the type of 'assert_' (line 189)
        assert__429638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 189)
        assert__call_result_429646 = invoke(stypy.reporting.localization.Localization(__file__, 189, 8), assert__429638, *[result_and__429644], **kwargs_429645)
        
        
        # ################# End of 'test_onenormest_table_6_t_1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_onenormest_table_6_t_1' in the type store
        # Getting the type of 'stypy_return_type' (line 145)
        stypy_return_type_429647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_429647)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_onenormest_table_6_t_1'
        return stypy_return_type_429647


    @norecursion
    def _help_product_norm_slow(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_help_product_norm_slow'
        module_type_store = module_type_store.open_function_context('_help_product_norm_slow', 191, 4, False)
        # Assigning a type to the variable 'self' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOnenormest._help_product_norm_slow.__dict__.__setitem__('stypy_localization', localization)
        TestOnenormest._help_product_norm_slow.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOnenormest._help_product_norm_slow.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOnenormest._help_product_norm_slow.__dict__.__setitem__('stypy_function_name', 'TestOnenormest._help_product_norm_slow')
        TestOnenormest._help_product_norm_slow.__dict__.__setitem__('stypy_param_names_list', ['A', 'B'])
        TestOnenormest._help_product_norm_slow.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOnenormest._help_product_norm_slow.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOnenormest._help_product_norm_slow.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOnenormest._help_product_norm_slow.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOnenormest._help_product_norm_slow.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOnenormest._help_product_norm_slow.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOnenormest._help_product_norm_slow', ['A', 'B'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_help_product_norm_slow', localization, ['A', 'B'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_help_product_norm_slow(...)' code ##################

        
        # Assigning a Call to a Name (line 193):
        
        # Assigning a Call to a Name (line 193):
        
        # Call to dot(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'A' (line 193)
        A_429650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 19), 'A', False)
        # Getting the type of 'B' (line 193)
        B_429651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 22), 'B', False)
        # Processing the call keyword arguments (line 193)
        kwargs_429652 = {}
        # Getting the type of 'np' (line 193)
        np_429648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'np', False)
        # Obtaining the member 'dot' of a type (line 193)
        dot_429649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), np_429648, 'dot')
        # Calling dot(args, kwargs) (line 193)
        dot_call_result_429653 = invoke(stypy.reporting.localization.Localization(__file__, 193, 12), dot_429649, *[A_429650, B_429651], **kwargs_429652)
        
        # Assigning a type to the variable 'C' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'C', dot_call_result_429653)
        
        # Call to norm(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'C' (line 194)
        C_429657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 33), 'C', False)
        int_429658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 36), 'int')
        # Processing the call keyword arguments (line 194)
        kwargs_429659 = {}
        # Getting the type of 'scipy' (line 194)
        scipy_429654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 15), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 194)
        linalg_429655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 15), scipy_429654, 'linalg')
        # Obtaining the member 'norm' of a type (line 194)
        norm_429656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 15), linalg_429655, 'norm')
        # Calling norm(args, kwargs) (line 194)
        norm_call_result_429660 = invoke(stypy.reporting.localization.Localization(__file__, 194, 15), norm_429656, *[C_429657, int_429658], **kwargs_429659)
        
        # Assigning a type to the variable 'stypy_return_type' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'stypy_return_type', norm_call_result_429660)
        
        # ################# End of '_help_product_norm_slow(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_help_product_norm_slow' in the type store
        # Getting the type of 'stypy_return_type' (line 191)
        stypy_return_type_429661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_429661)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_help_product_norm_slow'
        return stypy_return_type_429661


    @norecursion
    def _help_product_norm_fast(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_help_product_norm_fast'
        module_type_store = module_type_store.open_function_context('_help_product_norm_fast', 196, 4, False)
        # Assigning a type to the variable 'self' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOnenormest._help_product_norm_fast.__dict__.__setitem__('stypy_localization', localization)
        TestOnenormest._help_product_norm_fast.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOnenormest._help_product_norm_fast.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOnenormest._help_product_norm_fast.__dict__.__setitem__('stypy_function_name', 'TestOnenormest._help_product_norm_fast')
        TestOnenormest._help_product_norm_fast.__dict__.__setitem__('stypy_param_names_list', ['A', 'B'])
        TestOnenormest._help_product_norm_fast.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOnenormest._help_product_norm_fast.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOnenormest._help_product_norm_fast.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOnenormest._help_product_norm_fast.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOnenormest._help_product_norm_fast.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOnenormest._help_product_norm_fast.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOnenormest._help_product_norm_fast', ['A', 'B'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_help_product_norm_fast', localization, ['A', 'B'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_help_product_norm_fast(...)' code ##################

        
        # Assigning a Num to a Name (line 198):
        
        # Assigning a Num to a Name (line 198):
        int_429662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 12), 'int')
        # Assigning a type to the variable 't' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 't', int_429662)
        
        # Assigning a Num to a Name (line 199):
        
        # Assigning a Num to a Name (line 199):
        int_429663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 16), 'int')
        # Assigning a type to the variable 'itmax' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'itmax', int_429663)
        
        # Assigning a Call to a Name (line 200):
        
        # Assigning a Call to a Name (line 200):
        
        # Call to MatrixProductOperator(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'A' (line 200)
        A_429665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 34), 'A', False)
        # Getting the type of 'B' (line 200)
        B_429666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 37), 'B', False)
        # Processing the call keyword arguments (line 200)
        kwargs_429667 = {}
        # Getting the type of 'MatrixProductOperator' (line 200)
        MatrixProductOperator_429664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'MatrixProductOperator', False)
        # Calling MatrixProductOperator(args, kwargs) (line 200)
        MatrixProductOperator_call_result_429668 = invoke(stypy.reporting.localization.Localization(__file__, 200, 12), MatrixProductOperator_429664, *[A_429665, B_429666], **kwargs_429667)
        
        # Assigning a type to the variable 'D' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'D', MatrixProductOperator_call_result_429668)
        
        # Assigning a Call to a Tuple (line 201):
        
        # Assigning a Subscript to a Name (line 201):
        
        # Obtaining the type of the subscript
        int_429669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 8), 'int')
        
        # Call to _onenormest_core(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'D' (line 201)
        D_429671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 57), 'D', False)
        # Getting the type of 'D' (line 201)
        D_429672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 60), 'D', False)
        # Obtaining the member 'T' of a type (line 201)
        T_429673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 60), D_429672, 'T')
        # Getting the type of 't' (line 201)
        t_429674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 65), 't', False)
        # Getting the type of 'itmax' (line 201)
        itmax_429675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 68), 'itmax', False)
        # Processing the call keyword arguments (line 201)
        kwargs_429676 = {}
        # Getting the type of '_onenormest_core' (line 201)
        _onenormest_core_429670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 40), '_onenormest_core', False)
        # Calling _onenormest_core(args, kwargs) (line 201)
        _onenormest_core_call_result_429677 = invoke(stypy.reporting.localization.Localization(__file__, 201, 40), _onenormest_core_429670, *[D_429671, T_429673, t_429674, itmax_429675], **kwargs_429676)
        
        # Obtaining the member '__getitem__' of a type (line 201)
        getitem___429678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), _onenormest_core_call_result_429677, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 201)
        subscript_call_result_429679 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), getitem___429678, int_429669)
        
        # Assigning a type to the variable 'tuple_var_assignment_428716' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'tuple_var_assignment_428716', subscript_call_result_429679)
        
        # Assigning a Subscript to a Name (line 201):
        
        # Obtaining the type of the subscript
        int_429680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 8), 'int')
        
        # Call to _onenormest_core(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'D' (line 201)
        D_429682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 57), 'D', False)
        # Getting the type of 'D' (line 201)
        D_429683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 60), 'D', False)
        # Obtaining the member 'T' of a type (line 201)
        T_429684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 60), D_429683, 'T')
        # Getting the type of 't' (line 201)
        t_429685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 65), 't', False)
        # Getting the type of 'itmax' (line 201)
        itmax_429686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 68), 'itmax', False)
        # Processing the call keyword arguments (line 201)
        kwargs_429687 = {}
        # Getting the type of '_onenormest_core' (line 201)
        _onenormest_core_429681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 40), '_onenormest_core', False)
        # Calling _onenormest_core(args, kwargs) (line 201)
        _onenormest_core_call_result_429688 = invoke(stypy.reporting.localization.Localization(__file__, 201, 40), _onenormest_core_429681, *[D_429682, T_429684, t_429685, itmax_429686], **kwargs_429687)
        
        # Obtaining the member '__getitem__' of a type (line 201)
        getitem___429689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), _onenormest_core_call_result_429688, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 201)
        subscript_call_result_429690 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), getitem___429689, int_429680)
        
        # Assigning a type to the variable 'tuple_var_assignment_428717' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'tuple_var_assignment_428717', subscript_call_result_429690)
        
        # Assigning a Subscript to a Name (line 201):
        
        # Obtaining the type of the subscript
        int_429691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 8), 'int')
        
        # Call to _onenormest_core(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'D' (line 201)
        D_429693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 57), 'D', False)
        # Getting the type of 'D' (line 201)
        D_429694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 60), 'D', False)
        # Obtaining the member 'T' of a type (line 201)
        T_429695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 60), D_429694, 'T')
        # Getting the type of 't' (line 201)
        t_429696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 65), 't', False)
        # Getting the type of 'itmax' (line 201)
        itmax_429697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 68), 'itmax', False)
        # Processing the call keyword arguments (line 201)
        kwargs_429698 = {}
        # Getting the type of '_onenormest_core' (line 201)
        _onenormest_core_429692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 40), '_onenormest_core', False)
        # Calling _onenormest_core(args, kwargs) (line 201)
        _onenormest_core_call_result_429699 = invoke(stypy.reporting.localization.Localization(__file__, 201, 40), _onenormest_core_429692, *[D_429693, T_429695, t_429696, itmax_429697], **kwargs_429698)
        
        # Obtaining the member '__getitem__' of a type (line 201)
        getitem___429700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), _onenormest_core_call_result_429699, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 201)
        subscript_call_result_429701 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), getitem___429700, int_429691)
        
        # Assigning a type to the variable 'tuple_var_assignment_428718' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'tuple_var_assignment_428718', subscript_call_result_429701)
        
        # Assigning a Subscript to a Name (line 201):
        
        # Obtaining the type of the subscript
        int_429702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 8), 'int')
        
        # Call to _onenormest_core(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'D' (line 201)
        D_429704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 57), 'D', False)
        # Getting the type of 'D' (line 201)
        D_429705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 60), 'D', False)
        # Obtaining the member 'T' of a type (line 201)
        T_429706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 60), D_429705, 'T')
        # Getting the type of 't' (line 201)
        t_429707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 65), 't', False)
        # Getting the type of 'itmax' (line 201)
        itmax_429708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 68), 'itmax', False)
        # Processing the call keyword arguments (line 201)
        kwargs_429709 = {}
        # Getting the type of '_onenormest_core' (line 201)
        _onenormest_core_429703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 40), '_onenormest_core', False)
        # Calling _onenormest_core(args, kwargs) (line 201)
        _onenormest_core_call_result_429710 = invoke(stypy.reporting.localization.Localization(__file__, 201, 40), _onenormest_core_429703, *[D_429704, T_429706, t_429707, itmax_429708], **kwargs_429709)
        
        # Obtaining the member '__getitem__' of a type (line 201)
        getitem___429711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), _onenormest_core_call_result_429710, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 201)
        subscript_call_result_429712 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), getitem___429711, int_429702)
        
        # Assigning a type to the variable 'tuple_var_assignment_428719' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'tuple_var_assignment_428719', subscript_call_result_429712)
        
        # Assigning a Subscript to a Name (line 201):
        
        # Obtaining the type of the subscript
        int_429713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 8), 'int')
        
        # Call to _onenormest_core(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'D' (line 201)
        D_429715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 57), 'D', False)
        # Getting the type of 'D' (line 201)
        D_429716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 60), 'D', False)
        # Obtaining the member 'T' of a type (line 201)
        T_429717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 60), D_429716, 'T')
        # Getting the type of 't' (line 201)
        t_429718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 65), 't', False)
        # Getting the type of 'itmax' (line 201)
        itmax_429719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 68), 'itmax', False)
        # Processing the call keyword arguments (line 201)
        kwargs_429720 = {}
        # Getting the type of '_onenormest_core' (line 201)
        _onenormest_core_429714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 40), '_onenormest_core', False)
        # Calling _onenormest_core(args, kwargs) (line 201)
        _onenormest_core_call_result_429721 = invoke(stypy.reporting.localization.Localization(__file__, 201, 40), _onenormest_core_429714, *[D_429715, T_429717, t_429718, itmax_429719], **kwargs_429720)
        
        # Obtaining the member '__getitem__' of a type (line 201)
        getitem___429722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), _onenormest_core_call_result_429721, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 201)
        subscript_call_result_429723 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), getitem___429722, int_429713)
        
        # Assigning a type to the variable 'tuple_var_assignment_428720' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'tuple_var_assignment_428720', subscript_call_result_429723)
        
        # Assigning a Name to a Name (line 201):
        # Getting the type of 'tuple_var_assignment_428716' (line 201)
        tuple_var_assignment_428716_429724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'tuple_var_assignment_428716')
        # Assigning a type to the variable 'est' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'est', tuple_var_assignment_428716_429724)
        
        # Assigning a Name to a Name (line 201):
        # Getting the type of 'tuple_var_assignment_428717' (line 201)
        tuple_var_assignment_428717_429725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'tuple_var_assignment_428717')
        # Assigning a type to the variable 'v' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 13), 'v', tuple_var_assignment_428717_429725)
        
        # Assigning a Name to a Name (line 201):
        # Getting the type of 'tuple_var_assignment_428718' (line 201)
        tuple_var_assignment_428718_429726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'tuple_var_assignment_428718')
        # Assigning a type to the variable 'w' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'w', tuple_var_assignment_428718_429726)
        
        # Assigning a Name to a Name (line 201):
        # Getting the type of 'tuple_var_assignment_428719' (line 201)
        tuple_var_assignment_428719_429727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'tuple_var_assignment_428719')
        # Assigning a type to the variable 'nmults' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 19), 'nmults', tuple_var_assignment_428719_429727)
        
        # Assigning a Name to a Name (line 201):
        # Getting the type of 'tuple_var_assignment_428720' (line 201)
        tuple_var_assignment_428720_429728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'tuple_var_assignment_428720')
        # Assigning a type to the variable 'nresamples' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 27), 'nresamples', tuple_var_assignment_428720_429728)
        # Getting the type of 'est' (line 202)
        est_429729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 15), 'est')
        # Assigning a type to the variable 'stypy_return_type' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'stypy_return_type', est_429729)
        
        # ################# End of '_help_product_norm_fast(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_help_product_norm_fast' in the type store
        # Getting the type of 'stypy_return_type' (line 196)
        stypy_return_type_429730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_429730)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_help_product_norm_fast'
        return stypy_return_type_429730


    @norecursion
    def test_onenormest_linear_operator(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_onenormest_linear_operator'
        module_type_store = module_type_store.open_function_context('test_onenormest_linear_operator', 204, 4, False)
        # Assigning a type to the variable 'self' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOnenormest.test_onenormest_linear_operator.__dict__.__setitem__('stypy_localization', localization)
        TestOnenormest.test_onenormest_linear_operator.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOnenormest.test_onenormest_linear_operator.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOnenormest.test_onenormest_linear_operator.__dict__.__setitem__('stypy_function_name', 'TestOnenormest.test_onenormest_linear_operator')
        TestOnenormest.test_onenormest_linear_operator.__dict__.__setitem__('stypy_param_names_list', [])
        TestOnenormest.test_onenormest_linear_operator.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOnenormest.test_onenormest_linear_operator.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOnenormest.test_onenormest_linear_operator.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOnenormest.test_onenormest_linear_operator.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOnenormest.test_onenormest_linear_operator.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOnenormest.test_onenormest_linear_operator.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOnenormest.test_onenormest_linear_operator', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_onenormest_linear_operator', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_onenormest_linear_operator(...)' code ##################

        
        # Call to seed(...): (line 211)
        # Processing the call arguments (line 211)
        int_429734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 23), 'int')
        # Processing the call keyword arguments (line 211)
        kwargs_429735 = {}
        # Getting the type of 'np' (line 211)
        np_429731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 211)
        random_429732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), np_429731, 'random')
        # Obtaining the member 'seed' of a type (line 211)
        seed_429733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), random_429732, 'seed')
        # Calling seed(args, kwargs) (line 211)
        seed_call_result_429736 = invoke(stypy.reporting.localization.Localization(__file__, 211, 8), seed_429733, *[int_429734], **kwargs_429735)
        
        
        # Assigning a Num to a Name (line 212):
        
        # Assigning a Num to a Name (line 212):
        int_429737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 12), 'int')
        # Assigning a type to the variable 'n' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'n', int_429737)
        
        # Assigning a Num to a Name (line 213):
        
        # Assigning a Num to a Name (line 213):
        int_429738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 12), 'int')
        # Assigning a type to the variable 'k' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'k', int_429738)
        
        # Assigning a Call to a Name (line 214):
        
        # Assigning a Call to a Name (line 214):
        
        # Call to randn(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'n' (line 214)
        n_429742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 28), 'n', False)
        # Getting the type of 'k' (line 214)
        k_429743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 31), 'k', False)
        # Processing the call keyword arguments (line 214)
        kwargs_429744 = {}
        # Getting the type of 'np' (line 214)
        np_429739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 214)
        random_429740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 12), np_429739, 'random')
        # Obtaining the member 'randn' of a type (line 214)
        randn_429741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 12), random_429740, 'randn')
        # Calling randn(args, kwargs) (line 214)
        randn_call_result_429745 = invoke(stypy.reporting.localization.Localization(__file__, 214, 12), randn_429741, *[n_429742, k_429743], **kwargs_429744)
        
        # Assigning a type to the variable 'A' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'A', randn_call_result_429745)
        
        # Assigning a Call to a Name (line 215):
        
        # Assigning a Call to a Name (line 215):
        
        # Call to randn(...): (line 215)
        # Processing the call arguments (line 215)
        # Getting the type of 'k' (line 215)
        k_429749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 28), 'k', False)
        # Getting the type of 'n' (line 215)
        n_429750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 31), 'n', False)
        # Processing the call keyword arguments (line 215)
        kwargs_429751 = {}
        # Getting the type of 'np' (line 215)
        np_429746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 215)
        random_429747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 12), np_429746, 'random')
        # Obtaining the member 'randn' of a type (line 215)
        randn_429748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 12), random_429747, 'randn')
        # Calling randn(args, kwargs) (line 215)
        randn_call_result_429752 = invoke(stypy.reporting.localization.Localization(__file__, 215, 12), randn_429748, *[k_429749, n_429750], **kwargs_429751)
        
        # Assigning a type to the variable 'B' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'B', randn_call_result_429752)
        
        # Assigning a Call to a Name (line 216):
        
        # Assigning a Call to a Name (line 216):
        
        # Call to _help_product_norm_fast(...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 'A' (line 216)
        A_429755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 53), 'A', False)
        # Getting the type of 'B' (line 216)
        B_429756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 56), 'B', False)
        # Processing the call keyword arguments (line 216)
        kwargs_429757 = {}
        # Getting the type of 'self' (line 216)
        self_429753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 24), 'self', False)
        # Obtaining the member '_help_product_norm_fast' of a type (line 216)
        _help_product_norm_fast_429754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 24), self_429753, '_help_product_norm_fast')
        # Calling _help_product_norm_fast(args, kwargs) (line 216)
        _help_product_norm_fast_call_result_429758 = invoke(stypy.reporting.localization.Localization(__file__, 216, 24), _help_product_norm_fast_429754, *[A_429755, B_429756], **kwargs_429757)
        
        # Assigning a type to the variable 'fast_estimate' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'fast_estimate', _help_product_norm_fast_call_result_429758)
        
        # Assigning a Call to a Name (line 217):
        
        # Assigning a Call to a Name (line 217):
        
        # Call to _help_product_norm_slow(...): (line 217)
        # Processing the call arguments (line 217)
        # Getting the type of 'A' (line 217)
        A_429761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 51), 'A', False)
        # Getting the type of 'B' (line 217)
        B_429762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 54), 'B', False)
        # Processing the call keyword arguments (line 217)
        kwargs_429763 = {}
        # Getting the type of 'self' (line 217)
        self_429759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 22), 'self', False)
        # Obtaining the member '_help_product_norm_slow' of a type (line 217)
        _help_product_norm_slow_429760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 22), self_429759, '_help_product_norm_slow')
        # Calling _help_product_norm_slow(args, kwargs) (line 217)
        _help_product_norm_slow_call_result_429764 = invoke(stypy.reporting.localization.Localization(__file__, 217, 22), _help_product_norm_slow_429760, *[A_429761, B_429762], **kwargs_429763)
        
        # Assigning a type to the variable 'exact_value' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'exact_value', _help_product_norm_slow_call_result_429764)
        
        # Call to assert_(...): (line 218)
        # Processing the call arguments (line 218)
        
        # Getting the type of 'fast_estimate' (line 218)
        fast_estimate_429766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'fast_estimate', False)
        # Getting the type of 'exact_value' (line 218)
        exact_value_429767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 33), 'exact_value', False)
        # Applying the binary operator '<=' (line 218)
        result_le_429768 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 16), '<=', fast_estimate_429766, exact_value_429767)
        int_429769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 48), 'int')
        # Getting the type of 'fast_estimate' (line 218)
        fast_estimate_429770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 50), 'fast_estimate', False)
        # Applying the binary operator '*' (line 218)
        result_mul_429771 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 48), '*', int_429769, fast_estimate_429770)
        
        # Applying the binary operator '<=' (line 218)
        result_le_429772 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 16), '<=', exact_value_429767, result_mul_429771)
        # Applying the binary operator '&' (line 218)
        result_and__429773 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 16), '&', result_le_429768, result_le_429772)
        
        str_429774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 16), 'str', 'fast: %g\nexact:%g')
        
        # Obtaining an instance of the builtin type 'tuple' (line 219)
        tuple_429775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 219)
        # Adding element type (line 219)
        # Getting the type of 'fast_estimate' (line 219)
        fast_estimate_429776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 40), 'fast_estimate', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 40), tuple_429775, fast_estimate_429776)
        # Adding element type (line 219)
        # Getting the type of 'exact_value' (line 219)
        exact_value_429777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 55), 'exact_value', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 40), tuple_429775, exact_value_429777)
        
        # Applying the binary operator '%' (line 219)
        result_mod_429778 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 16), '%', str_429774, tuple_429775)
        
        # Processing the call keyword arguments (line 218)
        kwargs_429779 = {}
        # Getting the type of 'assert_' (line 218)
        assert__429765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 218)
        assert__call_result_429780 = invoke(stypy.reporting.localization.Localization(__file__, 218, 8), assert__429765, *[result_and__429773, result_mod_429778], **kwargs_429779)
        
        
        # ################# End of 'test_onenormest_linear_operator(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_onenormest_linear_operator' in the type store
        # Getting the type of 'stypy_return_type' (line 204)
        stypy_return_type_429781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_429781)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_onenormest_linear_operator'
        return stypy_return_type_429781


    @norecursion
    def test_returns(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_returns'
        module_type_store = module_type_store.open_function_context('test_returns', 221, 4, False)
        # Assigning a type to the variable 'self' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOnenormest.test_returns.__dict__.__setitem__('stypy_localization', localization)
        TestOnenormest.test_returns.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOnenormest.test_returns.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOnenormest.test_returns.__dict__.__setitem__('stypy_function_name', 'TestOnenormest.test_returns')
        TestOnenormest.test_returns.__dict__.__setitem__('stypy_param_names_list', [])
        TestOnenormest.test_returns.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOnenormest.test_returns.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOnenormest.test_returns.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOnenormest.test_returns.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOnenormest.test_returns.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOnenormest.test_returns.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOnenormest.test_returns', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_returns', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_returns(...)' code ##################

        
        # Call to seed(...): (line 222)
        # Processing the call arguments (line 222)
        int_429785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 23), 'int')
        # Processing the call keyword arguments (line 222)
        kwargs_429786 = {}
        # Getting the type of 'np' (line 222)
        np_429782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 222)
        random_429783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), np_429782, 'random')
        # Obtaining the member 'seed' of a type (line 222)
        seed_429784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), random_429783, 'seed')
        # Calling seed(args, kwargs) (line 222)
        seed_call_result_429787 = invoke(stypy.reporting.localization.Localization(__file__, 222, 8), seed_429784, *[int_429785], **kwargs_429786)
        
        
        # Assigning a Call to a Name (line 223):
        
        # Assigning a Call to a Name (line 223):
        
        # Call to rand(...): (line 223)
        # Processing the call arguments (line 223)
        int_429791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 30), 'int')
        int_429792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 34), 'int')
        float_429793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 38), 'float')
        # Processing the call keyword arguments (line 223)
        kwargs_429794 = {}
        # Getting the type of 'scipy' (line 223)
        scipy_429788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 223)
        sparse_429789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 12), scipy_429788, 'sparse')
        # Obtaining the member 'rand' of a type (line 223)
        rand_429790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 12), sparse_429789, 'rand')
        # Calling rand(args, kwargs) (line 223)
        rand_call_result_429795 = invoke(stypy.reporting.localization.Localization(__file__, 223, 12), rand_429790, *[int_429791, int_429792, float_429793], **kwargs_429794)
        
        # Assigning a type to the variable 'A' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'A', rand_call_result_429795)
        
        # Assigning a Call to a Name (line 225):
        
        # Assigning a Call to a Name (line 225):
        
        # Call to norm(...): (line 225)
        # Processing the call arguments (line 225)
        
        # Call to todense(...): (line 225)
        # Processing the call keyword arguments (line 225)
        kwargs_429801 = {}
        # Getting the type of 'A' (line 225)
        A_429799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 31), 'A', False)
        # Obtaining the member 'todense' of a type (line 225)
        todense_429800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 31), A_429799, 'todense')
        # Calling todense(args, kwargs) (line 225)
        todense_call_result_429802 = invoke(stypy.reporting.localization.Localization(__file__, 225, 31), todense_429800, *[], **kwargs_429801)
        
        int_429803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 44), 'int')
        # Processing the call keyword arguments (line 225)
        kwargs_429804 = {}
        # Getting the type of 'scipy' (line 225)
        scipy_429796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 13), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 225)
        linalg_429797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 13), scipy_429796, 'linalg')
        # Obtaining the member 'norm' of a type (line 225)
        norm_429798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 13), linalg_429797, 'norm')
        # Calling norm(args, kwargs) (line 225)
        norm_call_result_429805 = invoke(stypy.reporting.localization.Localization(__file__, 225, 13), norm_429798, *[todense_call_result_429802, int_429803], **kwargs_429804)
        
        # Assigning a type to the variable 's0' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 's0', norm_call_result_429805)
        
        # Assigning a Call to a Tuple (line 226):
        
        # Assigning a Subscript to a Name (line 226):
        
        # Obtaining the type of the subscript
        int_429806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 8), 'int')
        
        # Call to onenormest(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'A' (line 226)
        A_429811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 47), 'A', False)
        # Processing the call keyword arguments (line 226)
        # Getting the type of 'True' (line 226)
        True_429812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 60), 'True', False)
        keyword_429813 = True_429812
        kwargs_429814 = {'compute_v': keyword_429813}
        # Getting the type of 'scipy' (line 226)
        scipy_429807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 16), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 226)
        sparse_429808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 16), scipy_429807, 'sparse')
        # Obtaining the member 'linalg' of a type (line 226)
        linalg_429809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 16), sparse_429808, 'linalg')
        # Obtaining the member 'onenormest' of a type (line 226)
        onenormest_429810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 16), linalg_429809, 'onenormest')
        # Calling onenormest(args, kwargs) (line 226)
        onenormest_call_result_429815 = invoke(stypy.reporting.localization.Localization(__file__, 226, 16), onenormest_429810, *[A_429811], **kwargs_429814)
        
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___429816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), onenormest_call_result_429815, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_429817 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), getitem___429816, int_429806)
        
        # Assigning a type to the variable 'tuple_var_assignment_428721' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_428721', subscript_call_result_429817)
        
        # Assigning a Subscript to a Name (line 226):
        
        # Obtaining the type of the subscript
        int_429818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 8), 'int')
        
        # Call to onenormest(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'A' (line 226)
        A_429823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 47), 'A', False)
        # Processing the call keyword arguments (line 226)
        # Getting the type of 'True' (line 226)
        True_429824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 60), 'True', False)
        keyword_429825 = True_429824
        kwargs_429826 = {'compute_v': keyword_429825}
        # Getting the type of 'scipy' (line 226)
        scipy_429819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 16), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 226)
        sparse_429820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 16), scipy_429819, 'sparse')
        # Obtaining the member 'linalg' of a type (line 226)
        linalg_429821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 16), sparse_429820, 'linalg')
        # Obtaining the member 'onenormest' of a type (line 226)
        onenormest_429822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 16), linalg_429821, 'onenormest')
        # Calling onenormest(args, kwargs) (line 226)
        onenormest_call_result_429827 = invoke(stypy.reporting.localization.Localization(__file__, 226, 16), onenormest_429822, *[A_429823], **kwargs_429826)
        
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___429828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), onenormest_call_result_429827, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_429829 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), getitem___429828, int_429818)
        
        # Assigning a type to the variable 'tuple_var_assignment_428722' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_428722', subscript_call_result_429829)
        
        # Assigning a Name to a Name (line 226):
        # Getting the type of 'tuple_var_assignment_428721' (line 226)
        tuple_var_assignment_428721_429830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_428721')
        # Assigning a type to the variable 's1' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 's1', tuple_var_assignment_428721_429830)
        
        # Assigning a Name to a Name (line 226):
        # Getting the type of 'tuple_var_assignment_428722' (line 226)
        tuple_var_assignment_428722_429831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_428722')
        # Assigning a type to the variable 'v' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'v', tuple_var_assignment_428722_429831)
        
        # Assigning a Call to a Tuple (line 227):
        
        # Assigning a Subscript to a Name (line 227):
        
        # Obtaining the type of the subscript
        int_429832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 8), 'int')
        
        # Call to onenormest(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'A' (line 227)
        A_429837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 47), 'A', False)
        # Processing the call keyword arguments (line 227)
        # Getting the type of 'True' (line 227)
        True_429838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 60), 'True', False)
        keyword_429839 = True_429838
        kwargs_429840 = {'compute_w': keyword_429839}
        # Getting the type of 'scipy' (line 227)
        scipy_429833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 227)
        sparse_429834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 16), scipy_429833, 'sparse')
        # Obtaining the member 'linalg' of a type (line 227)
        linalg_429835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 16), sparse_429834, 'linalg')
        # Obtaining the member 'onenormest' of a type (line 227)
        onenormest_429836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 16), linalg_429835, 'onenormest')
        # Calling onenormest(args, kwargs) (line 227)
        onenormest_call_result_429841 = invoke(stypy.reporting.localization.Localization(__file__, 227, 16), onenormest_429836, *[A_429837], **kwargs_429840)
        
        # Obtaining the member '__getitem__' of a type (line 227)
        getitem___429842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), onenormest_call_result_429841, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 227)
        subscript_call_result_429843 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), getitem___429842, int_429832)
        
        # Assigning a type to the variable 'tuple_var_assignment_428723' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_428723', subscript_call_result_429843)
        
        # Assigning a Subscript to a Name (line 227):
        
        # Obtaining the type of the subscript
        int_429844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 8), 'int')
        
        # Call to onenormest(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'A' (line 227)
        A_429849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 47), 'A', False)
        # Processing the call keyword arguments (line 227)
        # Getting the type of 'True' (line 227)
        True_429850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 60), 'True', False)
        keyword_429851 = True_429850
        kwargs_429852 = {'compute_w': keyword_429851}
        # Getting the type of 'scipy' (line 227)
        scipy_429845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 227)
        sparse_429846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 16), scipy_429845, 'sparse')
        # Obtaining the member 'linalg' of a type (line 227)
        linalg_429847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 16), sparse_429846, 'linalg')
        # Obtaining the member 'onenormest' of a type (line 227)
        onenormest_429848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 16), linalg_429847, 'onenormest')
        # Calling onenormest(args, kwargs) (line 227)
        onenormest_call_result_429853 = invoke(stypy.reporting.localization.Localization(__file__, 227, 16), onenormest_429848, *[A_429849], **kwargs_429852)
        
        # Obtaining the member '__getitem__' of a type (line 227)
        getitem___429854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), onenormest_call_result_429853, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 227)
        subscript_call_result_429855 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), getitem___429854, int_429844)
        
        # Assigning a type to the variable 'tuple_var_assignment_428724' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_428724', subscript_call_result_429855)
        
        # Assigning a Name to a Name (line 227):
        # Getting the type of 'tuple_var_assignment_428723' (line 227)
        tuple_var_assignment_428723_429856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_428723')
        # Assigning a type to the variable 's2' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 's2', tuple_var_assignment_428723_429856)
        
        # Assigning a Name to a Name (line 227):
        # Getting the type of 'tuple_var_assignment_428724' (line 227)
        tuple_var_assignment_428724_429857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_428724')
        # Assigning a type to the variable 'w' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'w', tuple_var_assignment_428724_429857)
        
        # Assigning a Call to a Tuple (line 228):
        
        # Assigning a Subscript to a Name (line 228):
        
        # Obtaining the type of the subscript
        int_429858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 8), 'int')
        
        # Call to onenormest(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'A' (line 228)
        A_429863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 52), 'A', False)
        # Processing the call keyword arguments (line 228)
        # Getting the type of 'True' (line 228)
        True_429864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 65), 'True', False)
        keyword_429865 = True_429864
        # Getting the type of 'True' (line 228)
        True_429866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 81), 'True', False)
        keyword_429867 = True_429866
        kwargs_429868 = {'compute_w': keyword_429865, 'compute_v': keyword_429867}
        # Getting the type of 'scipy' (line 228)
        scipy_429859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 21), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 228)
        sparse_429860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 21), scipy_429859, 'sparse')
        # Obtaining the member 'linalg' of a type (line 228)
        linalg_429861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 21), sparse_429860, 'linalg')
        # Obtaining the member 'onenormest' of a type (line 228)
        onenormest_429862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 21), linalg_429861, 'onenormest')
        # Calling onenormest(args, kwargs) (line 228)
        onenormest_call_result_429869 = invoke(stypy.reporting.localization.Localization(__file__, 228, 21), onenormest_429862, *[A_429863], **kwargs_429868)
        
        # Obtaining the member '__getitem__' of a type (line 228)
        getitem___429870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 8), onenormest_call_result_429869, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 228)
        subscript_call_result_429871 = invoke(stypy.reporting.localization.Localization(__file__, 228, 8), getitem___429870, int_429858)
        
        # Assigning a type to the variable 'tuple_var_assignment_428725' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'tuple_var_assignment_428725', subscript_call_result_429871)
        
        # Assigning a Subscript to a Name (line 228):
        
        # Obtaining the type of the subscript
        int_429872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 8), 'int')
        
        # Call to onenormest(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'A' (line 228)
        A_429877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 52), 'A', False)
        # Processing the call keyword arguments (line 228)
        # Getting the type of 'True' (line 228)
        True_429878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 65), 'True', False)
        keyword_429879 = True_429878
        # Getting the type of 'True' (line 228)
        True_429880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 81), 'True', False)
        keyword_429881 = True_429880
        kwargs_429882 = {'compute_w': keyword_429879, 'compute_v': keyword_429881}
        # Getting the type of 'scipy' (line 228)
        scipy_429873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 21), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 228)
        sparse_429874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 21), scipy_429873, 'sparse')
        # Obtaining the member 'linalg' of a type (line 228)
        linalg_429875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 21), sparse_429874, 'linalg')
        # Obtaining the member 'onenormest' of a type (line 228)
        onenormest_429876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 21), linalg_429875, 'onenormest')
        # Calling onenormest(args, kwargs) (line 228)
        onenormest_call_result_429883 = invoke(stypy.reporting.localization.Localization(__file__, 228, 21), onenormest_429876, *[A_429877], **kwargs_429882)
        
        # Obtaining the member '__getitem__' of a type (line 228)
        getitem___429884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 8), onenormest_call_result_429883, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 228)
        subscript_call_result_429885 = invoke(stypy.reporting.localization.Localization(__file__, 228, 8), getitem___429884, int_429872)
        
        # Assigning a type to the variable 'tuple_var_assignment_428726' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'tuple_var_assignment_428726', subscript_call_result_429885)
        
        # Assigning a Subscript to a Name (line 228):
        
        # Obtaining the type of the subscript
        int_429886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 8), 'int')
        
        # Call to onenormest(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'A' (line 228)
        A_429891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 52), 'A', False)
        # Processing the call keyword arguments (line 228)
        # Getting the type of 'True' (line 228)
        True_429892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 65), 'True', False)
        keyword_429893 = True_429892
        # Getting the type of 'True' (line 228)
        True_429894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 81), 'True', False)
        keyword_429895 = True_429894
        kwargs_429896 = {'compute_w': keyword_429893, 'compute_v': keyword_429895}
        # Getting the type of 'scipy' (line 228)
        scipy_429887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 21), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 228)
        sparse_429888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 21), scipy_429887, 'sparse')
        # Obtaining the member 'linalg' of a type (line 228)
        linalg_429889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 21), sparse_429888, 'linalg')
        # Obtaining the member 'onenormest' of a type (line 228)
        onenormest_429890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 21), linalg_429889, 'onenormest')
        # Calling onenormest(args, kwargs) (line 228)
        onenormest_call_result_429897 = invoke(stypy.reporting.localization.Localization(__file__, 228, 21), onenormest_429890, *[A_429891], **kwargs_429896)
        
        # Obtaining the member '__getitem__' of a type (line 228)
        getitem___429898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 8), onenormest_call_result_429897, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 228)
        subscript_call_result_429899 = invoke(stypy.reporting.localization.Localization(__file__, 228, 8), getitem___429898, int_429886)
        
        # Assigning a type to the variable 'tuple_var_assignment_428727' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'tuple_var_assignment_428727', subscript_call_result_429899)
        
        # Assigning a Name to a Name (line 228):
        # Getting the type of 'tuple_var_assignment_428725' (line 228)
        tuple_var_assignment_428725_429900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'tuple_var_assignment_428725')
        # Assigning a type to the variable 's3' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 's3', tuple_var_assignment_428725_429900)
        
        # Assigning a Name to a Name (line 228):
        # Getting the type of 'tuple_var_assignment_428726' (line 228)
        tuple_var_assignment_428726_429901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'tuple_var_assignment_428726')
        # Assigning a type to the variable 'v2' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'v2', tuple_var_assignment_428726_429901)
        
        # Assigning a Name to a Name (line 228):
        # Getting the type of 'tuple_var_assignment_428727' (line 228)
        tuple_var_assignment_428727_429902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'tuple_var_assignment_428727')
        # Assigning a type to the variable 'w2' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 16), 'w2', tuple_var_assignment_428727_429902)
        
        # Call to assert_allclose(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 's1' (line 230)
        s1_429904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 24), 's1', False)
        # Getting the type of 's0' (line 230)
        s0_429905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 28), 's0', False)
        # Processing the call keyword arguments (line 230)
        float_429906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 37), 'float')
        keyword_429907 = float_429906
        kwargs_429908 = {'rtol': keyword_429907}
        # Getting the type of 'assert_allclose' (line 230)
        assert_allclose_429903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 230)
        assert_allclose_call_result_429909 = invoke(stypy.reporting.localization.Localization(__file__, 230, 8), assert_allclose_429903, *[s1_429904, s0_429905], **kwargs_429908)
        
        
        # Call to assert_allclose(...): (line 231)
        # Processing the call arguments (line 231)
        
        # Call to norm(...): (line 231)
        # Processing the call arguments (line 231)
        
        # Call to dot(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'v' (line 231)
        v_429916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 45), 'v', False)
        # Processing the call keyword arguments (line 231)
        kwargs_429917 = {}
        # Getting the type of 'A' (line 231)
        A_429914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 39), 'A', False)
        # Obtaining the member 'dot' of a type (line 231)
        dot_429915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 39), A_429914, 'dot')
        # Calling dot(args, kwargs) (line 231)
        dot_call_result_429918 = invoke(stypy.reporting.localization.Localization(__file__, 231, 39), dot_429915, *[v_429916], **kwargs_429917)
        
        int_429919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 49), 'int')
        # Processing the call keyword arguments (line 231)
        kwargs_429920 = {}
        # Getting the type of 'np' (line 231)
        np_429911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 24), 'np', False)
        # Obtaining the member 'linalg' of a type (line 231)
        linalg_429912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 24), np_429911, 'linalg')
        # Obtaining the member 'norm' of a type (line 231)
        norm_429913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 24), linalg_429912, 'norm')
        # Calling norm(args, kwargs) (line 231)
        norm_call_result_429921 = invoke(stypy.reporting.localization.Localization(__file__, 231, 24), norm_429913, *[dot_call_result_429918, int_429919], **kwargs_429920)
        
        # Getting the type of 's0' (line 231)
        s0_429922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 53), 's0', False)
        
        # Call to norm(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'v' (line 231)
        v_429926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 71), 'v', False)
        int_429927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 74), 'int')
        # Processing the call keyword arguments (line 231)
        kwargs_429928 = {}
        # Getting the type of 'np' (line 231)
        np_429923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 56), 'np', False)
        # Obtaining the member 'linalg' of a type (line 231)
        linalg_429924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 56), np_429923, 'linalg')
        # Obtaining the member 'norm' of a type (line 231)
        norm_429925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 56), linalg_429924, 'norm')
        # Calling norm(args, kwargs) (line 231)
        norm_call_result_429929 = invoke(stypy.reporting.localization.Localization(__file__, 231, 56), norm_429925, *[v_429926, int_429927], **kwargs_429928)
        
        # Applying the binary operator '*' (line 231)
        result_mul_429930 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 53), '*', s0_429922, norm_call_result_429929)
        
        # Processing the call keyword arguments (line 231)
        float_429931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 83), 'float')
        keyword_429932 = float_429931
        kwargs_429933 = {'rtol': keyword_429932}
        # Getting the type of 'assert_allclose' (line 231)
        assert_allclose_429910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 231)
        assert_allclose_call_result_429934 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), assert_allclose_429910, *[norm_call_result_429921, result_mul_429930], **kwargs_429933)
        
        
        # Call to assert_allclose(...): (line 232)
        # Processing the call arguments (line 232)
        
        # Call to dot(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'v' (line 232)
        v_429938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 30), 'v', False)
        # Processing the call keyword arguments (line 232)
        kwargs_429939 = {}
        # Getting the type of 'A' (line 232)
        A_429936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 24), 'A', False)
        # Obtaining the member 'dot' of a type (line 232)
        dot_429937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 24), A_429936, 'dot')
        # Calling dot(args, kwargs) (line 232)
        dot_call_result_429940 = invoke(stypy.reporting.localization.Localization(__file__, 232, 24), dot_429937, *[v_429938], **kwargs_429939)
        
        # Getting the type of 'w' (line 232)
        w_429941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 34), 'w', False)
        # Processing the call keyword arguments (line 232)
        float_429942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 42), 'float')
        keyword_429943 = float_429942
        kwargs_429944 = {'rtol': keyword_429943}
        # Getting the type of 'assert_allclose' (line 232)
        assert_allclose_429935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 232)
        assert_allclose_call_result_429945 = invoke(stypy.reporting.localization.Localization(__file__, 232, 8), assert_allclose_429935, *[dot_call_result_429940, w_429941], **kwargs_429944)
        
        
        # ################# End of 'test_returns(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_returns' in the type store
        # Getting the type of 'stypy_return_type' (line 221)
        stypy_return_type_429946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_429946)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_returns'
        return stypy_return_type_429946


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOnenormest.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestOnenormest' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'TestOnenormest', TestOnenormest)
# Declaration of the 'TestAlgorithm_2_2' class

class TestAlgorithm_2_2(object, ):

    @norecursion
    def test_randn_inv(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_randn_inv'
        module_type_store = module_type_store.open_function_context('test_randn_inv', 237, 4, False)
        # Assigning a type to the variable 'self' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestAlgorithm_2_2.test_randn_inv.__dict__.__setitem__('stypy_localization', localization)
        TestAlgorithm_2_2.test_randn_inv.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestAlgorithm_2_2.test_randn_inv.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestAlgorithm_2_2.test_randn_inv.__dict__.__setitem__('stypy_function_name', 'TestAlgorithm_2_2.test_randn_inv')
        TestAlgorithm_2_2.test_randn_inv.__dict__.__setitem__('stypy_param_names_list', [])
        TestAlgorithm_2_2.test_randn_inv.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestAlgorithm_2_2.test_randn_inv.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestAlgorithm_2_2.test_randn_inv.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestAlgorithm_2_2.test_randn_inv.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestAlgorithm_2_2.test_randn_inv.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestAlgorithm_2_2.test_randn_inv.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestAlgorithm_2_2.test_randn_inv', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_randn_inv', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_randn_inv(...)' code ##################

        
        # Call to seed(...): (line 238)
        # Processing the call arguments (line 238)
        int_429950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 23), 'int')
        # Processing the call keyword arguments (line 238)
        kwargs_429951 = {}
        # Getting the type of 'np' (line 238)
        np_429947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 238)
        random_429948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 8), np_429947, 'random')
        # Obtaining the member 'seed' of a type (line 238)
        seed_429949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 8), random_429948, 'seed')
        # Calling seed(args, kwargs) (line 238)
        seed_call_result_429952 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), seed_429949, *[int_429950], **kwargs_429951)
        
        
        # Assigning a Num to a Name (line 239):
        
        # Assigning a Num to a Name (line 239):
        int_429953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 12), 'int')
        # Assigning a type to the variable 'n' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'n', int_429953)
        
        # Assigning a Num to a Name (line 240):
        
        # Assigning a Num to a Name (line 240):
        int_429954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 19), 'int')
        # Assigning a type to the variable 'nsamples' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'nsamples', int_429954)
        
        
        # Call to range(...): (line 241)
        # Processing the call arguments (line 241)
        # Getting the type of 'nsamples' (line 241)
        nsamples_429956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 23), 'nsamples', False)
        # Processing the call keyword arguments (line 241)
        kwargs_429957 = {}
        # Getting the type of 'range' (line 241)
        range_429955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 17), 'range', False)
        # Calling range(args, kwargs) (line 241)
        range_call_result_429958 = invoke(stypy.reporting.localization.Localization(__file__, 241, 17), range_429955, *[nsamples_429956], **kwargs_429957)
        
        # Testing the type of a for loop iterable (line 241)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 241, 8), range_call_result_429958)
        # Getting the type of the for loop variable (line 241)
        for_loop_var_429959 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 241, 8), range_call_result_429958)
        # Assigning a type to the variable 'i' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'i', for_loop_var_429959)
        # SSA begins for a for statement (line 241)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 244):
        
        # Assigning a Call to a Name (line 244):
        
        # Call to randint(...): (line 244)
        # Processing the call arguments (line 244)
        int_429963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 34), 'int')
        int_429964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 37), 'int')
        # Processing the call keyword arguments (line 244)
        kwargs_429965 = {}
        # Getting the type of 'np' (line 244)
        np_429960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 16), 'np', False)
        # Obtaining the member 'random' of a type (line 244)
        random_429961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 16), np_429960, 'random')
        # Obtaining the member 'randint' of a type (line 244)
        randint_429962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 16), random_429961, 'randint')
        # Calling randint(args, kwargs) (line 244)
        randint_call_result_429966 = invoke(stypy.reporting.localization.Localization(__file__, 244, 16), randint_429962, *[int_429963, int_429964], **kwargs_429965)
        
        # Assigning a type to the variable 't' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 't', randint_call_result_429966)
        
        # Assigning a Call to a Name (line 247):
        
        # Assigning a Call to a Name (line 247):
        
        # Call to randint(...): (line 247)
        # Processing the call arguments (line 247)
        int_429970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 34), 'int')
        int_429971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 38), 'int')
        # Processing the call keyword arguments (line 247)
        kwargs_429972 = {}
        # Getting the type of 'np' (line 247)
        np_429967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'np', False)
        # Obtaining the member 'random' of a type (line 247)
        random_429968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 16), np_429967, 'random')
        # Obtaining the member 'randint' of a type (line 247)
        randint_429969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 16), random_429968, 'randint')
        # Calling randint(args, kwargs) (line 247)
        randint_call_result_429973 = invoke(stypy.reporting.localization.Localization(__file__, 247, 16), randint_429969, *[int_429970, int_429971], **kwargs_429972)
        
        # Assigning a type to the variable 'n' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'n', randint_call_result_429973)
        
        # Assigning a Call to a Name (line 250):
        
        # Assigning a Call to a Name (line 250):
        
        # Call to inv(...): (line 250)
        # Processing the call arguments (line 250)
        
        # Call to randn(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'n' (line 250)
        n_429980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 49), 'n', False)
        # Getting the type of 'n' (line 250)
        n_429981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 52), 'n', False)
        # Processing the call keyword arguments (line 250)
        kwargs_429982 = {}
        # Getting the type of 'np' (line 250)
        np_429977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 33), 'np', False)
        # Obtaining the member 'random' of a type (line 250)
        random_429978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 33), np_429977, 'random')
        # Obtaining the member 'randn' of a type (line 250)
        randn_429979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 33), random_429978, 'randn')
        # Calling randn(args, kwargs) (line 250)
        randn_call_result_429983 = invoke(stypy.reporting.localization.Localization(__file__, 250, 33), randn_429979, *[n_429980, n_429981], **kwargs_429982)
        
        # Processing the call keyword arguments (line 250)
        kwargs_429984 = {}
        # Getting the type of 'scipy' (line 250)
        scipy_429974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 250)
        linalg_429975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 16), scipy_429974, 'linalg')
        # Obtaining the member 'inv' of a type (line 250)
        inv_429976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 16), linalg_429975, 'inv')
        # Calling inv(args, kwargs) (line 250)
        inv_call_result_429985 = invoke(stypy.reporting.localization.Localization(__file__, 250, 16), inv_429976, *[randn_call_result_429983], **kwargs_429984)
        
        # Assigning a type to the variable 'A' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'A', inv_call_result_429985)
        
        # Assigning a Call to a Tuple (line 253):
        
        # Assigning a Subscript to a Name (line 253):
        
        # Obtaining the type of the subscript
        int_429986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 12), 'int')
        
        # Call to _algorithm_2_2(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'A' (line 253)
        A_429988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 36), 'A', False)
        # Getting the type of 'A' (line 253)
        A_429989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 39), 'A', False)
        # Obtaining the member 'T' of a type (line 253)
        T_429990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 39), A_429989, 'T')
        # Getting the type of 't' (line 253)
        t_429991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 44), 't', False)
        # Processing the call keyword arguments (line 253)
        kwargs_429992 = {}
        # Getting the type of '_algorithm_2_2' (line 253)
        _algorithm_2_2_429987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 21), '_algorithm_2_2', False)
        # Calling _algorithm_2_2(args, kwargs) (line 253)
        _algorithm_2_2_call_result_429993 = invoke(stypy.reporting.localization.Localization(__file__, 253, 21), _algorithm_2_2_429987, *[A_429988, T_429990, t_429991], **kwargs_429992)
        
        # Obtaining the member '__getitem__' of a type (line 253)
        getitem___429994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 12), _algorithm_2_2_call_result_429993, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 253)
        subscript_call_result_429995 = invoke(stypy.reporting.localization.Localization(__file__, 253, 12), getitem___429994, int_429986)
        
        # Assigning a type to the variable 'tuple_var_assignment_428728' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'tuple_var_assignment_428728', subscript_call_result_429995)
        
        # Assigning a Subscript to a Name (line 253):
        
        # Obtaining the type of the subscript
        int_429996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 12), 'int')
        
        # Call to _algorithm_2_2(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'A' (line 253)
        A_429998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 36), 'A', False)
        # Getting the type of 'A' (line 253)
        A_429999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 39), 'A', False)
        # Obtaining the member 'T' of a type (line 253)
        T_430000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 39), A_429999, 'T')
        # Getting the type of 't' (line 253)
        t_430001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 44), 't', False)
        # Processing the call keyword arguments (line 253)
        kwargs_430002 = {}
        # Getting the type of '_algorithm_2_2' (line 253)
        _algorithm_2_2_429997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 21), '_algorithm_2_2', False)
        # Calling _algorithm_2_2(args, kwargs) (line 253)
        _algorithm_2_2_call_result_430003 = invoke(stypy.reporting.localization.Localization(__file__, 253, 21), _algorithm_2_2_429997, *[A_429998, T_430000, t_430001], **kwargs_430002)
        
        # Obtaining the member '__getitem__' of a type (line 253)
        getitem___430004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 12), _algorithm_2_2_call_result_430003, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 253)
        subscript_call_result_430005 = invoke(stypy.reporting.localization.Localization(__file__, 253, 12), getitem___430004, int_429996)
        
        # Assigning a type to the variable 'tuple_var_assignment_428729' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'tuple_var_assignment_428729', subscript_call_result_430005)
        
        # Assigning a Name to a Name (line 253):
        # Getting the type of 'tuple_var_assignment_428728' (line 253)
        tuple_var_assignment_428728_430006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'tuple_var_assignment_428728')
        # Assigning a type to the variable 'g' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'g', tuple_var_assignment_428728_430006)
        
        # Assigning a Name to a Name (line 253):
        # Getting the type of 'tuple_var_assignment_428729' (line 253)
        tuple_var_assignment_428729_430007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'tuple_var_assignment_428729')
        # Assigning a type to the variable 'ind' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 15), 'ind', tuple_var_assignment_428729_430007)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_randn_inv(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_randn_inv' in the type store
        # Getting the type of 'stypy_return_type' (line 237)
        stypy_return_type_430008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_430008)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_randn_inv'
        return stypy_return_type_430008


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 235, 0, False)
        # Assigning a type to the variable 'self' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestAlgorithm_2_2.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestAlgorithm_2_2' (line 235)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 0), 'TestAlgorithm_2_2', TestAlgorithm_2_2)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
