
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Unit tests for trust-region iterative subproblem.
3: 
4: To run it in its simplest form::
5:   nosetests test_optimize.py
6: 
7: '''
8: from __future__ import division, print_function, absolute_import
9: 
10: import numpy as np
11: from scipy.optimize._trustregion_exact import (
12:     estimate_smallest_singular_value,
13:     singular_leading_submatrix,
14:     IterativeSubproblem)
15: from scipy.linalg import (svd, get_lapack_funcs, det,
16:                           cho_factor, cho_solve, qr,
17:                           eigvalsh, eig, norm)
18: from numpy.testing import (assert_, assert_array_equal,
19:                            assert_equal, assert_array_almost_equal,
20:                            assert_array_less)
21: 
22: 
23: def random_entry(n, min_eig, max_eig, case):
24: 
25:     # Generate random matrix
26:     rand = np.random.uniform(-1, 1, (n, n))
27: 
28:     # QR decomposition
29:     Q, _, _ = qr(rand, pivoting='True')
30: 
31:     # Generate random eigenvalues
32:     eigvalues = np.random.uniform(min_eig, max_eig, n)
33:     eigvalues = np.sort(eigvalues)[::-1]
34: 
35:     # Generate matrix
36:     Qaux = np.multiply(eigvalues, Q)
37:     A = np.dot(Qaux, Q.T)
38: 
39:     # Generate gradient vector accordingly
40:     # to the case is being tested.
41:     if case == 'hard':
42:         g = np.zeros(n)
43:         g[:-1] = np.random.uniform(-1, 1, n-1)
44:         g = np.dot(Q, g)
45:     elif case == 'jac_equal_zero':
46:         g = np.zeros(n)
47:     else:
48:         g = np.random.uniform(-1, 1, n)
49: 
50:     return A, g
51: 
52: 
53: class TestEstimateSmallestSingularValue(object):
54: 
55:     def test_for_ill_condiotioned_matrix(self):
56: 
57:         # Ill-conditioned triangular matrix
58:         C = np.array([[1, 2, 3, 4],
59:                       [0, 0.05, 60, 7],
60:                       [0, 0, 0.8, 9],
61:                       [0, 0, 0, 10]])
62: 
63:         # Get svd decomposition
64:         U, s, Vt = svd(C)
65: 
66:         # Get smallest singular value and correspondent right singular vector.
67:         smin_svd = s[-1]
68:         zmin_svd = Vt[-1, :]
69: 
70:         # Estimate smallest singular value
71:         smin, zmin = estimate_smallest_singular_value(C)
72: 
73:         # Check the estimation
74:         assert_array_almost_equal(smin, smin_svd, decimal=8)
75:         assert_array_almost_equal(abs(zmin), abs(zmin_svd), decimal=8)
76: 
77: 
78: class TestSingularLeadingSubmatrix(object):
79: 
80:     def test_for_already_singular_leading_submatrix(self):
81: 
82:         # Define test matrix A.
83:         # Note that the leading 2x2 submatrix is singular.
84:         A = np.array([[1, 2, 3],
85:                       [2, 4, 5],
86:                       [3, 5, 6]])
87: 
88:         # Get Cholesky from lapack functions
89:         cholesky, = get_lapack_funcs(('potrf',), (A,))
90: 
91:         # Compute Cholesky Decomposition
92:         c, k = cholesky(A, lower=False, overwrite_a=False, clean=True)
93: 
94:         delta, v = singular_leading_submatrix(A, c, k)
95: 
96:         A[k-1, k-1] += delta
97: 
98:         # Check if the leading submatrix is singular.
99:         assert_array_almost_equal(det(A[:k, :k]), 0)
100: 
101:         # Check if `v` fullfil the specified properties
102:         quadratic_term = np.dot(v, np.dot(A, v))
103:         assert_array_almost_equal(quadratic_term, 0)
104: 
105:     def test_for_simetric_indefinite_matrix(self):
106: 
107:         # Define test matrix A.
108:         # Note that the leading 5x5 submatrix is indefinite.
109:         A = np.asarray([[1, 2, 3, 7, 8],
110:                         [2, 5, 5, 9, 0],
111:                         [3, 5, 11, 1, 2],
112:                         [7, 9, 1, 7, 5],
113:                         [8, 0, 2, 5, 8]])
114: 
115:         # Get Cholesky from lapack functions
116:         cholesky, = get_lapack_funcs(('potrf',), (A,))
117: 
118:         # Compute Cholesky Decomposition
119:         c, k = cholesky(A, lower=False, overwrite_a=False, clean=True)
120: 
121:         delta, v = singular_leading_submatrix(A, c, k)
122: 
123:         A[k-1, k-1] += delta
124: 
125:         # Check if the leading submatrix is singular.
126:         assert_array_almost_equal(det(A[:k, :k]), 0)
127: 
128:         # Check if `v` fullfil the specified properties
129:         quadratic_term = np.dot(v, np.dot(A, v))
130:         assert_array_almost_equal(quadratic_term, 0)
131: 
132:     def test_for_first_element_equal_to_zero(self):
133: 
134:         # Define test matrix A.
135:         # Note that the leading 2x2 submatrix is singular.
136:         A = np.array([[0, 3, 11],
137:                       [3, 12, 5],
138:                       [11, 5, 6]])
139: 
140:         # Get Cholesky from lapack functions
141:         cholesky, = get_lapack_funcs(('potrf',), (A,))
142: 
143:         # Compute Cholesky Decomposition
144:         c, k = cholesky(A, lower=False, overwrite_a=False, clean=True)
145: 
146:         delta, v = singular_leading_submatrix(A, c, k)
147: 
148:         A[k-1, k-1] += delta
149: 
150:         # Check if the leading submatrix is singular
151:         assert_array_almost_equal(det(A[:k, :k]), 0)
152: 
153:         # Check if `v` fullfil the specified properties
154:         quadratic_term = np.dot(v, np.dot(A, v))
155:         assert_array_almost_equal(quadratic_term, 0)
156: 
157: 
158: class TestIterativeSubproblem(object):
159: 
160:     def test_for_the_easy_case(self):
161: 
162:         # `H` is chosen such that `g` is not orthogonal to the
163:         # eigenvector associated with the smallest eigenvalue `s`.
164:         H = [[10, 2, 3, 4],
165:              [2, 1, 7, 1],
166:              [3, 7, 1, 7],
167:              [4, 1, 7, 2]]
168:         g = [1, 1, 1, 1]
169: 
170:         # Trust Radius
171:         trust_radius = 1
172: 
173:         # Solve Subproblem
174:         subprob = IterativeSubproblem(x=0,
175:                                       fun=lambda x: 0,
176:                                       jac=lambda x: np.array(g),
177:                                       hess=lambda x: np.array(H),
178:                                       k_easy=1e-10,
179:                                       k_hard=1e-10)
180:         p, hits_boundary = subprob.solve(trust_radius)
181: 
182:         assert_array_almost_equal(p, [0.00393332, -0.55260862,
183:                                       0.67065477, -0.49480341])
184:         assert_array_almost_equal(hits_boundary, True)
185: 
186:     def test_for_the_hard_case(self):
187: 
188:         # `H` is chosen such that `g` is orthogonal to the
189:         # eigenvector associated with the smallest eigenvalue `s`.
190:         H = [[10, 2, 3, 4],
191:              [2, 1, 7, 1],
192:              [3, 7, 1, 7],
193:              [4, 1, 7, 2]]
194:         g = [6.4852641521327437, 1, 1, 1]
195:         s = -8.2151519874416614
196: 
197:         # Trust Radius
198:         trust_radius = 1
199: 
200:         # Solve Subproblem
201:         subprob = IterativeSubproblem(x=0,
202:                                       fun=lambda x: 0,
203:                                       jac=lambda x: np.array(g),
204:                                       hess=lambda x: np.array(H),
205:                                       k_easy=1e-10,
206:                                       k_hard=1e-10)
207:         p, hits_boundary = subprob.solve(trust_radius)
208: 
209:         assert_array_almost_equal(-s, subprob.lambda_current)
210: 
211:     def test_for_interior_convergence(self):
212: 
213:         H = [[1.812159, 0.82687265, 0.21838879, -0.52487006, 0.25436988],
214:              [0.82687265, 2.66380283, 0.31508988, -0.40144163, 0.08811588],
215:              [0.21838879, 0.31508988, 2.38020726, -0.3166346, 0.27363867],
216:              [-0.52487006, -0.40144163, -0.3166346, 1.61927182, -0.42140166],
217:              [0.25436988, 0.08811588, 0.27363867, -0.42140166, 1.33243101]]
218: 
219:         g = [0.75798952, 0.01421945, 0.33847612, 0.83725004, -0.47909534]
220: 
221:         # Solve Subproblem
222:         subprob = IterativeSubproblem(x=0,
223:                                       fun=lambda x: 0,
224:                                       jac=lambda x: np.array(g),
225:                                       hess=lambda x: np.array(H))
226:         p, hits_boundary = subprob.solve(1.1)
227: 
228:         assert_array_almost_equal(p, [-0.68585435, 0.1222621, -0.22090999,
229:                                       -0.67005053, 0.31586769])
230:         assert_array_almost_equal(hits_boundary, False)
231:         assert_array_almost_equal(subprob.lambda_current, 0)
232:         assert_array_almost_equal(subprob.niter, 1)
233: 
234:     def test_for_jac_equal_zero(self):
235: 
236:         H = [[0.88547534, 2.90692271, 0.98440885, -0.78911503, -0.28035809],
237:              [2.90692271, -0.04618819, 0.32867263, -0.83737945, 0.17116396],
238:              [0.98440885, 0.32867263, -0.87355957, -0.06521957, -1.43030957],
239:              [-0.78911503, -0.83737945, -0.06521957, -1.645709, -0.33887298],
240:              [-0.28035809, 0.17116396, -1.43030957, -0.33887298, -1.68586978]]
241: 
242:         g = [0, 0, 0, 0, 0]
243: 
244:         # Solve Subproblem
245:         subprob = IterativeSubproblem(x=0,
246:                                       fun=lambda x: 0,
247:                                       jac=lambda x: np.array(g),
248:                                       hess=lambda x: np.array(H),
249:                                       k_easy=1e-10,
250:                                       k_hard=1e-10)
251:         p, hits_boundary = subprob.solve(1.1)
252: 
253:         assert_array_almost_equal(p, [0.06910534, -0.01432721,
254:                                       -0.65311947, -0.23815972,
255:                                       -0.84954934])
256:         assert_array_almost_equal(hits_boundary, True)
257: 
258:     def test_for_jac_very_close_to_zero(self):
259: 
260:         H = [[0.88547534, 2.90692271, 0.98440885, -0.78911503, -0.28035809],
261:              [2.90692271, -0.04618819, 0.32867263, -0.83737945, 0.17116396],
262:              [0.98440885, 0.32867263, -0.87355957, -0.06521957, -1.43030957],
263:              [-0.78911503, -0.83737945, -0.06521957, -1.645709, -0.33887298],
264:              [-0.28035809, 0.17116396, -1.43030957, -0.33887298, -1.68586978]]
265: 
266:         g = [0, 0, 0, 0, 1e-15]
267: 
268:         # Solve Subproblem
269:         subprob = IterativeSubproblem(x=0,
270:                                       fun=lambda x: 0,
271:                                       jac=lambda x: np.array(g),
272:                                       hess=lambda x: np.array(H),
273:                                       k_easy=1e-10,
274:                                       k_hard=1e-10)
275:         p, hits_boundary = subprob.solve(1.1)
276: 
277:         assert_array_almost_equal(p, [0.06910534, -0.01432721,
278:                                       -0.65311947, -0.23815972,
279:                                       -0.84954934])
280:         assert_array_almost_equal(hits_boundary, True)
281: 
282:     def test_for_random_entries(self):
283:         # Seed
284:         np.random.seed(1)
285: 
286:         # Dimension
287:         n = 5
288: 
289:         for case in ('easy', 'hard', 'jac_equal_zero'):
290: 
291:             eig_limits = [(-20, -15),
292:                           (-10, -5),
293:                           (-10, 0),
294:                           (-5, 5),
295:                           (-10, 10),
296:                           (0, 10),
297:                           (5, 10),
298:                           (15, 20)]
299: 
300:             for min_eig, max_eig in eig_limits:
301:                 # Generate random symmetric matrix H with
302:                 # eigenvalues between min_eig and max_eig.
303:                 H, g = random_entry(n, min_eig, max_eig, case)
304: 
305:                 # Trust radius
306:                 trust_radius_list = [0.1, 0.3, 0.6, 0.8, 1, 1.2, 3.3, 5.5, 10]
307: 
308:                 for trust_radius in trust_radius_list:
309:                     # Solve subproblem with very high accuracy
310:                     subprob_ac = IterativeSubproblem(0,
311:                                                      lambda x: 0,
312:                                                      lambda x: g,
313:                                                      lambda x: H,
314:                                                      k_easy=1e-10,
315:                                                      k_hard=1e-10)
316: 
317:                     p_ac, hits_boundary_ac = subprob_ac.solve(trust_radius)
318: 
319:                     # Compute objective function value
320:                     J_ac = 1/2*np.dot(p_ac, np.dot(H, p_ac))+np.dot(g, p_ac)
321: 
322:                     stop_criteria = [(0.1, 2),
323:                                      (0.5, 1.1),
324:                                      (0.9, 1.01)]
325: 
326:                     for k_opt, k_trf in stop_criteria:
327: 
328:                         # k_easy and k_hard computed in function
329:                         # of k_opt and k_trf accordingly to
330:                         # Conn, A. R., Gould, N. I., & Toint, P. L. (2000).
331:                         # "Trust region methods". Siam. p. 197.
332:                         k_easy = min(k_trf-1,
333:                                      1-np.sqrt(k_opt))
334:                         k_hard = 1-k_opt
335: 
336:                         # Solve subproblem
337:                         subprob = IterativeSubproblem(0,
338:                                                       lambda x: 0,
339:                                                       lambda x: g,
340:                                                       lambda x: H,
341:                                                       k_easy=k_easy,
342:                                                       k_hard=k_hard)
343:                         p, hits_boundary = subprob.solve(trust_radius)
344: 
345:                         # Compute objective function value
346:                         J = 1/2*np.dot(p, np.dot(H, p))+np.dot(g, p)
347: 
348:                         # Check if it respect k_trf
349:                         if hits_boundary:
350:                             assert_array_equal(np.abs(norm(p)-trust_radius) <=
351:                                                (k_trf-1)*trust_radius, True)
352:                         else:
353:                             assert_equal(norm(p) <= trust_radius, True)
354: 
355:                         # Check if it respect k_opt
356:                         assert_equal(J <= k_opt*J_ac, True)
357: 
358: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_234296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', '\nUnit tests for trust-region iterative subproblem.\n\nTo run it in its simplest form::\n  nosetests test_optimize.py\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import numpy' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_234297 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy')

if (type(import_234297) is not StypyTypeError):

    if (import_234297 != 'pyd_module'):
        __import__(import_234297)
        sys_modules_234298 = sys.modules[import_234297]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', sys_modules_234298.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', import_234297)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.optimize._trustregion_exact import estimate_smallest_singular_value, singular_leading_submatrix, IterativeSubproblem' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_234299 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize._trustregion_exact')

if (type(import_234299) is not StypyTypeError):

    if (import_234299 != 'pyd_module'):
        __import__(import_234299)
        sys_modules_234300 = sys.modules[import_234299]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize._trustregion_exact', sys_modules_234300.module_type_store, module_type_store, ['estimate_smallest_singular_value', 'singular_leading_submatrix', 'IterativeSubproblem'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_234300, sys_modules_234300.module_type_store, module_type_store)
    else:
        from scipy.optimize._trustregion_exact import estimate_smallest_singular_value, singular_leading_submatrix, IterativeSubproblem

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize._trustregion_exact', None, module_type_store, ['estimate_smallest_singular_value', 'singular_leading_submatrix', 'IterativeSubproblem'], [estimate_smallest_singular_value, singular_leading_submatrix, IterativeSubproblem])

else:
    # Assigning a type to the variable 'scipy.optimize._trustregion_exact' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize._trustregion_exact', import_234299)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from scipy.linalg import svd, get_lapack_funcs, det, cho_factor, cho_solve, qr, eigvalsh, eig, norm' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_234301 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.linalg')

if (type(import_234301) is not StypyTypeError):

    if (import_234301 != 'pyd_module'):
        __import__(import_234301)
        sys_modules_234302 = sys.modules[import_234301]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.linalg', sys_modules_234302.module_type_store, module_type_store, ['svd', 'get_lapack_funcs', 'det', 'cho_factor', 'cho_solve', 'qr', 'eigvalsh', 'eig', 'norm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_234302, sys_modules_234302.module_type_store, module_type_store)
    else:
        from scipy.linalg import svd, get_lapack_funcs, det, cho_factor, cho_solve, qr, eigvalsh, eig, norm

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.linalg', None, module_type_store, ['svd', 'get_lapack_funcs', 'det', 'cho_factor', 'cho_solve', 'qr', 'eigvalsh', 'eig', 'norm'], [svd, get_lapack_funcs, det, cho_factor, cho_solve, qr, eigvalsh, eig, norm])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.linalg', import_234301)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from numpy.testing import assert_, assert_array_equal, assert_equal, assert_array_almost_equal, assert_array_less' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_234303 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.testing')

if (type(import_234303) is not StypyTypeError):

    if (import_234303 != 'pyd_module'):
        __import__(import_234303)
        sys_modules_234304 = sys.modules[import_234303]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.testing', sys_modules_234304.module_type_store, module_type_store, ['assert_', 'assert_array_equal', 'assert_equal', 'assert_array_almost_equal', 'assert_array_less'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_234304, sys_modules_234304.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_array_equal, assert_equal, assert_array_almost_equal, assert_array_less

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_array_equal', 'assert_equal', 'assert_array_almost_equal', 'assert_array_less'], [assert_, assert_array_equal, assert_equal, assert_array_almost_equal, assert_array_less])

else:
    # Assigning a type to the variable 'numpy.testing' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.testing', import_234303)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')


@norecursion
def random_entry(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'random_entry'
    module_type_store = module_type_store.open_function_context('random_entry', 23, 0, False)
    
    # Passed parameters checking function
    random_entry.stypy_localization = localization
    random_entry.stypy_type_of_self = None
    random_entry.stypy_type_store = module_type_store
    random_entry.stypy_function_name = 'random_entry'
    random_entry.stypy_param_names_list = ['n', 'min_eig', 'max_eig', 'case']
    random_entry.stypy_varargs_param_name = None
    random_entry.stypy_kwargs_param_name = None
    random_entry.stypy_call_defaults = defaults
    random_entry.stypy_call_varargs = varargs
    random_entry.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'random_entry', ['n', 'min_eig', 'max_eig', 'case'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'random_entry', localization, ['n', 'min_eig', 'max_eig', 'case'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'random_entry(...)' code ##################

    
    # Assigning a Call to a Name (line 26):
    
    # Assigning a Call to a Name (line 26):
    
    # Call to uniform(...): (line 26)
    # Processing the call arguments (line 26)
    int_234308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 29), 'int')
    int_234309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 33), 'int')
    
    # Obtaining an instance of the builtin type 'tuple' (line 26)
    tuple_234310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 26)
    # Adding element type (line 26)
    # Getting the type of 'n' (line 26)
    n_234311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 37), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 37), tuple_234310, n_234311)
    # Adding element type (line 26)
    # Getting the type of 'n' (line 26)
    n_234312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 40), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 37), tuple_234310, n_234312)
    
    # Processing the call keyword arguments (line 26)
    kwargs_234313 = {}
    # Getting the type of 'np' (line 26)
    np_234305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'np', False)
    # Obtaining the member 'random' of a type (line 26)
    random_234306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 11), np_234305, 'random')
    # Obtaining the member 'uniform' of a type (line 26)
    uniform_234307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 11), random_234306, 'uniform')
    # Calling uniform(args, kwargs) (line 26)
    uniform_call_result_234314 = invoke(stypy.reporting.localization.Localization(__file__, 26, 11), uniform_234307, *[int_234308, int_234309, tuple_234310], **kwargs_234313)
    
    # Assigning a type to the variable 'rand' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'rand', uniform_call_result_234314)
    
    # Assigning a Call to a Tuple (line 29):
    
    # Assigning a Subscript to a Name (line 29):
    
    # Obtaining the type of the subscript
    int_234315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'int')
    
    # Call to qr(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'rand' (line 29)
    rand_234317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 17), 'rand', False)
    # Processing the call keyword arguments (line 29)
    str_234318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 32), 'str', 'True')
    keyword_234319 = str_234318
    kwargs_234320 = {'pivoting': keyword_234319}
    # Getting the type of 'qr' (line 29)
    qr_234316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 14), 'qr', False)
    # Calling qr(args, kwargs) (line 29)
    qr_call_result_234321 = invoke(stypy.reporting.localization.Localization(__file__, 29, 14), qr_234316, *[rand_234317], **kwargs_234320)
    
    # Obtaining the member '__getitem__' of a type (line 29)
    getitem___234322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 4), qr_call_result_234321, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 29)
    subscript_call_result_234323 = invoke(stypy.reporting.localization.Localization(__file__, 29, 4), getitem___234322, int_234315)
    
    # Assigning a type to the variable 'tuple_var_assignment_234257' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'tuple_var_assignment_234257', subscript_call_result_234323)
    
    # Assigning a Subscript to a Name (line 29):
    
    # Obtaining the type of the subscript
    int_234324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'int')
    
    # Call to qr(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'rand' (line 29)
    rand_234326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 17), 'rand', False)
    # Processing the call keyword arguments (line 29)
    str_234327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 32), 'str', 'True')
    keyword_234328 = str_234327
    kwargs_234329 = {'pivoting': keyword_234328}
    # Getting the type of 'qr' (line 29)
    qr_234325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 14), 'qr', False)
    # Calling qr(args, kwargs) (line 29)
    qr_call_result_234330 = invoke(stypy.reporting.localization.Localization(__file__, 29, 14), qr_234325, *[rand_234326], **kwargs_234329)
    
    # Obtaining the member '__getitem__' of a type (line 29)
    getitem___234331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 4), qr_call_result_234330, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 29)
    subscript_call_result_234332 = invoke(stypy.reporting.localization.Localization(__file__, 29, 4), getitem___234331, int_234324)
    
    # Assigning a type to the variable 'tuple_var_assignment_234258' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'tuple_var_assignment_234258', subscript_call_result_234332)
    
    # Assigning a Subscript to a Name (line 29):
    
    # Obtaining the type of the subscript
    int_234333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'int')
    
    # Call to qr(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'rand' (line 29)
    rand_234335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 17), 'rand', False)
    # Processing the call keyword arguments (line 29)
    str_234336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 32), 'str', 'True')
    keyword_234337 = str_234336
    kwargs_234338 = {'pivoting': keyword_234337}
    # Getting the type of 'qr' (line 29)
    qr_234334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 14), 'qr', False)
    # Calling qr(args, kwargs) (line 29)
    qr_call_result_234339 = invoke(stypy.reporting.localization.Localization(__file__, 29, 14), qr_234334, *[rand_234335], **kwargs_234338)
    
    # Obtaining the member '__getitem__' of a type (line 29)
    getitem___234340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 4), qr_call_result_234339, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 29)
    subscript_call_result_234341 = invoke(stypy.reporting.localization.Localization(__file__, 29, 4), getitem___234340, int_234333)
    
    # Assigning a type to the variable 'tuple_var_assignment_234259' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'tuple_var_assignment_234259', subscript_call_result_234341)
    
    # Assigning a Name to a Name (line 29):
    # Getting the type of 'tuple_var_assignment_234257' (line 29)
    tuple_var_assignment_234257_234342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'tuple_var_assignment_234257')
    # Assigning a type to the variable 'Q' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'Q', tuple_var_assignment_234257_234342)
    
    # Assigning a Name to a Name (line 29):
    # Getting the type of 'tuple_var_assignment_234258' (line 29)
    tuple_var_assignment_234258_234343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'tuple_var_assignment_234258')
    # Assigning a type to the variable '_' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 7), '_', tuple_var_assignment_234258_234343)
    
    # Assigning a Name to a Name (line 29):
    # Getting the type of 'tuple_var_assignment_234259' (line 29)
    tuple_var_assignment_234259_234344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'tuple_var_assignment_234259')
    # Assigning a type to the variable '_' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 10), '_', tuple_var_assignment_234259_234344)
    
    # Assigning a Call to a Name (line 32):
    
    # Assigning a Call to a Name (line 32):
    
    # Call to uniform(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'min_eig' (line 32)
    min_eig_234348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 34), 'min_eig', False)
    # Getting the type of 'max_eig' (line 32)
    max_eig_234349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 43), 'max_eig', False)
    # Getting the type of 'n' (line 32)
    n_234350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 52), 'n', False)
    # Processing the call keyword arguments (line 32)
    kwargs_234351 = {}
    # Getting the type of 'np' (line 32)
    np_234345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 16), 'np', False)
    # Obtaining the member 'random' of a type (line 32)
    random_234346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 16), np_234345, 'random')
    # Obtaining the member 'uniform' of a type (line 32)
    uniform_234347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 16), random_234346, 'uniform')
    # Calling uniform(args, kwargs) (line 32)
    uniform_call_result_234352 = invoke(stypy.reporting.localization.Localization(__file__, 32, 16), uniform_234347, *[min_eig_234348, max_eig_234349, n_234350], **kwargs_234351)
    
    # Assigning a type to the variable 'eigvalues' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'eigvalues', uniform_call_result_234352)
    
    # Assigning a Subscript to a Name (line 33):
    
    # Assigning a Subscript to a Name (line 33):
    
    # Obtaining the type of the subscript
    int_234353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 37), 'int')
    slice_234354 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 33, 16), None, None, int_234353)
    
    # Call to sort(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'eigvalues' (line 33)
    eigvalues_234357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 24), 'eigvalues', False)
    # Processing the call keyword arguments (line 33)
    kwargs_234358 = {}
    # Getting the type of 'np' (line 33)
    np_234355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 16), 'np', False)
    # Obtaining the member 'sort' of a type (line 33)
    sort_234356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 16), np_234355, 'sort')
    # Calling sort(args, kwargs) (line 33)
    sort_call_result_234359 = invoke(stypy.reporting.localization.Localization(__file__, 33, 16), sort_234356, *[eigvalues_234357], **kwargs_234358)
    
    # Obtaining the member '__getitem__' of a type (line 33)
    getitem___234360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 16), sort_call_result_234359, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 33)
    subscript_call_result_234361 = invoke(stypy.reporting.localization.Localization(__file__, 33, 16), getitem___234360, slice_234354)
    
    # Assigning a type to the variable 'eigvalues' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'eigvalues', subscript_call_result_234361)
    
    # Assigning a Call to a Name (line 36):
    
    # Assigning a Call to a Name (line 36):
    
    # Call to multiply(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'eigvalues' (line 36)
    eigvalues_234364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 23), 'eigvalues', False)
    # Getting the type of 'Q' (line 36)
    Q_234365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 34), 'Q', False)
    # Processing the call keyword arguments (line 36)
    kwargs_234366 = {}
    # Getting the type of 'np' (line 36)
    np_234362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 11), 'np', False)
    # Obtaining the member 'multiply' of a type (line 36)
    multiply_234363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 11), np_234362, 'multiply')
    # Calling multiply(args, kwargs) (line 36)
    multiply_call_result_234367 = invoke(stypy.reporting.localization.Localization(__file__, 36, 11), multiply_234363, *[eigvalues_234364, Q_234365], **kwargs_234366)
    
    # Assigning a type to the variable 'Qaux' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'Qaux', multiply_call_result_234367)
    
    # Assigning a Call to a Name (line 37):
    
    # Assigning a Call to a Name (line 37):
    
    # Call to dot(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'Qaux' (line 37)
    Qaux_234370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'Qaux', False)
    # Getting the type of 'Q' (line 37)
    Q_234371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 21), 'Q', False)
    # Obtaining the member 'T' of a type (line 37)
    T_234372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 21), Q_234371, 'T')
    # Processing the call keyword arguments (line 37)
    kwargs_234373 = {}
    # Getting the type of 'np' (line 37)
    np_234368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'np', False)
    # Obtaining the member 'dot' of a type (line 37)
    dot_234369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), np_234368, 'dot')
    # Calling dot(args, kwargs) (line 37)
    dot_call_result_234374 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), dot_234369, *[Qaux_234370, T_234372], **kwargs_234373)
    
    # Assigning a type to the variable 'A' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'A', dot_call_result_234374)
    
    
    # Getting the type of 'case' (line 41)
    case_234375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 7), 'case')
    str_234376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 15), 'str', 'hard')
    # Applying the binary operator '==' (line 41)
    result_eq_234377 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 7), '==', case_234375, str_234376)
    
    # Testing the type of an if condition (line 41)
    if_condition_234378 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 4), result_eq_234377)
    # Assigning a type to the variable 'if_condition_234378' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'if_condition_234378', if_condition_234378)
    # SSA begins for if statement (line 41)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 42):
    
    # Assigning a Call to a Name (line 42):
    
    # Call to zeros(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'n' (line 42)
    n_234381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 21), 'n', False)
    # Processing the call keyword arguments (line 42)
    kwargs_234382 = {}
    # Getting the type of 'np' (line 42)
    np_234379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'np', False)
    # Obtaining the member 'zeros' of a type (line 42)
    zeros_234380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 12), np_234379, 'zeros')
    # Calling zeros(args, kwargs) (line 42)
    zeros_call_result_234383 = invoke(stypy.reporting.localization.Localization(__file__, 42, 12), zeros_234380, *[n_234381], **kwargs_234382)
    
    # Assigning a type to the variable 'g' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'g', zeros_call_result_234383)
    
    # Assigning a Call to a Subscript (line 43):
    
    # Assigning a Call to a Subscript (line 43):
    
    # Call to uniform(...): (line 43)
    # Processing the call arguments (line 43)
    int_234387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 35), 'int')
    int_234388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 39), 'int')
    # Getting the type of 'n' (line 43)
    n_234389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 42), 'n', False)
    int_234390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 44), 'int')
    # Applying the binary operator '-' (line 43)
    result_sub_234391 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 42), '-', n_234389, int_234390)
    
    # Processing the call keyword arguments (line 43)
    kwargs_234392 = {}
    # Getting the type of 'np' (line 43)
    np_234384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 17), 'np', False)
    # Obtaining the member 'random' of a type (line 43)
    random_234385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 17), np_234384, 'random')
    # Obtaining the member 'uniform' of a type (line 43)
    uniform_234386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 17), random_234385, 'uniform')
    # Calling uniform(args, kwargs) (line 43)
    uniform_call_result_234393 = invoke(stypy.reporting.localization.Localization(__file__, 43, 17), uniform_234386, *[int_234387, int_234388, result_sub_234391], **kwargs_234392)
    
    # Getting the type of 'g' (line 43)
    g_234394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'g')
    int_234395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 11), 'int')
    slice_234396 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 43, 8), None, int_234395, None)
    # Storing an element on a container (line 43)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 8), g_234394, (slice_234396, uniform_call_result_234393))
    
    # Assigning a Call to a Name (line 44):
    
    # Assigning a Call to a Name (line 44):
    
    # Call to dot(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'Q' (line 44)
    Q_234399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 19), 'Q', False)
    # Getting the type of 'g' (line 44)
    g_234400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 22), 'g', False)
    # Processing the call keyword arguments (line 44)
    kwargs_234401 = {}
    # Getting the type of 'np' (line 44)
    np_234397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'np', False)
    # Obtaining the member 'dot' of a type (line 44)
    dot_234398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 12), np_234397, 'dot')
    # Calling dot(args, kwargs) (line 44)
    dot_call_result_234402 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), dot_234398, *[Q_234399, g_234400], **kwargs_234401)
    
    # Assigning a type to the variable 'g' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'g', dot_call_result_234402)
    # SSA branch for the else part of an if statement (line 41)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'case' (line 45)
    case_234403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 9), 'case')
    str_234404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 17), 'str', 'jac_equal_zero')
    # Applying the binary operator '==' (line 45)
    result_eq_234405 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 9), '==', case_234403, str_234404)
    
    # Testing the type of an if condition (line 45)
    if_condition_234406 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 9), result_eq_234405)
    # Assigning a type to the variable 'if_condition_234406' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 9), 'if_condition_234406', if_condition_234406)
    # SSA begins for if statement (line 45)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 46):
    
    # Assigning a Call to a Name (line 46):
    
    # Call to zeros(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'n' (line 46)
    n_234409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 21), 'n', False)
    # Processing the call keyword arguments (line 46)
    kwargs_234410 = {}
    # Getting the type of 'np' (line 46)
    np_234407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'np', False)
    # Obtaining the member 'zeros' of a type (line 46)
    zeros_234408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 12), np_234407, 'zeros')
    # Calling zeros(args, kwargs) (line 46)
    zeros_call_result_234411 = invoke(stypy.reporting.localization.Localization(__file__, 46, 12), zeros_234408, *[n_234409], **kwargs_234410)
    
    # Assigning a type to the variable 'g' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'g', zeros_call_result_234411)
    # SSA branch for the else part of an if statement (line 45)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 48):
    
    # Assigning a Call to a Name (line 48):
    
    # Call to uniform(...): (line 48)
    # Processing the call arguments (line 48)
    int_234415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 30), 'int')
    int_234416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 34), 'int')
    # Getting the type of 'n' (line 48)
    n_234417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 37), 'n', False)
    # Processing the call keyword arguments (line 48)
    kwargs_234418 = {}
    # Getting the type of 'np' (line 48)
    np_234412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'np', False)
    # Obtaining the member 'random' of a type (line 48)
    random_234413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 12), np_234412, 'random')
    # Obtaining the member 'uniform' of a type (line 48)
    uniform_234414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 12), random_234413, 'uniform')
    # Calling uniform(args, kwargs) (line 48)
    uniform_call_result_234419 = invoke(stypy.reporting.localization.Localization(__file__, 48, 12), uniform_234414, *[int_234415, int_234416, n_234417], **kwargs_234418)
    
    # Assigning a type to the variable 'g' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'g', uniform_call_result_234419)
    # SSA join for if statement (line 45)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 41)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 50)
    tuple_234420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 50)
    # Adding element type (line 50)
    # Getting the type of 'A' (line 50)
    A_234421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'A')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 11), tuple_234420, A_234421)
    # Adding element type (line 50)
    # Getting the type of 'g' (line 50)
    g_234422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 14), 'g')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 11), tuple_234420, g_234422)
    
    # Assigning a type to the variable 'stypy_return_type' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type', tuple_234420)
    
    # ################# End of 'random_entry(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'random_entry' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_234423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_234423)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'random_entry'
    return stypy_return_type_234423

# Assigning a type to the variable 'random_entry' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'random_entry', random_entry)
# Declaration of the 'TestEstimateSmallestSingularValue' class

class TestEstimateSmallestSingularValue(object, ):

    @norecursion
    def test_for_ill_condiotioned_matrix(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_for_ill_condiotioned_matrix'
        module_type_store = module_type_store.open_function_context('test_for_ill_condiotioned_matrix', 55, 4, False)
        # Assigning a type to the variable 'self' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestEstimateSmallestSingularValue.test_for_ill_condiotioned_matrix.__dict__.__setitem__('stypy_localization', localization)
        TestEstimateSmallestSingularValue.test_for_ill_condiotioned_matrix.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestEstimateSmallestSingularValue.test_for_ill_condiotioned_matrix.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestEstimateSmallestSingularValue.test_for_ill_condiotioned_matrix.__dict__.__setitem__('stypy_function_name', 'TestEstimateSmallestSingularValue.test_for_ill_condiotioned_matrix')
        TestEstimateSmallestSingularValue.test_for_ill_condiotioned_matrix.__dict__.__setitem__('stypy_param_names_list', [])
        TestEstimateSmallestSingularValue.test_for_ill_condiotioned_matrix.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestEstimateSmallestSingularValue.test_for_ill_condiotioned_matrix.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestEstimateSmallestSingularValue.test_for_ill_condiotioned_matrix.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestEstimateSmallestSingularValue.test_for_ill_condiotioned_matrix.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestEstimateSmallestSingularValue.test_for_ill_condiotioned_matrix.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestEstimateSmallestSingularValue.test_for_ill_condiotioned_matrix.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestEstimateSmallestSingularValue.test_for_ill_condiotioned_matrix', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_for_ill_condiotioned_matrix', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_for_ill_condiotioned_matrix(...)' code ##################

        
        # Assigning a Call to a Name (line 58):
        
        # Assigning a Call to a Name (line 58):
        
        # Call to array(...): (line 58)
        # Processing the call arguments (line 58)
        
        # Obtaining an instance of the builtin type 'list' (line 58)
        list_234426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 58)
        # Adding element type (line 58)
        
        # Obtaining an instance of the builtin type 'list' (line 58)
        list_234427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 58)
        # Adding element type (line 58)
        int_234428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 22), list_234427, int_234428)
        # Adding element type (line 58)
        int_234429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 22), list_234427, int_234429)
        # Adding element type (line 58)
        int_234430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 22), list_234427, int_234430)
        # Adding element type (line 58)
        int_234431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 22), list_234427, int_234431)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 21), list_234426, list_234427)
        # Adding element type (line 58)
        
        # Obtaining an instance of the builtin type 'list' (line 59)
        list_234432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 59)
        # Adding element type (line 59)
        int_234433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 22), list_234432, int_234433)
        # Adding element type (line 59)
        float_234434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 22), list_234432, float_234434)
        # Adding element type (line 59)
        int_234435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 22), list_234432, int_234435)
        # Adding element type (line 59)
        int_234436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 22), list_234432, int_234436)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 21), list_234426, list_234432)
        # Adding element type (line 58)
        
        # Obtaining an instance of the builtin type 'list' (line 60)
        list_234437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 60)
        # Adding element type (line 60)
        int_234438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 22), list_234437, int_234438)
        # Adding element type (line 60)
        int_234439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 22), list_234437, int_234439)
        # Adding element type (line 60)
        float_234440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 22), list_234437, float_234440)
        # Adding element type (line 60)
        int_234441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 22), list_234437, int_234441)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 21), list_234426, list_234437)
        # Adding element type (line 58)
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_234442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        int_234443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 22), list_234442, int_234443)
        # Adding element type (line 61)
        int_234444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 22), list_234442, int_234444)
        # Adding element type (line 61)
        int_234445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 22), list_234442, int_234445)
        # Adding element type (line 61)
        int_234446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 22), list_234442, int_234446)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 21), list_234426, list_234442)
        
        # Processing the call keyword arguments (line 58)
        kwargs_234447 = {}
        # Getting the type of 'np' (line 58)
        np_234424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 58)
        array_234425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 12), np_234424, 'array')
        # Calling array(args, kwargs) (line 58)
        array_call_result_234448 = invoke(stypy.reporting.localization.Localization(__file__, 58, 12), array_234425, *[list_234426], **kwargs_234447)
        
        # Assigning a type to the variable 'C' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'C', array_call_result_234448)
        
        # Assigning a Call to a Tuple (line 64):
        
        # Assigning a Subscript to a Name (line 64):
        
        # Obtaining the type of the subscript
        int_234449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'int')
        
        # Call to svd(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'C' (line 64)
        C_234451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 23), 'C', False)
        # Processing the call keyword arguments (line 64)
        kwargs_234452 = {}
        # Getting the type of 'svd' (line 64)
        svd_234450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 19), 'svd', False)
        # Calling svd(args, kwargs) (line 64)
        svd_call_result_234453 = invoke(stypy.reporting.localization.Localization(__file__, 64, 19), svd_234450, *[C_234451], **kwargs_234452)
        
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___234454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), svd_call_result_234453, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_234455 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), getitem___234454, int_234449)
        
        # Assigning a type to the variable 'tuple_var_assignment_234260' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_234260', subscript_call_result_234455)
        
        # Assigning a Subscript to a Name (line 64):
        
        # Obtaining the type of the subscript
        int_234456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'int')
        
        # Call to svd(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'C' (line 64)
        C_234458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 23), 'C', False)
        # Processing the call keyword arguments (line 64)
        kwargs_234459 = {}
        # Getting the type of 'svd' (line 64)
        svd_234457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 19), 'svd', False)
        # Calling svd(args, kwargs) (line 64)
        svd_call_result_234460 = invoke(stypy.reporting.localization.Localization(__file__, 64, 19), svd_234457, *[C_234458], **kwargs_234459)
        
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___234461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), svd_call_result_234460, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_234462 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), getitem___234461, int_234456)
        
        # Assigning a type to the variable 'tuple_var_assignment_234261' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_234261', subscript_call_result_234462)
        
        # Assigning a Subscript to a Name (line 64):
        
        # Obtaining the type of the subscript
        int_234463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'int')
        
        # Call to svd(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'C' (line 64)
        C_234465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 23), 'C', False)
        # Processing the call keyword arguments (line 64)
        kwargs_234466 = {}
        # Getting the type of 'svd' (line 64)
        svd_234464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 19), 'svd', False)
        # Calling svd(args, kwargs) (line 64)
        svd_call_result_234467 = invoke(stypy.reporting.localization.Localization(__file__, 64, 19), svd_234464, *[C_234465], **kwargs_234466)
        
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___234468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), svd_call_result_234467, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_234469 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), getitem___234468, int_234463)
        
        # Assigning a type to the variable 'tuple_var_assignment_234262' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_234262', subscript_call_result_234469)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'tuple_var_assignment_234260' (line 64)
        tuple_var_assignment_234260_234470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_234260')
        # Assigning a type to the variable 'U' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'U', tuple_var_assignment_234260_234470)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'tuple_var_assignment_234261' (line 64)
        tuple_var_assignment_234261_234471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_234261')
        # Assigning a type to the variable 's' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 's', tuple_var_assignment_234261_234471)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'tuple_var_assignment_234262' (line 64)
        tuple_var_assignment_234262_234472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_234262')
        # Assigning a type to the variable 'Vt' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 14), 'Vt', tuple_var_assignment_234262_234472)
        
        # Assigning a Subscript to a Name (line 67):
        
        # Assigning a Subscript to a Name (line 67):
        
        # Obtaining the type of the subscript
        int_234473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 21), 'int')
        # Getting the type of 's' (line 67)
        s_234474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 19), 's')
        # Obtaining the member '__getitem__' of a type (line 67)
        getitem___234475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 19), s_234474, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 67)
        subscript_call_result_234476 = invoke(stypy.reporting.localization.Localization(__file__, 67, 19), getitem___234475, int_234473)
        
        # Assigning a type to the variable 'smin_svd' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'smin_svd', subscript_call_result_234476)
        
        # Assigning a Subscript to a Name (line 68):
        
        # Assigning a Subscript to a Name (line 68):
        
        # Obtaining the type of the subscript
        int_234477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 22), 'int')
        slice_234478 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 68, 19), None, None, None)
        # Getting the type of 'Vt' (line 68)
        Vt_234479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 19), 'Vt')
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___234480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 19), Vt_234479, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 68)
        subscript_call_result_234481 = invoke(stypy.reporting.localization.Localization(__file__, 68, 19), getitem___234480, (int_234477, slice_234478))
        
        # Assigning a type to the variable 'zmin_svd' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'zmin_svd', subscript_call_result_234481)
        
        # Assigning a Call to a Tuple (line 71):
        
        # Assigning a Subscript to a Name (line 71):
        
        # Obtaining the type of the subscript
        int_234482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 8), 'int')
        
        # Call to estimate_smallest_singular_value(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'C' (line 71)
        C_234484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 54), 'C', False)
        # Processing the call keyword arguments (line 71)
        kwargs_234485 = {}
        # Getting the type of 'estimate_smallest_singular_value' (line 71)
        estimate_smallest_singular_value_234483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 21), 'estimate_smallest_singular_value', False)
        # Calling estimate_smallest_singular_value(args, kwargs) (line 71)
        estimate_smallest_singular_value_call_result_234486 = invoke(stypy.reporting.localization.Localization(__file__, 71, 21), estimate_smallest_singular_value_234483, *[C_234484], **kwargs_234485)
        
        # Obtaining the member '__getitem__' of a type (line 71)
        getitem___234487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), estimate_smallest_singular_value_call_result_234486, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 71)
        subscript_call_result_234488 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), getitem___234487, int_234482)
        
        # Assigning a type to the variable 'tuple_var_assignment_234263' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'tuple_var_assignment_234263', subscript_call_result_234488)
        
        # Assigning a Subscript to a Name (line 71):
        
        # Obtaining the type of the subscript
        int_234489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 8), 'int')
        
        # Call to estimate_smallest_singular_value(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'C' (line 71)
        C_234491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 54), 'C', False)
        # Processing the call keyword arguments (line 71)
        kwargs_234492 = {}
        # Getting the type of 'estimate_smallest_singular_value' (line 71)
        estimate_smallest_singular_value_234490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 21), 'estimate_smallest_singular_value', False)
        # Calling estimate_smallest_singular_value(args, kwargs) (line 71)
        estimate_smallest_singular_value_call_result_234493 = invoke(stypy.reporting.localization.Localization(__file__, 71, 21), estimate_smallest_singular_value_234490, *[C_234491], **kwargs_234492)
        
        # Obtaining the member '__getitem__' of a type (line 71)
        getitem___234494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), estimate_smallest_singular_value_call_result_234493, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 71)
        subscript_call_result_234495 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), getitem___234494, int_234489)
        
        # Assigning a type to the variable 'tuple_var_assignment_234264' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'tuple_var_assignment_234264', subscript_call_result_234495)
        
        # Assigning a Name to a Name (line 71):
        # Getting the type of 'tuple_var_assignment_234263' (line 71)
        tuple_var_assignment_234263_234496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'tuple_var_assignment_234263')
        # Assigning a type to the variable 'smin' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'smin', tuple_var_assignment_234263_234496)
        
        # Assigning a Name to a Name (line 71):
        # Getting the type of 'tuple_var_assignment_234264' (line 71)
        tuple_var_assignment_234264_234497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'tuple_var_assignment_234264')
        # Assigning a type to the variable 'zmin' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 14), 'zmin', tuple_var_assignment_234264_234497)
        
        # Call to assert_array_almost_equal(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'smin' (line 74)
        smin_234499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 34), 'smin', False)
        # Getting the type of 'smin_svd' (line 74)
        smin_svd_234500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 40), 'smin_svd', False)
        # Processing the call keyword arguments (line 74)
        int_234501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 58), 'int')
        keyword_234502 = int_234501
        kwargs_234503 = {'decimal': keyword_234502}
        # Getting the type of 'assert_array_almost_equal' (line 74)
        assert_array_almost_equal_234498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 74)
        assert_array_almost_equal_call_result_234504 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), assert_array_almost_equal_234498, *[smin_234499, smin_svd_234500], **kwargs_234503)
        
        
        # Call to assert_array_almost_equal(...): (line 75)
        # Processing the call arguments (line 75)
        
        # Call to abs(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'zmin' (line 75)
        zmin_234507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 38), 'zmin', False)
        # Processing the call keyword arguments (line 75)
        kwargs_234508 = {}
        # Getting the type of 'abs' (line 75)
        abs_234506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 34), 'abs', False)
        # Calling abs(args, kwargs) (line 75)
        abs_call_result_234509 = invoke(stypy.reporting.localization.Localization(__file__, 75, 34), abs_234506, *[zmin_234507], **kwargs_234508)
        
        
        # Call to abs(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'zmin_svd' (line 75)
        zmin_svd_234511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 49), 'zmin_svd', False)
        # Processing the call keyword arguments (line 75)
        kwargs_234512 = {}
        # Getting the type of 'abs' (line 75)
        abs_234510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 45), 'abs', False)
        # Calling abs(args, kwargs) (line 75)
        abs_call_result_234513 = invoke(stypy.reporting.localization.Localization(__file__, 75, 45), abs_234510, *[zmin_svd_234511], **kwargs_234512)
        
        # Processing the call keyword arguments (line 75)
        int_234514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 68), 'int')
        keyword_234515 = int_234514
        kwargs_234516 = {'decimal': keyword_234515}
        # Getting the type of 'assert_array_almost_equal' (line 75)
        assert_array_almost_equal_234505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 75)
        assert_array_almost_equal_call_result_234517 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), assert_array_almost_equal_234505, *[abs_call_result_234509, abs_call_result_234513], **kwargs_234516)
        
        
        # ################# End of 'test_for_ill_condiotioned_matrix(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_for_ill_condiotioned_matrix' in the type store
        # Getting the type of 'stypy_return_type' (line 55)
        stypy_return_type_234518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_234518)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_for_ill_condiotioned_matrix'
        return stypy_return_type_234518


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 53, 0, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestEstimateSmallestSingularValue.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestEstimateSmallestSingularValue' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'TestEstimateSmallestSingularValue', TestEstimateSmallestSingularValue)
# Declaration of the 'TestSingularLeadingSubmatrix' class

class TestSingularLeadingSubmatrix(object, ):

    @norecursion
    def test_for_already_singular_leading_submatrix(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_for_already_singular_leading_submatrix'
        module_type_store = module_type_store.open_function_context('test_for_already_singular_leading_submatrix', 80, 4, False)
        # Assigning a type to the variable 'self' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSingularLeadingSubmatrix.test_for_already_singular_leading_submatrix.__dict__.__setitem__('stypy_localization', localization)
        TestSingularLeadingSubmatrix.test_for_already_singular_leading_submatrix.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSingularLeadingSubmatrix.test_for_already_singular_leading_submatrix.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSingularLeadingSubmatrix.test_for_already_singular_leading_submatrix.__dict__.__setitem__('stypy_function_name', 'TestSingularLeadingSubmatrix.test_for_already_singular_leading_submatrix')
        TestSingularLeadingSubmatrix.test_for_already_singular_leading_submatrix.__dict__.__setitem__('stypy_param_names_list', [])
        TestSingularLeadingSubmatrix.test_for_already_singular_leading_submatrix.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSingularLeadingSubmatrix.test_for_already_singular_leading_submatrix.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSingularLeadingSubmatrix.test_for_already_singular_leading_submatrix.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSingularLeadingSubmatrix.test_for_already_singular_leading_submatrix.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSingularLeadingSubmatrix.test_for_already_singular_leading_submatrix.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSingularLeadingSubmatrix.test_for_already_singular_leading_submatrix.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSingularLeadingSubmatrix.test_for_already_singular_leading_submatrix', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_for_already_singular_leading_submatrix', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_for_already_singular_leading_submatrix(...)' code ##################

        
        # Assigning a Call to a Name (line 84):
        
        # Assigning a Call to a Name (line 84):
        
        # Call to array(...): (line 84)
        # Processing the call arguments (line 84)
        
        # Obtaining an instance of the builtin type 'list' (line 84)
        list_234521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 84)
        # Adding element type (line 84)
        
        # Obtaining an instance of the builtin type 'list' (line 84)
        list_234522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 84)
        # Adding element type (line 84)
        int_234523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 22), list_234522, int_234523)
        # Adding element type (line 84)
        int_234524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 22), list_234522, int_234524)
        # Adding element type (line 84)
        int_234525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 22), list_234522, int_234525)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 21), list_234521, list_234522)
        # Adding element type (line 84)
        
        # Obtaining an instance of the builtin type 'list' (line 85)
        list_234526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 85)
        # Adding element type (line 85)
        int_234527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 22), list_234526, int_234527)
        # Adding element type (line 85)
        int_234528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 22), list_234526, int_234528)
        # Adding element type (line 85)
        int_234529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 22), list_234526, int_234529)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 21), list_234521, list_234526)
        # Adding element type (line 84)
        
        # Obtaining an instance of the builtin type 'list' (line 86)
        list_234530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 86)
        # Adding element type (line 86)
        int_234531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 22), list_234530, int_234531)
        # Adding element type (line 86)
        int_234532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 22), list_234530, int_234532)
        # Adding element type (line 86)
        int_234533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 22), list_234530, int_234533)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 21), list_234521, list_234530)
        
        # Processing the call keyword arguments (line 84)
        kwargs_234534 = {}
        # Getting the type of 'np' (line 84)
        np_234519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 84)
        array_234520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 12), np_234519, 'array')
        # Calling array(args, kwargs) (line 84)
        array_call_result_234535 = invoke(stypy.reporting.localization.Localization(__file__, 84, 12), array_234520, *[list_234521], **kwargs_234534)
        
        # Assigning a type to the variable 'A' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'A', array_call_result_234535)
        
        # Assigning a Call to a Tuple (line 89):
        
        # Assigning a Subscript to a Name (line 89):
        
        # Obtaining the type of the subscript
        int_234536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 8), 'int')
        
        # Call to get_lapack_funcs(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Obtaining an instance of the builtin type 'tuple' (line 89)
        tuple_234538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 89)
        # Adding element type (line 89)
        str_234539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 38), 'str', 'potrf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 38), tuple_234538, str_234539)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 89)
        tuple_234540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 89)
        # Adding element type (line 89)
        # Getting the type of 'A' (line 89)
        A_234541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 50), 'A', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 50), tuple_234540, A_234541)
        
        # Processing the call keyword arguments (line 89)
        kwargs_234542 = {}
        # Getting the type of 'get_lapack_funcs' (line 89)
        get_lapack_funcs_234537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 20), 'get_lapack_funcs', False)
        # Calling get_lapack_funcs(args, kwargs) (line 89)
        get_lapack_funcs_call_result_234543 = invoke(stypy.reporting.localization.Localization(__file__, 89, 20), get_lapack_funcs_234537, *[tuple_234538, tuple_234540], **kwargs_234542)
        
        # Obtaining the member '__getitem__' of a type (line 89)
        getitem___234544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), get_lapack_funcs_call_result_234543, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 89)
        subscript_call_result_234545 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), getitem___234544, int_234536)
        
        # Assigning a type to the variable 'tuple_var_assignment_234265' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'tuple_var_assignment_234265', subscript_call_result_234545)
        
        # Assigning a Name to a Name (line 89):
        # Getting the type of 'tuple_var_assignment_234265' (line 89)
        tuple_var_assignment_234265_234546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'tuple_var_assignment_234265')
        # Assigning a type to the variable 'cholesky' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'cholesky', tuple_var_assignment_234265_234546)
        
        # Assigning a Call to a Tuple (line 92):
        
        # Assigning a Subscript to a Name (line 92):
        
        # Obtaining the type of the subscript
        int_234547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 8), 'int')
        
        # Call to cholesky(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'A' (line 92)
        A_234549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'A', False)
        # Processing the call keyword arguments (line 92)
        # Getting the type of 'False' (line 92)
        False_234550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 33), 'False', False)
        keyword_234551 = False_234550
        # Getting the type of 'False' (line 92)
        False_234552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 52), 'False', False)
        keyword_234553 = False_234552
        # Getting the type of 'True' (line 92)
        True_234554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 65), 'True', False)
        keyword_234555 = True_234554
        kwargs_234556 = {'lower': keyword_234551, 'overwrite_a': keyword_234553, 'clean': keyword_234555}
        # Getting the type of 'cholesky' (line 92)
        cholesky_234548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 15), 'cholesky', False)
        # Calling cholesky(args, kwargs) (line 92)
        cholesky_call_result_234557 = invoke(stypy.reporting.localization.Localization(__file__, 92, 15), cholesky_234548, *[A_234549], **kwargs_234556)
        
        # Obtaining the member '__getitem__' of a type (line 92)
        getitem___234558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), cholesky_call_result_234557, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 92)
        subscript_call_result_234559 = invoke(stypy.reporting.localization.Localization(__file__, 92, 8), getitem___234558, int_234547)
        
        # Assigning a type to the variable 'tuple_var_assignment_234266' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'tuple_var_assignment_234266', subscript_call_result_234559)
        
        # Assigning a Subscript to a Name (line 92):
        
        # Obtaining the type of the subscript
        int_234560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 8), 'int')
        
        # Call to cholesky(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'A' (line 92)
        A_234562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'A', False)
        # Processing the call keyword arguments (line 92)
        # Getting the type of 'False' (line 92)
        False_234563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 33), 'False', False)
        keyword_234564 = False_234563
        # Getting the type of 'False' (line 92)
        False_234565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 52), 'False', False)
        keyword_234566 = False_234565
        # Getting the type of 'True' (line 92)
        True_234567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 65), 'True', False)
        keyword_234568 = True_234567
        kwargs_234569 = {'lower': keyword_234564, 'overwrite_a': keyword_234566, 'clean': keyword_234568}
        # Getting the type of 'cholesky' (line 92)
        cholesky_234561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 15), 'cholesky', False)
        # Calling cholesky(args, kwargs) (line 92)
        cholesky_call_result_234570 = invoke(stypy.reporting.localization.Localization(__file__, 92, 15), cholesky_234561, *[A_234562], **kwargs_234569)
        
        # Obtaining the member '__getitem__' of a type (line 92)
        getitem___234571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), cholesky_call_result_234570, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 92)
        subscript_call_result_234572 = invoke(stypy.reporting.localization.Localization(__file__, 92, 8), getitem___234571, int_234560)
        
        # Assigning a type to the variable 'tuple_var_assignment_234267' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'tuple_var_assignment_234267', subscript_call_result_234572)
        
        # Assigning a Name to a Name (line 92):
        # Getting the type of 'tuple_var_assignment_234266' (line 92)
        tuple_var_assignment_234266_234573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'tuple_var_assignment_234266')
        # Assigning a type to the variable 'c' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'c', tuple_var_assignment_234266_234573)
        
        # Assigning a Name to a Name (line 92):
        # Getting the type of 'tuple_var_assignment_234267' (line 92)
        tuple_var_assignment_234267_234574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'tuple_var_assignment_234267')
        # Assigning a type to the variable 'k' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'k', tuple_var_assignment_234267_234574)
        
        # Assigning a Call to a Tuple (line 94):
        
        # Assigning a Subscript to a Name (line 94):
        
        # Obtaining the type of the subscript
        int_234575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 8), 'int')
        
        # Call to singular_leading_submatrix(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'A' (line 94)
        A_234577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 46), 'A', False)
        # Getting the type of 'c' (line 94)
        c_234578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 49), 'c', False)
        # Getting the type of 'k' (line 94)
        k_234579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 52), 'k', False)
        # Processing the call keyword arguments (line 94)
        kwargs_234580 = {}
        # Getting the type of 'singular_leading_submatrix' (line 94)
        singular_leading_submatrix_234576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 19), 'singular_leading_submatrix', False)
        # Calling singular_leading_submatrix(args, kwargs) (line 94)
        singular_leading_submatrix_call_result_234581 = invoke(stypy.reporting.localization.Localization(__file__, 94, 19), singular_leading_submatrix_234576, *[A_234577, c_234578, k_234579], **kwargs_234580)
        
        # Obtaining the member '__getitem__' of a type (line 94)
        getitem___234582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), singular_leading_submatrix_call_result_234581, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 94)
        subscript_call_result_234583 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), getitem___234582, int_234575)
        
        # Assigning a type to the variable 'tuple_var_assignment_234268' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'tuple_var_assignment_234268', subscript_call_result_234583)
        
        # Assigning a Subscript to a Name (line 94):
        
        # Obtaining the type of the subscript
        int_234584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 8), 'int')
        
        # Call to singular_leading_submatrix(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'A' (line 94)
        A_234586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 46), 'A', False)
        # Getting the type of 'c' (line 94)
        c_234587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 49), 'c', False)
        # Getting the type of 'k' (line 94)
        k_234588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 52), 'k', False)
        # Processing the call keyword arguments (line 94)
        kwargs_234589 = {}
        # Getting the type of 'singular_leading_submatrix' (line 94)
        singular_leading_submatrix_234585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 19), 'singular_leading_submatrix', False)
        # Calling singular_leading_submatrix(args, kwargs) (line 94)
        singular_leading_submatrix_call_result_234590 = invoke(stypy.reporting.localization.Localization(__file__, 94, 19), singular_leading_submatrix_234585, *[A_234586, c_234587, k_234588], **kwargs_234589)
        
        # Obtaining the member '__getitem__' of a type (line 94)
        getitem___234591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), singular_leading_submatrix_call_result_234590, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 94)
        subscript_call_result_234592 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), getitem___234591, int_234584)
        
        # Assigning a type to the variable 'tuple_var_assignment_234269' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'tuple_var_assignment_234269', subscript_call_result_234592)
        
        # Assigning a Name to a Name (line 94):
        # Getting the type of 'tuple_var_assignment_234268' (line 94)
        tuple_var_assignment_234268_234593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'tuple_var_assignment_234268')
        # Assigning a type to the variable 'delta' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'delta', tuple_var_assignment_234268_234593)
        
        # Assigning a Name to a Name (line 94):
        # Getting the type of 'tuple_var_assignment_234269' (line 94)
        tuple_var_assignment_234269_234594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'tuple_var_assignment_234269')
        # Assigning a type to the variable 'v' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 15), 'v', tuple_var_assignment_234269_234594)
        
        # Getting the type of 'A' (line 96)
        A_234595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'A')
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 96)
        tuple_234596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 10), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 96)
        # Adding element type (line 96)
        # Getting the type of 'k' (line 96)
        k_234597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 10), 'k')
        int_234598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 12), 'int')
        # Applying the binary operator '-' (line 96)
        result_sub_234599 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 10), '-', k_234597, int_234598)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 10), tuple_234596, result_sub_234599)
        # Adding element type (line 96)
        # Getting the type of 'k' (line 96)
        k_234600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 15), 'k')
        int_234601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 17), 'int')
        # Applying the binary operator '-' (line 96)
        result_sub_234602 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 15), '-', k_234600, int_234601)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 10), tuple_234596, result_sub_234602)
        
        # Getting the type of 'A' (line 96)
        A_234603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'A')
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___234604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), A_234603, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 96)
        subscript_call_result_234605 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), getitem___234604, tuple_234596)
        
        # Getting the type of 'delta' (line 96)
        delta_234606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 23), 'delta')
        # Applying the binary operator '+=' (line 96)
        result_iadd_234607 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 8), '+=', subscript_call_result_234605, delta_234606)
        # Getting the type of 'A' (line 96)
        A_234608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 96)
        tuple_234609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 10), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 96)
        # Adding element type (line 96)
        # Getting the type of 'k' (line 96)
        k_234610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 10), 'k')
        int_234611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 12), 'int')
        # Applying the binary operator '-' (line 96)
        result_sub_234612 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 10), '-', k_234610, int_234611)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 10), tuple_234609, result_sub_234612)
        # Adding element type (line 96)
        # Getting the type of 'k' (line 96)
        k_234613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 15), 'k')
        int_234614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 17), 'int')
        # Applying the binary operator '-' (line 96)
        result_sub_234615 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 15), '-', k_234613, int_234614)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 10), tuple_234609, result_sub_234615)
        
        # Storing an element on a container (line 96)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 8), A_234608, (tuple_234609, result_iadd_234607))
        
        
        # Call to assert_array_almost_equal(...): (line 99)
        # Processing the call arguments (line 99)
        
        # Call to det(...): (line 99)
        # Processing the call arguments (line 99)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 99)
        k_234618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 41), 'k', False)
        slice_234619 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 99, 38), None, k_234618, None)
        # Getting the type of 'k' (line 99)
        k_234620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 45), 'k', False)
        slice_234621 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 99, 38), None, k_234620, None)
        # Getting the type of 'A' (line 99)
        A_234622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 38), 'A', False)
        # Obtaining the member '__getitem__' of a type (line 99)
        getitem___234623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 38), A_234622, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 99)
        subscript_call_result_234624 = invoke(stypy.reporting.localization.Localization(__file__, 99, 38), getitem___234623, (slice_234619, slice_234621))
        
        # Processing the call keyword arguments (line 99)
        kwargs_234625 = {}
        # Getting the type of 'det' (line 99)
        det_234617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 34), 'det', False)
        # Calling det(args, kwargs) (line 99)
        det_call_result_234626 = invoke(stypy.reporting.localization.Localization(__file__, 99, 34), det_234617, *[subscript_call_result_234624], **kwargs_234625)
        
        int_234627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 50), 'int')
        # Processing the call keyword arguments (line 99)
        kwargs_234628 = {}
        # Getting the type of 'assert_array_almost_equal' (line 99)
        assert_array_almost_equal_234616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 99)
        assert_array_almost_equal_call_result_234629 = invoke(stypy.reporting.localization.Localization(__file__, 99, 8), assert_array_almost_equal_234616, *[det_call_result_234626, int_234627], **kwargs_234628)
        
        
        # Assigning a Call to a Name (line 102):
        
        # Assigning a Call to a Name (line 102):
        
        # Call to dot(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'v' (line 102)
        v_234632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 32), 'v', False)
        
        # Call to dot(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'A' (line 102)
        A_234635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 42), 'A', False)
        # Getting the type of 'v' (line 102)
        v_234636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 45), 'v', False)
        # Processing the call keyword arguments (line 102)
        kwargs_234637 = {}
        # Getting the type of 'np' (line 102)
        np_234633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 35), 'np', False)
        # Obtaining the member 'dot' of a type (line 102)
        dot_234634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 35), np_234633, 'dot')
        # Calling dot(args, kwargs) (line 102)
        dot_call_result_234638 = invoke(stypy.reporting.localization.Localization(__file__, 102, 35), dot_234634, *[A_234635, v_234636], **kwargs_234637)
        
        # Processing the call keyword arguments (line 102)
        kwargs_234639 = {}
        # Getting the type of 'np' (line 102)
        np_234630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 25), 'np', False)
        # Obtaining the member 'dot' of a type (line 102)
        dot_234631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 25), np_234630, 'dot')
        # Calling dot(args, kwargs) (line 102)
        dot_call_result_234640 = invoke(stypy.reporting.localization.Localization(__file__, 102, 25), dot_234631, *[v_234632, dot_call_result_234638], **kwargs_234639)
        
        # Assigning a type to the variable 'quadratic_term' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'quadratic_term', dot_call_result_234640)
        
        # Call to assert_array_almost_equal(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'quadratic_term' (line 103)
        quadratic_term_234642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 34), 'quadratic_term', False)
        int_234643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 50), 'int')
        # Processing the call keyword arguments (line 103)
        kwargs_234644 = {}
        # Getting the type of 'assert_array_almost_equal' (line 103)
        assert_array_almost_equal_234641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 103)
        assert_array_almost_equal_call_result_234645 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), assert_array_almost_equal_234641, *[quadratic_term_234642, int_234643], **kwargs_234644)
        
        
        # ################# End of 'test_for_already_singular_leading_submatrix(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_for_already_singular_leading_submatrix' in the type store
        # Getting the type of 'stypy_return_type' (line 80)
        stypy_return_type_234646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_234646)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_for_already_singular_leading_submatrix'
        return stypy_return_type_234646


    @norecursion
    def test_for_simetric_indefinite_matrix(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_for_simetric_indefinite_matrix'
        module_type_store = module_type_store.open_function_context('test_for_simetric_indefinite_matrix', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSingularLeadingSubmatrix.test_for_simetric_indefinite_matrix.__dict__.__setitem__('stypy_localization', localization)
        TestSingularLeadingSubmatrix.test_for_simetric_indefinite_matrix.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSingularLeadingSubmatrix.test_for_simetric_indefinite_matrix.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSingularLeadingSubmatrix.test_for_simetric_indefinite_matrix.__dict__.__setitem__('stypy_function_name', 'TestSingularLeadingSubmatrix.test_for_simetric_indefinite_matrix')
        TestSingularLeadingSubmatrix.test_for_simetric_indefinite_matrix.__dict__.__setitem__('stypy_param_names_list', [])
        TestSingularLeadingSubmatrix.test_for_simetric_indefinite_matrix.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSingularLeadingSubmatrix.test_for_simetric_indefinite_matrix.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSingularLeadingSubmatrix.test_for_simetric_indefinite_matrix.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSingularLeadingSubmatrix.test_for_simetric_indefinite_matrix.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSingularLeadingSubmatrix.test_for_simetric_indefinite_matrix.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSingularLeadingSubmatrix.test_for_simetric_indefinite_matrix.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSingularLeadingSubmatrix.test_for_simetric_indefinite_matrix', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_for_simetric_indefinite_matrix', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_for_simetric_indefinite_matrix(...)' code ##################

        
        # Assigning a Call to a Name (line 109):
        
        # Assigning a Call to a Name (line 109):
        
        # Call to asarray(...): (line 109)
        # Processing the call arguments (line 109)
        
        # Obtaining an instance of the builtin type 'list' (line 109)
        list_234649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 109)
        # Adding element type (line 109)
        
        # Obtaining an instance of the builtin type 'list' (line 109)
        list_234650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 109)
        # Adding element type (line 109)
        int_234651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 24), list_234650, int_234651)
        # Adding element type (line 109)
        int_234652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 24), list_234650, int_234652)
        # Adding element type (line 109)
        int_234653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 24), list_234650, int_234653)
        # Adding element type (line 109)
        int_234654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 24), list_234650, int_234654)
        # Adding element type (line 109)
        int_234655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 24), list_234650, int_234655)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 23), list_234649, list_234650)
        # Adding element type (line 109)
        
        # Obtaining an instance of the builtin type 'list' (line 110)
        list_234656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 110)
        # Adding element type (line 110)
        int_234657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 24), list_234656, int_234657)
        # Adding element type (line 110)
        int_234658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 24), list_234656, int_234658)
        # Adding element type (line 110)
        int_234659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 24), list_234656, int_234659)
        # Adding element type (line 110)
        int_234660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 24), list_234656, int_234660)
        # Adding element type (line 110)
        int_234661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 24), list_234656, int_234661)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 23), list_234649, list_234656)
        # Adding element type (line 109)
        
        # Obtaining an instance of the builtin type 'list' (line 111)
        list_234662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 111)
        # Adding element type (line 111)
        int_234663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 24), list_234662, int_234663)
        # Adding element type (line 111)
        int_234664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 24), list_234662, int_234664)
        # Adding element type (line 111)
        int_234665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 24), list_234662, int_234665)
        # Adding element type (line 111)
        int_234666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 24), list_234662, int_234666)
        # Adding element type (line 111)
        int_234667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 24), list_234662, int_234667)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 23), list_234649, list_234662)
        # Adding element type (line 109)
        
        # Obtaining an instance of the builtin type 'list' (line 112)
        list_234668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 112)
        # Adding element type (line 112)
        int_234669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 24), list_234668, int_234669)
        # Adding element type (line 112)
        int_234670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 24), list_234668, int_234670)
        # Adding element type (line 112)
        int_234671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 24), list_234668, int_234671)
        # Adding element type (line 112)
        int_234672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 24), list_234668, int_234672)
        # Adding element type (line 112)
        int_234673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 24), list_234668, int_234673)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 23), list_234649, list_234668)
        # Adding element type (line 109)
        
        # Obtaining an instance of the builtin type 'list' (line 113)
        list_234674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 113)
        # Adding element type (line 113)
        int_234675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 24), list_234674, int_234675)
        # Adding element type (line 113)
        int_234676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 24), list_234674, int_234676)
        # Adding element type (line 113)
        int_234677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 24), list_234674, int_234677)
        # Adding element type (line 113)
        int_234678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 24), list_234674, int_234678)
        # Adding element type (line 113)
        int_234679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 24), list_234674, int_234679)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 23), list_234649, list_234674)
        
        # Processing the call keyword arguments (line 109)
        kwargs_234680 = {}
        # Getting the type of 'np' (line 109)
        np_234647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 109)
        asarray_234648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 12), np_234647, 'asarray')
        # Calling asarray(args, kwargs) (line 109)
        asarray_call_result_234681 = invoke(stypy.reporting.localization.Localization(__file__, 109, 12), asarray_234648, *[list_234649], **kwargs_234680)
        
        # Assigning a type to the variable 'A' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'A', asarray_call_result_234681)
        
        # Assigning a Call to a Tuple (line 116):
        
        # Assigning a Subscript to a Name (line 116):
        
        # Obtaining the type of the subscript
        int_234682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 8), 'int')
        
        # Call to get_lapack_funcs(...): (line 116)
        # Processing the call arguments (line 116)
        
        # Obtaining an instance of the builtin type 'tuple' (line 116)
        tuple_234684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 116)
        # Adding element type (line 116)
        str_234685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 38), 'str', 'potrf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 38), tuple_234684, str_234685)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 116)
        tuple_234686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 116)
        # Adding element type (line 116)
        # Getting the type of 'A' (line 116)
        A_234687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 50), 'A', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 50), tuple_234686, A_234687)
        
        # Processing the call keyword arguments (line 116)
        kwargs_234688 = {}
        # Getting the type of 'get_lapack_funcs' (line 116)
        get_lapack_funcs_234683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 20), 'get_lapack_funcs', False)
        # Calling get_lapack_funcs(args, kwargs) (line 116)
        get_lapack_funcs_call_result_234689 = invoke(stypy.reporting.localization.Localization(__file__, 116, 20), get_lapack_funcs_234683, *[tuple_234684, tuple_234686], **kwargs_234688)
        
        # Obtaining the member '__getitem__' of a type (line 116)
        getitem___234690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), get_lapack_funcs_call_result_234689, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 116)
        subscript_call_result_234691 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), getitem___234690, int_234682)
        
        # Assigning a type to the variable 'tuple_var_assignment_234270' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'tuple_var_assignment_234270', subscript_call_result_234691)
        
        # Assigning a Name to a Name (line 116):
        # Getting the type of 'tuple_var_assignment_234270' (line 116)
        tuple_var_assignment_234270_234692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'tuple_var_assignment_234270')
        # Assigning a type to the variable 'cholesky' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'cholesky', tuple_var_assignment_234270_234692)
        
        # Assigning a Call to a Tuple (line 119):
        
        # Assigning a Subscript to a Name (line 119):
        
        # Obtaining the type of the subscript
        int_234693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 8), 'int')
        
        # Call to cholesky(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'A' (line 119)
        A_234695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 24), 'A', False)
        # Processing the call keyword arguments (line 119)
        # Getting the type of 'False' (line 119)
        False_234696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 33), 'False', False)
        keyword_234697 = False_234696
        # Getting the type of 'False' (line 119)
        False_234698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 52), 'False', False)
        keyword_234699 = False_234698
        # Getting the type of 'True' (line 119)
        True_234700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 65), 'True', False)
        keyword_234701 = True_234700
        kwargs_234702 = {'lower': keyword_234697, 'overwrite_a': keyword_234699, 'clean': keyword_234701}
        # Getting the type of 'cholesky' (line 119)
        cholesky_234694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'cholesky', False)
        # Calling cholesky(args, kwargs) (line 119)
        cholesky_call_result_234703 = invoke(stypy.reporting.localization.Localization(__file__, 119, 15), cholesky_234694, *[A_234695], **kwargs_234702)
        
        # Obtaining the member '__getitem__' of a type (line 119)
        getitem___234704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), cholesky_call_result_234703, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 119)
        subscript_call_result_234705 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), getitem___234704, int_234693)
        
        # Assigning a type to the variable 'tuple_var_assignment_234271' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'tuple_var_assignment_234271', subscript_call_result_234705)
        
        # Assigning a Subscript to a Name (line 119):
        
        # Obtaining the type of the subscript
        int_234706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 8), 'int')
        
        # Call to cholesky(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'A' (line 119)
        A_234708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 24), 'A', False)
        # Processing the call keyword arguments (line 119)
        # Getting the type of 'False' (line 119)
        False_234709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 33), 'False', False)
        keyword_234710 = False_234709
        # Getting the type of 'False' (line 119)
        False_234711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 52), 'False', False)
        keyword_234712 = False_234711
        # Getting the type of 'True' (line 119)
        True_234713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 65), 'True', False)
        keyword_234714 = True_234713
        kwargs_234715 = {'lower': keyword_234710, 'overwrite_a': keyword_234712, 'clean': keyword_234714}
        # Getting the type of 'cholesky' (line 119)
        cholesky_234707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'cholesky', False)
        # Calling cholesky(args, kwargs) (line 119)
        cholesky_call_result_234716 = invoke(stypy.reporting.localization.Localization(__file__, 119, 15), cholesky_234707, *[A_234708], **kwargs_234715)
        
        # Obtaining the member '__getitem__' of a type (line 119)
        getitem___234717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), cholesky_call_result_234716, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 119)
        subscript_call_result_234718 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), getitem___234717, int_234706)
        
        # Assigning a type to the variable 'tuple_var_assignment_234272' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'tuple_var_assignment_234272', subscript_call_result_234718)
        
        # Assigning a Name to a Name (line 119):
        # Getting the type of 'tuple_var_assignment_234271' (line 119)
        tuple_var_assignment_234271_234719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'tuple_var_assignment_234271')
        # Assigning a type to the variable 'c' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'c', tuple_var_assignment_234271_234719)
        
        # Assigning a Name to a Name (line 119):
        # Getting the type of 'tuple_var_assignment_234272' (line 119)
        tuple_var_assignment_234272_234720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'tuple_var_assignment_234272')
        # Assigning a type to the variable 'k' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 11), 'k', tuple_var_assignment_234272_234720)
        
        # Assigning a Call to a Tuple (line 121):
        
        # Assigning a Subscript to a Name (line 121):
        
        # Obtaining the type of the subscript
        int_234721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 8), 'int')
        
        # Call to singular_leading_submatrix(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'A' (line 121)
        A_234723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 46), 'A', False)
        # Getting the type of 'c' (line 121)
        c_234724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 49), 'c', False)
        # Getting the type of 'k' (line 121)
        k_234725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 52), 'k', False)
        # Processing the call keyword arguments (line 121)
        kwargs_234726 = {}
        # Getting the type of 'singular_leading_submatrix' (line 121)
        singular_leading_submatrix_234722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'singular_leading_submatrix', False)
        # Calling singular_leading_submatrix(args, kwargs) (line 121)
        singular_leading_submatrix_call_result_234727 = invoke(stypy.reporting.localization.Localization(__file__, 121, 19), singular_leading_submatrix_234722, *[A_234723, c_234724, k_234725], **kwargs_234726)
        
        # Obtaining the member '__getitem__' of a type (line 121)
        getitem___234728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), singular_leading_submatrix_call_result_234727, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 121)
        subscript_call_result_234729 = invoke(stypy.reporting.localization.Localization(__file__, 121, 8), getitem___234728, int_234721)
        
        # Assigning a type to the variable 'tuple_var_assignment_234273' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'tuple_var_assignment_234273', subscript_call_result_234729)
        
        # Assigning a Subscript to a Name (line 121):
        
        # Obtaining the type of the subscript
        int_234730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 8), 'int')
        
        # Call to singular_leading_submatrix(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'A' (line 121)
        A_234732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 46), 'A', False)
        # Getting the type of 'c' (line 121)
        c_234733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 49), 'c', False)
        # Getting the type of 'k' (line 121)
        k_234734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 52), 'k', False)
        # Processing the call keyword arguments (line 121)
        kwargs_234735 = {}
        # Getting the type of 'singular_leading_submatrix' (line 121)
        singular_leading_submatrix_234731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'singular_leading_submatrix', False)
        # Calling singular_leading_submatrix(args, kwargs) (line 121)
        singular_leading_submatrix_call_result_234736 = invoke(stypy.reporting.localization.Localization(__file__, 121, 19), singular_leading_submatrix_234731, *[A_234732, c_234733, k_234734], **kwargs_234735)
        
        # Obtaining the member '__getitem__' of a type (line 121)
        getitem___234737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), singular_leading_submatrix_call_result_234736, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 121)
        subscript_call_result_234738 = invoke(stypy.reporting.localization.Localization(__file__, 121, 8), getitem___234737, int_234730)
        
        # Assigning a type to the variable 'tuple_var_assignment_234274' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'tuple_var_assignment_234274', subscript_call_result_234738)
        
        # Assigning a Name to a Name (line 121):
        # Getting the type of 'tuple_var_assignment_234273' (line 121)
        tuple_var_assignment_234273_234739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'tuple_var_assignment_234273')
        # Assigning a type to the variable 'delta' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'delta', tuple_var_assignment_234273_234739)
        
        # Assigning a Name to a Name (line 121):
        # Getting the type of 'tuple_var_assignment_234274' (line 121)
        tuple_var_assignment_234274_234740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'tuple_var_assignment_234274')
        # Assigning a type to the variable 'v' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'v', tuple_var_assignment_234274_234740)
        
        # Getting the type of 'A' (line 123)
        A_234741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'A')
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 123)
        tuple_234742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 10), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 123)
        # Adding element type (line 123)
        # Getting the type of 'k' (line 123)
        k_234743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 10), 'k')
        int_234744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 12), 'int')
        # Applying the binary operator '-' (line 123)
        result_sub_234745 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 10), '-', k_234743, int_234744)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 10), tuple_234742, result_sub_234745)
        # Adding element type (line 123)
        # Getting the type of 'k' (line 123)
        k_234746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 15), 'k')
        int_234747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 17), 'int')
        # Applying the binary operator '-' (line 123)
        result_sub_234748 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 15), '-', k_234746, int_234747)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 10), tuple_234742, result_sub_234748)
        
        # Getting the type of 'A' (line 123)
        A_234749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'A')
        # Obtaining the member '__getitem__' of a type (line 123)
        getitem___234750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), A_234749, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 123)
        subscript_call_result_234751 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), getitem___234750, tuple_234742)
        
        # Getting the type of 'delta' (line 123)
        delta_234752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 23), 'delta')
        # Applying the binary operator '+=' (line 123)
        result_iadd_234753 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 8), '+=', subscript_call_result_234751, delta_234752)
        # Getting the type of 'A' (line 123)
        A_234754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 123)
        tuple_234755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 10), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 123)
        # Adding element type (line 123)
        # Getting the type of 'k' (line 123)
        k_234756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 10), 'k')
        int_234757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 12), 'int')
        # Applying the binary operator '-' (line 123)
        result_sub_234758 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 10), '-', k_234756, int_234757)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 10), tuple_234755, result_sub_234758)
        # Adding element type (line 123)
        # Getting the type of 'k' (line 123)
        k_234759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 15), 'k')
        int_234760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 17), 'int')
        # Applying the binary operator '-' (line 123)
        result_sub_234761 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 15), '-', k_234759, int_234760)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 10), tuple_234755, result_sub_234761)
        
        # Storing an element on a container (line 123)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 8), A_234754, (tuple_234755, result_iadd_234753))
        
        
        # Call to assert_array_almost_equal(...): (line 126)
        # Processing the call arguments (line 126)
        
        # Call to det(...): (line 126)
        # Processing the call arguments (line 126)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 126)
        k_234764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 41), 'k', False)
        slice_234765 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 126, 38), None, k_234764, None)
        # Getting the type of 'k' (line 126)
        k_234766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 45), 'k', False)
        slice_234767 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 126, 38), None, k_234766, None)
        # Getting the type of 'A' (line 126)
        A_234768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 38), 'A', False)
        # Obtaining the member '__getitem__' of a type (line 126)
        getitem___234769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 38), A_234768, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 126)
        subscript_call_result_234770 = invoke(stypy.reporting.localization.Localization(__file__, 126, 38), getitem___234769, (slice_234765, slice_234767))
        
        # Processing the call keyword arguments (line 126)
        kwargs_234771 = {}
        # Getting the type of 'det' (line 126)
        det_234763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 34), 'det', False)
        # Calling det(args, kwargs) (line 126)
        det_call_result_234772 = invoke(stypy.reporting.localization.Localization(__file__, 126, 34), det_234763, *[subscript_call_result_234770], **kwargs_234771)
        
        int_234773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 50), 'int')
        # Processing the call keyword arguments (line 126)
        kwargs_234774 = {}
        # Getting the type of 'assert_array_almost_equal' (line 126)
        assert_array_almost_equal_234762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 126)
        assert_array_almost_equal_call_result_234775 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), assert_array_almost_equal_234762, *[det_call_result_234772, int_234773], **kwargs_234774)
        
        
        # Assigning a Call to a Name (line 129):
        
        # Assigning a Call to a Name (line 129):
        
        # Call to dot(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'v' (line 129)
        v_234778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 32), 'v', False)
        
        # Call to dot(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'A' (line 129)
        A_234781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 42), 'A', False)
        # Getting the type of 'v' (line 129)
        v_234782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 45), 'v', False)
        # Processing the call keyword arguments (line 129)
        kwargs_234783 = {}
        # Getting the type of 'np' (line 129)
        np_234779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 35), 'np', False)
        # Obtaining the member 'dot' of a type (line 129)
        dot_234780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 35), np_234779, 'dot')
        # Calling dot(args, kwargs) (line 129)
        dot_call_result_234784 = invoke(stypy.reporting.localization.Localization(__file__, 129, 35), dot_234780, *[A_234781, v_234782], **kwargs_234783)
        
        # Processing the call keyword arguments (line 129)
        kwargs_234785 = {}
        # Getting the type of 'np' (line 129)
        np_234776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 25), 'np', False)
        # Obtaining the member 'dot' of a type (line 129)
        dot_234777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 25), np_234776, 'dot')
        # Calling dot(args, kwargs) (line 129)
        dot_call_result_234786 = invoke(stypy.reporting.localization.Localization(__file__, 129, 25), dot_234777, *[v_234778, dot_call_result_234784], **kwargs_234785)
        
        # Assigning a type to the variable 'quadratic_term' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'quadratic_term', dot_call_result_234786)
        
        # Call to assert_array_almost_equal(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'quadratic_term' (line 130)
        quadratic_term_234788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 34), 'quadratic_term', False)
        int_234789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 50), 'int')
        # Processing the call keyword arguments (line 130)
        kwargs_234790 = {}
        # Getting the type of 'assert_array_almost_equal' (line 130)
        assert_array_almost_equal_234787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 130)
        assert_array_almost_equal_call_result_234791 = invoke(stypy.reporting.localization.Localization(__file__, 130, 8), assert_array_almost_equal_234787, *[quadratic_term_234788, int_234789], **kwargs_234790)
        
        
        # ################# End of 'test_for_simetric_indefinite_matrix(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_for_simetric_indefinite_matrix' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_234792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_234792)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_for_simetric_indefinite_matrix'
        return stypy_return_type_234792


    @norecursion
    def test_for_first_element_equal_to_zero(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_for_first_element_equal_to_zero'
        module_type_store = module_type_store.open_function_context('test_for_first_element_equal_to_zero', 132, 4, False)
        # Assigning a type to the variable 'self' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSingularLeadingSubmatrix.test_for_first_element_equal_to_zero.__dict__.__setitem__('stypy_localization', localization)
        TestSingularLeadingSubmatrix.test_for_first_element_equal_to_zero.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSingularLeadingSubmatrix.test_for_first_element_equal_to_zero.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSingularLeadingSubmatrix.test_for_first_element_equal_to_zero.__dict__.__setitem__('stypy_function_name', 'TestSingularLeadingSubmatrix.test_for_first_element_equal_to_zero')
        TestSingularLeadingSubmatrix.test_for_first_element_equal_to_zero.__dict__.__setitem__('stypy_param_names_list', [])
        TestSingularLeadingSubmatrix.test_for_first_element_equal_to_zero.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSingularLeadingSubmatrix.test_for_first_element_equal_to_zero.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSingularLeadingSubmatrix.test_for_first_element_equal_to_zero.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSingularLeadingSubmatrix.test_for_first_element_equal_to_zero.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSingularLeadingSubmatrix.test_for_first_element_equal_to_zero.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSingularLeadingSubmatrix.test_for_first_element_equal_to_zero.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSingularLeadingSubmatrix.test_for_first_element_equal_to_zero', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_for_first_element_equal_to_zero', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_for_first_element_equal_to_zero(...)' code ##################

        
        # Assigning a Call to a Name (line 136):
        
        # Assigning a Call to a Name (line 136):
        
        # Call to array(...): (line 136)
        # Processing the call arguments (line 136)
        
        # Obtaining an instance of the builtin type 'list' (line 136)
        list_234795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 136)
        # Adding element type (line 136)
        
        # Obtaining an instance of the builtin type 'list' (line 136)
        list_234796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 136)
        # Adding element type (line 136)
        int_234797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 22), list_234796, int_234797)
        # Adding element type (line 136)
        int_234798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 22), list_234796, int_234798)
        # Adding element type (line 136)
        int_234799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 22), list_234796, int_234799)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 21), list_234795, list_234796)
        # Adding element type (line 136)
        
        # Obtaining an instance of the builtin type 'list' (line 137)
        list_234800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 137)
        # Adding element type (line 137)
        int_234801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 22), list_234800, int_234801)
        # Adding element type (line 137)
        int_234802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 22), list_234800, int_234802)
        # Adding element type (line 137)
        int_234803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 22), list_234800, int_234803)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 21), list_234795, list_234800)
        # Adding element type (line 136)
        
        # Obtaining an instance of the builtin type 'list' (line 138)
        list_234804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 138)
        # Adding element type (line 138)
        int_234805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 22), list_234804, int_234805)
        # Adding element type (line 138)
        int_234806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 22), list_234804, int_234806)
        # Adding element type (line 138)
        int_234807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 22), list_234804, int_234807)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 21), list_234795, list_234804)
        
        # Processing the call keyword arguments (line 136)
        kwargs_234808 = {}
        # Getting the type of 'np' (line 136)
        np_234793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 136)
        array_234794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 12), np_234793, 'array')
        # Calling array(args, kwargs) (line 136)
        array_call_result_234809 = invoke(stypy.reporting.localization.Localization(__file__, 136, 12), array_234794, *[list_234795], **kwargs_234808)
        
        # Assigning a type to the variable 'A' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'A', array_call_result_234809)
        
        # Assigning a Call to a Tuple (line 141):
        
        # Assigning a Subscript to a Name (line 141):
        
        # Obtaining the type of the subscript
        int_234810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 8), 'int')
        
        # Call to get_lapack_funcs(...): (line 141)
        # Processing the call arguments (line 141)
        
        # Obtaining an instance of the builtin type 'tuple' (line 141)
        tuple_234812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 141)
        # Adding element type (line 141)
        str_234813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 38), 'str', 'potrf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 38), tuple_234812, str_234813)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 141)
        tuple_234814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 141)
        # Adding element type (line 141)
        # Getting the type of 'A' (line 141)
        A_234815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 50), 'A', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 50), tuple_234814, A_234815)
        
        # Processing the call keyword arguments (line 141)
        kwargs_234816 = {}
        # Getting the type of 'get_lapack_funcs' (line 141)
        get_lapack_funcs_234811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 20), 'get_lapack_funcs', False)
        # Calling get_lapack_funcs(args, kwargs) (line 141)
        get_lapack_funcs_call_result_234817 = invoke(stypy.reporting.localization.Localization(__file__, 141, 20), get_lapack_funcs_234811, *[tuple_234812, tuple_234814], **kwargs_234816)
        
        # Obtaining the member '__getitem__' of a type (line 141)
        getitem___234818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), get_lapack_funcs_call_result_234817, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 141)
        subscript_call_result_234819 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), getitem___234818, int_234810)
        
        # Assigning a type to the variable 'tuple_var_assignment_234275' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'tuple_var_assignment_234275', subscript_call_result_234819)
        
        # Assigning a Name to a Name (line 141):
        # Getting the type of 'tuple_var_assignment_234275' (line 141)
        tuple_var_assignment_234275_234820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'tuple_var_assignment_234275')
        # Assigning a type to the variable 'cholesky' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'cholesky', tuple_var_assignment_234275_234820)
        
        # Assigning a Call to a Tuple (line 144):
        
        # Assigning a Subscript to a Name (line 144):
        
        # Obtaining the type of the subscript
        int_234821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 8), 'int')
        
        # Call to cholesky(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'A' (line 144)
        A_234823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 24), 'A', False)
        # Processing the call keyword arguments (line 144)
        # Getting the type of 'False' (line 144)
        False_234824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 33), 'False', False)
        keyword_234825 = False_234824
        # Getting the type of 'False' (line 144)
        False_234826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 52), 'False', False)
        keyword_234827 = False_234826
        # Getting the type of 'True' (line 144)
        True_234828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 65), 'True', False)
        keyword_234829 = True_234828
        kwargs_234830 = {'lower': keyword_234825, 'overwrite_a': keyword_234827, 'clean': keyword_234829}
        # Getting the type of 'cholesky' (line 144)
        cholesky_234822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 15), 'cholesky', False)
        # Calling cholesky(args, kwargs) (line 144)
        cholesky_call_result_234831 = invoke(stypy.reporting.localization.Localization(__file__, 144, 15), cholesky_234822, *[A_234823], **kwargs_234830)
        
        # Obtaining the member '__getitem__' of a type (line 144)
        getitem___234832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), cholesky_call_result_234831, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 144)
        subscript_call_result_234833 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), getitem___234832, int_234821)
        
        # Assigning a type to the variable 'tuple_var_assignment_234276' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'tuple_var_assignment_234276', subscript_call_result_234833)
        
        # Assigning a Subscript to a Name (line 144):
        
        # Obtaining the type of the subscript
        int_234834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 8), 'int')
        
        # Call to cholesky(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'A' (line 144)
        A_234836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 24), 'A', False)
        # Processing the call keyword arguments (line 144)
        # Getting the type of 'False' (line 144)
        False_234837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 33), 'False', False)
        keyword_234838 = False_234837
        # Getting the type of 'False' (line 144)
        False_234839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 52), 'False', False)
        keyword_234840 = False_234839
        # Getting the type of 'True' (line 144)
        True_234841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 65), 'True', False)
        keyword_234842 = True_234841
        kwargs_234843 = {'lower': keyword_234838, 'overwrite_a': keyword_234840, 'clean': keyword_234842}
        # Getting the type of 'cholesky' (line 144)
        cholesky_234835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 15), 'cholesky', False)
        # Calling cholesky(args, kwargs) (line 144)
        cholesky_call_result_234844 = invoke(stypy.reporting.localization.Localization(__file__, 144, 15), cholesky_234835, *[A_234836], **kwargs_234843)
        
        # Obtaining the member '__getitem__' of a type (line 144)
        getitem___234845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), cholesky_call_result_234844, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 144)
        subscript_call_result_234846 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), getitem___234845, int_234834)
        
        # Assigning a type to the variable 'tuple_var_assignment_234277' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'tuple_var_assignment_234277', subscript_call_result_234846)
        
        # Assigning a Name to a Name (line 144):
        # Getting the type of 'tuple_var_assignment_234276' (line 144)
        tuple_var_assignment_234276_234847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'tuple_var_assignment_234276')
        # Assigning a type to the variable 'c' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'c', tuple_var_assignment_234276_234847)
        
        # Assigning a Name to a Name (line 144):
        # Getting the type of 'tuple_var_assignment_234277' (line 144)
        tuple_var_assignment_234277_234848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'tuple_var_assignment_234277')
        # Assigning a type to the variable 'k' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 11), 'k', tuple_var_assignment_234277_234848)
        
        # Assigning a Call to a Tuple (line 146):
        
        # Assigning a Subscript to a Name (line 146):
        
        # Obtaining the type of the subscript
        int_234849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 8), 'int')
        
        # Call to singular_leading_submatrix(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'A' (line 146)
        A_234851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 46), 'A', False)
        # Getting the type of 'c' (line 146)
        c_234852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 49), 'c', False)
        # Getting the type of 'k' (line 146)
        k_234853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 52), 'k', False)
        # Processing the call keyword arguments (line 146)
        kwargs_234854 = {}
        # Getting the type of 'singular_leading_submatrix' (line 146)
        singular_leading_submatrix_234850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 19), 'singular_leading_submatrix', False)
        # Calling singular_leading_submatrix(args, kwargs) (line 146)
        singular_leading_submatrix_call_result_234855 = invoke(stypy.reporting.localization.Localization(__file__, 146, 19), singular_leading_submatrix_234850, *[A_234851, c_234852, k_234853], **kwargs_234854)
        
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___234856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), singular_leading_submatrix_call_result_234855, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_234857 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), getitem___234856, int_234849)
        
        # Assigning a type to the variable 'tuple_var_assignment_234278' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_234278', subscript_call_result_234857)
        
        # Assigning a Subscript to a Name (line 146):
        
        # Obtaining the type of the subscript
        int_234858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 8), 'int')
        
        # Call to singular_leading_submatrix(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'A' (line 146)
        A_234860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 46), 'A', False)
        # Getting the type of 'c' (line 146)
        c_234861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 49), 'c', False)
        # Getting the type of 'k' (line 146)
        k_234862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 52), 'k', False)
        # Processing the call keyword arguments (line 146)
        kwargs_234863 = {}
        # Getting the type of 'singular_leading_submatrix' (line 146)
        singular_leading_submatrix_234859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 19), 'singular_leading_submatrix', False)
        # Calling singular_leading_submatrix(args, kwargs) (line 146)
        singular_leading_submatrix_call_result_234864 = invoke(stypy.reporting.localization.Localization(__file__, 146, 19), singular_leading_submatrix_234859, *[A_234860, c_234861, k_234862], **kwargs_234863)
        
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___234865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), singular_leading_submatrix_call_result_234864, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_234866 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), getitem___234865, int_234858)
        
        # Assigning a type to the variable 'tuple_var_assignment_234279' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_234279', subscript_call_result_234866)
        
        # Assigning a Name to a Name (line 146):
        # Getting the type of 'tuple_var_assignment_234278' (line 146)
        tuple_var_assignment_234278_234867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_234278')
        # Assigning a type to the variable 'delta' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'delta', tuple_var_assignment_234278_234867)
        
        # Assigning a Name to a Name (line 146):
        # Getting the type of 'tuple_var_assignment_234279' (line 146)
        tuple_var_assignment_234279_234868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_234279')
        # Assigning a type to the variable 'v' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 15), 'v', tuple_var_assignment_234279_234868)
        
        # Getting the type of 'A' (line 148)
        A_234869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'A')
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 148)
        tuple_234870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 10), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 148)
        # Adding element type (line 148)
        # Getting the type of 'k' (line 148)
        k_234871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 10), 'k')
        int_234872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 12), 'int')
        # Applying the binary operator '-' (line 148)
        result_sub_234873 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 10), '-', k_234871, int_234872)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 10), tuple_234870, result_sub_234873)
        # Adding element type (line 148)
        # Getting the type of 'k' (line 148)
        k_234874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 15), 'k')
        int_234875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 17), 'int')
        # Applying the binary operator '-' (line 148)
        result_sub_234876 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 15), '-', k_234874, int_234875)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 10), tuple_234870, result_sub_234876)
        
        # Getting the type of 'A' (line 148)
        A_234877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'A')
        # Obtaining the member '__getitem__' of a type (line 148)
        getitem___234878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), A_234877, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 148)
        subscript_call_result_234879 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), getitem___234878, tuple_234870)
        
        # Getting the type of 'delta' (line 148)
        delta_234880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 23), 'delta')
        # Applying the binary operator '+=' (line 148)
        result_iadd_234881 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 8), '+=', subscript_call_result_234879, delta_234880)
        # Getting the type of 'A' (line 148)
        A_234882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 148)
        tuple_234883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 10), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 148)
        # Adding element type (line 148)
        # Getting the type of 'k' (line 148)
        k_234884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 10), 'k')
        int_234885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 12), 'int')
        # Applying the binary operator '-' (line 148)
        result_sub_234886 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 10), '-', k_234884, int_234885)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 10), tuple_234883, result_sub_234886)
        # Adding element type (line 148)
        # Getting the type of 'k' (line 148)
        k_234887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 15), 'k')
        int_234888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 17), 'int')
        # Applying the binary operator '-' (line 148)
        result_sub_234889 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 15), '-', k_234887, int_234888)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 10), tuple_234883, result_sub_234889)
        
        # Storing an element on a container (line 148)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 8), A_234882, (tuple_234883, result_iadd_234881))
        
        
        # Call to assert_array_almost_equal(...): (line 151)
        # Processing the call arguments (line 151)
        
        # Call to det(...): (line 151)
        # Processing the call arguments (line 151)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 151)
        k_234892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 41), 'k', False)
        slice_234893 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 151, 38), None, k_234892, None)
        # Getting the type of 'k' (line 151)
        k_234894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 45), 'k', False)
        slice_234895 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 151, 38), None, k_234894, None)
        # Getting the type of 'A' (line 151)
        A_234896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 38), 'A', False)
        # Obtaining the member '__getitem__' of a type (line 151)
        getitem___234897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 38), A_234896, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 151)
        subscript_call_result_234898 = invoke(stypy.reporting.localization.Localization(__file__, 151, 38), getitem___234897, (slice_234893, slice_234895))
        
        # Processing the call keyword arguments (line 151)
        kwargs_234899 = {}
        # Getting the type of 'det' (line 151)
        det_234891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 34), 'det', False)
        # Calling det(args, kwargs) (line 151)
        det_call_result_234900 = invoke(stypy.reporting.localization.Localization(__file__, 151, 34), det_234891, *[subscript_call_result_234898], **kwargs_234899)
        
        int_234901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 50), 'int')
        # Processing the call keyword arguments (line 151)
        kwargs_234902 = {}
        # Getting the type of 'assert_array_almost_equal' (line 151)
        assert_array_almost_equal_234890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 151)
        assert_array_almost_equal_call_result_234903 = invoke(stypy.reporting.localization.Localization(__file__, 151, 8), assert_array_almost_equal_234890, *[det_call_result_234900, int_234901], **kwargs_234902)
        
        
        # Assigning a Call to a Name (line 154):
        
        # Assigning a Call to a Name (line 154):
        
        # Call to dot(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'v' (line 154)
        v_234906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 32), 'v', False)
        
        # Call to dot(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'A' (line 154)
        A_234909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 42), 'A', False)
        # Getting the type of 'v' (line 154)
        v_234910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 45), 'v', False)
        # Processing the call keyword arguments (line 154)
        kwargs_234911 = {}
        # Getting the type of 'np' (line 154)
        np_234907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 35), 'np', False)
        # Obtaining the member 'dot' of a type (line 154)
        dot_234908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 35), np_234907, 'dot')
        # Calling dot(args, kwargs) (line 154)
        dot_call_result_234912 = invoke(stypy.reporting.localization.Localization(__file__, 154, 35), dot_234908, *[A_234909, v_234910], **kwargs_234911)
        
        # Processing the call keyword arguments (line 154)
        kwargs_234913 = {}
        # Getting the type of 'np' (line 154)
        np_234904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 25), 'np', False)
        # Obtaining the member 'dot' of a type (line 154)
        dot_234905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 25), np_234904, 'dot')
        # Calling dot(args, kwargs) (line 154)
        dot_call_result_234914 = invoke(stypy.reporting.localization.Localization(__file__, 154, 25), dot_234905, *[v_234906, dot_call_result_234912], **kwargs_234913)
        
        # Assigning a type to the variable 'quadratic_term' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'quadratic_term', dot_call_result_234914)
        
        # Call to assert_array_almost_equal(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'quadratic_term' (line 155)
        quadratic_term_234916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 34), 'quadratic_term', False)
        int_234917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 50), 'int')
        # Processing the call keyword arguments (line 155)
        kwargs_234918 = {}
        # Getting the type of 'assert_array_almost_equal' (line 155)
        assert_array_almost_equal_234915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 155)
        assert_array_almost_equal_call_result_234919 = invoke(stypy.reporting.localization.Localization(__file__, 155, 8), assert_array_almost_equal_234915, *[quadratic_term_234916, int_234917], **kwargs_234918)
        
        
        # ################# End of 'test_for_first_element_equal_to_zero(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_for_first_element_equal_to_zero' in the type store
        # Getting the type of 'stypy_return_type' (line 132)
        stypy_return_type_234920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_234920)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_for_first_element_equal_to_zero'
        return stypy_return_type_234920


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 78, 0, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSingularLeadingSubmatrix.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSingularLeadingSubmatrix' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'TestSingularLeadingSubmatrix', TestSingularLeadingSubmatrix)
# Declaration of the 'TestIterativeSubproblem' class

class TestIterativeSubproblem(object, ):

    @norecursion
    def test_for_the_easy_case(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_for_the_easy_case'
        module_type_store = module_type_store.open_function_context('test_for_the_easy_case', 160, 4, False)
        # Assigning a type to the variable 'self' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIterativeSubproblem.test_for_the_easy_case.__dict__.__setitem__('stypy_localization', localization)
        TestIterativeSubproblem.test_for_the_easy_case.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIterativeSubproblem.test_for_the_easy_case.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIterativeSubproblem.test_for_the_easy_case.__dict__.__setitem__('stypy_function_name', 'TestIterativeSubproblem.test_for_the_easy_case')
        TestIterativeSubproblem.test_for_the_easy_case.__dict__.__setitem__('stypy_param_names_list', [])
        TestIterativeSubproblem.test_for_the_easy_case.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIterativeSubproblem.test_for_the_easy_case.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIterativeSubproblem.test_for_the_easy_case.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIterativeSubproblem.test_for_the_easy_case.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIterativeSubproblem.test_for_the_easy_case.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIterativeSubproblem.test_for_the_easy_case.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIterativeSubproblem.test_for_the_easy_case', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_for_the_easy_case', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_for_the_easy_case(...)' code ##################

        
        # Assigning a List to a Name (line 164):
        
        # Assigning a List to a Name (line 164):
        
        # Obtaining an instance of the builtin type 'list' (line 164)
        list_234921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 164)
        # Adding element type (line 164)
        
        # Obtaining an instance of the builtin type 'list' (line 164)
        list_234922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 164)
        # Adding element type (line 164)
        int_234923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 13), list_234922, int_234923)
        # Adding element type (line 164)
        int_234924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 13), list_234922, int_234924)
        # Adding element type (line 164)
        int_234925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 13), list_234922, int_234925)
        # Adding element type (line 164)
        int_234926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 13), list_234922, int_234926)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 12), list_234921, list_234922)
        # Adding element type (line 164)
        
        # Obtaining an instance of the builtin type 'list' (line 165)
        list_234927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 165)
        # Adding element type (line 165)
        int_234928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 13), list_234927, int_234928)
        # Adding element type (line 165)
        int_234929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 13), list_234927, int_234929)
        # Adding element type (line 165)
        int_234930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 13), list_234927, int_234930)
        # Adding element type (line 165)
        int_234931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 13), list_234927, int_234931)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 12), list_234921, list_234927)
        # Adding element type (line 164)
        
        # Obtaining an instance of the builtin type 'list' (line 166)
        list_234932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 166)
        # Adding element type (line 166)
        int_234933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 13), list_234932, int_234933)
        # Adding element type (line 166)
        int_234934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 13), list_234932, int_234934)
        # Adding element type (line 166)
        int_234935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 13), list_234932, int_234935)
        # Adding element type (line 166)
        int_234936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 13), list_234932, int_234936)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 12), list_234921, list_234932)
        # Adding element type (line 164)
        
        # Obtaining an instance of the builtin type 'list' (line 167)
        list_234937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 167)
        # Adding element type (line 167)
        int_234938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 13), list_234937, int_234938)
        # Adding element type (line 167)
        int_234939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 13), list_234937, int_234939)
        # Adding element type (line 167)
        int_234940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 13), list_234937, int_234940)
        # Adding element type (line 167)
        int_234941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 13), list_234937, int_234941)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 12), list_234921, list_234937)
        
        # Assigning a type to the variable 'H' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'H', list_234921)
        
        # Assigning a List to a Name (line 168):
        
        # Assigning a List to a Name (line 168):
        
        # Obtaining an instance of the builtin type 'list' (line 168)
        list_234942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 168)
        # Adding element type (line 168)
        int_234943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 12), list_234942, int_234943)
        # Adding element type (line 168)
        int_234944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 12), list_234942, int_234944)
        # Adding element type (line 168)
        int_234945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 12), list_234942, int_234945)
        # Adding element type (line 168)
        int_234946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 12), list_234942, int_234946)
        
        # Assigning a type to the variable 'g' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'g', list_234942)
        
        # Assigning a Num to a Name (line 171):
        
        # Assigning a Num to a Name (line 171):
        int_234947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 23), 'int')
        # Assigning a type to the variable 'trust_radius' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'trust_radius', int_234947)
        
        # Assigning a Call to a Name (line 174):
        
        # Assigning a Call to a Name (line 174):
        
        # Call to IterativeSubproblem(...): (line 174)
        # Processing the call keyword arguments (line 174)
        int_234949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 40), 'int')
        keyword_234950 = int_234949

        @norecursion
        def _stypy_temp_lambda_116(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_116'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_116', 175, 42, True)
            # Passed parameters checking function
            _stypy_temp_lambda_116.stypy_localization = localization
            _stypy_temp_lambda_116.stypy_type_of_self = None
            _stypy_temp_lambda_116.stypy_type_store = module_type_store
            _stypy_temp_lambda_116.stypy_function_name = '_stypy_temp_lambda_116'
            _stypy_temp_lambda_116.stypy_param_names_list = ['x']
            _stypy_temp_lambda_116.stypy_varargs_param_name = None
            _stypy_temp_lambda_116.stypy_kwargs_param_name = None
            _stypy_temp_lambda_116.stypy_call_defaults = defaults
            _stypy_temp_lambda_116.stypy_call_varargs = varargs
            _stypy_temp_lambda_116.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_116', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_116', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_234951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 52), 'int')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 175)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 42), 'stypy_return_type', int_234951)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_116' in the type store
            # Getting the type of 'stypy_return_type' (line 175)
            stypy_return_type_234952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 42), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_234952)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_116'
            return stypy_return_type_234952

        # Assigning a type to the variable '_stypy_temp_lambda_116' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 42), '_stypy_temp_lambda_116', _stypy_temp_lambda_116)
        # Getting the type of '_stypy_temp_lambda_116' (line 175)
        _stypy_temp_lambda_116_234953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 42), '_stypy_temp_lambda_116')
        keyword_234954 = _stypy_temp_lambda_116_234953

        @norecursion
        def _stypy_temp_lambda_117(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_117'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_117', 176, 42, True)
            # Passed parameters checking function
            _stypy_temp_lambda_117.stypy_localization = localization
            _stypy_temp_lambda_117.stypy_type_of_self = None
            _stypy_temp_lambda_117.stypy_type_store = module_type_store
            _stypy_temp_lambda_117.stypy_function_name = '_stypy_temp_lambda_117'
            _stypy_temp_lambda_117.stypy_param_names_list = ['x']
            _stypy_temp_lambda_117.stypy_varargs_param_name = None
            _stypy_temp_lambda_117.stypy_kwargs_param_name = None
            _stypy_temp_lambda_117.stypy_call_defaults = defaults
            _stypy_temp_lambda_117.stypy_call_varargs = varargs
            _stypy_temp_lambda_117.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_117', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_117', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to array(...): (line 176)
            # Processing the call arguments (line 176)
            # Getting the type of 'g' (line 176)
            g_234957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 61), 'g', False)
            # Processing the call keyword arguments (line 176)
            kwargs_234958 = {}
            # Getting the type of 'np' (line 176)
            np_234955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 52), 'np', False)
            # Obtaining the member 'array' of a type (line 176)
            array_234956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 52), np_234955, 'array')
            # Calling array(args, kwargs) (line 176)
            array_call_result_234959 = invoke(stypy.reporting.localization.Localization(__file__, 176, 52), array_234956, *[g_234957], **kwargs_234958)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 176)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 42), 'stypy_return_type', array_call_result_234959)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_117' in the type store
            # Getting the type of 'stypy_return_type' (line 176)
            stypy_return_type_234960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 42), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_234960)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_117'
            return stypy_return_type_234960

        # Assigning a type to the variable '_stypy_temp_lambda_117' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 42), '_stypy_temp_lambda_117', _stypy_temp_lambda_117)
        # Getting the type of '_stypy_temp_lambda_117' (line 176)
        _stypy_temp_lambda_117_234961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 42), '_stypy_temp_lambda_117')
        keyword_234962 = _stypy_temp_lambda_117_234961

        @norecursion
        def _stypy_temp_lambda_118(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_118'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_118', 177, 43, True)
            # Passed parameters checking function
            _stypy_temp_lambda_118.stypy_localization = localization
            _stypy_temp_lambda_118.stypy_type_of_self = None
            _stypy_temp_lambda_118.stypy_type_store = module_type_store
            _stypy_temp_lambda_118.stypy_function_name = '_stypy_temp_lambda_118'
            _stypy_temp_lambda_118.stypy_param_names_list = ['x']
            _stypy_temp_lambda_118.stypy_varargs_param_name = None
            _stypy_temp_lambda_118.stypy_kwargs_param_name = None
            _stypy_temp_lambda_118.stypy_call_defaults = defaults
            _stypy_temp_lambda_118.stypy_call_varargs = varargs
            _stypy_temp_lambda_118.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_118', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_118', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to array(...): (line 177)
            # Processing the call arguments (line 177)
            # Getting the type of 'H' (line 177)
            H_234965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 62), 'H', False)
            # Processing the call keyword arguments (line 177)
            kwargs_234966 = {}
            # Getting the type of 'np' (line 177)
            np_234963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 53), 'np', False)
            # Obtaining the member 'array' of a type (line 177)
            array_234964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 53), np_234963, 'array')
            # Calling array(args, kwargs) (line 177)
            array_call_result_234967 = invoke(stypy.reporting.localization.Localization(__file__, 177, 53), array_234964, *[H_234965], **kwargs_234966)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 177)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 43), 'stypy_return_type', array_call_result_234967)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_118' in the type store
            # Getting the type of 'stypy_return_type' (line 177)
            stypy_return_type_234968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 43), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_234968)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_118'
            return stypy_return_type_234968

        # Assigning a type to the variable '_stypy_temp_lambda_118' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 43), '_stypy_temp_lambda_118', _stypy_temp_lambda_118)
        # Getting the type of '_stypy_temp_lambda_118' (line 177)
        _stypy_temp_lambda_118_234969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 43), '_stypy_temp_lambda_118')
        keyword_234970 = _stypy_temp_lambda_118_234969
        float_234971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 45), 'float')
        keyword_234972 = float_234971
        float_234973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 45), 'float')
        keyword_234974 = float_234973
        kwargs_234975 = {'k_hard': keyword_234974, 'k_easy': keyword_234972, 'fun': keyword_234954, 'x': keyword_234950, 'hess': keyword_234970, 'jac': keyword_234962}
        # Getting the type of 'IterativeSubproblem' (line 174)
        IterativeSubproblem_234948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 18), 'IterativeSubproblem', False)
        # Calling IterativeSubproblem(args, kwargs) (line 174)
        IterativeSubproblem_call_result_234976 = invoke(stypy.reporting.localization.Localization(__file__, 174, 18), IterativeSubproblem_234948, *[], **kwargs_234975)
        
        # Assigning a type to the variable 'subprob' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'subprob', IterativeSubproblem_call_result_234976)
        
        # Assigning a Call to a Tuple (line 180):
        
        # Assigning a Subscript to a Name (line 180):
        
        # Obtaining the type of the subscript
        int_234977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 8), 'int')
        
        # Call to solve(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'trust_radius' (line 180)
        trust_radius_234980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 41), 'trust_radius', False)
        # Processing the call keyword arguments (line 180)
        kwargs_234981 = {}
        # Getting the type of 'subprob' (line 180)
        subprob_234978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 27), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 180)
        solve_234979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 27), subprob_234978, 'solve')
        # Calling solve(args, kwargs) (line 180)
        solve_call_result_234982 = invoke(stypy.reporting.localization.Localization(__file__, 180, 27), solve_234979, *[trust_radius_234980], **kwargs_234981)
        
        # Obtaining the member '__getitem__' of a type (line 180)
        getitem___234983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), solve_call_result_234982, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 180)
        subscript_call_result_234984 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), getitem___234983, int_234977)
        
        # Assigning a type to the variable 'tuple_var_assignment_234280' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'tuple_var_assignment_234280', subscript_call_result_234984)
        
        # Assigning a Subscript to a Name (line 180):
        
        # Obtaining the type of the subscript
        int_234985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 8), 'int')
        
        # Call to solve(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'trust_radius' (line 180)
        trust_radius_234988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 41), 'trust_radius', False)
        # Processing the call keyword arguments (line 180)
        kwargs_234989 = {}
        # Getting the type of 'subprob' (line 180)
        subprob_234986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 27), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 180)
        solve_234987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 27), subprob_234986, 'solve')
        # Calling solve(args, kwargs) (line 180)
        solve_call_result_234990 = invoke(stypy.reporting.localization.Localization(__file__, 180, 27), solve_234987, *[trust_radius_234988], **kwargs_234989)
        
        # Obtaining the member '__getitem__' of a type (line 180)
        getitem___234991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), solve_call_result_234990, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 180)
        subscript_call_result_234992 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), getitem___234991, int_234985)
        
        # Assigning a type to the variable 'tuple_var_assignment_234281' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'tuple_var_assignment_234281', subscript_call_result_234992)
        
        # Assigning a Name to a Name (line 180):
        # Getting the type of 'tuple_var_assignment_234280' (line 180)
        tuple_var_assignment_234280_234993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'tuple_var_assignment_234280')
        # Assigning a type to the variable 'p' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'p', tuple_var_assignment_234280_234993)
        
        # Assigning a Name to a Name (line 180):
        # Getting the type of 'tuple_var_assignment_234281' (line 180)
        tuple_var_assignment_234281_234994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'tuple_var_assignment_234281')
        # Assigning a type to the variable 'hits_boundary' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 11), 'hits_boundary', tuple_var_assignment_234281_234994)
        
        # Call to assert_array_almost_equal(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 'p' (line 182)
        p_234996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 34), 'p', False)
        
        # Obtaining an instance of the builtin type 'list' (line 182)
        list_234997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 182)
        # Adding element type (line 182)
        float_234998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 37), list_234997, float_234998)
        # Adding element type (line 182)
        float_234999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 37), list_234997, float_234999)
        # Adding element type (line 182)
        float_235000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 37), list_234997, float_235000)
        # Adding element type (line 182)
        float_235001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 37), list_234997, float_235001)
        
        # Processing the call keyword arguments (line 182)
        kwargs_235002 = {}
        # Getting the type of 'assert_array_almost_equal' (line 182)
        assert_array_almost_equal_234995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 182)
        assert_array_almost_equal_call_result_235003 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), assert_array_almost_equal_234995, *[p_234996, list_234997], **kwargs_235002)
        
        
        # Call to assert_array_almost_equal(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'hits_boundary' (line 184)
        hits_boundary_235005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 34), 'hits_boundary', False)
        # Getting the type of 'True' (line 184)
        True_235006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 49), 'True', False)
        # Processing the call keyword arguments (line 184)
        kwargs_235007 = {}
        # Getting the type of 'assert_array_almost_equal' (line 184)
        assert_array_almost_equal_235004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 184)
        assert_array_almost_equal_call_result_235008 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), assert_array_almost_equal_235004, *[hits_boundary_235005, True_235006], **kwargs_235007)
        
        
        # ################# End of 'test_for_the_easy_case(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_for_the_easy_case' in the type store
        # Getting the type of 'stypy_return_type' (line 160)
        stypy_return_type_235009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_235009)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_for_the_easy_case'
        return stypy_return_type_235009


    @norecursion
    def test_for_the_hard_case(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_for_the_hard_case'
        module_type_store = module_type_store.open_function_context('test_for_the_hard_case', 186, 4, False)
        # Assigning a type to the variable 'self' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIterativeSubproblem.test_for_the_hard_case.__dict__.__setitem__('stypy_localization', localization)
        TestIterativeSubproblem.test_for_the_hard_case.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIterativeSubproblem.test_for_the_hard_case.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIterativeSubproblem.test_for_the_hard_case.__dict__.__setitem__('stypy_function_name', 'TestIterativeSubproblem.test_for_the_hard_case')
        TestIterativeSubproblem.test_for_the_hard_case.__dict__.__setitem__('stypy_param_names_list', [])
        TestIterativeSubproblem.test_for_the_hard_case.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIterativeSubproblem.test_for_the_hard_case.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIterativeSubproblem.test_for_the_hard_case.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIterativeSubproblem.test_for_the_hard_case.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIterativeSubproblem.test_for_the_hard_case.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIterativeSubproblem.test_for_the_hard_case.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIterativeSubproblem.test_for_the_hard_case', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_for_the_hard_case', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_for_the_hard_case(...)' code ##################

        
        # Assigning a List to a Name (line 190):
        
        # Assigning a List to a Name (line 190):
        
        # Obtaining an instance of the builtin type 'list' (line 190)
        list_235010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 190)
        # Adding element type (line 190)
        
        # Obtaining an instance of the builtin type 'list' (line 190)
        list_235011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 190)
        # Adding element type (line 190)
        int_235012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 13), list_235011, int_235012)
        # Adding element type (line 190)
        int_235013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 13), list_235011, int_235013)
        # Adding element type (line 190)
        int_235014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 13), list_235011, int_235014)
        # Adding element type (line 190)
        int_235015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 13), list_235011, int_235015)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 12), list_235010, list_235011)
        # Adding element type (line 190)
        
        # Obtaining an instance of the builtin type 'list' (line 191)
        list_235016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 191)
        # Adding element type (line 191)
        int_235017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 13), list_235016, int_235017)
        # Adding element type (line 191)
        int_235018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 13), list_235016, int_235018)
        # Adding element type (line 191)
        int_235019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 13), list_235016, int_235019)
        # Adding element type (line 191)
        int_235020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 13), list_235016, int_235020)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 12), list_235010, list_235016)
        # Adding element type (line 190)
        
        # Obtaining an instance of the builtin type 'list' (line 192)
        list_235021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 192)
        # Adding element type (line 192)
        int_235022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 13), list_235021, int_235022)
        # Adding element type (line 192)
        int_235023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 13), list_235021, int_235023)
        # Adding element type (line 192)
        int_235024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 13), list_235021, int_235024)
        # Adding element type (line 192)
        int_235025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 13), list_235021, int_235025)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 12), list_235010, list_235021)
        # Adding element type (line 190)
        
        # Obtaining an instance of the builtin type 'list' (line 193)
        list_235026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 193)
        # Adding element type (line 193)
        int_235027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 13), list_235026, int_235027)
        # Adding element type (line 193)
        int_235028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 13), list_235026, int_235028)
        # Adding element type (line 193)
        int_235029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 13), list_235026, int_235029)
        # Adding element type (line 193)
        int_235030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 13), list_235026, int_235030)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 12), list_235010, list_235026)
        
        # Assigning a type to the variable 'H' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'H', list_235010)
        
        # Assigning a List to a Name (line 194):
        
        # Assigning a List to a Name (line 194):
        
        # Obtaining an instance of the builtin type 'list' (line 194)
        list_235031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 194)
        # Adding element type (line 194)
        float_235032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 12), list_235031, float_235032)
        # Adding element type (line 194)
        int_235033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 12), list_235031, int_235033)
        # Adding element type (line 194)
        int_235034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 12), list_235031, int_235034)
        # Adding element type (line 194)
        int_235035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 12), list_235031, int_235035)
        
        # Assigning a type to the variable 'g' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'g', list_235031)
        
        # Assigning a Num to a Name (line 195):
        
        # Assigning a Num to a Name (line 195):
        float_235036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 12), 'float')
        # Assigning a type to the variable 's' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 's', float_235036)
        
        # Assigning a Num to a Name (line 198):
        
        # Assigning a Num to a Name (line 198):
        int_235037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 23), 'int')
        # Assigning a type to the variable 'trust_radius' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'trust_radius', int_235037)
        
        # Assigning a Call to a Name (line 201):
        
        # Assigning a Call to a Name (line 201):
        
        # Call to IterativeSubproblem(...): (line 201)
        # Processing the call keyword arguments (line 201)
        int_235039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 40), 'int')
        keyword_235040 = int_235039

        @norecursion
        def _stypy_temp_lambda_119(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_119'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_119', 202, 42, True)
            # Passed parameters checking function
            _stypy_temp_lambda_119.stypy_localization = localization
            _stypy_temp_lambda_119.stypy_type_of_self = None
            _stypy_temp_lambda_119.stypy_type_store = module_type_store
            _stypy_temp_lambda_119.stypy_function_name = '_stypy_temp_lambda_119'
            _stypy_temp_lambda_119.stypy_param_names_list = ['x']
            _stypy_temp_lambda_119.stypy_varargs_param_name = None
            _stypy_temp_lambda_119.stypy_kwargs_param_name = None
            _stypy_temp_lambda_119.stypy_call_defaults = defaults
            _stypy_temp_lambda_119.stypy_call_varargs = varargs
            _stypy_temp_lambda_119.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_119', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_119', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_235041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 52), 'int')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 202)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 42), 'stypy_return_type', int_235041)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_119' in the type store
            # Getting the type of 'stypy_return_type' (line 202)
            stypy_return_type_235042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 42), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235042)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_119'
            return stypy_return_type_235042

        # Assigning a type to the variable '_stypy_temp_lambda_119' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 42), '_stypy_temp_lambda_119', _stypy_temp_lambda_119)
        # Getting the type of '_stypy_temp_lambda_119' (line 202)
        _stypy_temp_lambda_119_235043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 42), '_stypy_temp_lambda_119')
        keyword_235044 = _stypy_temp_lambda_119_235043

        @norecursion
        def _stypy_temp_lambda_120(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_120'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_120', 203, 42, True)
            # Passed parameters checking function
            _stypy_temp_lambda_120.stypy_localization = localization
            _stypy_temp_lambda_120.stypy_type_of_self = None
            _stypy_temp_lambda_120.stypy_type_store = module_type_store
            _stypy_temp_lambda_120.stypy_function_name = '_stypy_temp_lambda_120'
            _stypy_temp_lambda_120.stypy_param_names_list = ['x']
            _stypy_temp_lambda_120.stypy_varargs_param_name = None
            _stypy_temp_lambda_120.stypy_kwargs_param_name = None
            _stypy_temp_lambda_120.stypy_call_defaults = defaults
            _stypy_temp_lambda_120.stypy_call_varargs = varargs
            _stypy_temp_lambda_120.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_120', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_120', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to array(...): (line 203)
            # Processing the call arguments (line 203)
            # Getting the type of 'g' (line 203)
            g_235047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 61), 'g', False)
            # Processing the call keyword arguments (line 203)
            kwargs_235048 = {}
            # Getting the type of 'np' (line 203)
            np_235045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 52), 'np', False)
            # Obtaining the member 'array' of a type (line 203)
            array_235046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 52), np_235045, 'array')
            # Calling array(args, kwargs) (line 203)
            array_call_result_235049 = invoke(stypy.reporting.localization.Localization(__file__, 203, 52), array_235046, *[g_235047], **kwargs_235048)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 203)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 42), 'stypy_return_type', array_call_result_235049)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_120' in the type store
            # Getting the type of 'stypy_return_type' (line 203)
            stypy_return_type_235050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 42), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235050)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_120'
            return stypy_return_type_235050

        # Assigning a type to the variable '_stypy_temp_lambda_120' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 42), '_stypy_temp_lambda_120', _stypy_temp_lambda_120)
        # Getting the type of '_stypy_temp_lambda_120' (line 203)
        _stypy_temp_lambda_120_235051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 42), '_stypy_temp_lambda_120')
        keyword_235052 = _stypy_temp_lambda_120_235051

        @norecursion
        def _stypy_temp_lambda_121(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_121'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_121', 204, 43, True)
            # Passed parameters checking function
            _stypy_temp_lambda_121.stypy_localization = localization
            _stypy_temp_lambda_121.stypy_type_of_self = None
            _stypy_temp_lambda_121.stypy_type_store = module_type_store
            _stypy_temp_lambda_121.stypy_function_name = '_stypy_temp_lambda_121'
            _stypy_temp_lambda_121.stypy_param_names_list = ['x']
            _stypy_temp_lambda_121.stypy_varargs_param_name = None
            _stypy_temp_lambda_121.stypy_kwargs_param_name = None
            _stypy_temp_lambda_121.stypy_call_defaults = defaults
            _stypy_temp_lambda_121.stypy_call_varargs = varargs
            _stypy_temp_lambda_121.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_121', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_121', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to array(...): (line 204)
            # Processing the call arguments (line 204)
            # Getting the type of 'H' (line 204)
            H_235055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 62), 'H', False)
            # Processing the call keyword arguments (line 204)
            kwargs_235056 = {}
            # Getting the type of 'np' (line 204)
            np_235053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 53), 'np', False)
            # Obtaining the member 'array' of a type (line 204)
            array_235054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 53), np_235053, 'array')
            # Calling array(args, kwargs) (line 204)
            array_call_result_235057 = invoke(stypy.reporting.localization.Localization(__file__, 204, 53), array_235054, *[H_235055], **kwargs_235056)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 204)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 43), 'stypy_return_type', array_call_result_235057)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_121' in the type store
            # Getting the type of 'stypy_return_type' (line 204)
            stypy_return_type_235058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 43), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235058)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_121'
            return stypy_return_type_235058

        # Assigning a type to the variable '_stypy_temp_lambda_121' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 43), '_stypy_temp_lambda_121', _stypy_temp_lambda_121)
        # Getting the type of '_stypy_temp_lambda_121' (line 204)
        _stypy_temp_lambda_121_235059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 43), '_stypy_temp_lambda_121')
        keyword_235060 = _stypy_temp_lambda_121_235059
        float_235061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 45), 'float')
        keyword_235062 = float_235061
        float_235063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 45), 'float')
        keyword_235064 = float_235063
        kwargs_235065 = {'k_hard': keyword_235064, 'k_easy': keyword_235062, 'fun': keyword_235044, 'x': keyword_235040, 'hess': keyword_235060, 'jac': keyword_235052}
        # Getting the type of 'IterativeSubproblem' (line 201)
        IterativeSubproblem_235038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 18), 'IterativeSubproblem', False)
        # Calling IterativeSubproblem(args, kwargs) (line 201)
        IterativeSubproblem_call_result_235066 = invoke(stypy.reporting.localization.Localization(__file__, 201, 18), IterativeSubproblem_235038, *[], **kwargs_235065)
        
        # Assigning a type to the variable 'subprob' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'subprob', IterativeSubproblem_call_result_235066)
        
        # Assigning a Call to a Tuple (line 207):
        
        # Assigning a Subscript to a Name (line 207):
        
        # Obtaining the type of the subscript
        int_235067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 8), 'int')
        
        # Call to solve(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'trust_radius' (line 207)
        trust_radius_235070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 41), 'trust_radius', False)
        # Processing the call keyword arguments (line 207)
        kwargs_235071 = {}
        # Getting the type of 'subprob' (line 207)
        subprob_235068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 27), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 207)
        solve_235069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 27), subprob_235068, 'solve')
        # Calling solve(args, kwargs) (line 207)
        solve_call_result_235072 = invoke(stypy.reporting.localization.Localization(__file__, 207, 27), solve_235069, *[trust_radius_235070], **kwargs_235071)
        
        # Obtaining the member '__getitem__' of a type (line 207)
        getitem___235073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), solve_call_result_235072, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 207)
        subscript_call_result_235074 = invoke(stypy.reporting.localization.Localization(__file__, 207, 8), getitem___235073, int_235067)
        
        # Assigning a type to the variable 'tuple_var_assignment_234282' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'tuple_var_assignment_234282', subscript_call_result_235074)
        
        # Assigning a Subscript to a Name (line 207):
        
        # Obtaining the type of the subscript
        int_235075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 8), 'int')
        
        # Call to solve(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'trust_radius' (line 207)
        trust_radius_235078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 41), 'trust_radius', False)
        # Processing the call keyword arguments (line 207)
        kwargs_235079 = {}
        # Getting the type of 'subprob' (line 207)
        subprob_235076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 27), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 207)
        solve_235077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 27), subprob_235076, 'solve')
        # Calling solve(args, kwargs) (line 207)
        solve_call_result_235080 = invoke(stypy.reporting.localization.Localization(__file__, 207, 27), solve_235077, *[trust_radius_235078], **kwargs_235079)
        
        # Obtaining the member '__getitem__' of a type (line 207)
        getitem___235081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), solve_call_result_235080, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 207)
        subscript_call_result_235082 = invoke(stypy.reporting.localization.Localization(__file__, 207, 8), getitem___235081, int_235075)
        
        # Assigning a type to the variable 'tuple_var_assignment_234283' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'tuple_var_assignment_234283', subscript_call_result_235082)
        
        # Assigning a Name to a Name (line 207):
        # Getting the type of 'tuple_var_assignment_234282' (line 207)
        tuple_var_assignment_234282_235083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'tuple_var_assignment_234282')
        # Assigning a type to the variable 'p' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'p', tuple_var_assignment_234282_235083)
        
        # Assigning a Name to a Name (line 207):
        # Getting the type of 'tuple_var_assignment_234283' (line 207)
        tuple_var_assignment_234283_235084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'tuple_var_assignment_234283')
        # Assigning a type to the variable 'hits_boundary' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 11), 'hits_boundary', tuple_var_assignment_234283_235084)
        
        # Call to assert_array_almost_equal(...): (line 209)
        # Processing the call arguments (line 209)
        
        # Getting the type of 's' (line 209)
        s_235086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 35), 's', False)
        # Applying the 'usub' unary operator (line 209)
        result___neg___235087 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 34), 'usub', s_235086)
        
        # Getting the type of 'subprob' (line 209)
        subprob_235088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 38), 'subprob', False)
        # Obtaining the member 'lambda_current' of a type (line 209)
        lambda_current_235089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 38), subprob_235088, 'lambda_current')
        # Processing the call keyword arguments (line 209)
        kwargs_235090 = {}
        # Getting the type of 'assert_array_almost_equal' (line 209)
        assert_array_almost_equal_235085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 209)
        assert_array_almost_equal_call_result_235091 = invoke(stypy.reporting.localization.Localization(__file__, 209, 8), assert_array_almost_equal_235085, *[result___neg___235087, lambda_current_235089], **kwargs_235090)
        
        
        # ################# End of 'test_for_the_hard_case(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_for_the_hard_case' in the type store
        # Getting the type of 'stypy_return_type' (line 186)
        stypy_return_type_235092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_235092)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_for_the_hard_case'
        return stypy_return_type_235092


    @norecursion
    def test_for_interior_convergence(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_for_interior_convergence'
        module_type_store = module_type_store.open_function_context('test_for_interior_convergence', 211, 4, False)
        # Assigning a type to the variable 'self' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIterativeSubproblem.test_for_interior_convergence.__dict__.__setitem__('stypy_localization', localization)
        TestIterativeSubproblem.test_for_interior_convergence.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIterativeSubproblem.test_for_interior_convergence.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIterativeSubproblem.test_for_interior_convergence.__dict__.__setitem__('stypy_function_name', 'TestIterativeSubproblem.test_for_interior_convergence')
        TestIterativeSubproblem.test_for_interior_convergence.__dict__.__setitem__('stypy_param_names_list', [])
        TestIterativeSubproblem.test_for_interior_convergence.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIterativeSubproblem.test_for_interior_convergence.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIterativeSubproblem.test_for_interior_convergence.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIterativeSubproblem.test_for_interior_convergence.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIterativeSubproblem.test_for_interior_convergence.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIterativeSubproblem.test_for_interior_convergence.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIterativeSubproblem.test_for_interior_convergence', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_for_interior_convergence', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_for_interior_convergence(...)' code ##################

        
        # Assigning a List to a Name (line 213):
        
        # Assigning a List to a Name (line 213):
        
        # Obtaining an instance of the builtin type 'list' (line 213)
        list_235093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 213)
        # Adding element type (line 213)
        
        # Obtaining an instance of the builtin type 'list' (line 213)
        list_235094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 213)
        # Adding element type (line 213)
        float_235095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 13), list_235094, float_235095)
        # Adding element type (line 213)
        float_235096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 13), list_235094, float_235096)
        # Adding element type (line 213)
        float_235097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 13), list_235094, float_235097)
        # Adding element type (line 213)
        float_235098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 13), list_235094, float_235098)
        # Adding element type (line 213)
        float_235099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 61), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 13), list_235094, float_235099)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 12), list_235093, list_235094)
        # Adding element type (line 213)
        
        # Obtaining an instance of the builtin type 'list' (line 214)
        list_235100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 214)
        # Adding element type (line 214)
        float_235101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 13), list_235100, float_235101)
        # Adding element type (line 214)
        float_235102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 13), list_235100, float_235102)
        # Adding element type (line 214)
        float_235103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 13), list_235100, float_235103)
        # Adding element type (line 214)
        float_235104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 13), list_235100, float_235104)
        # Adding element type (line 214)
        float_235105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 63), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 13), list_235100, float_235105)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 12), list_235093, list_235100)
        # Adding element type (line 213)
        
        # Obtaining an instance of the builtin type 'list' (line 215)
        list_235106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 215)
        # Adding element type (line 215)
        float_235107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 13), list_235106, float_235107)
        # Adding element type (line 215)
        float_235108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 13), list_235106, float_235108)
        # Adding element type (line 215)
        float_235109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 13), list_235106, float_235109)
        # Adding element type (line 215)
        float_235110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 13), list_235106, float_235110)
        # Adding element type (line 215)
        float_235111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 62), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 13), list_235106, float_235111)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 12), list_235093, list_235106)
        # Adding element type (line 213)
        
        # Obtaining an instance of the builtin type 'list' (line 216)
        list_235112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 216)
        # Adding element type (line 216)
        float_235113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 13), list_235112, float_235113)
        # Adding element type (line 216)
        float_235114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 13), list_235112, float_235114)
        # Adding element type (line 216)
        float_235115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 13), list_235112, float_235115)
        # Adding element type (line 216)
        float_235116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 13), list_235112, float_235116)
        # Adding element type (line 216)
        float_235117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 64), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 13), list_235112, float_235117)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 12), list_235093, list_235112)
        # Adding element type (line 213)
        
        # Obtaining an instance of the builtin type 'list' (line 217)
        list_235118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 217)
        # Adding element type (line 217)
        float_235119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 13), list_235118, float_235119)
        # Adding element type (line 217)
        float_235120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 13), list_235118, float_235120)
        # Adding element type (line 217)
        float_235121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 13), list_235118, float_235121)
        # Adding element type (line 217)
        float_235122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 13), list_235118, float_235122)
        # Adding element type (line 217)
        float_235123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 63), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 13), list_235118, float_235123)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 12), list_235093, list_235118)
        
        # Assigning a type to the variable 'H' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'H', list_235093)
        
        # Assigning a List to a Name (line 219):
        
        # Assigning a List to a Name (line 219):
        
        # Obtaining an instance of the builtin type 'list' (line 219)
        list_235124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 219)
        # Adding element type (line 219)
        float_235125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 12), list_235124, float_235125)
        # Adding element type (line 219)
        float_235126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 12), list_235124, float_235126)
        # Adding element type (line 219)
        float_235127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 12), list_235124, float_235127)
        # Adding element type (line 219)
        float_235128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 12), list_235124, float_235128)
        # Adding element type (line 219)
        float_235129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 61), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 12), list_235124, float_235129)
        
        # Assigning a type to the variable 'g' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'g', list_235124)
        
        # Assigning a Call to a Name (line 222):
        
        # Assigning a Call to a Name (line 222):
        
        # Call to IterativeSubproblem(...): (line 222)
        # Processing the call keyword arguments (line 222)
        int_235131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 40), 'int')
        keyword_235132 = int_235131

        @norecursion
        def _stypy_temp_lambda_122(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_122'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_122', 223, 42, True)
            # Passed parameters checking function
            _stypy_temp_lambda_122.stypy_localization = localization
            _stypy_temp_lambda_122.stypy_type_of_self = None
            _stypy_temp_lambda_122.stypy_type_store = module_type_store
            _stypy_temp_lambda_122.stypy_function_name = '_stypy_temp_lambda_122'
            _stypy_temp_lambda_122.stypy_param_names_list = ['x']
            _stypy_temp_lambda_122.stypy_varargs_param_name = None
            _stypy_temp_lambda_122.stypy_kwargs_param_name = None
            _stypy_temp_lambda_122.stypy_call_defaults = defaults
            _stypy_temp_lambda_122.stypy_call_varargs = varargs
            _stypy_temp_lambda_122.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_122', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_122', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_235133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 52), 'int')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 223)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 42), 'stypy_return_type', int_235133)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_122' in the type store
            # Getting the type of 'stypy_return_type' (line 223)
            stypy_return_type_235134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 42), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235134)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_122'
            return stypy_return_type_235134

        # Assigning a type to the variable '_stypy_temp_lambda_122' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 42), '_stypy_temp_lambda_122', _stypy_temp_lambda_122)
        # Getting the type of '_stypy_temp_lambda_122' (line 223)
        _stypy_temp_lambda_122_235135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 42), '_stypy_temp_lambda_122')
        keyword_235136 = _stypy_temp_lambda_122_235135

        @norecursion
        def _stypy_temp_lambda_123(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_123'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_123', 224, 42, True)
            # Passed parameters checking function
            _stypy_temp_lambda_123.stypy_localization = localization
            _stypy_temp_lambda_123.stypy_type_of_self = None
            _stypy_temp_lambda_123.stypy_type_store = module_type_store
            _stypy_temp_lambda_123.stypy_function_name = '_stypy_temp_lambda_123'
            _stypy_temp_lambda_123.stypy_param_names_list = ['x']
            _stypy_temp_lambda_123.stypy_varargs_param_name = None
            _stypy_temp_lambda_123.stypy_kwargs_param_name = None
            _stypy_temp_lambda_123.stypy_call_defaults = defaults
            _stypy_temp_lambda_123.stypy_call_varargs = varargs
            _stypy_temp_lambda_123.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_123', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_123', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to array(...): (line 224)
            # Processing the call arguments (line 224)
            # Getting the type of 'g' (line 224)
            g_235139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 61), 'g', False)
            # Processing the call keyword arguments (line 224)
            kwargs_235140 = {}
            # Getting the type of 'np' (line 224)
            np_235137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 52), 'np', False)
            # Obtaining the member 'array' of a type (line 224)
            array_235138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 52), np_235137, 'array')
            # Calling array(args, kwargs) (line 224)
            array_call_result_235141 = invoke(stypy.reporting.localization.Localization(__file__, 224, 52), array_235138, *[g_235139], **kwargs_235140)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 224)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 42), 'stypy_return_type', array_call_result_235141)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_123' in the type store
            # Getting the type of 'stypy_return_type' (line 224)
            stypy_return_type_235142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 42), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235142)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_123'
            return stypy_return_type_235142

        # Assigning a type to the variable '_stypy_temp_lambda_123' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 42), '_stypy_temp_lambda_123', _stypy_temp_lambda_123)
        # Getting the type of '_stypy_temp_lambda_123' (line 224)
        _stypy_temp_lambda_123_235143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 42), '_stypy_temp_lambda_123')
        keyword_235144 = _stypy_temp_lambda_123_235143

        @norecursion
        def _stypy_temp_lambda_124(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_124'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_124', 225, 43, True)
            # Passed parameters checking function
            _stypy_temp_lambda_124.stypy_localization = localization
            _stypy_temp_lambda_124.stypy_type_of_self = None
            _stypy_temp_lambda_124.stypy_type_store = module_type_store
            _stypy_temp_lambda_124.stypy_function_name = '_stypy_temp_lambda_124'
            _stypy_temp_lambda_124.stypy_param_names_list = ['x']
            _stypy_temp_lambda_124.stypy_varargs_param_name = None
            _stypy_temp_lambda_124.stypy_kwargs_param_name = None
            _stypy_temp_lambda_124.stypy_call_defaults = defaults
            _stypy_temp_lambda_124.stypy_call_varargs = varargs
            _stypy_temp_lambda_124.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_124', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_124', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to array(...): (line 225)
            # Processing the call arguments (line 225)
            # Getting the type of 'H' (line 225)
            H_235147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 62), 'H', False)
            # Processing the call keyword arguments (line 225)
            kwargs_235148 = {}
            # Getting the type of 'np' (line 225)
            np_235145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 53), 'np', False)
            # Obtaining the member 'array' of a type (line 225)
            array_235146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 53), np_235145, 'array')
            # Calling array(args, kwargs) (line 225)
            array_call_result_235149 = invoke(stypy.reporting.localization.Localization(__file__, 225, 53), array_235146, *[H_235147], **kwargs_235148)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 225)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 43), 'stypy_return_type', array_call_result_235149)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_124' in the type store
            # Getting the type of 'stypy_return_type' (line 225)
            stypy_return_type_235150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 43), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235150)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_124'
            return stypy_return_type_235150

        # Assigning a type to the variable '_stypy_temp_lambda_124' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 43), '_stypy_temp_lambda_124', _stypy_temp_lambda_124)
        # Getting the type of '_stypy_temp_lambda_124' (line 225)
        _stypy_temp_lambda_124_235151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 43), '_stypy_temp_lambda_124')
        keyword_235152 = _stypy_temp_lambda_124_235151
        kwargs_235153 = {'fun': keyword_235136, 'x': keyword_235132, 'hess': keyword_235152, 'jac': keyword_235144}
        # Getting the type of 'IterativeSubproblem' (line 222)
        IterativeSubproblem_235130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 18), 'IterativeSubproblem', False)
        # Calling IterativeSubproblem(args, kwargs) (line 222)
        IterativeSubproblem_call_result_235154 = invoke(stypy.reporting.localization.Localization(__file__, 222, 18), IterativeSubproblem_235130, *[], **kwargs_235153)
        
        # Assigning a type to the variable 'subprob' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'subprob', IterativeSubproblem_call_result_235154)
        
        # Assigning a Call to a Tuple (line 226):
        
        # Assigning a Subscript to a Name (line 226):
        
        # Obtaining the type of the subscript
        int_235155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 8), 'int')
        
        # Call to solve(...): (line 226)
        # Processing the call arguments (line 226)
        float_235158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 41), 'float')
        # Processing the call keyword arguments (line 226)
        kwargs_235159 = {}
        # Getting the type of 'subprob' (line 226)
        subprob_235156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 27), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 226)
        solve_235157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 27), subprob_235156, 'solve')
        # Calling solve(args, kwargs) (line 226)
        solve_call_result_235160 = invoke(stypy.reporting.localization.Localization(__file__, 226, 27), solve_235157, *[float_235158], **kwargs_235159)
        
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___235161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), solve_call_result_235160, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_235162 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), getitem___235161, int_235155)
        
        # Assigning a type to the variable 'tuple_var_assignment_234284' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_234284', subscript_call_result_235162)
        
        # Assigning a Subscript to a Name (line 226):
        
        # Obtaining the type of the subscript
        int_235163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 8), 'int')
        
        # Call to solve(...): (line 226)
        # Processing the call arguments (line 226)
        float_235166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 41), 'float')
        # Processing the call keyword arguments (line 226)
        kwargs_235167 = {}
        # Getting the type of 'subprob' (line 226)
        subprob_235164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 27), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 226)
        solve_235165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 27), subprob_235164, 'solve')
        # Calling solve(args, kwargs) (line 226)
        solve_call_result_235168 = invoke(stypy.reporting.localization.Localization(__file__, 226, 27), solve_235165, *[float_235166], **kwargs_235167)
        
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___235169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), solve_call_result_235168, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_235170 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), getitem___235169, int_235163)
        
        # Assigning a type to the variable 'tuple_var_assignment_234285' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_234285', subscript_call_result_235170)
        
        # Assigning a Name to a Name (line 226):
        # Getting the type of 'tuple_var_assignment_234284' (line 226)
        tuple_var_assignment_234284_235171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_234284')
        # Assigning a type to the variable 'p' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'p', tuple_var_assignment_234284_235171)
        
        # Assigning a Name to a Name (line 226):
        # Getting the type of 'tuple_var_assignment_234285' (line 226)
        tuple_var_assignment_234285_235172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_234285')
        # Assigning a type to the variable 'hits_boundary' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 11), 'hits_boundary', tuple_var_assignment_234285_235172)
        
        # Call to assert_array_almost_equal(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'p' (line 228)
        p_235174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 34), 'p', False)
        
        # Obtaining an instance of the builtin type 'list' (line 228)
        list_235175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 228)
        # Adding element type (line 228)
        float_235176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 37), list_235175, float_235176)
        # Adding element type (line 228)
        float_235177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 37), list_235175, float_235177)
        # Adding element type (line 228)
        float_235178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 62), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 37), list_235175, float_235178)
        # Adding element type (line 228)
        float_235179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 37), list_235175, float_235179)
        # Adding element type (line 228)
        float_235180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 37), list_235175, float_235180)
        
        # Processing the call keyword arguments (line 228)
        kwargs_235181 = {}
        # Getting the type of 'assert_array_almost_equal' (line 228)
        assert_array_almost_equal_235173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 228)
        assert_array_almost_equal_call_result_235182 = invoke(stypy.reporting.localization.Localization(__file__, 228, 8), assert_array_almost_equal_235173, *[p_235174, list_235175], **kwargs_235181)
        
        
        # Call to assert_array_almost_equal(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'hits_boundary' (line 230)
        hits_boundary_235184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 34), 'hits_boundary', False)
        # Getting the type of 'False' (line 230)
        False_235185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 49), 'False', False)
        # Processing the call keyword arguments (line 230)
        kwargs_235186 = {}
        # Getting the type of 'assert_array_almost_equal' (line 230)
        assert_array_almost_equal_235183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 230)
        assert_array_almost_equal_call_result_235187 = invoke(stypy.reporting.localization.Localization(__file__, 230, 8), assert_array_almost_equal_235183, *[hits_boundary_235184, False_235185], **kwargs_235186)
        
        
        # Call to assert_array_almost_equal(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'subprob' (line 231)
        subprob_235189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 34), 'subprob', False)
        # Obtaining the member 'lambda_current' of a type (line 231)
        lambda_current_235190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 34), subprob_235189, 'lambda_current')
        int_235191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 58), 'int')
        # Processing the call keyword arguments (line 231)
        kwargs_235192 = {}
        # Getting the type of 'assert_array_almost_equal' (line 231)
        assert_array_almost_equal_235188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 231)
        assert_array_almost_equal_call_result_235193 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), assert_array_almost_equal_235188, *[lambda_current_235190, int_235191], **kwargs_235192)
        
        
        # Call to assert_array_almost_equal(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'subprob' (line 232)
        subprob_235195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 34), 'subprob', False)
        # Obtaining the member 'niter' of a type (line 232)
        niter_235196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 34), subprob_235195, 'niter')
        int_235197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 49), 'int')
        # Processing the call keyword arguments (line 232)
        kwargs_235198 = {}
        # Getting the type of 'assert_array_almost_equal' (line 232)
        assert_array_almost_equal_235194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 232)
        assert_array_almost_equal_call_result_235199 = invoke(stypy.reporting.localization.Localization(__file__, 232, 8), assert_array_almost_equal_235194, *[niter_235196, int_235197], **kwargs_235198)
        
        
        # ################# End of 'test_for_interior_convergence(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_for_interior_convergence' in the type store
        # Getting the type of 'stypy_return_type' (line 211)
        stypy_return_type_235200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_235200)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_for_interior_convergence'
        return stypy_return_type_235200


    @norecursion
    def test_for_jac_equal_zero(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_for_jac_equal_zero'
        module_type_store = module_type_store.open_function_context('test_for_jac_equal_zero', 234, 4, False)
        # Assigning a type to the variable 'self' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIterativeSubproblem.test_for_jac_equal_zero.__dict__.__setitem__('stypy_localization', localization)
        TestIterativeSubproblem.test_for_jac_equal_zero.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIterativeSubproblem.test_for_jac_equal_zero.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIterativeSubproblem.test_for_jac_equal_zero.__dict__.__setitem__('stypy_function_name', 'TestIterativeSubproblem.test_for_jac_equal_zero')
        TestIterativeSubproblem.test_for_jac_equal_zero.__dict__.__setitem__('stypy_param_names_list', [])
        TestIterativeSubproblem.test_for_jac_equal_zero.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIterativeSubproblem.test_for_jac_equal_zero.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIterativeSubproblem.test_for_jac_equal_zero.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIterativeSubproblem.test_for_jac_equal_zero.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIterativeSubproblem.test_for_jac_equal_zero.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIterativeSubproblem.test_for_jac_equal_zero.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIterativeSubproblem.test_for_jac_equal_zero', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_for_jac_equal_zero', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_for_jac_equal_zero(...)' code ##################

        
        # Assigning a List to a Name (line 236):
        
        # Assigning a List to a Name (line 236):
        
        # Obtaining an instance of the builtin type 'list' (line 236)
        list_235201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 236)
        # Adding element type (line 236)
        
        # Obtaining an instance of the builtin type 'list' (line 236)
        list_235202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 236)
        # Adding element type (line 236)
        float_235203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 13), list_235202, float_235203)
        # Adding element type (line 236)
        float_235204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 13), list_235202, float_235204)
        # Adding element type (line 236)
        float_235205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 13), list_235202, float_235205)
        # Adding element type (line 236)
        float_235206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 13), list_235202, float_235206)
        # Adding element type (line 236)
        float_235207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 63), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 13), list_235202, float_235207)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 12), list_235201, list_235202)
        # Adding element type (line 236)
        
        # Obtaining an instance of the builtin type 'list' (line 237)
        list_235208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 237)
        # Adding element type (line 237)
        float_235209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 13), list_235208, float_235209)
        # Adding element type (line 237)
        float_235210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 13), list_235208, float_235210)
        # Adding element type (line 237)
        float_235211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 13), list_235208, float_235211)
        # Adding element type (line 237)
        float_235212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 13), list_235208, float_235212)
        # Adding element type (line 237)
        float_235213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 64), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 13), list_235208, float_235213)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 12), list_235201, list_235208)
        # Adding element type (line 236)
        
        # Obtaining an instance of the builtin type 'list' (line 238)
        list_235214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 238)
        # Adding element type (line 238)
        float_235215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 13), list_235214, float_235215)
        # Adding element type (line 238)
        float_235216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 13), list_235214, float_235216)
        # Adding element type (line 238)
        float_235217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 13), list_235214, float_235217)
        # Adding element type (line 238)
        float_235218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 13), list_235214, float_235218)
        # Adding element type (line 238)
        float_235219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 64), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 13), list_235214, float_235219)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 12), list_235201, list_235214)
        # Adding element type (line 236)
        
        # Obtaining an instance of the builtin type 'list' (line 239)
        list_235220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 239)
        # Adding element type (line 239)
        float_235221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 13), list_235220, float_235221)
        # Adding element type (line 239)
        float_235222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 13), list_235220, float_235222)
        # Adding element type (line 239)
        float_235223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 13), list_235220, float_235223)
        # Adding element type (line 239)
        float_235224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 13), list_235220, float_235224)
        # Adding element type (line 239)
        float_235225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 64), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 13), list_235220, float_235225)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 12), list_235201, list_235220)
        # Adding element type (line 236)
        
        # Obtaining an instance of the builtin type 'list' (line 240)
        list_235226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 240)
        # Adding element type (line 240)
        float_235227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 13), list_235226, float_235227)
        # Adding element type (line 240)
        float_235228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 13), list_235226, float_235228)
        # Adding element type (line 240)
        float_235229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 13), list_235226, float_235229)
        # Adding element type (line 240)
        float_235230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 13), list_235226, float_235230)
        # Adding element type (line 240)
        float_235231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 13), list_235226, float_235231)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 12), list_235201, list_235226)
        
        # Assigning a type to the variable 'H' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'H', list_235201)
        
        # Assigning a List to a Name (line 242):
        
        # Assigning a List to a Name (line 242):
        
        # Obtaining an instance of the builtin type 'list' (line 242)
        list_235232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 242)
        # Adding element type (line 242)
        int_235233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 12), list_235232, int_235233)
        # Adding element type (line 242)
        int_235234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 12), list_235232, int_235234)
        # Adding element type (line 242)
        int_235235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 12), list_235232, int_235235)
        # Adding element type (line 242)
        int_235236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 12), list_235232, int_235236)
        # Adding element type (line 242)
        int_235237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 12), list_235232, int_235237)
        
        # Assigning a type to the variable 'g' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'g', list_235232)
        
        # Assigning a Call to a Name (line 245):
        
        # Assigning a Call to a Name (line 245):
        
        # Call to IterativeSubproblem(...): (line 245)
        # Processing the call keyword arguments (line 245)
        int_235239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 40), 'int')
        keyword_235240 = int_235239

        @norecursion
        def _stypy_temp_lambda_125(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_125'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_125', 246, 42, True)
            # Passed parameters checking function
            _stypy_temp_lambda_125.stypy_localization = localization
            _stypy_temp_lambda_125.stypy_type_of_self = None
            _stypy_temp_lambda_125.stypy_type_store = module_type_store
            _stypy_temp_lambda_125.stypy_function_name = '_stypy_temp_lambda_125'
            _stypy_temp_lambda_125.stypy_param_names_list = ['x']
            _stypy_temp_lambda_125.stypy_varargs_param_name = None
            _stypy_temp_lambda_125.stypy_kwargs_param_name = None
            _stypy_temp_lambda_125.stypy_call_defaults = defaults
            _stypy_temp_lambda_125.stypy_call_varargs = varargs
            _stypy_temp_lambda_125.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_125', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_125', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_235241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 52), 'int')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 246)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 42), 'stypy_return_type', int_235241)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_125' in the type store
            # Getting the type of 'stypy_return_type' (line 246)
            stypy_return_type_235242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 42), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235242)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_125'
            return stypy_return_type_235242

        # Assigning a type to the variable '_stypy_temp_lambda_125' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 42), '_stypy_temp_lambda_125', _stypy_temp_lambda_125)
        # Getting the type of '_stypy_temp_lambda_125' (line 246)
        _stypy_temp_lambda_125_235243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 42), '_stypy_temp_lambda_125')
        keyword_235244 = _stypy_temp_lambda_125_235243

        @norecursion
        def _stypy_temp_lambda_126(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_126'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_126', 247, 42, True)
            # Passed parameters checking function
            _stypy_temp_lambda_126.stypy_localization = localization
            _stypy_temp_lambda_126.stypy_type_of_self = None
            _stypy_temp_lambda_126.stypy_type_store = module_type_store
            _stypy_temp_lambda_126.stypy_function_name = '_stypy_temp_lambda_126'
            _stypy_temp_lambda_126.stypy_param_names_list = ['x']
            _stypy_temp_lambda_126.stypy_varargs_param_name = None
            _stypy_temp_lambda_126.stypy_kwargs_param_name = None
            _stypy_temp_lambda_126.stypy_call_defaults = defaults
            _stypy_temp_lambda_126.stypy_call_varargs = varargs
            _stypy_temp_lambda_126.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_126', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_126', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to array(...): (line 247)
            # Processing the call arguments (line 247)
            # Getting the type of 'g' (line 247)
            g_235247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 61), 'g', False)
            # Processing the call keyword arguments (line 247)
            kwargs_235248 = {}
            # Getting the type of 'np' (line 247)
            np_235245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 52), 'np', False)
            # Obtaining the member 'array' of a type (line 247)
            array_235246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 52), np_235245, 'array')
            # Calling array(args, kwargs) (line 247)
            array_call_result_235249 = invoke(stypy.reporting.localization.Localization(__file__, 247, 52), array_235246, *[g_235247], **kwargs_235248)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 247)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 42), 'stypy_return_type', array_call_result_235249)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_126' in the type store
            # Getting the type of 'stypy_return_type' (line 247)
            stypy_return_type_235250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 42), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235250)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_126'
            return stypy_return_type_235250

        # Assigning a type to the variable '_stypy_temp_lambda_126' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 42), '_stypy_temp_lambda_126', _stypy_temp_lambda_126)
        # Getting the type of '_stypy_temp_lambda_126' (line 247)
        _stypy_temp_lambda_126_235251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 42), '_stypy_temp_lambda_126')
        keyword_235252 = _stypy_temp_lambda_126_235251

        @norecursion
        def _stypy_temp_lambda_127(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_127'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_127', 248, 43, True)
            # Passed parameters checking function
            _stypy_temp_lambda_127.stypy_localization = localization
            _stypy_temp_lambda_127.stypy_type_of_self = None
            _stypy_temp_lambda_127.stypy_type_store = module_type_store
            _stypy_temp_lambda_127.stypy_function_name = '_stypy_temp_lambda_127'
            _stypy_temp_lambda_127.stypy_param_names_list = ['x']
            _stypy_temp_lambda_127.stypy_varargs_param_name = None
            _stypy_temp_lambda_127.stypy_kwargs_param_name = None
            _stypy_temp_lambda_127.stypy_call_defaults = defaults
            _stypy_temp_lambda_127.stypy_call_varargs = varargs
            _stypy_temp_lambda_127.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_127', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_127', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to array(...): (line 248)
            # Processing the call arguments (line 248)
            # Getting the type of 'H' (line 248)
            H_235255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 62), 'H', False)
            # Processing the call keyword arguments (line 248)
            kwargs_235256 = {}
            # Getting the type of 'np' (line 248)
            np_235253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 53), 'np', False)
            # Obtaining the member 'array' of a type (line 248)
            array_235254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 53), np_235253, 'array')
            # Calling array(args, kwargs) (line 248)
            array_call_result_235257 = invoke(stypy.reporting.localization.Localization(__file__, 248, 53), array_235254, *[H_235255], **kwargs_235256)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 248)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 43), 'stypy_return_type', array_call_result_235257)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_127' in the type store
            # Getting the type of 'stypy_return_type' (line 248)
            stypy_return_type_235258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 43), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235258)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_127'
            return stypy_return_type_235258

        # Assigning a type to the variable '_stypy_temp_lambda_127' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 43), '_stypy_temp_lambda_127', _stypy_temp_lambda_127)
        # Getting the type of '_stypy_temp_lambda_127' (line 248)
        _stypy_temp_lambda_127_235259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 43), '_stypy_temp_lambda_127')
        keyword_235260 = _stypy_temp_lambda_127_235259
        float_235261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 45), 'float')
        keyword_235262 = float_235261
        float_235263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 45), 'float')
        keyword_235264 = float_235263
        kwargs_235265 = {'k_hard': keyword_235264, 'k_easy': keyword_235262, 'fun': keyword_235244, 'x': keyword_235240, 'hess': keyword_235260, 'jac': keyword_235252}
        # Getting the type of 'IterativeSubproblem' (line 245)
        IterativeSubproblem_235238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 18), 'IterativeSubproblem', False)
        # Calling IterativeSubproblem(args, kwargs) (line 245)
        IterativeSubproblem_call_result_235266 = invoke(stypy.reporting.localization.Localization(__file__, 245, 18), IterativeSubproblem_235238, *[], **kwargs_235265)
        
        # Assigning a type to the variable 'subprob' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'subprob', IterativeSubproblem_call_result_235266)
        
        # Assigning a Call to a Tuple (line 251):
        
        # Assigning a Subscript to a Name (line 251):
        
        # Obtaining the type of the subscript
        int_235267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 8), 'int')
        
        # Call to solve(...): (line 251)
        # Processing the call arguments (line 251)
        float_235270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 41), 'float')
        # Processing the call keyword arguments (line 251)
        kwargs_235271 = {}
        # Getting the type of 'subprob' (line 251)
        subprob_235268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 27), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 251)
        solve_235269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 27), subprob_235268, 'solve')
        # Calling solve(args, kwargs) (line 251)
        solve_call_result_235272 = invoke(stypy.reporting.localization.Localization(__file__, 251, 27), solve_235269, *[float_235270], **kwargs_235271)
        
        # Obtaining the member '__getitem__' of a type (line 251)
        getitem___235273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 8), solve_call_result_235272, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 251)
        subscript_call_result_235274 = invoke(stypy.reporting.localization.Localization(__file__, 251, 8), getitem___235273, int_235267)
        
        # Assigning a type to the variable 'tuple_var_assignment_234286' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'tuple_var_assignment_234286', subscript_call_result_235274)
        
        # Assigning a Subscript to a Name (line 251):
        
        # Obtaining the type of the subscript
        int_235275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 8), 'int')
        
        # Call to solve(...): (line 251)
        # Processing the call arguments (line 251)
        float_235278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 41), 'float')
        # Processing the call keyword arguments (line 251)
        kwargs_235279 = {}
        # Getting the type of 'subprob' (line 251)
        subprob_235276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 27), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 251)
        solve_235277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 27), subprob_235276, 'solve')
        # Calling solve(args, kwargs) (line 251)
        solve_call_result_235280 = invoke(stypy.reporting.localization.Localization(__file__, 251, 27), solve_235277, *[float_235278], **kwargs_235279)
        
        # Obtaining the member '__getitem__' of a type (line 251)
        getitem___235281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 8), solve_call_result_235280, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 251)
        subscript_call_result_235282 = invoke(stypy.reporting.localization.Localization(__file__, 251, 8), getitem___235281, int_235275)
        
        # Assigning a type to the variable 'tuple_var_assignment_234287' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'tuple_var_assignment_234287', subscript_call_result_235282)
        
        # Assigning a Name to a Name (line 251):
        # Getting the type of 'tuple_var_assignment_234286' (line 251)
        tuple_var_assignment_234286_235283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'tuple_var_assignment_234286')
        # Assigning a type to the variable 'p' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'p', tuple_var_assignment_234286_235283)
        
        # Assigning a Name to a Name (line 251):
        # Getting the type of 'tuple_var_assignment_234287' (line 251)
        tuple_var_assignment_234287_235284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'tuple_var_assignment_234287')
        # Assigning a type to the variable 'hits_boundary' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 11), 'hits_boundary', tuple_var_assignment_234287_235284)
        
        # Call to assert_array_almost_equal(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'p' (line 253)
        p_235286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 34), 'p', False)
        
        # Obtaining an instance of the builtin type 'list' (line 253)
        list_235287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 253)
        # Adding element type (line 253)
        float_235288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 37), list_235287, float_235288)
        # Adding element type (line 253)
        float_235289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 37), list_235287, float_235289)
        # Adding element type (line 253)
        float_235290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 37), list_235287, float_235290)
        # Adding element type (line 253)
        float_235291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 37), list_235287, float_235291)
        # Adding element type (line 253)
        float_235292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 37), list_235287, float_235292)
        
        # Processing the call keyword arguments (line 253)
        kwargs_235293 = {}
        # Getting the type of 'assert_array_almost_equal' (line 253)
        assert_array_almost_equal_235285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 253)
        assert_array_almost_equal_call_result_235294 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), assert_array_almost_equal_235285, *[p_235286, list_235287], **kwargs_235293)
        
        
        # Call to assert_array_almost_equal(...): (line 256)
        # Processing the call arguments (line 256)
        # Getting the type of 'hits_boundary' (line 256)
        hits_boundary_235296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 34), 'hits_boundary', False)
        # Getting the type of 'True' (line 256)
        True_235297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 49), 'True', False)
        # Processing the call keyword arguments (line 256)
        kwargs_235298 = {}
        # Getting the type of 'assert_array_almost_equal' (line 256)
        assert_array_almost_equal_235295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 256)
        assert_array_almost_equal_call_result_235299 = invoke(stypy.reporting.localization.Localization(__file__, 256, 8), assert_array_almost_equal_235295, *[hits_boundary_235296, True_235297], **kwargs_235298)
        
        
        # ################# End of 'test_for_jac_equal_zero(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_for_jac_equal_zero' in the type store
        # Getting the type of 'stypy_return_type' (line 234)
        stypy_return_type_235300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_235300)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_for_jac_equal_zero'
        return stypy_return_type_235300


    @norecursion
    def test_for_jac_very_close_to_zero(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_for_jac_very_close_to_zero'
        module_type_store = module_type_store.open_function_context('test_for_jac_very_close_to_zero', 258, 4, False)
        # Assigning a type to the variable 'self' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIterativeSubproblem.test_for_jac_very_close_to_zero.__dict__.__setitem__('stypy_localization', localization)
        TestIterativeSubproblem.test_for_jac_very_close_to_zero.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIterativeSubproblem.test_for_jac_very_close_to_zero.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIterativeSubproblem.test_for_jac_very_close_to_zero.__dict__.__setitem__('stypy_function_name', 'TestIterativeSubproblem.test_for_jac_very_close_to_zero')
        TestIterativeSubproblem.test_for_jac_very_close_to_zero.__dict__.__setitem__('stypy_param_names_list', [])
        TestIterativeSubproblem.test_for_jac_very_close_to_zero.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIterativeSubproblem.test_for_jac_very_close_to_zero.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIterativeSubproblem.test_for_jac_very_close_to_zero.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIterativeSubproblem.test_for_jac_very_close_to_zero.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIterativeSubproblem.test_for_jac_very_close_to_zero.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIterativeSubproblem.test_for_jac_very_close_to_zero.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIterativeSubproblem.test_for_jac_very_close_to_zero', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_for_jac_very_close_to_zero', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_for_jac_very_close_to_zero(...)' code ##################

        
        # Assigning a List to a Name (line 260):
        
        # Assigning a List to a Name (line 260):
        
        # Obtaining an instance of the builtin type 'list' (line 260)
        list_235301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 260)
        # Adding element type (line 260)
        
        # Obtaining an instance of the builtin type 'list' (line 260)
        list_235302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 260)
        # Adding element type (line 260)
        float_235303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 13), list_235302, float_235303)
        # Adding element type (line 260)
        float_235304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 13), list_235302, float_235304)
        # Adding element type (line 260)
        float_235305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 13), list_235302, float_235305)
        # Adding element type (line 260)
        float_235306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 13), list_235302, float_235306)
        # Adding element type (line 260)
        float_235307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 63), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 13), list_235302, float_235307)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 12), list_235301, list_235302)
        # Adding element type (line 260)
        
        # Obtaining an instance of the builtin type 'list' (line 261)
        list_235308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 261)
        # Adding element type (line 261)
        float_235309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 13), list_235308, float_235309)
        # Adding element type (line 261)
        float_235310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 13), list_235308, float_235310)
        # Adding element type (line 261)
        float_235311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 13), list_235308, float_235311)
        # Adding element type (line 261)
        float_235312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 13), list_235308, float_235312)
        # Adding element type (line 261)
        float_235313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 64), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 13), list_235308, float_235313)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 12), list_235301, list_235308)
        # Adding element type (line 260)
        
        # Obtaining an instance of the builtin type 'list' (line 262)
        list_235314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 262)
        # Adding element type (line 262)
        float_235315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 13), list_235314, float_235315)
        # Adding element type (line 262)
        float_235316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 13), list_235314, float_235316)
        # Adding element type (line 262)
        float_235317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 13), list_235314, float_235317)
        # Adding element type (line 262)
        float_235318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 13), list_235314, float_235318)
        # Adding element type (line 262)
        float_235319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 64), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 13), list_235314, float_235319)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 12), list_235301, list_235314)
        # Adding element type (line 260)
        
        # Obtaining an instance of the builtin type 'list' (line 263)
        list_235320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 263)
        # Adding element type (line 263)
        float_235321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 13), list_235320, float_235321)
        # Adding element type (line 263)
        float_235322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 13), list_235320, float_235322)
        # Adding element type (line 263)
        float_235323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 13), list_235320, float_235323)
        # Adding element type (line 263)
        float_235324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 13), list_235320, float_235324)
        # Adding element type (line 263)
        float_235325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 64), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 13), list_235320, float_235325)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 12), list_235301, list_235320)
        # Adding element type (line 260)
        
        # Obtaining an instance of the builtin type 'list' (line 264)
        list_235326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 264)
        # Adding element type (line 264)
        float_235327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 13), list_235326, float_235327)
        # Adding element type (line 264)
        float_235328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 13), list_235326, float_235328)
        # Adding element type (line 264)
        float_235329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 13), list_235326, float_235329)
        # Adding element type (line 264)
        float_235330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 13), list_235326, float_235330)
        # Adding element type (line 264)
        float_235331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 13), list_235326, float_235331)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 12), list_235301, list_235326)
        
        # Assigning a type to the variable 'H' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'H', list_235301)
        
        # Assigning a List to a Name (line 266):
        
        # Assigning a List to a Name (line 266):
        
        # Obtaining an instance of the builtin type 'list' (line 266)
        list_235332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 266)
        # Adding element type (line 266)
        int_235333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 12), list_235332, int_235333)
        # Adding element type (line 266)
        int_235334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 12), list_235332, int_235334)
        # Adding element type (line 266)
        int_235335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 12), list_235332, int_235335)
        # Adding element type (line 266)
        int_235336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 12), list_235332, int_235336)
        # Adding element type (line 266)
        float_235337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 12), list_235332, float_235337)
        
        # Assigning a type to the variable 'g' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'g', list_235332)
        
        # Assigning a Call to a Name (line 269):
        
        # Assigning a Call to a Name (line 269):
        
        # Call to IterativeSubproblem(...): (line 269)
        # Processing the call keyword arguments (line 269)
        int_235339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 40), 'int')
        keyword_235340 = int_235339

        @norecursion
        def _stypy_temp_lambda_128(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_128'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_128', 270, 42, True)
            # Passed parameters checking function
            _stypy_temp_lambda_128.stypy_localization = localization
            _stypy_temp_lambda_128.stypy_type_of_self = None
            _stypy_temp_lambda_128.stypy_type_store = module_type_store
            _stypy_temp_lambda_128.stypy_function_name = '_stypy_temp_lambda_128'
            _stypy_temp_lambda_128.stypy_param_names_list = ['x']
            _stypy_temp_lambda_128.stypy_varargs_param_name = None
            _stypy_temp_lambda_128.stypy_kwargs_param_name = None
            _stypy_temp_lambda_128.stypy_call_defaults = defaults
            _stypy_temp_lambda_128.stypy_call_varargs = varargs
            _stypy_temp_lambda_128.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_128', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_128', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_235341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 52), 'int')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 270)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 42), 'stypy_return_type', int_235341)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_128' in the type store
            # Getting the type of 'stypy_return_type' (line 270)
            stypy_return_type_235342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 42), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235342)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_128'
            return stypy_return_type_235342

        # Assigning a type to the variable '_stypy_temp_lambda_128' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 42), '_stypy_temp_lambda_128', _stypy_temp_lambda_128)
        # Getting the type of '_stypy_temp_lambda_128' (line 270)
        _stypy_temp_lambda_128_235343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 42), '_stypy_temp_lambda_128')
        keyword_235344 = _stypy_temp_lambda_128_235343

        @norecursion
        def _stypy_temp_lambda_129(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_129'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_129', 271, 42, True)
            # Passed parameters checking function
            _stypy_temp_lambda_129.stypy_localization = localization
            _stypy_temp_lambda_129.stypy_type_of_self = None
            _stypy_temp_lambda_129.stypy_type_store = module_type_store
            _stypy_temp_lambda_129.stypy_function_name = '_stypy_temp_lambda_129'
            _stypy_temp_lambda_129.stypy_param_names_list = ['x']
            _stypy_temp_lambda_129.stypy_varargs_param_name = None
            _stypy_temp_lambda_129.stypy_kwargs_param_name = None
            _stypy_temp_lambda_129.stypy_call_defaults = defaults
            _stypy_temp_lambda_129.stypy_call_varargs = varargs
            _stypy_temp_lambda_129.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_129', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_129', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to array(...): (line 271)
            # Processing the call arguments (line 271)
            # Getting the type of 'g' (line 271)
            g_235347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 61), 'g', False)
            # Processing the call keyword arguments (line 271)
            kwargs_235348 = {}
            # Getting the type of 'np' (line 271)
            np_235345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 52), 'np', False)
            # Obtaining the member 'array' of a type (line 271)
            array_235346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 52), np_235345, 'array')
            # Calling array(args, kwargs) (line 271)
            array_call_result_235349 = invoke(stypy.reporting.localization.Localization(__file__, 271, 52), array_235346, *[g_235347], **kwargs_235348)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 271)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 42), 'stypy_return_type', array_call_result_235349)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_129' in the type store
            # Getting the type of 'stypy_return_type' (line 271)
            stypy_return_type_235350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 42), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235350)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_129'
            return stypy_return_type_235350

        # Assigning a type to the variable '_stypy_temp_lambda_129' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 42), '_stypy_temp_lambda_129', _stypy_temp_lambda_129)
        # Getting the type of '_stypy_temp_lambda_129' (line 271)
        _stypy_temp_lambda_129_235351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 42), '_stypy_temp_lambda_129')
        keyword_235352 = _stypy_temp_lambda_129_235351

        @norecursion
        def _stypy_temp_lambda_130(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_130'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_130', 272, 43, True)
            # Passed parameters checking function
            _stypy_temp_lambda_130.stypy_localization = localization
            _stypy_temp_lambda_130.stypy_type_of_self = None
            _stypy_temp_lambda_130.stypy_type_store = module_type_store
            _stypy_temp_lambda_130.stypy_function_name = '_stypy_temp_lambda_130'
            _stypy_temp_lambda_130.stypy_param_names_list = ['x']
            _stypy_temp_lambda_130.stypy_varargs_param_name = None
            _stypy_temp_lambda_130.stypy_kwargs_param_name = None
            _stypy_temp_lambda_130.stypy_call_defaults = defaults
            _stypy_temp_lambda_130.stypy_call_varargs = varargs
            _stypy_temp_lambda_130.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_130', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_130', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to array(...): (line 272)
            # Processing the call arguments (line 272)
            # Getting the type of 'H' (line 272)
            H_235355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 62), 'H', False)
            # Processing the call keyword arguments (line 272)
            kwargs_235356 = {}
            # Getting the type of 'np' (line 272)
            np_235353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 53), 'np', False)
            # Obtaining the member 'array' of a type (line 272)
            array_235354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 53), np_235353, 'array')
            # Calling array(args, kwargs) (line 272)
            array_call_result_235357 = invoke(stypy.reporting.localization.Localization(__file__, 272, 53), array_235354, *[H_235355], **kwargs_235356)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 272)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 43), 'stypy_return_type', array_call_result_235357)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_130' in the type store
            # Getting the type of 'stypy_return_type' (line 272)
            stypy_return_type_235358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 43), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235358)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_130'
            return stypy_return_type_235358

        # Assigning a type to the variable '_stypy_temp_lambda_130' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 43), '_stypy_temp_lambda_130', _stypy_temp_lambda_130)
        # Getting the type of '_stypy_temp_lambda_130' (line 272)
        _stypy_temp_lambda_130_235359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 43), '_stypy_temp_lambda_130')
        keyword_235360 = _stypy_temp_lambda_130_235359
        float_235361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 45), 'float')
        keyword_235362 = float_235361
        float_235363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 45), 'float')
        keyword_235364 = float_235363
        kwargs_235365 = {'k_hard': keyword_235364, 'k_easy': keyword_235362, 'fun': keyword_235344, 'x': keyword_235340, 'hess': keyword_235360, 'jac': keyword_235352}
        # Getting the type of 'IterativeSubproblem' (line 269)
        IterativeSubproblem_235338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 18), 'IterativeSubproblem', False)
        # Calling IterativeSubproblem(args, kwargs) (line 269)
        IterativeSubproblem_call_result_235366 = invoke(stypy.reporting.localization.Localization(__file__, 269, 18), IterativeSubproblem_235338, *[], **kwargs_235365)
        
        # Assigning a type to the variable 'subprob' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'subprob', IterativeSubproblem_call_result_235366)
        
        # Assigning a Call to a Tuple (line 275):
        
        # Assigning a Subscript to a Name (line 275):
        
        # Obtaining the type of the subscript
        int_235367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 8), 'int')
        
        # Call to solve(...): (line 275)
        # Processing the call arguments (line 275)
        float_235370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 41), 'float')
        # Processing the call keyword arguments (line 275)
        kwargs_235371 = {}
        # Getting the type of 'subprob' (line 275)
        subprob_235368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 27), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 275)
        solve_235369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 27), subprob_235368, 'solve')
        # Calling solve(args, kwargs) (line 275)
        solve_call_result_235372 = invoke(stypy.reporting.localization.Localization(__file__, 275, 27), solve_235369, *[float_235370], **kwargs_235371)
        
        # Obtaining the member '__getitem__' of a type (line 275)
        getitem___235373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), solve_call_result_235372, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 275)
        subscript_call_result_235374 = invoke(stypy.reporting.localization.Localization(__file__, 275, 8), getitem___235373, int_235367)
        
        # Assigning a type to the variable 'tuple_var_assignment_234288' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'tuple_var_assignment_234288', subscript_call_result_235374)
        
        # Assigning a Subscript to a Name (line 275):
        
        # Obtaining the type of the subscript
        int_235375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 8), 'int')
        
        # Call to solve(...): (line 275)
        # Processing the call arguments (line 275)
        float_235378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 41), 'float')
        # Processing the call keyword arguments (line 275)
        kwargs_235379 = {}
        # Getting the type of 'subprob' (line 275)
        subprob_235376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 27), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 275)
        solve_235377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 27), subprob_235376, 'solve')
        # Calling solve(args, kwargs) (line 275)
        solve_call_result_235380 = invoke(stypy.reporting.localization.Localization(__file__, 275, 27), solve_235377, *[float_235378], **kwargs_235379)
        
        # Obtaining the member '__getitem__' of a type (line 275)
        getitem___235381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), solve_call_result_235380, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 275)
        subscript_call_result_235382 = invoke(stypy.reporting.localization.Localization(__file__, 275, 8), getitem___235381, int_235375)
        
        # Assigning a type to the variable 'tuple_var_assignment_234289' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'tuple_var_assignment_234289', subscript_call_result_235382)
        
        # Assigning a Name to a Name (line 275):
        # Getting the type of 'tuple_var_assignment_234288' (line 275)
        tuple_var_assignment_234288_235383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'tuple_var_assignment_234288')
        # Assigning a type to the variable 'p' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'p', tuple_var_assignment_234288_235383)
        
        # Assigning a Name to a Name (line 275):
        # Getting the type of 'tuple_var_assignment_234289' (line 275)
        tuple_var_assignment_234289_235384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'tuple_var_assignment_234289')
        # Assigning a type to the variable 'hits_boundary' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 11), 'hits_boundary', tuple_var_assignment_234289_235384)
        
        # Call to assert_array_almost_equal(...): (line 277)
        # Processing the call arguments (line 277)
        # Getting the type of 'p' (line 277)
        p_235386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 34), 'p', False)
        
        # Obtaining an instance of the builtin type 'list' (line 277)
        list_235387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 277)
        # Adding element type (line 277)
        float_235388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 37), list_235387, float_235388)
        # Adding element type (line 277)
        float_235389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 37), list_235387, float_235389)
        # Adding element type (line 277)
        float_235390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 37), list_235387, float_235390)
        # Adding element type (line 277)
        float_235391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 37), list_235387, float_235391)
        # Adding element type (line 277)
        float_235392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 37), list_235387, float_235392)
        
        # Processing the call keyword arguments (line 277)
        kwargs_235393 = {}
        # Getting the type of 'assert_array_almost_equal' (line 277)
        assert_array_almost_equal_235385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 277)
        assert_array_almost_equal_call_result_235394 = invoke(stypy.reporting.localization.Localization(__file__, 277, 8), assert_array_almost_equal_235385, *[p_235386, list_235387], **kwargs_235393)
        
        
        # Call to assert_array_almost_equal(...): (line 280)
        # Processing the call arguments (line 280)
        # Getting the type of 'hits_boundary' (line 280)
        hits_boundary_235396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 34), 'hits_boundary', False)
        # Getting the type of 'True' (line 280)
        True_235397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 49), 'True', False)
        # Processing the call keyword arguments (line 280)
        kwargs_235398 = {}
        # Getting the type of 'assert_array_almost_equal' (line 280)
        assert_array_almost_equal_235395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 280)
        assert_array_almost_equal_call_result_235399 = invoke(stypy.reporting.localization.Localization(__file__, 280, 8), assert_array_almost_equal_235395, *[hits_boundary_235396, True_235397], **kwargs_235398)
        
        
        # ################# End of 'test_for_jac_very_close_to_zero(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_for_jac_very_close_to_zero' in the type store
        # Getting the type of 'stypy_return_type' (line 258)
        stypy_return_type_235400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_235400)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_for_jac_very_close_to_zero'
        return stypy_return_type_235400


    @norecursion
    def test_for_random_entries(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_for_random_entries'
        module_type_store = module_type_store.open_function_context('test_for_random_entries', 282, 4, False)
        # Assigning a type to the variable 'self' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIterativeSubproblem.test_for_random_entries.__dict__.__setitem__('stypy_localization', localization)
        TestIterativeSubproblem.test_for_random_entries.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIterativeSubproblem.test_for_random_entries.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIterativeSubproblem.test_for_random_entries.__dict__.__setitem__('stypy_function_name', 'TestIterativeSubproblem.test_for_random_entries')
        TestIterativeSubproblem.test_for_random_entries.__dict__.__setitem__('stypy_param_names_list', [])
        TestIterativeSubproblem.test_for_random_entries.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIterativeSubproblem.test_for_random_entries.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIterativeSubproblem.test_for_random_entries.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIterativeSubproblem.test_for_random_entries.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIterativeSubproblem.test_for_random_entries.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIterativeSubproblem.test_for_random_entries.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIterativeSubproblem.test_for_random_entries', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_for_random_entries', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_for_random_entries(...)' code ##################

        
        # Call to seed(...): (line 284)
        # Processing the call arguments (line 284)
        int_235404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 23), 'int')
        # Processing the call keyword arguments (line 284)
        kwargs_235405 = {}
        # Getting the type of 'np' (line 284)
        np_235401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 284)
        random_235402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 8), np_235401, 'random')
        # Obtaining the member 'seed' of a type (line 284)
        seed_235403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 8), random_235402, 'seed')
        # Calling seed(args, kwargs) (line 284)
        seed_call_result_235406 = invoke(stypy.reporting.localization.Localization(__file__, 284, 8), seed_235403, *[int_235404], **kwargs_235405)
        
        
        # Assigning a Num to a Name (line 287):
        
        # Assigning a Num to a Name (line 287):
        int_235407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 12), 'int')
        # Assigning a type to the variable 'n' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'n', int_235407)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 289)
        tuple_235408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 289)
        # Adding element type (line 289)
        str_235409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 21), 'str', 'easy')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 21), tuple_235408, str_235409)
        # Adding element type (line 289)
        str_235410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 29), 'str', 'hard')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 21), tuple_235408, str_235410)
        # Adding element type (line 289)
        str_235411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 37), 'str', 'jac_equal_zero')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 21), tuple_235408, str_235411)
        
        # Testing the type of a for loop iterable (line 289)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 289, 8), tuple_235408)
        # Getting the type of the for loop variable (line 289)
        for_loop_var_235412 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 289, 8), tuple_235408)
        # Assigning a type to the variable 'case' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'case', for_loop_var_235412)
        # SSA begins for a for statement (line 289)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a List to a Name (line 291):
        
        # Assigning a List to a Name (line 291):
        
        # Obtaining an instance of the builtin type 'list' (line 291)
        list_235413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 291)
        # Adding element type (line 291)
        
        # Obtaining an instance of the builtin type 'tuple' (line 291)
        tuple_235414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 291)
        # Adding element type (line 291)
        int_235415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 27), tuple_235414, int_235415)
        # Adding element type (line 291)
        int_235416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 27), tuple_235414, int_235416)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 25), list_235413, tuple_235414)
        # Adding element type (line 291)
        
        # Obtaining an instance of the builtin type 'tuple' (line 292)
        tuple_235417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 292)
        # Adding element type (line 292)
        int_235418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 27), tuple_235417, int_235418)
        # Adding element type (line 292)
        int_235419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 27), tuple_235417, int_235419)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 25), list_235413, tuple_235417)
        # Adding element type (line 291)
        
        # Obtaining an instance of the builtin type 'tuple' (line 293)
        tuple_235420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 293)
        # Adding element type (line 293)
        int_235421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 27), tuple_235420, int_235421)
        # Adding element type (line 293)
        int_235422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 27), tuple_235420, int_235422)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 25), list_235413, tuple_235420)
        # Adding element type (line 291)
        
        # Obtaining an instance of the builtin type 'tuple' (line 294)
        tuple_235423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 294)
        # Adding element type (line 294)
        int_235424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 27), tuple_235423, int_235424)
        # Adding element type (line 294)
        int_235425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 27), tuple_235423, int_235425)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 25), list_235413, tuple_235423)
        # Adding element type (line 291)
        
        # Obtaining an instance of the builtin type 'tuple' (line 295)
        tuple_235426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 295)
        # Adding element type (line 295)
        int_235427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 27), tuple_235426, int_235427)
        # Adding element type (line 295)
        int_235428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 27), tuple_235426, int_235428)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 25), list_235413, tuple_235426)
        # Adding element type (line 291)
        
        # Obtaining an instance of the builtin type 'tuple' (line 296)
        tuple_235429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 296)
        # Adding element type (line 296)
        int_235430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 27), tuple_235429, int_235430)
        # Adding element type (line 296)
        int_235431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 27), tuple_235429, int_235431)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 25), list_235413, tuple_235429)
        # Adding element type (line 291)
        
        # Obtaining an instance of the builtin type 'tuple' (line 297)
        tuple_235432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 297)
        # Adding element type (line 297)
        int_235433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 27), tuple_235432, int_235433)
        # Adding element type (line 297)
        int_235434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 27), tuple_235432, int_235434)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 25), list_235413, tuple_235432)
        # Adding element type (line 291)
        
        # Obtaining an instance of the builtin type 'tuple' (line 298)
        tuple_235435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 298)
        # Adding element type (line 298)
        int_235436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 27), tuple_235435, int_235436)
        # Adding element type (line 298)
        int_235437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 27), tuple_235435, int_235437)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 25), list_235413, tuple_235435)
        
        # Assigning a type to the variable 'eig_limits' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'eig_limits', list_235413)
        
        # Getting the type of 'eig_limits' (line 300)
        eig_limits_235438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 36), 'eig_limits')
        # Testing the type of a for loop iterable (line 300)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 300, 12), eig_limits_235438)
        # Getting the type of the for loop variable (line 300)
        for_loop_var_235439 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 300, 12), eig_limits_235438)
        # Assigning a type to the variable 'min_eig' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'min_eig', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 12), for_loop_var_235439))
        # Assigning a type to the variable 'max_eig' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'max_eig', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 12), for_loop_var_235439))
        # SSA begins for a for statement (line 300)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 303):
        
        # Assigning a Subscript to a Name (line 303):
        
        # Obtaining the type of the subscript
        int_235440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 16), 'int')
        
        # Call to random_entry(...): (line 303)
        # Processing the call arguments (line 303)
        # Getting the type of 'n' (line 303)
        n_235442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 36), 'n', False)
        # Getting the type of 'min_eig' (line 303)
        min_eig_235443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 39), 'min_eig', False)
        # Getting the type of 'max_eig' (line 303)
        max_eig_235444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 48), 'max_eig', False)
        # Getting the type of 'case' (line 303)
        case_235445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 57), 'case', False)
        # Processing the call keyword arguments (line 303)
        kwargs_235446 = {}
        # Getting the type of 'random_entry' (line 303)
        random_entry_235441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 23), 'random_entry', False)
        # Calling random_entry(args, kwargs) (line 303)
        random_entry_call_result_235447 = invoke(stypy.reporting.localization.Localization(__file__, 303, 23), random_entry_235441, *[n_235442, min_eig_235443, max_eig_235444, case_235445], **kwargs_235446)
        
        # Obtaining the member '__getitem__' of a type (line 303)
        getitem___235448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 16), random_entry_call_result_235447, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 303)
        subscript_call_result_235449 = invoke(stypy.reporting.localization.Localization(__file__, 303, 16), getitem___235448, int_235440)
        
        # Assigning a type to the variable 'tuple_var_assignment_234290' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'tuple_var_assignment_234290', subscript_call_result_235449)
        
        # Assigning a Subscript to a Name (line 303):
        
        # Obtaining the type of the subscript
        int_235450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 16), 'int')
        
        # Call to random_entry(...): (line 303)
        # Processing the call arguments (line 303)
        # Getting the type of 'n' (line 303)
        n_235452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 36), 'n', False)
        # Getting the type of 'min_eig' (line 303)
        min_eig_235453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 39), 'min_eig', False)
        # Getting the type of 'max_eig' (line 303)
        max_eig_235454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 48), 'max_eig', False)
        # Getting the type of 'case' (line 303)
        case_235455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 57), 'case', False)
        # Processing the call keyword arguments (line 303)
        kwargs_235456 = {}
        # Getting the type of 'random_entry' (line 303)
        random_entry_235451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 23), 'random_entry', False)
        # Calling random_entry(args, kwargs) (line 303)
        random_entry_call_result_235457 = invoke(stypy.reporting.localization.Localization(__file__, 303, 23), random_entry_235451, *[n_235452, min_eig_235453, max_eig_235454, case_235455], **kwargs_235456)
        
        # Obtaining the member '__getitem__' of a type (line 303)
        getitem___235458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 16), random_entry_call_result_235457, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 303)
        subscript_call_result_235459 = invoke(stypy.reporting.localization.Localization(__file__, 303, 16), getitem___235458, int_235450)
        
        # Assigning a type to the variable 'tuple_var_assignment_234291' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'tuple_var_assignment_234291', subscript_call_result_235459)
        
        # Assigning a Name to a Name (line 303):
        # Getting the type of 'tuple_var_assignment_234290' (line 303)
        tuple_var_assignment_234290_235460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'tuple_var_assignment_234290')
        # Assigning a type to the variable 'H' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'H', tuple_var_assignment_234290_235460)
        
        # Assigning a Name to a Name (line 303):
        # Getting the type of 'tuple_var_assignment_234291' (line 303)
        tuple_var_assignment_234291_235461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'tuple_var_assignment_234291')
        # Assigning a type to the variable 'g' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 19), 'g', tuple_var_assignment_234291_235461)
        
        # Assigning a List to a Name (line 306):
        
        # Assigning a List to a Name (line 306):
        
        # Obtaining an instance of the builtin type 'list' (line 306)
        list_235462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 306)
        # Adding element type (line 306)
        float_235463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 36), list_235462, float_235463)
        # Adding element type (line 306)
        float_235464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 36), list_235462, float_235464)
        # Adding element type (line 306)
        float_235465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 36), list_235462, float_235465)
        # Adding element type (line 306)
        float_235466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 36), list_235462, float_235466)
        # Adding element type (line 306)
        int_235467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 36), list_235462, int_235467)
        # Adding element type (line 306)
        float_235468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 36), list_235462, float_235468)
        # Adding element type (line 306)
        float_235469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 36), list_235462, float_235469)
        # Adding element type (line 306)
        float_235470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 70), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 36), list_235462, float_235470)
        # Adding element type (line 306)
        int_235471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 75), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 36), list_235462, int_235471)
        
        # Assigning a type to the variable 'trust_radius_list' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'trust_radius_list', list_235462)
        
        # Getting the type of 'trust_radius_list' (line 308)
        trust_radius_list_235472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 36), 'trust_radius_list')
        # Testing the type of a for loop iterable (line 308)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 308, 16), trust_radius_list_235472)
        # Getting the type of the for loop variable (line 308)
        for_loop_var_235473 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 308, 16), trust_radius_list_235472)
        # Assigning a type to the variable 'trust_radius' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 16), 'trust_radius', for_loop_var_235473)
        # SSA begins for a for statement (line 308)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 310):
        
        # Assigning a Call to a Name (line 310):
        
        # Call to IterativeSubproblem(...): (line 310)
        # Processing the call arguments (line 310)
        int_235475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 53), 'int')

        @norecursion
        def _stypy_temp_lambda_131(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_131'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_131', 311, 53, True)
            # Passed parameters checking function
            _stypy_temp_lambda_131.stypy_localization = localization
            _stypy_temp_lambda_131.stypy_type_of_self = None
            _stypy_temp_lambda_131.stypy_type_store = module_type_store
            _stypy_temp_lambda_131.stypy_function_name = '_stypy_temp_lambda_131'
            _stypy_temp_lambda_131.stypy_param_names_list = ['x']
            _stypy_temp_lambda_131.stypy_varargs_param_name = None
            _stypy_temp_lambda_131.stypy_kwargs_param_name = None
            _stypy_temp_lambda_131.stypy_call_defaults = defaults
            _stypy_temp_lambda_131.stypy_call_varargs = varargs
            _stypy_temp_lambda_131.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_131', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_131', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_235476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 63), 'int')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 311)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 53), 'stypy_return_type', int_235476)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_131' in the type store
            # Getting the type of 'stypy_return_type' (line 311)
            stypy_return_type_235477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 53), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235477)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_131'
            return stypy_return_type_235477

        # Assigning a type to the variable '_stypy_temp_lambda_131' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 53), '_stypy_temp_lambda_131', _stypy_temp_lambda_131)
        # Getting the type of '_stypy_temp_lambda_131' (line 311)
        _stypy_temp_lambda_131_235478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 53), '_stypy_temp_lambda_131')

        @norecursion
        def _stypy_temp_lambda_132(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_132'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_132', 312, 53, True)
            # Passed parameters checking function
            _stypy_temp_lambda_132.stypy_localization = localization
            _stypy_temp_lambda_132.stypy_type_of_self = None
            _stypy_temp_lambda_132.stypy_type_store = module_type_store
            _stypy_temp_lambda_132.stypy_function_name = '_stypy_temp_lambda_132'
            _stypy_temp_lambda_132.stypy_param_names_list = ['x']
            _stypy_temp_lambda_132.stypy_varargs_param_name = None
            _stypy_temp_lambda_132.stypy_kwargs_param_name = None
            _stypy_temp_lambda_132.stypy_call_defaults = defaults
            _stypy_temp_lambda_132.stypy_call_varargs = varargs
            _stypy_temp_lambda_132.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_132', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_132', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'g' (line 312)
            g_235479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 63), 'g', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 312)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 53), 'stypy_return_type', g_235479)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_132' in the type store
            # Getting the type of 'stypy_return_type' (line 312)
            stypy_return_type_235480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 53), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235480)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_132'
            return stypy_return_type_235480

        # Assigning a type to the variable '_stypy_temp_lambda_132' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 53), '_stypy_temp_lambda_132', _stypy_temp_lambda_132)
        # Getting the type of '_stypy_temp_lambda_132' (line 312)
        _stypy_temp_lambda_132_235481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 53), '_stypy_temp_lambda_132')

        @norecursion
        def _stypy_temp_lambda_133(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_133'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_133', 313, 53, True)
            # Passed parameters checking function
            _stypy_temp_lambda_133.stypy_localization = localization
            _stypy_temp_lambda_133.stypy_type_of_self = None
            _stypy_temp_lambda_133.stypy_type_store = module_type_store
            _stypy_temp_lambda_133.stypy_function_name = '_stypy_temp_lambda_133'
            _stypy_temp_lambda_133.stypy_param_names_list = ['x']
            _stypy_temp_lambda_133.stypy_varargs_param_name = None
            _stypy_temp_lambda_133.stypy_kwargs_param_name = None
            _stypy_temp_lambda_133.stypy_call_defaults = defaults
            _stypy_temp_lambda_133.stypy_call_varargs = varargs
            _stypy_temp_lambda_133.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_133', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_133', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'H' (line 313)
            H_235482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 63), 'H', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 313)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 53), 'stypy_return_type', H_235482)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_133' in the type store
            # Getting the type of 'stypy_return_type' (line 313)
            stypy_return_type_235483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 53), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235483)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_133'
            return stypy_return_type_235483

        # Assigning a type to the variable '_stypy_temp_lambda_133' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 53), '_stypy_temp_lambda_133', _stypy_temp_lambda_133)
        # Getting the type of '_stypy_temp_lambda_133' (line 313)
        _stypy_temp_lambda_133_235484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 53), '_stypy_temp_lambda_133')
        # Processing the call keyword arguments (line 310)
        float_235485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 60), 'float')
        keyword_235486 = float_235485
        float_235487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 60), 'float')
        keyword_235488 = float_235487
        kwargs_235489 = {'k_easy': keyword_235486, 'k_hard': keyword_235488}
        # Getting the type of 'IterativeSubproblem' (line 310)
        IterativeSubproblem_235474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 33), 'IterativeSubproblem', False)
        # Calling IterativeSubproblem(args, kwargs) (line 310)
        IterativeSubproblem_call_result_235490 = invoke(stypy.reporting.localization.Localization(__file__, 310, 33), IterativeSubproblem_235474, *[int_235475, _stypy_temp_lambda_131_235478, _stypy_temp_lambda_132_235481, _stypy_temp_lambda_133_235484], **kwargs_235489)
        
        # Assigning a type to the variable 'subprob_ac' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 20), 'subprob_ac', IterativeSubproblem_call_result_235490)
        
        # Assigning a Call to a Tuple (line 317):
        
        # Assigning a Subscript to a Name (line 317):
        
        # Obtaining the type of the subscript
        int_235491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 20), 'int')
        
        # Call to solve(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'trust_radius' (line 317)
        trust_radius_235494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 62), 'trust_radius', False)
        # Processing the call keyword arguments (line 317)
        kwargs_235495 = {}
        # Getting the type of 'subprob_ac' (line 317)
        subprob_ac_235492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 45), 'subprob_ac', False)
        # Obtaining the member 'solve' of a type (line 317)
        solve_235493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 45), subprob_ac_235492, 'solve')
        # Calling solve(args, kwargs) (line 317)
        solve_call_result_235496 = invoke(stypy.reporting.localization.Localization(__file__, 317, 45), solve_235493, *[trust_radius_235494], **kwargs_235495)
        
        # Obtaining the member '__getitem__' of a type (line 317)
        getitem___235497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 20), solve_call_result_235496, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 317)
        subscript_call_result_235498 = invoke(stypy.reporting.localization.Localization(__file__, 317, 20), getitem___235497, int_235491)
        
        # Assigning a type to the variable 'tuple_var_assignment_234292' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 20), 'tuple_var_assignment_234292', subscript_call_result_235498)
        
        # Assigning a Subscript to a Name (line 317):
        
        # Obtaining the type of the subscript
        int_235499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 20), 'int')
        
        # Call to solve(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'trust_radius' (line 317)
        trust_radius_235502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 62), 'trust_radius', False)
        # Processing the call keyword arguments (line 317)
        kwargs_235503 = {}
        # Getting the type of 'subprob_ac' (line 317)
        subprob_ac_235500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 45), 'subprob_ac', False)
        # Obtaining the member 'solve' of a type (line 317)
        solve_235501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 45), subprob_ac_235500, 'solve')
        # Calling solve(args, kwargs) (line 317)
        solve_call_result_235504 = invoke(stypy.reporting.localization.Localization(__file__, 317, 45), solve_235501, *[trust_radius_235502], **kwargs_235503)
        
        # Obtaining the member '__getitem__' of a type (line 317)
        getitem___235505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 20), solve_call_result_235504, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 317)
        subscript_call_result_235506 = invoke(stypy.reporting.localization.Localization(__file__, 317, 20), getitem___235505, int_235499)
        
        # Assigning a type to the variable 'tuple_var_assignment_234293' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 20), 'tuple_var_assignment_234293', subscript_call_result_235506)
        
        # Assigning a Name to a Name (line 317):
        # Getting the type of 'tuple_var_assignment_234292' (line 317)
        tuple_var_assignment_234292_235507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 20), 'tuple_var_assignment_234292')
        # Assigning a type to the variable 'p_ac' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 20), 'p_ac', tuple_var_assignment_234292_235507)
        
        # Assigning a Name to a Name (line 317):
        # Getting the type of 'tuple_var_assignment_234293' (line 317)
        tuple_var_assignment_234293_235508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 20), 'tuple_var_assignment_234293')
        # Assigning a type to the variable 'hits_boundary_ac' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 26), 'hits_boundary_ac', tuple_var_assignment_234293_235508)
        
        # Assigning a BinOp to a Name (line 320):
        
        # Assigning a BinOp to a Name (line 320):
        int_235509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 27), 'int')
        int_235510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 29), 'int')
        # Applying the binary operator 'div' (line 320)
        result_div_235511 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 27), 'div', int_235509, int_235510)
        
        
        # Call to dot(...): (line 320)
        # Processing the call arguments (line 320)
        # Getting the type of 'p_ac' (line 320)
        p_ac_235514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 38), 'p_ac', False)
        
        # Call to dot(...): (line 320)
        # Processing the call arguments (line 320)
        # Getting the type of 'H' (line 320)
        H_235517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 51), 'H', False)
        # Getting the type of 'p_ac' (line 320)
        p_ac_235518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 54), 'p_ac', False)
        # Processing the call keyword arguments (line 320)
        kwargs_235519 = {}
        # Getting the type of 'np' (line 320)
        np_235515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 44), 'np', False)
        # Obtaining the member 'dot' of a type (line 320)
        dot_235516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 44), np_235515, 'dot')
        # Calling dot(args, kwargs) (line 320)
        dot_call_result_235520 = invoke(stypy.reporting.localization.Localization(__file__, 320, 44), dot_235516, *[H_235517, p_ac_235518], **kwargs_235519)
        
        # Processing the call keyword arguments (line 320)
        kwargs_235521 = {}
        # Getting the type of 'np' (line 320)
        np_235512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 31), 'np', False)
        # Obtaining the member 'dot' of a type (line 320)
        dot_235513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 31), np_235512, 'dot')
        # Calling dot(args, kwargs) (line 320)
        dot_call_result_235522 = invoke(stypy.reporting.localization.Localization(__file__, 320, 31), dot_235513, *[p_ac_235514, dot_call_result_235520], **kwargs_235521)
        
        # Applying the binary operator '*' (line 320)
        result_mul_235523 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 30), '*', result_div_235511, dot_call_result_235522)
        
        
        # Call to dot(...): (line 320)
        # Processing the call arguments (line 320)
        # Getting the type of 'g' (line 320)
        g_235526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 68), 'g', False)
        # Getting the type of 'p_ac' (line 320)
        p_ac_235527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 71), 'p_ac', False)
        # Processing the call keyword arguments (line 320)
        kwargs_235528 = {}
        # Getting the type of 'np' (line 320)
        np_235524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 61), 'np', False)
        # Obtaining the member 'dot' of a type (line 320)
        dot_235525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 61), np_235524, 'dot')
        # Calling dot(args, kwargs) (line 320)
        dot_call_result_235529 = invoke(stypy.reporting.localization.Localization(__file__, 320, 61), dot_235525, *[g_235526, p_ac_235527], **kwargs_235528)
        
        # Applying the binary operator '+' (line 320)
        result_add_235530 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 27), '+', result_mul_235523, dot_call_result_235529)
        
        # Assigning a type to the variable 'J_ac' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 20), 'J_ac', result_add_235530)
        
        # Assigning a List to a Name (line 322):
        
        # Assigning a List to a Name (line 322):
        
        # Obtaining an instance of the builtin type 'list' (line 322)
        list_235531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 322)
        # Adding element type (line 322)
        
        # Obtaining an instance of the builtin type 'tuple' (line 322)
        tuple_235532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 322)
        # Adding element type (line 322)
        float_235533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 38), tuple_235532, float_235533)
        # Adding element type (line 322)
        int_235534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 38), tuple_235532, int_235534)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 36), list_235531, tuple_235532)
        # Adding element type (line 322)
        
        # Obtaining an instance of the builtin type 'tuple' (line 323)
        tuple_235535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 323)
        # Adding element type (line 323)
        float_235536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 38), tuple_235535, float_235536)
        # Adding element type (line 323)
        float_235537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 38), tuple_235535, float_235537)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 36), list_235531, tuple_235535)
        # Adding element type (line 322)
        
        # Obtaining an instance of the builtin type 'tuple' (line 324)
        tuple_235538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 324)
        # Adding element type (line 324)
        float_235539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 38), tuple_235538, float_235539)
        # Adding element type (line 324)
        float_235540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 38), tuple_235538, float_235540)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 36), list_235531, tuple_235538)
        
        # Assigning a type to the variable 'stop_criteria' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 20), 'stop_criteria', list_235531)
        
        # Getting the type of 'stop_criteria' (line 326)
        stop_criteria_235541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 40), 'stop_criteria')
        # Testing the type of a for loop iterable (line 326)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 326, 20), stop_criteria_235541)
        # Getting the type of the for loop variable (line 326)
        for_loop_var_235542 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 326, 20), stop_criteria_235541)
        # Assigning a type to the variable 'k_opt' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 20), 'k_opt', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 20), for_loop_var_235542))
        # Assigning a type to the variable 'k_trf' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 20), 'k_trf', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 20), for_loop_var_235542))
        # SSA begins for a for statement (line 326)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 332):
        
        # Assigning a Call to a Name (line 332):
        
        # Call to min(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 'k_trf' (line 332)
        k_trf_235544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 37), 'k_trf', False)
        int_235545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 43), 'int')
        # Applying the binary operator '-' (line 332)
        result_sub_235546 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 37), '-', k_trf_235544, int_235545)
        
        int_235547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 37), 'int')
        
        # Call to sqrt(...): (line 333)
        # Processing the call arguments (line 333)
        # Getting the type of 'k_opt' (line 333)
        k_opt_235550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 47), 'k_opt', False)
        # Processing the call keyword arguments (line 333)
        kwargs_235551 = {}
        # Getting the type of 'np' (line 333)
        np_235548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 39), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 333)
        sqrt_235549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 39), np_235548, 'sqrt')
        # Calling sqrt(args, kwargs) (line 333)
        sqrt_call_result_235552 = invoke(stypy.reporting.localization.Localization(__file__, 333, 39), sqrt_235549, *[k_opt_235550], **kwargs_235551)
        
        # Applying the binary operator '-' (line 333)
        result_sub_235553 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 37), '-', int_235547, sqrt_call_result_235552)
        
        # Processing the call keyword arguments (line 332)
        kwargs_235554 = {}
        # Getting the type of 'min' (line 332)
        min_235543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 33), 'min', False)
        # Calling min(args, kwargs) (line 332)
        min_call_result_235555 = invoke(stypy.reporting.localization.Localization(__file__, 332, 33), min_235543, *[result_sub_235546, result_sub_235553], **kwargs_235554)
        
        # Assigning a type to the variable 'k_easy' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 24), 'k_easy', min_call_result_235555)
        
        # Assigning a BinOp to a Name (line 334):
        
        # Assigning a BinOp to a Name (line 334):
        int_235556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 33), 'int')
        # Getting the type of 'k_opt' (line 334)
        k_opt_235557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 35), 'k_opt')
        # Applying the binary operator '-' (line 334)
        result_sub_235558 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 33), '-', int_235556, k_opt_235557)
        
        # Assigning a type to the variable 'k_hard' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 24), 'k_hard', result_sub_235558)
        
        # Assigning a Call to a Name (line 337):
        
        # Assigning a Call to a Name (line 337):
        
        # Call to IterativeSubproblem(...): (line 337)
        # Processing the call arguments (line 337)
        int_235560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 54), 'int')

        @norecursion
        def _stypy_temp_lambda_134(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_134'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_134', 338, 54, True)
            # Passed parameters checking function
            _stypy_temp_lambda_134.stypy_localization = localization
            _stypy_temp_lambda_134.stypy_type_of_self = None
            _stypy_temp_lambda_134.stypy_type_store = module_type_store
            _stypy_temp_lambda_134.stypy_function_name = '_stypy_temp_lambda_134'
            _stypy_temp_lambda_134.stypy_param_names_list = ['x']
            _stypy_temp_lambda_134.stypy_varargs_param_name = None
            _stypy_temp_lambda_134.stypy_kwargs_param_name = None
            _stypy_temp_lambda_134.stypy_call_defaults = defaults
            _stypy_temp_lambda_134.stypy_call_varargs = varargs
            _stypy_temp_lambda_134.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_134', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_134', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_235561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 64), 'int')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 338)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 54), 'stypy_return_type', int_235561)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_134' in the type store
            # Getting the type of 'stypy_return_type' (line 338)
            stypy_return_type_235562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 54), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235562)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_134'
            return stypy_return_type_235562

        # Assigning a type to the variable '_stypy_temp_lambda_134' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 54), '_stypy_temp_lambda_134', _stypy_temp_lambda_134)
        # Getting the type of '_stypy_temp_lambda_134' (line 338)
        _stypy_temp_lambda_134_235563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 54), '_stypy_temp_lambda_134')

        @norecursion
        def _stypy_temp_lambda_135(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_135'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_135', 339, 54, True)
            # Passed parameters checking function
            _stypy_temp_lambda_135.stypy_localization = localization
            _stypy_temp_lambda_135.stypy_type_of_self = None
            _stypy_temp_lambda_135.stypy_type_store = module_type_store
            _stypy_temp_lambda_135.stypy_function_name = '_stypy_temp_lambda_135'
            _stypy_temp_lambda_135.stypy_param_names_list = ['x']
            _stypy_temp_lambda_135.stypy_varargs_param_name = None
            _stypy_temp_lambda_135.stypy_kwargs_param_name = None
            _stypy_temp_lambda_135.stypy_call_defaults = defaults
            _stypy_temp_lambda_135.stypy_call_varargs = varargs
            _stypy_temp_lambda_135.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_135', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_135', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'g' (line 339)
            g_235564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 64), 'g', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 339)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 54), 'stypy_return_type', g_235564)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_135' in the type store
            # Getting the type of 'stypy_return_type' (line 339)
            stypy_return_type_235565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 54), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235565)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_135'
            return stypy_return_type_235565

        # Assigning a type to the variable '_stypy_temp_lambda_135' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 54), '_stypy_temp_lambda_135', _stypy_temp_lambda_135)
        # Getting the type of '_stypy_temp_lambda_135' (line 339)
        _stypy_temp_lambda_135_235566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 54), '_stypy_temp_lambda_135')

        @norecursion
        def _stypy_temp_lambda_136(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_136'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_136', 340, 54, True)
            # Passed parameters checking function
            _stypy_temp_lambda_136.stypy_localization = localization
            _stypy_temp_lambda_136.stypy_type_of_self = None
            _stypy_temp_lambda_136.stypy_type_store = module_type_store
            _stypy_temp_lambda_136.stypy_function_name = '_stypy_temp_lambda_136'
            _stypy_temp_lambda_136.stypy_param_names_list = ['x']
            _stypy_temp_lambda_136.stypy_varargs_param_name = None
            _stypy_temp_lambda_136.stypy_kwargs_param_name = None
            _stypy_temp_lambda_136.stypy_call_defaults = defaults
            _stypy_temp_lambda_136.stypy_call_varargs = varargs
            _stypy_temp_lambda_136.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_136', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_136', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'H' (line 340)
            H_235567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 64), 'H', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 340)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 54), 'stypy_return_type', H_235567)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_136' in the type store
            # Getting the type of 'stypy_return_type' (line 340)
            stypy_return_type_235568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 54), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_235568)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_136'
            return stypy_return_type_235568

        # Assigning a type to the variable '_stypy_temp_lambda_136' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 54), '_stypy_temp_lambda_136', _stypy_temp_lambda_136)
        # Getting the type of '_stypy_temp_lambda_136' (line 340)
        _stypy_temp_lambda_136_235569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 54), '_stypy_temp_lambda_136')
        # Processing the call keyword arguments (line 337)
        # Getting the type of 'k_easy' (line 341)
        k_easy_235570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 61), 'k_easy', False)
        keyword_235571 = k_easy_235570
        # Getting the type of 'k_hard' (line 342)
        k_hard_235572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 61), 'k_hard', False)
        keyword_235573 = k_hard_235572
        kwargs_235574 = {'k_easy': keyword_235571, 'k_hard': keyword_235573}
        # Getting the type of 'IterativeSubproblem' (line 337)
        IterativeSubproblem_235559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 34), 'IterativeSubproblem', False)
        # Calling IterativeSubproblem(args, kwargs) (line 337)
        IterativeSubproblem_call_result_235575 = invoke(stypy.reporting.localization.Localization(__file__, 337, 34), IterativeSubproblem_235559, *[int_235560, _stypy_temp_lambda_134_235563, _stypy_temp_lambda_135_235566, _stypy_temp_lambda_136_235569], **kwargs_235574)
        
        # Assigning a type to the variable 'subprob' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 24), 'subprob', IterativeSubproblem_call_result_235575)
        
        # Assigning a Call to a Tuple (line 343):
        
        # Assigning a Subscript to a Name (line 343):
        
        # Obtaining the type of the subscript
        int_235576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 24), 'int')
        
        # Call to solve(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'trust_radius' (line 343)
        trust_radius_235579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 57), 'trust_radius', False)
        # Processing the call keyword arguments (line 343)
        kwargs_235580 = {}
        # Getting the type of 'subprob' (line 343)
        subprob_235577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 43), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 343)
        solve_235578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 43), subprob_235577, 'solve')
        # Calling solve(args, kwargs) (line 343)
        solve_call_result_235581 = invoke(stypy.reporting.localization.Localization(__file__, 343, 43), solve_235578, *[trust_radius_235579], **kwargs_235580)
        
        # Obtaining the member '__getitem__' of a type (line 343)
        getitem___235582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 24), solve_call_result_235581, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 343)
        subscript_call_result_235583 = invoke(stypy.reporting.localization.Localization(__file__, 343, 24), getitem___235582, int_235576)
        
        # Assigning a type to the variable 'tuple_var_assignment_234294' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 24), 'tuple_var_assignment_234294', subscript_call_result_235583)
        
        # Assigning a Subscript to a Name (line 343):
        
        # Obtaining the type of the subscript
        int_235584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 24), 'int')
        
        # Call to solve(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'trust_radius' (line 343)
        trust_radius_235587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 57), 'trust_radius', False)
        # Processing the call keyword arguments (line 343)
        kwargs_235588 = {}
        # Getting the type of 'subprob' (line 343)
        subprob_235585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 43), 'subprob', False)
        # Obtaining the member 'solve' of a type (line 343)
        solve_235586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 43), subprob_235585, 'solve')
        # Calling solve(args, kwargs) (line 343)
        solve_call_result_235589 = invoke(stypy.reporting.localization.Localization(__file__, 343, 43), solve_235586, *[trust_radius_235587], **kwargs_235588)
        
        # Obtaining the member '__getitem__' of a type (line 343)
        getitem___235590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 24), solve_call_result_235589, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 343)
        subscript_call_result_235591 = invoke(stypy.reporting.localization.Localization(__file__, 343, 24), getitem___235590, int_235584)
        
        # Assigning a type to the variable 'tuple_var_assignment_234295' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 24), 'tuple_var_assignment_234295', subscript_call_result_235591)
        
        # Assigning a Name to a Name (line 343):
        # Getting the type of 'tuple_var_assignment_234294' (line 343)
        tuple_var_assignment_234294_235592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 24), 'tuple_var_assignment_234294')
        # Assigning a type to the variable 'p' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 24), 'p', tuple_var_assignment_234294_235592)
        
        # Assigning a Name to a Name (line 343):
        # Getting the type of 'tuple_var_assignment_234295' (line 343)
        tuple_var_assignment_234295_235593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 24), 'tuple_var_assignment_234295')
        # Assigning a type to the variable 'hits_boundary' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 27), 'hits_boundary', tuple_var_assignment_234295_235593)
        
        # Assigning a BinOp to a Name (line 346):
        
        # Assigning a BinOp to a Name (line 346):
        int_235594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 28), 'int')
        int_235595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 30), 'int')
        # Applying the binary operator 'div' (line 346)
        result_div_235596 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 28), 'div', int_235594, int_235595)
        
        
        # Call to dot(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'p' (line 346)
        p_235599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 39), 'p', False)
        
        # Call to dot(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'H' (line 346)
        H_235602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 49), 'H', False)
        # Getting the type of 'p' (line 346)
        p_235603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 52), 'p', False)
        # Processing the call keyword arguments (line 346)
        kwargs_235604 = {}
        # Getting the type of 'np' (line 346)
        np_235600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 42), 'np', False)
        # Obtaining the member 'dot' of a type (line 346)
        dot_235601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 42), np_235600, 'dot')
        # Calling dot(args, kwargs) (line 346)
        dot_call_result_235605 = invoke(stypy.reporting.localization.Localization(__file__, 346, 42), dot_235601, *[H_235602, p_235603], **kwargs_235604)
        
        # Processing the call keyword arguments (line 346)
        kwargs_235606 = {}
        # Getting the type of 'np' (line 346)
        np_235597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 32), 'np', False)
        # Obtaining the member 'dot' of a type (line 346)
        dot_235598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 32), np_235597, 'dot')
        # Calling dot(args, kwargs) (line 346)
        dot_call_result_235607 = invoke(stypy.reporting.localization.Localization(__file__, 346, 32), dot_235598, *[p_235599, dot_call_result_235605], **kwargs_235606)
        
        # Applying the binary operator '*' (line 346)
        result_mul_235608 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 31), '*', result_div_235596, dot_call_result_235607)
        
        
        # Call to dot(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'g' (line 346)
        g_235611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 63), 'g', False)
        # Getting the type of 'p' (line 346)
        p_235612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 66), 'p', False)
        # Processing the call keyword arguments (line 346)
        kwargs_235613 = {}
        # Getting the type of 'np' (line 346)
        np_235609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 56), 'np', False)
        # Obtaining the member 'dot' of a type (line 346)
        dot_235610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 56), np_235609, 'dot')
        # Calling dot(args, kwargs) (line 346)
        dot_call_result_235614 = invoke(stypy.reporting.localization.Localization(__file__, 346, 56), dot_235610, *[g_235611, p_235612], **kwargs_235613)
        
        # Applying the binary operator '+' (line 346)
        result_add_235615 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 28), '+', result_mul_235608, dot_call_result_235614)
        
        # Assigning a type to the variable 'J' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 24), 'J', result_add_235615)
        
        # Getting the type of 'hits_boundary' (line 349)
        hits_boundary_235616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 27), 'hits_boundary')
        # Testing the type of an if condition (line 349)
        if_condition_235617 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 349, 24), hits_boundary_235616)
        # Assigning a type to the variable 'if_condition_235617' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 24), 'if_condition_235617', if_condition_235617)
        # SSA begins for if statement (line 349)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_array_equal(...): (line 350)
        # Processing the call arguments (line 350)
        
        
        # Call to abs(...): (line 350)
        # Processing the call arguments (line 350)
        
        # Call to norm(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 'p' (line 350)
        p_235622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 59), 'p', False)
        # Processing the call keyword arguments (line 350)
        kwargs_235623 = {}
        # Getting the type of 'norm' (line 350)
        norm_235621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 54), 'norm', False)
        # Calling norm(args, kwargs) (line 350)
        norm_call_result_235624 = invoke(stypy.reporting.localization.Localization(__file__, 350, 54), norm_235621, *[p_235622], **kwargs_235623)
        
        # Getting the type of 'trust_radius' (line 350)
        trust_radius_235625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 62), 'trust_radius', False)
        # Applying the binary operator '-' (line 350)
        result_sub_235626 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 54), '-', norm_call_result_235624, trust_radius_235625)
        
        # Processing the call keyword arguments (line 350)
        kwargs_235627 = {}
        # Getting the type of 'np' (line 350)
        np_235619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 47), 'np', False)
        # Obtaining the member 'abs' of a type (line 350)
        abs_235620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 47), np_235619, 'abs')
        # Calling abs(args, kwargs) (line 350)
        abs_call_result_235628 = invoke(stypy.reporting.localization.Localization(__file__, 350, 47), abs_235620, *[result_sub_235626], **kwargs_235627)
        
        # Getting the type of 'k_trf' (line 351)
        k_trf_235629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 48), 'k_trf', False)
        int_235630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 54), 'int')
        # Applying the binary operator '-' (line 351)
        result_sub_235631 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 48), '-', k_trf_235629, int_235630)
        
        # Getting the type of 'trust_radius' (line 351)
        trust_radius_235632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 57), 'trust_radius', False)
        # Applying the binary operator '*' (line 351)
        result_mul_235633 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 47), '*', result_sub_235631, trust_radius_235632)
        
        # Applying the binary operator '<=' (line 350)
        result_le_235634 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 47), '<=', abs_call_result_235628, result_mul_235633)
        
        # Getting the type of 'True' (line 351)
        True_235635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 71), 'True', False)
        # Processing the call keyword arguments (line 350)
        kwargs_235636 = {}
        # Getting the type of 'assert_array_equal' (line 350)
        assert_array_equal_235618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 28), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 350)
        assert_array_equal_call_result_235637 = invoke(stypy.reporting.localization.Localization(__file__, 350, 28), assert_array_equal_235618, *[result_le_235634, True_235635], **kwargs_235636)
        
        # SSA branch for the else part of an if statement (line 349)
        module_type_store.open_ssa_branch('else')
        
        # Call to assert_equal(...): (line 353)
        # Processing the call arguments (line 353)
        
        
        # Call to norm(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 'p' (line 353)
        p_235640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 46), 'p', False)
        # Processing the call keyword arguments (line 353)
        kwargs_235641 = {}
        # Getting the type of 'norm' (line 353)
        norm_235639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 41), 'norm', False)
        # Calling norm(args, kwargs) (line 353)
        norm_call_result_235642 = invoke(stypy.reporting.localization.Localization(__file__, 353, 41), norm_235639, *[p_235640], **kwargs_235641)
        
        # Getting the type of 'trust_radius' (line 353)
        trust_radius_235643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 52), 'trust_radius', False)
        # Applying the binary operator '<=' (line 353)
        result_le_235644 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 41), '<=', norm_call_result_235642, trust_radius_235643)
        
        # Getting the type of 'True' (line 353)
        True_235645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 66), 'True', False)
        # Processing the call keyword arguments (line 353)
        kwargs_235646 = {}
        # Getting the type of 'assert_equal' (line 353)
        assert_equal_235638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 28), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 353)
        assert_equal_call_result_235647 = invoke(stypy.reporting.localization.Localization(__file__, 353, 28), assert_equal_235638, *[result_le_235644, True_235645], **kwargs_235646)
        
        # SSA join for if statement (line 349)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_equal(...): (line 356)
        # Processing the call arguments (line 356)
        
        # Getting the type of 'J' (line 356)
        J_235649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 37), 'J', False)
        # Getting the type of 'k_opt' (line 356)
        k_opt_235650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 42), 'k_opt', False)
        # Getting the type of 'J_ac' (line 356)
        J_ac_235651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 48), 'J_ac', False)
        # Applying the binary operator '*' (line 356)
        result_mul_235652 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 42), '*', k_opt_235650, J_ac_235651)
        
        # Applying the binary operator '<=' (line 356)
        result_le_235653 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 37), '<=', J_235649, result_mul_235652)
        
        # Getting the type of 'True' (line 356)
        True_235654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 54), 'True', False)
        # Processing the call keyword arguments (line 356)
        kwargs_235655 = {}
        # Getting the type of 'assert_equal' (line 356)
        assert_equal_235648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 24), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 356)
        assert_equal_call_result_235656 = invoke(stypy.reporting.localization.Localization(__file__, 356, 24), assert_equal_235648, *[result_le_235653, True_235654], **kwargs_235655)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_for_random_entries(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_for_random_entries' in the type store
        # Getting the type of 'stypy_return_type' (line 282)
        stypy_return_type_235657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_235657)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_for_random_entries'
        return stypy_return_type_235657


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 158, 0, False)
        # Assigning a type to the variable 'self' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIterativeSubproblem.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestIterativeSubproblem' (line 158)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'TestIterativeSubproblem', TestIterativeSubproblem)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
