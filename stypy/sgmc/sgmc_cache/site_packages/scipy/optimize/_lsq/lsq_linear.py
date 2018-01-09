
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Linear least squares with bound constraints on independent variables.'''
2: from __future__ import division, print_function, absolute_import
3: 
4: import numpy as np
5: from numpy.linalg import norm
6: from scipy.sparse import issparse, csr_matrix
7: from scipy.sparse.linalg import LinearOperator, lsmr
8: from scipy.optimize import OptimizeResult
9: 
10: from .common import in_bounds, compute_grad
11: from .trf_linear import trf_linear
12: from .bvls import bvls
13: 
14: 
15: def prepare_bounds(bounds, n):
16:     lb, ub = [np.asarray(b, dtype=float) for b in bounds]
17: 
18:     if lb.ndim == 0:
19:         lb = np.resize(lb, n)
20: 
21:     if ub.ndim == 0:
22:         ub = np.resize(ub, n)
23: 
24:     return lb, ub
25: 
26: 
27: TERMINATION_MESSAGES = {
28:     -1: "The algorithm was not able to make progress on the last iteration.",
29:     0: "The maximum number of iterations is exceeded.",
30:     1: "The first-order optimality measure is less than `tol`.",
31:     2: "The relative change of the cost function is less than `tol`.",
32:     3: "The unconstrained solution is optimal."
33: }
34: 
35: 
36: def lsq_linear(A, b, bounds=(-np.inf, np.inf), method='trf', tol=1e-10,
37:                lsq_solver=None, lsmr_tol=None, max_iter=None, verbose=0):
38:     r'''Solve a linear least-squares problem with bounds on the variables.
39: 
40:     Given a m-by-n design matrix A and a target vector b with m elements,
41:     `lsq_linear` solves the following optimization problem::
42: 
43:         minimize 0.5 * ||A x - b||**2
44:         subject to lb <= x <= ub
45: 
46:     This optimization problem is convex, hence a found minimum (if iterations
47:     have converged) is guaranteed to be global.
48: 
49:     Parameters
50:     ----------
51:     A : array_like, sparse matrix of LinearOperator, shape (m, n)
52:         Design matrix. Can be `scipy.sparse.linalg.LinearOperator`.
53:     b : array_like, shape (m,)
54:         Target vector.
55:     bounds : 2-tuple of array_like, optional
56:         Lower and upper bounds on independent variables. Defaults to no bounds.
57:         Each array must have shape (n,) or be a scalar, in the latter
58:         case a bound will be the same for all variables. Use ``np.inf`` with
59:         an appropriate sign to disable bounds on all or some variables.
60:     method : 'trf' or 'bvls', optional
61:         Method to perform minimization.
62: 
63:             * 'trf' : Trust Region Reflective algorithm adapted for a linear
64:               least-squares problem. This is an interior-point-like method
65:               and the required number of iterations is weakly correlated with
66:               the number of variables.
67:             * 'bvls' : Bounded-Variable Least-Squares algorithm. This is
68:               an active set method, which requires the number of iterations
69:               comparable to the number of variables. Can't be used when `A` is
70:               sparse or LinearOperator.
71: 
72:         Default is 'trf'.
73:     tol : float, optional
74:         Tolerance parameter. The algorithm terminates if a relative change
75:         of the cost function is less than `tol` on the last iteration.
76:         Additionally the first-order optimality measure is considered:
77: 
78:             * ``method='trf'`` terminates if the uniform norm of the gradient,
79:               scaled to account for the presence of the bounds, is less than
80:               `tol`.
81:             * ``method='bvls'`` terminates if Karush-Kuhn-Tucker conditions
82:               are satisfied within `tol` tolerance.
83: 
84:     lsq_solver : {None, 'exact', 'lsmr'}, optional
85:         Method of solving unbounded least-squares problems throughout
86:         iterations:
87: 
88:             * 'exact' : Use dense QR or SVD decomposition approach. Can't be
89:               used when `A` is sparse or LinearOperator.
90:             * 'lsmr' : Use `scipy.sparse.linalg.lsmr` iterative procedure
91:               which requires only matrix-vector product evaluations. Can't
92:               be used with ``method='bvls'``.
93: 
94:         If None (default) the solver is chosen based on type of `A`.
95:     lsmr_tol : None, float or 'auto', optional
96:         Tolerance parameters 'atol' and 'btol' for `scipy.sparse.linalg.lsmr`
97:         If None (default), it is set to ``1e-2 * tol``. If 'auto', the
98:         tolerance will be adjusted based on the optimality of the current
99:         iterate, which can speed up the optimization process, but is not always
100:         reliable.
101:     max_iter : None or int, optional
102:         Maximum number of iterations before termination. If None (default), it
103:         is set to 100 for ``method='trf'`` or to the number of variables for
104:         ``method='bvls'`` (not counting iterations for 'bvls' initialization).
105:     verbose : {0, 1, 2}, optional
106:         Level of algorithm's verbosity:
107: 
108:             * 0 : work silently (default).
109:             * 1 : display a termination report.
110:             * 2 : display progress during iterations.
111: 
112:     Returns
113:     -------
114:     OptimizeResult with the following fields defined:
115:     x : ndarray, shape (n,)
116:         Solution found.
117:     cost : float
118:         Value of the cost function at the solution.
119:     fun : ndarray, shape (m,)
120:         Vector of residuals at the solution.
121:     optimality : float
122:         First-order optimality measure. The exact meaning depends on `method`,
123:         refer to the description of `tol` parameter.
124:     active_mask : ndarray of int, shape (n,)
125:         Each component shows whether a corresponding constraint is active
126:         (that is, whether a variable is at the bound):
127: 
128:             *  0 : a constraint is not active.
129:             * -1 : a lower bound is active.
130:             *  1 : an upper bound is active.
131: 
132:         Might be somewhat arbitrary for the `trf` method as it generates a
133:         sequence of strictly feasible iterates and active_mask is determined
134:         within a tolerance threshold.
135:     nit : int
136:         Number of iterations. Zero if the unconstrained solution is optimal.
137:     status : int
138:         Reason for algorithm termination:
139: 
140:             * -1 : the algorithm was not able to make progress on the last
141:               iteration.
142:             *  0 : the maximum number of iterations is exceeded.
143:             *  1 : the first-order optimality measure is less than `tol`.
144:             *  2 : the relative change of the cost function is less than `tol`.
145:             *  3 : the unconstrained solution is optimal.
146: 
147:     message : str
148:         Verbal description of the termination reason.
149:     success : bool
150:         True if one of the convergence criteria is satisfied (`status` > 0).
151: 
152:     See Also
153:     --------
154:     nnls : Linear least squares with non-negativity constraint.
155:     least_squares : Nonlinear least squares with bounds on the variables.                    
156: 
157:     Notes
158:     -----
159:     The algorithm first computes the unconstrained least-squares solution by
160:     `numpy.linalg.lstsq` or `scipy.sparse.linalg.lsmr` depending on
161:     `lsq_solver`. This solution is returned as optimal if it lies within the
162:     bounds.
163: 
164:     Method 'trf' runs the adaptation of the algorithm described in [STIR]_ for
165:     a linear least-squares problem. The iterations are essentially the same as
166:     in the nonlinear least-squares algorithm, but as the quadratic function
167:     model is always accurate, we don't need to track or modify the radius of
168:     a trust region. The line search (backtracking) is used as a safety net
169:     when a selected step does not decrease the cost function. Read more
170:     detailed description of the algorithm in `scipy.optimize.least_squares`.
171: 
172:     Method 'bvls' runs a Python implementation of the algorithm described in
173:     [BVLS]_. The algorithm maintains active and free sets of variables, on
174:     each iteration chooses a new variable to move from the active set to the
175:     free set and then solves the unconstrained least-squares problem on free
176:     variables. This algorithm is guaranteed to give an accurate solution
177:     eventually, but may require up to n iterations for a problem with n
178:     variables. Additionally, an ad-hoc initialization procedure is
179:     implemented, that determines which variables to set free or active
180:     initially. It takes some number of iterations before actual BVLS starts,
181:     but can significantly reduce the number of further iterations.
182: 
183:     References
184:     ----------
185:     .. [STIR] M. A. Branch, T. F. Coleman, and Y. Li, "A Subspace, Interior,
186:               and Conjugate Gradient Method for Large-Scale Bound-Constrained
187:               Minimization Problems," SIAM Journal on Scientific Computing,
188:               Vol. 21, Number 1, pp 1-23, 1999.
189:     .. [BVLS] P. B. Start and R. L. Parker, "Bounded-Variable Least-Squares:
190:               an Algorithm and Applications", Computational Statistics, 10,
191:               129-141, 1995.
192: 
193:     Examples
194:     --------
195:     In this example a problem with a large sparse matrix and bounds on the
196:     variables is solved.
197: 
198:     >>> from scipy.sparse import rand
199:     >>> from scipy.optimize import lsq_linear
200:     ...
201:     >>> np.random.seed(0)
202:     ...
203:     >>> m = 20000
204:     >>> n = 10000
205:     ...
206:     >>> A = rand(m, n, density=1e-4)
207:     >>> b = np.random.randn(m)
208:     ...
209:     >>> lb = np.random.randn(n)
210:     >>> ub = lb + 1
211:     ...
212:     >>> res = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto', verbose=1)
213:     # may vary
214:     The relative change of the cost function is less than `tol`.
215:     Number of iterations 16, initial cost 1.5039e+04, final cost 1.1112e+04,
216:     first-order optimality 4.66e-08.
217:     '''
218:     if method not in ['trf', 'bvls']:
219:         raise ValueError("`method` must be 'trf' or 'bvls'")
220: 
221:     if lsq_solver not in [None, 'exact', 'lsmr']:
222:         raise ValueError("`solver` must be None, 'exact' or 'lsmr'.")
223: 
224:     if verbose not in [0, 1, 2]:
225:         raise ValueError("`verbose` must be in [0, 1, 2].")
226: 
227:     if issparse(A):
228:         A = csr_matrix(A)
229:     elif not isinstance(A, LinearOperator):
230:         A = np.atleast_2d(A)
231: 
232:     if method == 'bvls':
233:         if lsq_solver == 'lsmr':
234:             raise ValueError("method='bvls' can't be used with "
235:                              "lsq_solver='lsmr'")
236: 
237:         if not isinstance(A, np.ndarray):
238:             raise ValueError("method='bvls' can't be used with `A` being "
239:                              "sparse or LinearOperator.")
240: 
241:     if lsq_solver is None:
242:         if isinstance(A, np.ndarray):
243:             lsq_solver = 'exact'
244:         else:
245:             lsq_solver = 'lsmr'
246:     elif lsq_solver == 'exact' and not isinstance(A, np.ndarray):
247:         raise ValueError("`exact` solver can't be used when `A` is "
248:                          "sparse or LinearOperator.")
249: 
250:     if len(A.shape) != 2:  # No ndim for LinearOperator.
251:         raise ValueError("`A` must have at most 2 dimensions.")
252: 
253:     if len(bounds) != 2:
254:         raise ValueError("`bounds` must contain 2 elements.")
255: 
256:     if max_iter is not None and max_iter <= 0:
257:         raise ValueError("`max_iter` must be None or positive integer.")
258: 
259:     m, n = A.shape
260: 
261:     b = np.atleast_1d(b)
262:     if b.ndim != 1:
263:         raise ValueError("`b` must have at most 1 dimension.")
264: 
265:     if b.size != m:
266:         raise ValueError("Inconsistent shapes between `A` and `b`.")
267: 
268:     lb, ub = prepare_bounds(bounds, n)
269: 
270:     if lb.shape != (n,) and ub.shape != (n,):
271:         raise ValueError("Bounds have wrong shape.")
272: 
273:     if np.any(lb >= ub):
274:         raise ValueError("Each lower bound must be strictly less than each "
275:                          "upper bound.")
276: 
277:     if lsq_solver == 'exact':
278:         x_lsq = np.linalg.lstsq(A, b, rcond=-1)[0]
279:     elif lsq_solver == 'lsmr':
280:         x_lsq = lsmr(A, b, atol=tol, btol=tol)[0]
281: 
282:     if in_bounds(x_lsq, lb, ub):
283:         r = A.dot(x_lsq) - b
284:         cost = 0.5 * np.dot(r, r)
285:         termination_status = 3
286:         termination_message = TERMINATION_MESSAGES[termination_status]
287:         g = compute_grad(A, r)
288:         g_norm = norm(g, ord=np.inf)
289: 
290:         if verbose > 0:
291:             print(termination_message)
292:             print("Final cost {0:.4e}, first-order optimality {1:.2e}"
293:                   .format(cost, g_norm))
294: 
295:         return OptimizeResult(
296:             x=x_lsq, fun=r, cost=cost, optimality=g_norm,
297:             active_mask=np.zeros(n), nit=0, status=termination_status,
298:             message=termination_message, success=True)
299: 
300:     if method == 'trf':
301:         res = trf_linear(A, b, x_lsq, lb, ub, tol, lsq_solver, lsmr_tol,
302:                          max_iter, verbose)
303:     elif method == 'bvls':
304:         res = bvls(A, b, x_lsq, lb, ub, tol, max_iter, verbose)
305: 
306:     res.message = TERMINATION_MESSAGES[res.status]
307:     res.success = res.status > 0
308: 
309:     if verbose > 0:
310:         print(res.message)
311:         print("Number of iterations {0}, initial cost {1:.4e}, "
312:               "final cost {2:.4e}, first-order optimality {3:.2e}."
313:               .format(res.nit, res.initial_cost, res.cost, res.optimality))
314: 
315:     del res.initial_cost
316: 
317:     return res
318: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_252052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Linear least squares with bound constraints on independent variables.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_252053 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_252053) is not StypyTypeError):

    if (import_252053 != 'pyd_module'):
        __import__(import_252053)
        sys_modules_252054 = sys.modules[import_252053]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_252054.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_252053)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy.linalg import norm' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_252055 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.linalg')

if (type(import_252055) is not StypyTypeError):

    if (import_252055 != 'pyd_module'):
        __import__(import_252055)
        sys_modules_252056 = sys.modules[import_252055]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.linalg', sys_modules_252056.module_type_store, module_type_store, ['norm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_252056, sys_modules_252056.module_type_store, module_type_store)
    else:
        from numpy.linalg import norm

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.linalg', None, module_type_store, ['norm'], [norm])

else:
    # Assigning a type to the variable 'numpy.linalg' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.linalg', import_252055)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.sparse import issparse, csr_matrix' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_252057 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse')

if (type(import_252057) is not StypyTypeError):

    if (import_252057 != 'pyd_module'):
        __import__(import_252057)
        sys_modules_252058 = sys.modules[import_252057]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse', sys_modules_252058.module_type_store, module_type_store, ['issparse', 'csr_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_252058, sys_modules_252058.module_type_store, module_type_store)
    else:
        from scipy.sparse import issparse, csr_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse', None, module_type_store, ['issparse', 'csr_matrix'], [issparse, csr_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse', import_252057)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.sparse.linalg import LinearOperator, lsmr' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_252059 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse.linalg')

if (type(import_252059) is not StypyTypeError):

    if (import_252059 != 'pyd_module'):
        __import__(import_252059)
        sys_modules_252060 = sys.modules[import_252059]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse.linalg', sys_modules_252060.module_type_store, module_type_store, ['LinearOperator', 'lsmr'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_252060, sys_modules_252060.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import LinearOperator, lsmr

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse.linalg', None, module_type_store, ['LinearOperator', 'lsmr'], [LinearOperator, lsmr])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse.linalg', import_252059)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.optimize import OptimizeResult' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_252061 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize')

if (type(import_252061) is not StypyTypeError):

    if (import_252061 != 'pyd_module'):
        __import__(import_252061)
        sys_modules_252062 = sys.modules[import_252061]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize', sys_modules_252062.module_type_store, module_type_store, ['OptimizeResult'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_252062, sys_modules_252062.module_type_store, module_type_store)
    else:
        from scipy.optimize import OptimizeResult

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize', None, module_type_store, ['OptimizeResult'], [OptimizeResult])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize', import_252061)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.optimize._lsq.common import in_bounds, compute_grad' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_252063 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.optimize._lsq.common')

if (type(import_252063) is not StypyTypeError):

    if (import_252063 != 'pyd_module'):
        __import__(import_252063)
        sys_modules_252064 = sys.modules[import_252063]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.optimize._lsq.common', sys_modules_252064.module_type_store, module_type_store, ['in_bounds', 'compute_grad'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_252064, sys_modules_252064.module_type_store, module_type_store)
    else:
        from scipy.optimize._lsq.common import in_bounds, compute_grad

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.optimize._lsq.common', None, module_type_store, ['in_bounds', 'compute_grad'], [in_bounds, compute_grad])

else:
    # Assigning a type to the variable 'scipy.optimize._lsq.common' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.optimize._lsq.common', import_252063)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.optimize._lsq.trf_linear import trf_linear' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_252065 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize._lsq.trf_linear')

if (type(import_252065) is not StypyTypeError):

    if (import_252065 != 'pyd_module'):
        __import__(import_252065)
        sys_modules_252066 = sys.modules[import_252065]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize._lsq.trf_linear', sys_modules_252066.module_type_store, module_type_store, ['trf_linear'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_252066, sys_modules_252066.module_type_store, module_type_store)
    else:
        from scipy.optimize._lsq.trf_linear import trf_linear

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize._lsq.trf_linear', None, module_type_store, ['trf_linear'], [trf_linear])

else:
    # Assigning a type to the variable 'scipy.optimize._lsq.trf_linear' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize._lsq.trf_linear', import_252065)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.optimize._lsq.bvls import bvls' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_252067 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.optimize._lsq.bvls')

if (type(import_252067) is not StypyTypeError):

    if (import_252067 != 'pyd_module'):
        __import__(import_252067)
        sys_modules_252068 = sys.modules[import_252067]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.optimize._lsq.bvls', sys_modules_252068.module_type_store, module_type_store, ['bvls'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_252068, sys_modules_252068.module_type_store, module_type_store)
    else:
        from scipy.optimize._lsq.bvls import bvls

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.optimize._lsq.bvls', None, module_type_store, ['bvls'], [bvls])

else:
    # Assigning a type to the variable 'scipy.optimize._lsq.bvls' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.optimize._lsq.bvls', import_252067)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')


@norecursion
def prepare_bounds(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'prepare_bounds'
    module_type_store = module_type_store.open_function_context('prepare_bounds', 15, 0, False)
    
    # Passed parameters checking function
    prepare_bounds.stypy_localization = localization
    prepare_bounds.stypy_type_of_self = None
    prepare_bounds.stypy_type_store = module_type_store
    prepare_bounds.stypy_function_name = 'prepare_bounds'
    prepare_bounds.stypy_param_names_list = ['bounds', 'n']
    prepare_bounds.stypy_varargs_param_name = None
    prepare_bounds.stypy_kwargs_param_name = None
    prepare_bounds.stypy_call_defaults = defaults
    prepare_bounds.stypy_call_varargs = varargs
    prepare_bounds.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'prepare_bounds', ['bounds', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'prepare_bounds', localization, ['bounds', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'prepare_bounds(...)' code ##################

    
    # Assigning a ListComp to a Tuple (line 16):
    
    # Assigning a Subscript to a Name (line 16):
    
    # Obtaining the type of the subscript
    int_252069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'bounds' (line 16)
    bounds_252077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 50), 'bounds')
    comprehension_252078 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 14), bounds_252077)
    # Assigning a type to the variable 'b' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 14), 'b', comprehension_252078)
    
    # Call to asarray(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'b' (line 16)
    b_252072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 25), 'b', False)
    # Processing the call keyword arguments (line 16)
    # Getting the type of 'float' (line 16)
    float_252073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 34), 'float', False)
    keyword_252074 = float_252073
    kwargs_252075 = {'dtype': keyword_252074}
    # Getting the type of 'np' (line 16)
    np_252070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 14), 'np', False)
    # Obtaining the member 'asarray' of a type (line 16)
    asarray_252071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 14), np_252070, 'asarray')
    # Calling asarray(args, kwargs) (line 16)
    asarray_call_result_252076 = invoke(stypy.reporting.localization.Localization(__file__, 16, 14), asarray_252071, *[b_252072], **kwargs_252075)
    
    list_252079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 14), list_252079, asarray_call_result_252076)
    # Obtaining the member '__getitem__' of a type (line 16)
    getitem___252080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), list_252079, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 16)
    subscript_call_result_252081 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), getitem___252080, int_252069)
    
    # Assigning a type to the variable 'tuple_var_assignment_252046' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_252046', subscript_call_result_252081)
    
    # Assigning a Subscript to a Name (line 16):
    
    # Obtaining the type of the subscript
    int_252082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'bounds' (line 16)
    bounds_252090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 50), 'bounds')
    comprehension_252091 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 14), bounds_252090)
    # Assigning a type to the variable 'b' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 14), 'b', comprehension_252091)
    
    # Call to asarray(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'b' (line 16)
    b_252085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 25), 'b', False)
    # Processing the call keyword arguments (line 16)
    # Getting the type of 'float' (line 16)
    float_252086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 34), 'float', False)
    keyword_252087 = float_252086
    kwargs_252088 = {'dtype': keyword_252087}
    # Getting the type of 'np' (line 16)
    np_252083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 14), 'np', False)
    # Obtaining the member 'asarray' of a type (line 16)
    asarray_252084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 14), np_252083, 'asarray')
    # Calling asarray(args, kwargs) (line 16)
    asarray_call_result_252089 = invoke(stypy.reporting.localization.Localization(__file__, 16, 14), asarray_252084, *[b_252085], **kwargs_252088)
    
    list_252092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 14), list_252092, asarray_call_result_252089)
    # Obtaining the member '__getitem__' of a type (line 16)
    getitem___252093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), list_252092, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 16)
    subscript_call_result_252094 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), getitem___252093, int_252082)
    
    # Assigning a type to the variable 'tuple_var_assignment_252047' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_252047', subscript_call_result_252094)
    
    # Assigning a Name to a Name (line 16):
    # Getting the type of 'tuple_var_assignment_252046' (line 16)
    tuple_var_assignment_252046_252095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_252046')
    # Assigning a type to the variable 'lb' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'lb', tuple_var_assignment_252046_252095)
    
    # Assigning a Name to a Name (line 16):
    # Getting the type of 'tuple_var_assignment_252047' (line 16)
    tuple_var_assignment_252047_252096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_252047')
    # Assigning a type to the variable 'ub' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'ub', tuple_var_assignment_252047_252096)
    
    
    # Getting the type of 'lb' (line 18)
    lb_252097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 7), 'lb')
    # Obtaining the member 'ndim' of a type (line 18)
    ndim_252098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 7), lb_252097, 'ndim')
    int_252099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 18), 'int')
    # Applying the binary operator '==' (line 18)
    result_eq_252100 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 7), '==', ndim_252098, int_252099)
    
    # Testing the type of an if condition (line 18)
    if_condition_252101 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 18, 4), result_eq_252100)
    # Assigning a type to the variable 'if_condition_252101' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'if_condition_252101', if_condition_252101)
    # SSA begins for if statement (line 18)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 19):
    
    # Assigning a Call to a Name (line 19):
    
    # Call to resize(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'lb' (line 19)
    lb_252104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 23), 'lb', False)
    # Getting the type of 'n' (line 19)
    n_252105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 27), 'n', False)
    # Processing the call keyword arguments (line 19)
    kwargs_252106 = {}
    # Getting the type of 'np' (line 19)
    np_252102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 13), 'np', False)
    # Obtaining the member 'resize' of a type (line 19)
    resize_252103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 13), np_252102, 'resize')
    # Calling resize(args, kwargs) (line 19)
    resize_call_result_252107 = invoke(stypy.reporting.localization.Localization(__file__, 19, 13), resize_252103, *[lb_252104, n_252105], **kwargs_252106)
    
    # Assigning a type to the variable 'lb' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'lb', resize_call_result_252107)
    # SSA join for if statement (line 18)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'ub' (line 21)
    ub_252108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 7), 'ub')
    # Obtaining the member 'ndim' of a type (line 21)
    ndim_252109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 7), ub_252108, 'ndim')
    int_252110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 18), 'int')
    # Applying the binary operator '==' (line 21)
    result_eq_252111 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 7), '==', ndim_252109, int_252110)
    
    # Testing the type of an if condition (line 21)
    if_condition_252112 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 21, 4), result_eq_252111)
    # Assigning a type to the variable 'if_condition_252112' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'if_condition_252112', if_condition_252112)
    # SSA begins for if statement (line 21)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 22):
    
    # Assigning a Call to a Name (line 22):
    
    # Call to resize(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'ub' (line 22)
    ub_252115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 23), 'ub', False)
    # Getting the type of 'n' (line 22)
    n_252116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 27), 'n', False)
    # Processing the call keyword arguments (line 22)
    kwargs_252117 = {}
    # Getting the type of 'np' (line 22)
    np_252113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 13), 'np', False)
    # Obtaining the member 'resize' of a type (line 22)
    resize_252114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 13), np_252113, 'resize')
    # Calling resize(args, kwargs) (line 22)
    resize_call_result_252118 = invoke(stypy.reporting.localization.Localization(__file__, 22, 13), resize_252114, *[ub_252115, n_252116], **kwargs_252117)
    
    # Assigning a type to the variable 'ub' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'ub', resize_call_result_252118)
    # SSA join for if statement (line 21)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 24)
    tuple_252119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 24)
    # Adding element type (line 24)
    # Getting the type of 'lb' (line 24)
    lb_252120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 11), 'lb')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 11), tuple_252119, lb_252120)
    # Adding element type (line 24)
    # Getting the type of 'ub' (line 24)
    ub_252121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 15), 'ub')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 11), tuple_252119, ub_252121)
    
    # Assigning a type to the variable 'stypy_return_type' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type', tuple_252119)
    
    # ################# End of 'prepare_bounds(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'prepare_bounds' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_252122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_252122)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'prepare_bounds'
    return stypy_return_type_252122

# Assigning a type to the variable 'prepare_bounds' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'prepare_bounds', prepare_bounds)

# Assigning a Dict to a Name (line 27):

# Assigning a Dict to a Name (line 27):

# Obtaining an instance of the builtin type 'dict' (line 27)
dict_252123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 23), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 27)
# Adding element type (key, value) (line 27)
int_252124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 4), 'int')
str_252125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 8), 'str', 'The algorithm was not able to make progress on the last iteration.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 23), dict_252123, (int_252124, str_252125))
# Adding element type (key, value) (line 27)
int_252126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'int')
str_252127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 7), 'str', 'The maximum number of iterations is exceeded.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 23), dict_252123, (int_252126, str_252127))
# Adding element type (key, value) (line 27)
int_252128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 4), 'int')
str_252129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 7), 'str', 'The first-order optimality measure is less than `tol`.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 23), dict_252123, (int_252128, str_252129))
# Adding element type (key, value) (line 27)
int_252130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 4), 'int')
str_252131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 7), 'str', 'The relative change of the cost function is less than `tol`.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 23), dict_252123, (int_252130, str_252131))
# Adding element type (key, value) (line 27)
int_252132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 4), 'int')
str_252133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 7), 'str', 'The unconstrained solution is optimal.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 23), dict_252123, (int_252132, str_252133))

# Assigning a type to the variable 'TERMINATION_MESSAGES' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'TERMINATION_MESSAGES', dict_252123)

@norecursion
def lsq_linear(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 36)
    tuple_252134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 36)
    # Adding element type (line 36)
    
    # Getting the type of 'np' (line 36)
    np_252135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 30), 'np')
    # Obtaining the member 'inf' of a type (line 36)
    inf_252136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 30), np_252135, 'inf')
    # Applying the 'usub' unary operator (line 36)
    result___neg___252137 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 29), 'usub', inf_252136)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 29), tuple_252134, result___neg___252137)
    # Adding element type (line 36)
    # Getting the type of 'np' (line 36)
    np_252138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 38), 'np')
    # Obtaining the member 'inf' of a type (line 36)
    inf_252139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 38), np_252138, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 29), tuple_252134, inf_252139)
    
    str_252140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 54), 'str', 'trf')
    float_252141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 65), 'float')
    # Getting the type of 'None' (line 37)
    None_252142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 26), 'None')
    # Getting the type of 'None' (line 37)
    None_252143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 41), 'None')
    # Getting the type of 'None' (line 37)
    None_252144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 56), 'None')
    int_252145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 70), 'int')
    defaults = [tuple_252134, str_252140, float_252141, None_252142, None_252143, None_252144, int_252145]
    # Create a new context for function 'lsq_linear'
    module_type_store = module_type_store.open_function_context('lsq_linear', 36, 0, False)
    
    # Passed parameters checking function
    lsq_linear.stypy_localization = localization
    lsq_linear.stypy_type_of_self = None
    lsq_linear.stypy_type_store = module_type_store
    lsq_linear.stypy_function_name = 'lsq_linear'
    lsq_linear.stypy_param_names_list = ['A', 'b', 'bounds', 'method', 'tol', 'lsq_solver', 'lsmr_tol', 'max_iter', 'verbose']
    lsq_linear.stypy_varargs_param_name = None
    lsq_linear.stypy_kwargs_param_name = None
    lsq_linear.stypy_call_defaults = defaults
    lsq_linear.stypy_call_varargs = varargs
    lsq_linear.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lsq_linear', ['A', 'b', 'bounds', 'method', 'tol', 'lsq_solver', 'lsmr_tol', 'max_iter', 'verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lsq_linear', localization, ['A', 'b', 'bounds', 'method', 'tol', 'lsq_solver', 'lsmr_tol', 'max_iter', 'verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lsq_linear(...)' code ##################

    str_252146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, (-1)), 'str', 'Solve a linear least-squares problem with bounds on the variables.\n\n    Given a m-by-n design matrix A and a target vector b with m elements,\n    `lsq_linear` solves the following optimization problem::\n\n        minimize 0.5 * ||A x - b||**2\n        subject to lb <= x <= ub\n\n    This optimization problem is convex, hence a found minimum (if iterations\n    have converged) is guaranteed to be global.\n\n    Parameters\n    ----------\n    A : array_like, sparse matrix of LinearOperator, shape (m, n)\n        Design matrix. Can be `scipy.sparse.linalg.LinearOperator`.\n    b : array_like, shape (m,)\n        Target vector.\n    bounds : 2-tuple of array_like, optional\n        Lower and upper bounds on independent variables. Defaults to no bounds.\n        Each array must have shape (n,) or be a scalar, in the latter\n        case a bound will be the same for all variables. Use ``np.inf`` with\n        an appropriate sign to disable bounds on all or some variables.\n    method : \'trf\' or \'bvls\', optional\n        Method to perform minimization.\n\n            * \'trf\' : Trust Region Reflective algorithm adapted for a linear\n              least-squares problem. This is an interior-point-like method\n              and the required number of iterations is weakly correlated with\n              the number of variables.\n            * \'bvls\' : Bounded-Variable Least-Squares algorithm. This is\n              an active set method, which requires the number of iterations\n              comparable to the number of variables. Can\'t be used when `A` is\n              sparse or LinearOperator.\n\n        Default is \'trf\'.\n    tol : float, optional\n        Tolerance parameter. The algorithm terminates if a relative change\n        of the cost function is less than `tol` on the last iteration.\n        Additionally the first-order optimality measure is considered:\n\n            * ``method=\'trf\'`` terminates if the uniform norm of the gradient,\n              scaled to account for the presence of the bounds, is less than\n              `tol`.\n            * ``method=\'bvls\'`` terminates if Karush-Kuhn-Tucker conditions\n              are satisfied within `tol` tolerance.\n\n    lsq_solver : {None, \'exact\', \'lsmr\'}, optional\n        Method of solving unbounded least-squares problems throughout\n        iterations:\n\n            * \'exact\' : Use dense QR or SVD decomposition approach. Can\'t be\n              used when `A` is sparse or LinearOperator.\n            * \'lsmr\' : Use `scipy.sparse.linalg.lsmr` iterative procedure\n              which requires only matrix-vector product evaluations. Can\'t\n              be used with ``method=\'bvls\'``.\n\n        If None (default) the solver is chosen based on type of `A`.\n    lsmr_tol : None, float or \'auto\', optional\n        Tolerance parameters \'atol\' and \'btol\' for `scipy.sparse.linalg.lsmr`\n        If None (default), it is set to ``1e-2 * tol``. If \'auto\', the\n        tolerance will be adjusted based on the optimality of the current\n        iterate, which can speed up the optimization process, but is not always\n        reliable.\n    max_iter : None or int, optional\n        Maximum number of iterations before termination. If None (default), it\n        is set to 100 for ``method=\'trf\'`` or to the number of variables for\n        ``method=\'bvls\'`` (not counting iterations for \'bvls\' initialization).\n    verbose : {0, 1, 2}, optional\n        Level of algorithm\'s verbosity:\n\n            * 0 : work silently (default).\n            * 1 : display a termination report.\n            * 2 : display progress during iterations.\n\n    Returns\n    -------\n    OptimizeResult with the following fields defined:\n    x : ndarray, shape (n,)\n        Solution found.\n    cost : float\n        Value of the cost function at the solution.\n    fun : ndarray, shape (m,)\n        Vector of residuals at the solution.\n    optimality : float\n        First-order optimality measure. The exact meaning depends on `method`,\n        refer to the description of `tol` parameter.\n    active_mask : ndarray of int, shape (n,)\n        Each component shows whether a corresponding constraint is active\n        (that is, whether a variable is at the bound):\n\n            *  0 : a constraint is not active.\n            * -1 : a lower bound is active.\n            *  1 : an upper bound is active.\n\n        Might be somewhat arbitrary for the `trf` method as it generates a\n        sequence of strictly feasible iterates and active_mask is determined\n        within a tolerance threshold.\n    nit : int\n        Number of iterations. Zero if the unconstrained solution is optimal.\n    status : int\n        Reason for algorithm termination:\n\n            * -1 : the algorithm was not able to make progress on the last\n              iteration.\n            *  0 : the maximum number of iterations is exceeded.\n            *  1 : the first-order optimality measure is less than `tol`.\n            *  2 : the relative change of the cost function is less than `tol`.\n            *  3 : the unconstrained solution is optimal.\n\n    message : str\n        Verbal description of the termination reason.\n    success : bool\n        True if one of the convergence criteria is satisfied (`status` > 0).\n\n    See Also\n    --------\n    nnls : Linear least squares with non-negativity constraint.\n    least_squares : Nonlinear least squares with bounds on the variables.                    \n\n    Notes\n    -----\n    The algorithm first computes the unconstrained least-squares solution by\n    `numpy.linalg.lstsq` or `scipy.sparse.linalg.lsmr` depending on\n    `lsq_solver`. This solution is returned as optimal if it lies within the\n    bounds.\n\n    Method \'trf\' runs the adaptation of the algorithm described in [STIR]_ for\n    a linear least-squares problem. The iterations are essentially the same as\n    in the nonlinear least-squares algorithm, but as the quadratic function\n    model is always accurate, we don\'t need to track or modify the radius of\n    a trust region. The line search (backtracking) is used as a safety net\n    when a selected step does not decrease the cost function. Read more\n    detailed description of the algorithm in `scipy.optimize.least_squares`.\n\n    Method \'bvls\' runs a Python implementation of the algorithm described in\n    [BVLS]_. The algorithm maintains active and free sets of variables, on\n    each iteration chooses a new variable to move from the active set to the\n    free set and then solves the unconstrained least-squares problem on free\n    variables. This algorithm is guaranteed to give an accurate solution\n    eventually, but may require up to n iterations for a problem with n\n    variables. Additionally, an ad-hoc initialization procedure is\n    implemented, that determines which variables to set free or active\n    initially. It takes some number of iterations before actual BVLS starts,\n    but can significantly reduce the number of further iterations.\n\n    References\n    ----------\n    .. [STIR] M. A. Branch, T. F. Coleman, and Y. Li, "A Subspace, Interior,\n              and Conjugate Gradient Method for Large-Scale Bound-Constrained\n              Minimization Problems," SIAM Journal on Scientific Computing,\n              Vol. 21, Number 1, pp 1-23, 1999.\n    .. [BVLS] P. B. Start and R. L. Parker, "Bounded-Variable Least-Squares:\n              an Algorithm and Applications", Computational Statistics, 10,\n              129-141, 1995.\n\n    Examples\n    --------\n    In this example a problem with a large sparse matrix and bounds on the\n    variables is solved.\n\n    >>> from scipy.sparse import rand\n    >>> from scipy.optimize import lsq_linear\n    ...\n    >>> np.random.seed(0)\n    ...\n    >>> m = 20000\n    >>> n = 10000\n    ...\n    >>> A = rand(m, n, density=1e-4)\n    >>> b = np.random.randn(m)\n    ...\n    >>> lb = np.random.randn(n)\n    >>> ub = lb + 1\n    ...\n    >>> res = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol=\'auto\', verbose=1)\n    # may vary\n    The relative change of the cost function is less than `tol`.\n    Number of iterations 16, initial cost 1.5039e+04, final cost 1.1112e+04,\n    first-order optimality 4.66e-08.\n    ')
    
    
    # Getting the type of 'method' (line 218)
    method_252147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 7), 'method')
    
    # Obtaining an instance of the builtin type 'list' (line 218)
    list_252148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 218)
    # Adding element type (line 218)
    str_252149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 22), 'str', 'trf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 21), list_252148, str_252149)
    # Adding element type (line 218)
    str_252150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 29), 'str', 'bvls')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 21), list_252148, str_252150)
    
    # Applying the binary operator 'notin' (line 218)
    result_contains_252151 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 7), 'notin', method_252147, list_252148)
    
    # Testing the type of an if condition (line 218)
    if_condition_252152 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 218, 4), result_contains_252151)
    # Assigning a type to the variable 'if_condition_252152' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'if_condition_252152', if_condition_252152)
    # SSA begins for if statement (line 218)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 219)
    # Processing the call arguments (line 219)
    str_252154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 25), 'str', "`method` must be 'trf' or 'bvls'")
    # Processing the call keyword arguments (line 219)
    kwargs_252155 = {}
    # Getting the type of 'ValueError' (line 219)
    ValueError_252153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 219)
    ValueError_call_result_252156 = invoke(stypy.reporting.localization.Localization(__file__, 219, 14), ValueError_252153, *[str_252154], **kwargs_252155)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 219, 8), ValueError_call_result_252156, 'raise parameter', BaseException)
    # SSA join for if statement (line 218)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'lsq_solver' (line 221)
    lsq_solver_252157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 7), 'lsq_solver')
    
    # Obtaining an instance of the builtin type 'list' (line 221)
    list_252158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 221)
    # Adding element type (line 221)
    # Getting the type of 'None' (line 221)
    None_252159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 26), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 25), list_252158, None_252159)
    # Adding element type (line 221)
    str_252160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 32), 'str', 'exact')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 25), list_252158, str_252160)
    # Adding element type (line 221)
    str_252161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 41), 'str', 'lsmr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 25), list_252158, str_252161)
    
    # Applying the binary operator 'notin' (line 221)
    result_contains_252162 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 7), 'notin', lsq_solver_252157, list_252158)
    
    # Testing the type of an if condition (line 221)
    if_condition_252163 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 4), result_contains_252162)
    # Assigning a type to the variable 'if_condition_252163' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'if_condition_252163', if_condition_252163)
    # SSA begins for if statement (line 221)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 222)
    # Processing the call arguments (line 222)
    str_252165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 25), 'str', "`solver` must be None, 'exact' or 'lsmr'.")
    # Processing the call keyword arguments (line 222)
    kwargs_252166 = {}
    # Getting the type of 'ValueError' (line 222)
    ValueError_252164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 222)
    ValueError_call_result_252167 = invoke(stypy.reporting.localization.Localization(__file__, 222, 14), ValueError_252164, *[str_252165], **kwargs_252166)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 222, 8), ValueError_call_result_252167, 'raise parameter', BaseException)
    # SSA join for if statement (line 221)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'verbose' (line 224)
    verbose_252168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 7), 'verbose')
    
    # Obtaining an instance of the builtin type 'list' (line 224)
    list_252169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 224)
    # Adding element type (line 224)
    int_252170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 22), list_252169, int_252170)
    # Adding element type (line 224)
    int_252171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 22), list_252169, int_252171)
    # Adding element type (line 224)
    int_252172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 22), list_252169, int_252172)
    
    # Applying the binary operator 'notin' (line 224)
    result_contains_252173 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 7), 'notin', verbose_252168, list_252169)
    
    # Testing the type of an if condition (line 224)
    if_condition_252174 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 224, 4), result_contains_252173)
    # Assigning a type to the variable 'if_condition_252174' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'if_condition_252174', if_condition_252174)
    # SSA begins for if statement (line 224)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 225)
    # Processing the call arguments (line 225)
    str_252176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 25), 'str', '`verbose` must be in [0, 1, 2].')
    # Processing the call keyword arguments (line 225)
    kwargs_252177 = {}
    # Getting the type of 'ValueError' (line 225)
    ValueError_252175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 225)
    ValueError_call_result_252178 = invoke(stypy.reporting.localization.Localization(__file__, 225, 14), ValueError_252175, *[str_252176], **kwargs_252177)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 225, 8), ValueError_call_result_252178, 'raise parameter', BaseException)
    # SSA join for if statement (line 224)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to issparse(...): (line 227)
    # Processing the call arguments (line 227)
    # Getting the type of 'A' (line 227)
    A_252180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), 'A', False)
    # Processing the call keyword arguments (line 227)
    kwargs_252181 = {}
    # Getting the type of 'issparse' (line 227)
    issparse_252179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 7), 'issparse', False)
    # Calling issparse(args, kwargs) (line 227)
    issparse_call_result_252182 = invoke(stypy.reporting.localization.Localization(__file__, 227, 7), issparse_252179, *[A_252180], **kwargs_252181)
    
    # Testing the type of an if condition (line 227)
    if_condition_252183 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 4), issparse_call_result_252182)
    # Assigning a type to the variable 'if_condition_252183' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'if_condition_252183', if_condition_252183)
    # SSA begins for if statement (line 227)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 228):
    
    # Assigning a Call to a Name (line 228):
    
    # Call to csr_matrix(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'A' (line 228)
    A_252185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 23), 'A', False)
    # Processing the call keyword arguments (line 228)
    kwargs_252186 = {}
    # Getting the type of 'csr_matrix' (line 228)
    csr_matrix_252184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'csr_matrix', False)
    # Calling csr_matrix(args, kwargs) (line 228)
    csr_matrix_call_result_252187 = invoke(stypy.reporting.localization.Localization(__file__, 228, 12), csr_matrix_252184, *[A_252185], **kwargs_252186)
    
    # Assigning a type to the variable 'A' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'A', csr_matrix_call_result_252187)
    # SSA branch for the else part of an if statement (line 227)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to isinstance(...): (line 229)
    # Processing the call arguments (line 229)
    # Getting the type of 'A' (line 229)
    A_252189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 24), 'A', False)
    # Getting the type of 'LinearOperator' (line 229)
    LinearOperator_252190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 27), 'LinearOperator', False)
    # Processing the call keyword arguments (line 229)
    kwargs_252191 = {}
    # Getting the type of 'isinstance' (line 229)
    isinstance_252188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 13), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 229)
    isinstance_call_result_252192 = invoke(stypy.reporting.localization.Localization(__file__, 229, 13), isinstance_252188, *[A_252189, LinearOperator_252190], **kwargs_252191)
    
    # Applying the 'not' unary operator (line 229)
    result_not__252193 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 9), 'not', isinstance_call_result_252192)
    
    # Testing the type of an if condition (line 229)
    if_condition_252194 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 229, 9), result_not__252193)
    # Assigning a type to the variable 'if_condition_252194' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 9), 'if_condition_252194', if_condition_252194)
    # SSA begins for if statement (line 229)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 230):
    
    # Assigning a Call to a Name (line 230):
    
    # Call to atleast_2d(...): (line 230)
    # Processing the call arguments (line 230)
    # Getting the type of 'A' (line 230)
    A_252197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 26), 'A', False)
    # Processing the call keyword arguments (line 230)
    kwargs_252198 = {}
    # Getting the type of 'np' (line 230)
    np_252195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'np', False)
    # Obtaining the member 'atleast_2d' of a type (line 230)
    atleast_2d_252196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 12), np_252195, 'atleast_2d')
    # Calling atleast_2d(args, kwargs) (line 230)
    atleast_2d_call_result_252199 = invoke(stypy.reporting.localization.Localization(__file__, 230, 12), atleast_2d_252196, *[A_252197], **kwargs_252198)
    
    # Assigning a type to the variable 'A' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'A', atleast_2d_call_result_252199)
    # SSA join for if statement (line 229)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 227)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'method' (line 232)
    method_252200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 7), 'method')
    str_252201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 17), 'str', 'bvls')
    # Applying the binary operator '==' (line 232)
    result_eq_252202 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 7), '==', method_252200, str_252201)
    
    # Testing the type of an if condition (line 232)
    if_condition_252203 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 232, 4), result_eq_252202)
    # Assigning a type to the variable 'if_condition_252203' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'if_condition_252203', if_condition_252203)
    # SSA begins for if statement (line 232)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'lsq_solver' (line 233)
    lsq_solver_252204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 11), 'lsq_solver')
    str_252205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 25), 'str', 'lsmr')
    # Applying the binary operator '==' (line 233)
    result_eq_252206 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 11), '==', lsq_solver_252204, str_252205)
    
    # Testing the type of an if condition (line 233)
    if_condition_252207 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 233, 8), result_eq_252206)
    # Assigning a type to the variable 'if_condition_252207' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'if_condition_252207', if_condition_252207)
    # SSA begins for if statement (line 233)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 234)
    # Processing the call arguments (line 234)
    str_252209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 29), 'str', "method='bvls' can't be used with lsq_solver='lsmr'")
    # Processing the call keyword arguments (line 234)
    kwargs_252210 = {}
    # Getting the type of 'ValueError' (line 234)
    ValueError_252208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 234)
    ValueError_call_result_252211 = invoke(stypy.reporting.localization.Localization(__file__, 234, 18), ValueError_252208, *[str_252209], **kwargs_252210)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 234, 12), ValueError_call_result_252211, 'raise parameter', BaseException)
    # SSA join for if statement (line 233)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to isinstance(...): (line 237)
    # Processing the call arguments (line 237)
    # Getting the type of 'A' (line 237)
    A_252213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 26), 'A', False)
    # Getting the type of 'np' (line 237)
    np_252214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 29), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 237)
    ndarray_252215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 29), np_252214, 'ndarray')
    # Processing the call keyword arguments (line 237)
    kwargs_252216 = {}
    # Getting the type of 'isinstance' (line 237)
    isinstance_252212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 15), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 237)
    isinstance_call_result_252217 = invoke(stypy.reporting.localization.Localization(__file__, 237, 15), isinstance_252212, *[A_252213, ndarray_252215], **kwargs_252216)
    
    # Applying the 'not' unary operator (line 237)
    result_not__252218 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 11), 'not', isinstance_call_result_252217)
    
    # Testing the type of an if condition (line 237)
    if_condition_252219 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 237, 8), result_not__252218)
    # Assigning a type to the variable 'if_condition_252219' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'if_condition_252219', if_condition_252219)
    # SSA begins for if statement (line 237)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 238)
    # Processing the call arguments (line 238)
    str_252221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 29), 'str', "method='bvls' can't be used with `A` being sparse or LinearOperator.")
    # Processing the call keyword arguments (line 238)
    kwargs_252222 = {}
    # Getting the type of 'ValueError' (line 238)
    ValueError_252220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 238)
    ValueError_call_result_252223 = invoke(stypy.reporting.localization.Localization(__file__, 238, 18), ValueError_252220, *[str_252221], **kwargs_252222)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 238, 12), ValueError_call_result_252223, 'raise parameter', BaseException)
    # SSA join for if statement (line 237)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 232)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 241)
    # Getting the type of 'lsq_solver' (line 241)
    lsq_solver_252224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 7), 'lsq_solver')
    # Getting the type of 'None' (line 241)
    None_252225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 21), 'None')
    
    (may_be_252226, more_types_in_union_252227) = may_be_none(lsq_solver_252224, None_252225)

    if may_be_252226:

        if more_types_in_union_252227:
            # Runtime conditional SSA (line 241)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Call to isinstance(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'A' (line 242)
        A_252229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 22), 'A', False)
        # Getting the type of 'np' (line 242)
        np_252230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 25), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 242)
        ndarray_252231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 25), np_252230, 'ndarray')
        # Processing the call keyword arguments (line 242)
        kwargs_252232 = {}
        # Getting the type of 'isinstance' (line 242)
        isinstance_252228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 242)
        isinstance_call_result_252233 = invoke(stypy.reporting.localization.Localization(__file__, 242, 11), isinstance_252228, *[A_252229, ndarray_252231], **kwargs_252232)
        
        # Testing the type of an if condition (line 242)
        if_condition_252234 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 8), isinstance_call_result_252233)
        # Assigning a type to the variable 'if_condition_252234' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'if_condition_252234', if_condition_252234)
        # SSA begins for if statement (line 242)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 243):
        
        # Assigning a Str to a Name (line 243):
        str_252235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 25), 'str', 'exact')
        # Assigning a type to the variable 'lsq_solver' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'lsq_solver', str_252235)
        # SSA branch for the else part of an if statement (line 242)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 245):
        
        # Assigning a Str to a Name (line 245):
        str_252236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 25), 'str', 'lsmr')
        # Assigning a type to the variable 'lsq_solver' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'lsq_solver', str_252236)
        # SSA join for if statement (line 242)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_252227:
            # Runtime conditional SSA for else branch (line 241)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_252226) or more_types_in_union_252227):
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'lsq_solver' (line 246)
        lsq_solver_252237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 9), 'lsq_solver')
        str_252238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 23), 'str', 'exact')
        # Applying the binary operator '==' (line 246)
        result_eq_252239 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 9), '==', lsq_solver_252237, str_252238)
        
        
        
        # Call to isinstance(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'A' (line 246)
        A_252241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 50), 'A', False)
        # Getting the type of 'np' (line 246)
        np_252242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 53), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 246)
        ndarray_252243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 53), np_252242, 'ndarray')
        # Processing the call keyword arguments (line 246)
        kwargs_252244 = {}
        # Getting the type of 'isinstance' (line 246)
        isinstance_252240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 39), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 246)
        isinstance_call_result_252245 = invoke(stypy.reporting.localization.Localization(__file__, 246, 39), isinstance_252240, *[A_252241, ndarray_252243], **kwargs_252244)
        
        # Applying the 'not' unary operator (line 246)
        result_not__252246 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 35), 'not', isinstance_call_result_252245)
        
        # Applying the binary operator 'and' (line 246)
        result_and_keyword_252247 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 9), 'and', result_eq_252239, result_not__252246)
        
        # Testing the type of an if condition (line 246)
        if_condition_252248 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 9), result_and_keyword_252247)
        # Assigning a type to the variable 'if_condition_252248' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 9), 'if_condition_252248', if_condition_252248)
        # SSA begins for if statement (line 246)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 247)
        # Processing the call arguments (line 247)
        str_252250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 25), 'str', "`exact` solver can't be used when `A` is sparse or LinearOperator.")
        # Processing the call keyword arguments (line 247)
        kwargs_252251 = {}
        # Getting the type of 'ValueError' (line 247)
        ValueError_252249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 247)
        ValueError_call_result_252252 = invoke(stypy.reporting.localization.Localization(__file__, 247, 14), ValueError_252249, *[str_252250], **kwargs_252251)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 247, 8), ValueError_call_result_252252, 'raise parameter', BaseException)
        # SSA join for if statement (line 246)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_252226 and more_types_in_union_252227):
            # SSA join for if statement (line 241)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    
    # Call to len(...): (line 250)
    # Processing the call arguments (line 250)
    # Getting the type of 'A' (line 250)
    A_252254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 11), 'A', False)
    # Obtaining the member 'shape' of a type (line 250)
    shape_252255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 11), A_252254, 'shape')
    # Processing the call keyword arguments (line 250)
    kwargs_252256 = {}
    # Getting the type of 'len' (line 250)
    len_252253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 7), 'len', False)
    # Calling len(args, kwargs) (line 250)
    len_call_result_252257 = invoke(stypy.reporting.localization.Localization(__file__, 250, 7), len_252253, *[shape_252255], **kwargs_252256)
    
    int_252258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 23), 'int')
    # Applying the binary operator '!=' (line 250)
    result_ne_252259 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 7), '!=', len_call_result_252257, int_252258)
    
    # Testing the type of an if condition (line 250)
    if_condition_252260 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 250, 4), result_ne_252259)
    # Assigning a type to the variable 'if_condition_252260' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'if_condition_252260', if_condition_252260)
    # SSA begins for if statement (line 250)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 251)
    # Processing the call arguments (line 251)
    str_252262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 25), 'str', '`A` must have at most 2 dimensions.')
    # Processing the call keyword arguments (line 251)
    kwargs_252263 = {}
    # Getting the type of 'ValueError' (line 251)
    ValueError_252261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 251)
    ValueError_call_result_252264 = invoke(stypy.reporting.localization.Localization(__file__, 251, 14), ValueError_252261, *[str_252262], **kwargs_252263)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 251, 8), ValueError_call_result_252264, 'raise parameter', BaseException)
    # SSA join for if statement (line 250)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 253)
    # Processing the call arguments (line 253)
    # Getting the type of 'bounds' (line 253)
    bounds_252266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 11), 'bounds', False)
    # Processing the call keyword arguments (line 253)
    kwargs_252267 = {}
    # Getting the type of 'len' (line 253)
    len_252265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 7), 'len', False)
    # Calling len(args, kwargs) (line 253)
    len_call_result_252268 = invoke(stypy.reporting.localization.Localization(__file__, 253, 7), len_252265, *[bounds_252266], **kwargs_252267)
    
    int_252269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 22), 'int')
    # Applying the binary operator '!=' (line 253)
    result_ne_252270 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 7), '!=', len_call_result_252268, int_252269)
    
    # Testing the type of an if condition (line 253)
    if_condition_252271 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 253, 4), result_ne_252270)
    # Assigning a type to the variable 'if_condition_252271' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'if_condition_252271', if_condition_252271)
    # SSA begins for if statement (line 253)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 254)
    # Processing the call arguments (line 254)
    str_252273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 25), 'str', '`bounds` must contain 2 elements.')
    # Processing the call keyword arguments (line 254)
    kwargs_252274 = {}
    # Getting the type of 'ValueError' (line 254)
    ValueError_252272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 254)
    ValueError_call_result_252275 = invoke(stypy.reporting.localization.Localization(__file__, 254, 14), ValueError_252272, *[str_252273], **kwargs_252274)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 254, 8), ValueError_call_result_252275, 'raise parameter', BaseException)
    # SSA join for if statement (line 253)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'max_iter' (line 256)
    max_iter_252276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 7), 'max_iter')
    # Getting the type of 'None' (line 256)
    None_252277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 23), 'None')
    # Applying the binary operator 'isnot' (line 256)
    result_is_not_252278 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 7), 'isnot', max_iter_252276, None_252277)
    
    
    # Getting the type of 'max_iter' (line 256)
    max_iter_252279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 32), 'max_iter')
    int_252280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 44), 'int')
    # Applying the binary operator '<=' (line 256)
    result_le_252281 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 32), '<=', max_iter_252279, int_252280)
    
    # Applying the binary operator 'and' (line 256)
    result_and_keyword_252282 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 7), 'and', result_is_not_252278, result_le_252281)
    
    # Testing the type of an if condition (line 256)
    if_condition_252283 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 256, 4), result_and_keyword_252282)
    # Assigning a type to the variable 'if_condition_252283' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'if_condition_252283', if_condition_252283)
    # SSA begins for if statement (line 256)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 257)
    # Processing the call arguments (line 257)
    str_252285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 25), 'str', '`max_iter` must be None or positive integer.')
    # Processing the call keyword arguments (line 257)
    kwargs_252286 = {}
    # Getting the type of 'ValueError' (line 257)
    ValueError_252284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 257)
    ValueError_call_result_252287 = invoke(stypy.reporting.localization.Localization(__file__, 257, 14), ValueError_252284, *[str_252285], **kwargs_252286)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 257, 8), ValueError_call_result_252287, 'raise parameter', BaseException)
    # SSA join for if statement (line 256)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Tuple (line 259):
    
    # Assigning a Subscript to a Name (line 259):
    
    # Obtaining the type of the subscript
    int_252288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 4), 'int')
    # Getting the type of 'A' (line 259)
    A_252289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 11), 'A')
    # Obtaining the member 'shape' of a type (line 259)
    shape_252290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 11), A_252289, 'shape')
    # Obtaining the member '__getitem__' of a type (line 259)
    getitem___252291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 4), shape_252290, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 259)
    subscript_call_result_252292 = invoke(stypy.reporting.localization.Localization(__file__, 259, 4), getitem___252291, int_252288)
    
    # Assigning a type to the variable 'tuple_var_assignment_252048' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'tuple_var_assignment_252048', subscript_call_result_252292)
    
    # Assigning a Subscript to a Name (line 259):
    
    # Obtaining the type of the subscript
    int_252293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 4), 'int')
    # Getting the type of 'A' (line 259)
    A_252294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 11), 'A')
    # Obtaining the member 'shape' of a type (line 259)
    shape_252295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 11), A_252294, 'shape')
    # Obtaining the member '__getitem__' of a type (line 259)
    getitem___252296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 4), shape_252295, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 259)
    subscript_call_result_252297 = invoke(stypy.reporting.localization.Localization(__file__, 259, 4), getitem___252296, int_252293)
    
    # Assigning a type to the variable 'tuple_var_assignment_252049' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'tuple_var_assignment_252049', subscript_call_result_252297)
    
    # Assigning a Name to a Name (line 259):
    # Getting the type of 'tuple_var_assignment_252048' (line 259)
    tuple_var_assignment_252048_252298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'tuple_var_assignment_252048')
    # Assigning a type to the variable 'm' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'm', tuple_var_assignment_252048_252298)
    
    # Assigning a Name to a Name (line 259):
    # Getting the type of 'tuple_var_assignment_252049' (line 259)
    tuple_var_assignment_252049_252299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'tuple_var_assignment_252049')
    # Assigning a type to the variable 'n' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 7), 'n', tuple_var_assignment_252049_252299)
    
    # Assigning a Call to a Name (line 261):
    
    # Assigning a Call to a Name (line 261):
    
    # Call to atleast_1d(...): (line 261)
    # Processing the call arguments (line 261)
    # Getting the type of 'b' (line 261)
    b_252302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 22), 'b', False)
    # Processing the call keyword arguments (line 261)
    kwargs_252303 = {}
    # Getting the type of 'np' (line 261)
    np_252300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 261)
    atleast_1d_252301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), np_252300, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 261)
    atleast_1d_call_result_252304 = invoke(stypy.reporting.localization.Localization(__file__, 261, 8), atleast_1d_252301, *[b_252302], **kwargs_252303)
    
    # Assigning a type to the variable 'b' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'b', atleast_1d_call_result_252304)
    
    
    # Getting the type of 'b' (line 262)
    b_252305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 7), 'b')
    # Obtaining the member 'ndim' of a type (line 262)
    ndim_252306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 7), b_252305, 'ndim')
    int_252307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 17), 'int')
    # Applying the binary operator '!=' (line 262)
    result_ne_252308 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 7), '!=', ndim_252306, int_252307)
    
    # Testing the type of an if condition (line 262)
    if_condition_252309 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 262, 4), result_ne_252308)
    # Assigning a type to the variable 'if_condition_252309' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'if_condition_252309', if_condition_252309)
    # SSA begins for if statement (line 262)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 263)
    # Processing the call arguments (line 263)
    str_252311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 25), 'str', '`b` must have at most 1 dimension.')
    # Processing the call keyword arguments (line 263)
    kwargs_252312 = {}
    # Getting the type of 'ValueError' (line 263)
    ValueError_252310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 263)
    ValueError_call_result_252313 = invoke(stypy.reporting.localization.Localization(__file__, 263, 14), ValueError_252310, *[str_252311], **kwargs_252312)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 263, 8), ValueError_call_result_252313, 'raise parameter', BaseException)
    # SSA join for if statement (line 262)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'b' (line 265)
    b_252314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 7), 'b')
    # Obtaining the member 'size' of a type (line 265)
    size_252315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 7), b_252314, 'size')
    # Getting the type of 'm' (line 265)
    m_252316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 17), 'm')
    # Applying the binary operator '!=' (line 265)
    result_ne_252317 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 7), '!=', size_252315, m_252316)
    
    # Testing the type of an if condition (line 265)
    if_condition_252318 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 4), result_ne_252317)
    # Assigning a type to the variable 'if_condition_252318' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'if_condition_252318', if_condition_252318)
    # SSA begins for if statement (line 265)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 266)
    # Processing the call arguments (line 266)
    str_252320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 25), 'str', 'Inconsistent shapes between `A` and `b`.')
    # Processing the call keyword arguments (line 266)
    kwargs_252321 = {}
    # Getting the type of 'ValueError' (line 266)
    ValueError_252319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 266)
    ValueError_call_result_252322 = invoke(stypy.reporting.localization.Localization(__file__, 266, 14), ValueError_252319, *[str_252320], **kwargs_252321)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 266, 8), ValueError_call_result_252322, 'raise parameter', BaseException)
    # SSA join for if statement (line 265)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 268):
    
    # Assigning a Subscript to a Name (line 268):
    
    # Obtaining the type of the subscript
    int_252323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 4), 'int')
    
    # Call to prepare_bounds(...): (line 268)
    # Processing the call arguments (line 268)
    # Getting the type of 'bounds' (line 268)
    bounds_252325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 28), 'bounds', False)
    # Getting the type of 'n' (line 268)
    n_252326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 36), 'n', False)
    # Processing the call keyword arguments (line 268)
    kwargs_252327 = {}
    # Getting the type of 'prepare_bounds' (line 268)
    prepare_bounds_252324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 13), 'prepare_bounds', False)
    # Calling prepare_bounds(args, kwargs) (line 268)
    prepare_bounds_call_result_252328 = invoke(stypy.reporting.localization.Localization(__file__, 268, 13), prepare_bounds_252324, *[bounds_252325, n_252326], **kwargs_252327)
    
    # Obtaining the member '__getitem__' of a type (line 268)
    getitem___252329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 4), prepare_bounds_call_result_252328, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 268)
    subscript_call_result_252330 = invoke(stypy.reporting.localization.Localization(__file__, 268, 4), getitem___252329, int_252323)
    
    # Assigning a type to the variable 'tuple_var_assignment_252050' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'tuple_var_assignment_252050', subscript_call_result_252330)
    
    # Assigning a Subscript to a Name (line 268):
    
    # Obtaining the type of the subscript
    int_252331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 4), 'int')
    
    # Call to prepare_bounds(...): (line 268)
    # Processing the call arguments (line 268)
    # Getting the type of 'bounds' (line 268)
    bounds_252333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 28), 'bounds', False)
    # Getting the type of 'n' (line 268)
    n_252334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 36), 'n', False)
    # Processing the call keyword arguments (line 268)
    kwargs_252335 = {}
    # Getting the type of 'prepare_bounds' (line 268)
    prepare_bounds_252332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 13), 'prepare_bounds', False)
    # Calling prepare_bounds(args, kwargs) (line 268)
    prepare_bounds_call_result_252336 = invoke(stypy.reporting.localization.Localization(__file__, 268, 13), prepare_bounds_252332, *[bounds_252333, n_252334], **kwargs_252335)
    
    # Obtaining the member '__getitem__' of a type (line 268)
    getitem___252337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 4), prepare_bounds_call_result_252336, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 268)
    subscript_call_result_252338 = invoke(stypy.reporting.localization.Localization(__file__, 268, 4), getitem___252337, int_252331)
    
    # Assigning a type to the variable 'tuple_var_assignment_252051' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'tuple_var_assignment_252051', subscript_call_result_252338)
    
    # Assigning a Name to a Name (line 268):
    # Getting the type of 'tuple_var_assignment_252050' (line 268)
    tuple_var_assignment_252050_252339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'tuple_var_assignment_252050')
    # Assigning a type to the variable 'lb' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'lb', tuple_var_assignment_252050_252339)
    
    # Assigning a Name to a Name (line 268):
    # Getting the type of 'tuple_var_assignment_252051' (line 268)
    tuple_var_assignment_252051_252340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'tuple_var_assignment_252051')
    # Assigning a type to the variable 'ub' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'ub', tuple_var_assignment_252051_252340)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'lb' (line 270)
    lb_252341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 7), 'lb')
    # Obtaining the member 'shape' of a type (line 270)
    shape_252342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 7), lb_252341, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 270)
    tuple_252343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 270)
    # Adding element type (line 270)
    # Getting the type of 'n' (line 270)
    n_252344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 20), 'n')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 20), tuple_252343, n_252344)
    
    # Applying the binary operator '!=' (line 270)
    result_ne_252345 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 7), '!=', shape_252342, tuple_252343)
    
    
    # Getting the type of 'ub' (line 270)
    ub_252346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 28), 'ub')
    # Obtaining the member 'shape' of a type (line 270)
    shape_252347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 28), ub_252346, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 270)
    tuple_252348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 270)
    # Adding element type (line 270)
    # Getting the type of 'n' (line 270)
    n_252349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 41), 'n')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 41), tuple_252348, n_252349)
    
    # Applying the binary operator '!=' (line 270)
    result_ne_252350 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 28), '!=', shape_252347, tuple_252348)
    
    # Applying the binary operator 'and' (line 270)
    result_and_keyword_252351 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 7), 'and', result_ne_252345, result_ne_252350)
    
    # Testing the type of an if condition (line 270)
    if_condition_252352 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 270, 4), result_and_keyword_252351)
    # Assigning a type to the variable 'if_condition_252352' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'if_condition_252352', if_condition_252352)
    # SSA begins for if statement (line 270)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 271)
    # Processing the call arguments (line 271)
    str_252354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 25), 'str', 'Bounds have wrong shape.')
    # Processing the call keyword arguments (line 271)
    kwargs_252355 = {}
    # Getting the type of 'ValueError' (line 271)
    ValueError_252353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 271)
    ValueError_call_result_252356 = invoke(stypy.reporting.localization.Localization(__file__, 271, 14), ValueError_252353, *[str_252354], **kwargs_252355)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 271, 8), ValueError_call_result_252356, 'raise parameter', BaseException)
    # SSA join for if statement (line 270)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to any(...): (line 273)
    # Processing the call arguments (line 273)
    
    # Getting the type of 'lb' (line 273)
    lb_252359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 14), 'lb', False)
    # Getting the type of 'ub' (line 273)
    ub_252360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 20), 'ub', False)
    # Applying the binary operator '>=' (line 273)
    result_ge_252361 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 14), '>=', lb_252359, ub_252360)
    
    # Processing the call keyword arguments (line 273)
    kwargs_252362 = {}
    # Getting the type of 'np' (line 273)
    np_252357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 7), 'np', False)
    # Obtaining the member 'any' of a type (line 273)
    any_252358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 7), np_252357, 'any')
    # Calling any(args, kwargs) (line 273)
    any_call_result_252363 = invoke(stypy.reporting.localization.Localization(__file__, 273, 7), any_252358, *[result_ge_252361], **kwargs_252362)
    
    # Testing the type of an if condition (line 273)
    if_condition_252364 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 4), any_call_result_252363)
    # Assigning a type to the variable 'if_condition_252364' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'if_condition_252364', if_condition_252364)
    # SSA begins for if statement (line 273)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 274)
    # Processing the call arguments (line 274)
    str_252366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 25), 'str', 'Each lower bound must be strictly less than each upper bound.')
    # Processing the call keyword arguments (line 274)
    kwargs_252367 = {}
    # Getting the type of 'ValueError' (line 274)
    ValueError_252365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 274)
    ValueError_call_result_252368 = invoke(stypy.reporting.localization.Localization(__file__, 274, 14), ValueError_252365, *[str_252366], **kwargs_252367)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 274, 8), ValueError_call_result_252368, 'raise parameter', BaseException)
    # SSA join for if statement (line 273)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'lsq_solver' (line 277)
    lsq_solver_252369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 7), 'lsq_solver')
    str_252370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 21), 'str', 'exact')
    # Applying the binary operator '==' (line 277)
    result_eq_252371 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 7), '==', lsq_solver_252369, str_252370)
    
    # Testing the type of an if condition (line 277)
    if_condition_252372 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 277, 4), result_eq_252371)
    # Assigning a type to the variable 'if_condition_252372' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'if_condition_252372', if_condition_252372)
    # SSA begins for if statement (line 277)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 278):
    
    # Assigning a Subscript to a Name (line 278):
    
    # Obtaining the type of the subscript
    int_252373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 48), 'int')
    
    # Call to lstsq(...): (line 278)
    # Processing the call arguments (line 278)
    # Getting the type of 'A' (line 278)
    A_252377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 32), 'A', False)
    # Getting the type of 'b' (line 278)
    b_252378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 35), 'b', False)
    # Processing the call keyword arguments (line 278)
    int_252379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 44), 'int')
    keyword_252380 = int_252379
    kwargs_252381 = {'rcond': keyword_252380}
    # Getting the type of 'np' (line 278)
    np_252374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 16), 'np', False)
    # Obtaining the member 'linalg' of a type (line 278)
    linalg_252375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 16), np_252374, 'linalg')
    # Obtaining the member 'lstsq' of a type (line 278)
    lstsq_252376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 16), linalg_252375, 'lstsq')
    # Calling lstsq(args, kwargs) (line 278)
    lstsq_call_result_252382 = invoke(stypy.reporting.localization.Localization(__file__, 278, 16), lstsq_252376, *[A_252377, b_252378], **kwargs_252381)
    
    # Obtaining the member '__getitem__' of a type (line 278)
    getitem___252383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 16), lstsq_call_result_252382, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 278)
    subscript_call_result_252384 = invoke(stypy.reporting.localization.Localization(__file__, 278, 16), getitem___252383, int_252373)
    
    # Assigning a type to the variable 'x_lsq' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'x_lsq', subscript_call_result_252384)
    # SSA branch for the else part of an if statement (line 277)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'lsq_solver' (line 279)
    lsq_solver_252385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 9), 'lsq_solver')
    str_252386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 23), 'str', 'lsmr')
    # Applying the binary operator '==' (line 279)
    result_eq_252387 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 9), '==', lsq_solver_252385, str_252386)
    
    # Testing the type of an if condition (line 279)
    if_condition_252388 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 9), result_eq_252387)
    # Assigning a type to the variable 'if_condition_252388' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 9), 'if_condition_252388', if_condition_252388)
    # SSA begins for if statement (line 279)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 280):
    
    # Assigning a Subscript to a Name (line 280):
    
    # Obtaining the type of the subscript
    int_252389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 47), 'int')
    
    # Call to lsmr(...): (line 280)
    # Processing the call arguments (line 280)
    # Getting the type of 'A' (line 280)
    A_252391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 21), 'A', False)
    # Getting the type of 'b' (line 280)
    b_252392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 24), 'b', False)
    # Processing the call keyword arguments (line 280)
    # Getting the type of 'tol' (line 280)
    tol_252393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 32), 'tol', False)
    keyword_252394 = tol_252393
    # Getting the type of 'tol' (line 280)
    tol_252395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 42), 'tol', False)
    keyword_252396 = tol_252395
    kwargs_252397 = {'btol': keyword_252396, 'atol': keyword_252394}
    # Getting the type of 'lsmr' (line 280)
    lsmr_252390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 16), 'lsmr', False)
    # Calling lsmr(args, kwargs) (line 280)
    lsmr_call_result_252398 = invoke(stypy.reporting.localization.Localization(__file__, 280, 16), lsmr_252390, *[A_252391, b_252392], **kwargs_252397)
    
    # Obtaining the member '__getitem__' of a type (line 280)
    getitem___252399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 16), lsmr_call_result_252398, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 280)
    subscript_call_result_252400 = invoke(stypy.reporting.localization.Localization(__file__, 280, 16), getitem___252399, int_252389)
    
    # Assigning a type to the variable 'x_lsq' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'x_lsq', subscript_call_result_252400)
    # SSA join for if statement (line 279)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 277)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to in_bounds(...): (line 282)
    # Processing the call arguments (line 282)
    # Getting the type of 'x_lsq' (line 282)
    x_lsq_252402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 17), 'x_lsq', False)
    # Getting the type of 'lb' (line 282)
    lb_252403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 24), 'lb', False)
    # Getting the type of 'ub' (line 282)
    ub_252404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 28), 'ub', False)
    # Processing the call keyword arguments (line 282)
    kwargs_252405 = {}
    # Getting the type of 'in_bounds' (line 282)
    in_bounds_252401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 7), 'in_bounds', False)
    # Calling in_bounds(args, kwargs) (line 282)
    in_bounds_call_result_252406 = invoke(stypy.reporting.localization.Localization(__file__, 282, 7), in_bounds_252401, *[x_lsq_252402, lb_252403, ub_252404], **kwargs_252405)
    
    # Testing the type of an if condition (line 282)
    if_condition_252407 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 282, 4), in_bounds_call_result_252406)
    # Assigning a type to the variable 'if_condition_252407' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'if_condition_252407', if_condition_252407)
    # SSA begins for if statement (line 282)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 283):
    
    # Assigning a BinOp to a Name (line 283):
    
    # Call to dot(...): (line 283)
    # Processing the call arguments (line 283)
    # Getting the type of 'x_lsq' (line 283)
    x_lsq_252410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 18), 'x_lsq', False)
    # Processing the call keyword arguments (line 283)
    kwargs_252411 = {}
    # Getting the type of 'A' (line 283)
    A_252408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'A', False)
    # Obtaining the member 'dot' of a type (line 283)
    dot_252409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 12), A_252408, 'dot')
    # Calling dot(args, kwargs) (line 283)
    dot_call_result_252412 = invoke(stypy.reporting.localization.Localization(__file__, 283, 12), dot_252409, *[x_lsq_252410], **kwargs_252411)
    
    # Getting the type of 'b' (line 283)
    b_252413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 27), 'b')
    # Applying the binary operator '-' (line 283)
    result_sub_252414 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 12), '-', dot_call_result_252412, b_252413)
    
    # Assigning a type to the variable 'r' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'r', result_sub_252414)
    
    # Assigning a BinOp to a Name (line 284):
    
    # Assigning a BinOp to a Name (line 284):
    float_252415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 15), 'float')
    
    # Call to dot(...): (line 284)
    # Processing the call arguments (line 284)
    # Getting the type of 'r' (line 284)
    r_252418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 28), 'r', False)
    # Getting the type of 'r' (line 284)
    r_252419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 31), 'r', False)
    # Processing the call keyword arguments (line 284)
    kwargs_252420 = {}
    # Getting the type of 'np' (line 284)
    np_252416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 21), 'np', False)
    # Obtaining the member 'dot' of a type (line 284)
    dot_252417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 21), np_252416, 'dot')
    # Calling dot(args, kwargs) (line 284)
    dot_call_result_252421 = invoke(stypy.reporting.localization.Localization(__file__, 284, 21), dot_252417, *[r_252418, r_252419], **kwargs_252420)
    
    # Applying the binary operator '*' (line 284)
    result_mul_252422 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 15), '*', float_252415, dot_call_result_252421)
    
    # Assigning a type to the variable 'cost' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'cost', result_mul_252422)
    
    # Assigning a Num to a Name (line 285):
    
    # Assigning a Num to a Name (line 285):
    int_252423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 29), 'int')
    # Assigning a type to the variable 'termination_status' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'termination_status', int_252423)
    
    # Assigning a Subscript to a Name (line 286):
    
    # Assigning a Subscript to a Name (line 286):
    
    # Obtaining the type of the subscript
    # Getting the type of 'termination_status' (line 286)
    termination_status_252424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 51), 'termination_status')
    # Getting the type of 'TERMINATION_MESSAGES' (line 286)
    TERMINATION_MESSAGES_252425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 30), 'TERMINATION_MESSAGES')
    # Obtaining the member '__getitem__' of a type (line 286)
    getitem___252426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 30), TERMINATION_MESSAGES_252425, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 286)
    subscript_call_result_252427 = invoke(stypy.reporting.localization.Localization(__file__, 286, 30), getitem___252426, termination_status_252424)
    
    # Assigning a type to the variable 'termination_message' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'termination_message', subscript_call_result_252427)
    
    # Assigning a Call to a Name (line 287):
    
    # Assigning a Call to a Name (line 287):
    
    # Call to compute_grad(...): (line 287)
    # Processing the call arguments (line 287)
    # Getting the type of 'A' (line 287)
    A_252429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 25), 'A', False)
    # Getting the type of 'r' (line 287)
    r_252430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 28), 'r', False)
    # Processing the call keyword arguments (line 287)
    kwargs_252431 = {}
    # Getting the type of 'compute_grad' (line 287)
    compute_grad_252428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'compute_grad', False)
    # Calling compute_grad(args, kwargs) (line 287)
    compute_grad_call_result_252432 = invoke(stypy.reporting.localization.Localization(__file__, 287, 12), compute_grad_252428, *[A_252429, r_252430], **kwargs_252431)
    
    # Assigning a type to the variable 'g' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'g', compute_grad_call_result_252432)
    
    # Assigning a Call to a Name (line 288):
    
    # Assigning a Call to a Name (line 288):
    
    # Call to norm(...): (line 288)
    # Processing the call arguments (line 288)
    # Getting the type of 'g' (line 288)
    g_252434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 22), 'g', False)
    # Processing the call keyword arguments (line 288)
    # Getting the type of 'np' (line 288)
    np_252435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 29), 'np', False)
    # Obtaining the member 'inf' of a type (line 288)
    inf_252436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 29), np_252435, 'inf')
    keyword_252437 = inf_252436
    kwargs_252438 = {'ord': keyword_252437}
    # Getting the type of 'norm' (line 288)
    norm_252433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 17), 'norm', False)
    # Calling norm(args, kwargs) (line 288)
    norm_call_result_252439 = invoke(stypy.reporting.localization.Localization(__file__, 288, 17), norm_252433, *[g_252434], **kwargs_252438)
    
    # Assigning a type to the variable 'g_norm' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'g_norm', norm_call_result_252439)
    
    
    # Getting the type of 'verbose' (line 290)
    verbose_252440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 11), 'verbose')
    int_252441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 21), 'int')
    # Applying the binary operator '>' (line 290)
    result_gt_252442 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 11), '>', verbose_252440, int_252441)
    
    # Testing the type of an if condition (line 290)
    if_condition_252443 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 290, 8), result_gt_252442)
    # Assigning a type to the variable 'if_condition_252443' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'if_condition_252443', if_condition_252443)
    # SSA begins for if statement (line 290)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 291)
    # Processing the call arguments (line 291)
    # Getting the type of 'termination_message' (line 291)
    termination_message_252445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 18), 'termination_message', False)
    # Processing the call keyword arguments (line 291)
    kwargs_252446 = {}
    # Getting the type of 'print' (line 291)
    print_252444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'print', False)
    # Calling print(args, kwargs) (line 291)
    print_call_result_252447 = invoke(stypy.reporting.localization.Localization(__file__, 291, 12), print_252444, *[termination_message_252445], **kwargs_252446)
    
    
    # Call to print(...): (line 292)
    # Processing the call arguments (line 292)
    
    # Call to format(...): (line 292)
    # Processing the call arguments (line 292)
    # Getting the type of 'cost' (line 293)
    cost_252451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 26), 'cost', False)
    # Getting the type of 'g_norm' (line 293)
    g_norm_252452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 32), 'g_norm', False)
    # Processing the call keyword arguments (line 292)
    kwargs_252453 = {}
    str_252449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 18), 'str', 'Final cost {0:.4e}, first-order optimality {1:.2e}')
    # Obtaining the member 'format' of a type (line 292)
    format_252450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 18), str_252449, 'format')
    # Calling format(args, kwargs) (line 292)
    format_call_result_252454 = invoke(stypy.reporting.localization.Localization(__file__, 292, 18), format_252450, *[cost_252451, g_norm_252452], **kwargs_252453)
    
    # Processing the call keyword arguments (line 292)
    kwargs_252455 = {}
    # Getting the type of 'print' (line 292)
    print_252448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'print', False)
    # Calling print(args, kwargs) (line 292)
    print_call_result_252456 = invoke(stypy.reporting.localization.Localization(__file__, 292, 12), print_252448, *[format_call_result_252454], **kwargs_252455)
    
    # SSA join for if statement (line 290)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to OptimizeResult(...): (line 295)
    # Processing the call keyword arguments (line 295)
    # Getting the type of 'x_lsq' (line 296)
    x_lsq_252458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 14), 'x_lsq', False)
    keyword_252459 = x_lsq_252458
    # Getting the type of 'r' (line 296)
    r_252460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 25), 'r', False)
    keyword_252461 = r_252460
    # Getting the type of 'cost' (line 296)
    cost_252462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 33), 'cost', False)
    keyword_252463 = cost_252462
    # Getting the type of 'g_norm' (line 296)
    g_norm_252464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 50), 'g_norm', False)
    keyword_252465 = g_norm_252464
    
    # Call to zeros(...): (line 297)
    # Processing the call arguments (line 297)
    # Getting the type of 'n' (line 297)
    n_252468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 33), 'n', False)
    # Processing the call keyword arguments (line 297)
    kwargs_252469 = {}
    # Getting the type of 'np' (line 297)
    np_252466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 24), 'np', False)
    # Obtaining the member 'zeros' of a type (line 297)
    zeros_252467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 24), np_252466, 'zeros')
    # Calling zeros(args, kwargs) (line 297)
    zeros_call_result_252470 = invoke(stypy.reporting.localization.Localization(__file__, 297, 24), zeros_252467, *[n_252468], **kwargs_252469)
    
    keyword_252471 = zeros_call_result_252470
    int_252472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 41), 'int')
    keyword_252473 = int_252472
    # Getting the type of 'termination_status' (line 297)
    termination_status_252474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 51), 'termination_status', False)
    keyword_252475 = termination_status_252474
    # Getting the type of 'termination_message' (line 298)
    termination_message_252476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 20), 'termination_message', False)
    keyword_252477 = termination_message_252476
    # Getting the type of 'True' (line 298)
    True_252478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 49), 'True', False)
    keyword_252479 = True_252478
    kwargs_252480 = {'status': keyword_252475, 'success': keyword_252479, 'active_mask': keyword_252471, 'cost': keyword_252463, 'optimality': keyword_252465, 'fun': keyword_252461, 'x': keyword_252459, 'message': keyword_252477, 'nit': keyword_252473}
    # Getting the type of 'OptimizeResult' (line 295)
    OptimizeResult_252457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 15), 'OptimizeResult', False)
    # Calling OptimizeResult(args, kwargs) (line 295)
    OptimizeResult_call_result_252481 = invoke(stypy.reporting.localization.Localization(__file__, 295, 15), OptimizeResult_252457, *[], **kwargs_252480)
    
    # Assigning a type to the variable 'stypy_return_type' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'stypy_return_type', OptimizeResult_call_result_252481)
    # SSA join for if statement (line 282)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'method' (line 300)
    method_252482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 7), 'method')
    str_252483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 17), 'str', 'trf')
    # Applying the binary operator '==' (line 300)
    result_eq_252484 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 7), '==', method_252482, str_252483)
    
    # Testing the type of an if condition (line 300)
    if_condition_252485 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 300, 4), result_eq_252484)
    # Assigning a type to the variable 'if_condition_252485' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'if_condition_252485', if_condition_252485)
    # SSA begins for if statement (line 300)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 301):
    
    # Assigning a Call to a Name (line 301):
    
    # Call to trf_linear(...): (line 301)
    # Processing the call arguments (line 301)
    # Getting the type of 'A' (line 301)
    A_252487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 25), 'A', False)
    # Getting the type of 'b' (line 301)
    b_252488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 28), 'b', False)
    # Getting the type of 'x_lsq' (line 301)
    x_lsq_252489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 31), 'x_lsq', False)
    # Getting the type of 'lb' (line 301)
    lb_252490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 38), 'lb', False)
    # Getting the type of 'ub' (line 301)
    ub_252491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 42), 'ub', False)
    # Getting the type of 'tol' (line 301)
    tol_252492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 46), 'tol', False)
    # Getting the type of 'lsq_solver' (line 301)
    lsq_solver_252493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 51), 'lsq_solver', False)
    # Getting the type of 'lsmr_tol' (line 301)
    lsmr_tol_252494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 63), 'lsmr_tol', False)
    # Getting the type of 'max_iter' (line 302)
    max_iter_252495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 25), 'max_iter', False)
    # Getting the type of 'verbose' (line 302)
    verbose_252496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 35), 'verbose', False)
    # Processing the call keyword arguments (line 301)
    kwargs_252497 = {}
    # Getting the type of 'trf_linear' (line 301)
    trf_linear_252486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 14), 'trf_linear', False)
    # Calling trf_linear(args, kwargs) (line 301)
    trf_linear_call_result_252498 = invoke(stypy.reporting.localization.Localization(__file__, 301, 14), trf_linear_252486, *[A_252487, b_252488, x_lsq_252489, lb_252490, ub_252491, tol_252492, lsq_solver_252493, lsmr_tol_252494, max_iter_252495, verbose_252496], **kwargs_252497)
    
    # Assigning a type to the variable 'res' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'res', trf_linear_call_result_252498)
    # SSA branch for the else part of an if statement (line 300)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'method' (line 303)
    method_252499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 9), 'method')
    str_252500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 19), 'str', 'bvls')
    # Applying the binary operator '==' (line 303)
    result_eq_252501 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 9), '==', method_252499, str_252500)
    
    # Testing the type of an if condition (line 303)
    if_condition_252502 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 303, 9), result_eq_252501)
    # Assigning a type to the variable 'if_condition_252502' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 9), 'if_condition_252502', if_condition_252502)
    # SSA begins for if statement (line 303)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 304):
    
    # Assigning a Call to a Name (line 304):
    
    # Call to bvls(...): (line 304)
    # Processing the call arguments (line 304)
    # Getting the type of 'A' (line 304)
    A_252504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 19), 'A', False)
    # Getting the type of 'b' (line 304)
    b_252505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 22), 'b', False)
    # Getting the type of 'x_lsq' (line 304)
    x_lsq_252506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 25), 'x_lsq', False)
    # Getting the type of 'lb' (line 304)
    lb_252507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 32), 'lb', False)
    # Getting the type of 'ub' (line 304)
    ub_252508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 36), 'ub', False)
    # Getting the type of 'tol' (line 304)
    tol_252509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 40), 'tol', False)
    # Getting the type of 'max_iter' (line 304)
    max_iter_252510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 45), 'max_iter', False)
    # Getting the type of 'verbose' (line 304)
    verbose_252511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 55), 'verbose', False)
    # Processing the call keyword arguments (line 304)
    kwargs_252512 = {}
    # Getting the type of 'bvls' (line 304)
    bvls_252503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 14), 'bvls', False)
    # Calling bvls(args, kwargs) (line 304)
    bvls_call_result_252513 = invoke(stypy.reporting.localization.Localization(__file__, 304, 14), bvls_252503, *[A_252504, b_252505, x_lsq_252506, lb_252507, ub_252508, tol_252509, max_iter_252510, verbose_252511], **kwargs_252512)
    
    # Assigning a type to the variable 'res' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'res', bvls_call_result_252513)
    # SSA join for if statement (line 303)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 300)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Attribute (line 306):
    
    # Assigning a Subscript to a Attribute (line 306):
    
    # Obtaining the type of the subscript
    # Getting the type of 'res' (line 306)
    res_252514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 39), 'res')
    # Obtaining the member 'status' of a type (line 306)
    status_252515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 39), res_252514, 'status')
    # Getting the type of 'TERMINATION_MESSAGES' (line 306)
    TERMINATION_MESSAGES_252516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 18), 'TERMINATION_MESSAGES')
    # Obtaining the member '__getitem__' of a type (line 306)
    getitem___252517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 18), TERMINATION_MESSAGES_252516, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 306)
    subscript_call_result_252518 = invoke(stypy.reporting.localization.Localization(__file__, 306, 18), getitem___252517, status_252515)
    
    # Getting the type of 'res' (line 306)
    res_252519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'res')
    # Setting the type of the member 'message' of a type (line 306)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 4), res_252519, 'message', subscript_call_result_252518)
    
    # Assigning a Compare to a Attribute (line 307):
    
    # Assigning a Compare to a Attribute (line 307):
    
    # Getting the type of 'res' (line 307)
    res_252520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 18), 'res')
    # Obtaining the member 'status' of a type (line 307)
    status_252521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 18), res_252520, 'status')
    int_252522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 31), 'int')
    # Applying the binary operator '>' (line 307)
    result_gt_252523 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 18), '>', status_252521, int_252522)
    
    # Getting the type of 'res' (line 307)
    res_252524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'res')
    # Setting the type of the member 'success' of a type (line 307)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 4), res_252524, 'success', result_gt_252523)
    
    
    # Getting the type of 'verbose' (line 309)
    verbose_252525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 7), 'verbose')
    int_252526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 17), 'int')
    # Applying the binary operator '>' (line 309)
    result_gt_252527 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 7), '>', verbose_252525, int_252526)
    
    # Testing the type of an if condition (line 309)
    if_condition_252528 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 309, 4), result_gt_252527)
    # Assigning a type to the variable 'if_condition_252528' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'if_condition_252528', if_condition_252528)
    # SSA begins for if statement (line 309)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 310)
    # Processing the call arguments (line 310)
    # Getting the type of 'res' (line 310)
    res_252530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 14), 'res', False)
    # Obtaining the member 'message' of a type (line 310)
    message_252531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 14), res_252530, 'message')
    # Processing the call keyword arguments (line 310)
    kwargs_252532 = {}
    # Getting the type of 'print' (line 310)
    print_252529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'print', False)
    # Calling print(args, kwargs) (line 310)
    print_call_result_252533 = invoke(stypy.reporting.localization.Localization(__file__, 310, 8), print_252529, *[message_252531], **kwargs_252532)
    
    
    # Call to print(...): (line 311)
    # Processing the call arguments (line 311)
    
    # Call to format(...): (line 311)
    # Processing the call arguments (line 311)
    # Getting the type of 'res' (line 313)
    res_252537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 22), 'res', False)
    # Obtaining the member 'nit' of a type (line 313)
    nit_252538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 22), res_252537, 'nit')
    # Getting the type of 'res' (line 313)
    res_252539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 31), 'res', False)
    # Obtaining the member 'initial_cost' of a type (line 313)
    initial_cost_252540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 31), res_252539, 'initial_cost')
    # Getting the type of 'res' (line 313)
    res_252541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 49), 'res', False)
    # Obtaining the member 'cost' of a type (line 313)
    cost_252542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 49), res_252541, 'cost')
    # Getting the type of 'res' (line 313)
    res_252543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 59), 'res', False)
    # Obtaining the member 'optimality' of a type (line 313)
    optimality_252544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 59), res_252543, 'optimality')
    # Processing the call keyword arguments (line 311)
    kwargs_252545 = {}
    str_252535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 14), 'str', 'Number of iterations {0}, initial cost {1:.4e}, final cost {2:.4e}, first-order optimality {3:.2e}.')
    # Obtaining the member 'format' of a type (line 311)
    format_252536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 14), str_252535, 'format')
    # Calling format(args, kwargs) (line 311)
    format_call_result_252546 = invoke(stypy.reporting.localization.Localization(__file__, 311, 14), format_252536, *[nit_252538, initial_cost_252540, cost_252542, optimality_252544], **kwargs_252545)
    
    # Processing the call keyword arguments (line 311)
    kwargs_252547 = {}
    # Getting the type of 'print' (line 311)
    print_252534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'print', False)
    # Calling print(args, kwargs) (line 311)
    print_call_result_252548 = invoke(stypy.reporting.localization.Localization(__file__, 311, 8), print_252534, *[format_call_result_252546], **kwargs_252547)
    
    # SSA join for if statement (line 309)
    module_type_store = module_type_store.join_ssa_context()
    
    # Deleting a member
    # Getting the type of 'res' (line 315)
    res_252549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'res')
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 315, 4), res_252549, 'initial_cost')
    # Getting the type of 'res' (line 317)
    res_252550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'stypy_return_type', res_252550)
    
    # ################# End of 'lsq_linear(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lsq_linear' in the type store
    # Getting the type of 'stypy_return_type' (line 36)
    stypy_return_type_252551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_252551)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lsq_linear'
    return stypy_return_type_252551

# Assigning a type to the variable 'lsq_linear' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'lsq_linear', lsq_linear)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
