
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Nearly exact trust-region optimization subproblem.'''
2: from __future__ import division, print_function, absolute_import
3: 
4: import numpy as np
5: from scipy.linalg import (norm, get_lapack_funcs, solve_triangular,
6:                           cho_solve)
7: from ._trustregion import (_minimize_trust_region, BaseQuadraticSubproblem)
8: 
9: __all__ = ['_minimize_trustregion_exact',
10:            'estimate_smallest_singular_value',
11:            'singular_leading_submatrix',
12:            'IterativeSubproblem']
13: 
14: 
15: def _minimize_trustregion_exact(fun, x0, args=(), jac=None, hess=None,
16:                                 **trust_region_options):
17:     '''
18:     Minimization of scalar function of one or more variables using
19:     a nearly exact trust-region algorithm.
20: 
21:     Options
22:     -------
23:     initial_tr_radius : float
24:         Initial trust-region radius.
25:     max_tr_radius : float
26:         Maximum value of the trust-region radius. No steps that are longer
27:         than this value will be proposed.
28:     eta : float
29:         Trust region related acceptance stringency for proposed steps.
30:     gtol : float
31:         Gradient norm must be less than ``gtol`` before successful
32:         termination.
33:     '''
34: 
35:     if jac is None:
36:         raise ValueError('Jacobian is required for trust region '
37:                          'exact minimization.')
38:     if hess is None:
39:         raise ValueError('Hessian matrix is required for trust region '
40:                          'exact minimization.')
41:     return _minimize_trust_region(fun, x0, args=args, jac=jac, hess=hess,
42:                                   subproblem=IterativeSubproblem,
43:                                   **trust_region_options)
44: 
45: 
46: def estimate_smallest_singular_value(U):
47:     '''Given upper triangular matrix ``U`` estimate the smallest singular
48:     value and the correspondent right singular vector in O(n**2) operations.
49: 
50:     Parameters
51:     ----------
52:     U : ndarray
53:         Square upper triangular matrix.
54: 
55:     Returns
56:     -------
57:     s_min : float
58:         Estimated smallest singular value of the provided matrix.
59:     z_min : ndarray
60:         Estimatied right singular vector.
61: 
62:     Notes
63:     -----
64:     The procedure is based on [1]_ and is done in two steps. First it finds
65:     a vector ``e`` with components selected from {+1, -1} such that the
66:     solution ``w`` from the system ``U.T w = e`` is as large as possible.
67:     Next it estimate ``U v = w``. The smallest singular value is close
68:     to ``norm(w)/norm(v)`` and the right singular vector is close
69:     to ``v/norm(v)``.
70: 
71:     The estimation will be better more ill-conditioned is the matrix.
72: 
73:     References
74:     ----------
75:     .. [1] Cline, A. K., Moler, C. B., Stewart, G. W., Wilkinson, J. H.
76:            An estimate for the condition number of a matrix.  1979.
77:            SIAM Journal on Numerical Analysis, 16(2), 368-375.
78:     '''
79: 
80:     U = np.atleast_2d(U)
81:     m, n = U.shape
82: 
83:     if m != n:
84:         raise ValueError("A square triangular matrix should be provided.")
85: 
86:     # A vector `e` with components selected from {+1, -1}
87:     # is selected so that the solution `w` to the system
88:     # `U.T w = e` is as large as possible. Implementation
89:     # based on algorithm 3.5.1, p. 142, from reference [2]
90:     # adapted for lower triangular matrix.
91: 
92:     p = np.zeros(n)
93:     w = np.empty(n)
94: 
95:     # Implemented according to:  Golub, G. H., Van Loan, C. F. (2013).
96:     # "Matrix computations". Forth Edition. JHU press. pp. 140-142.
97:     for k in range(n):
98:         wp = (1-p[k]) / U.T[k, k]
99:         wm = (-1-p[k]) / U.T[k, k]
100:         pp = p[k+1:] + U.T[k+1:, k]*wp
101:         pm = p[k+1:] + U.T[k+1:, k]*wm
102: 
103:         if abs(wp) + norm(pp, 1) >= abs(wm) + norm(pm, 1):
104:             w[k] = wp
105:             p[k+1:] = pp
106:         else:
107:             w[k] = wm
108:             p[k+1:] = pm
109: 
110:     # The system `U v = w` is solved using backward substitution.
111:     v = solve_triangular(U, w)
112: 
113:     v_norm = norm(v)
114:     w_norm = norm(w)
115: 
116:     # Smallest singular value
117:     s_min = w_norm / v_norm
118: 
119:     # Associated vector
120:     z_min = v / v_norm
121: 
122:     return s_min, z_min
123: 
124: 
125: def gershgorin_bounds(H):
126:     '''
127:     Given a square matrix ``H`` compute upper
128:     and lower bounds for its eigenvalues (Gregoshgorin Bounds).
129:     Defined ref. [1].
130: 
131:     References
132:     ----------
133:     .. [1] Conn, A. R., Gould, N. I., & Toint, P. L.
134:            Trust region methods. 2000. Siam. pp. 19.
135:     '''
136: 
137:     H_diag = np.diag(H)
138:     H_diag_abs = np.abs(H_diag)
139:     H_row_sums = np.sum(np.abs(H), axis=1)
140:     lb = np.min(H_diag + H_diag_abs - H_row_sums)
141:     ub = np.max(H_diag - H_diag_abs + H_row_sums)
142: 
143:     return lb, ub
144: 
145: 
146: def singular_leading_submatrix(A, U, k):
147:     '''
148:     Compute term that makes the leading ``k`` by ``k``
149:     submatrix from ``A`` singular.
150: 
151:     Parameters
152:     ----------
153:     A : ndarray
154:         Symmetric matrix that is not positive definite.
155:     U : ndarray
156:         Upper triangular matrix resulting of an incomplete
157:         Cholesky decomposition of matrix ``A``.
158:     k : int
159:         Positive integer such that the leading k by k submatrix from
160:         `A` is the first non-positive definite leading submatrix.
161: 
162:     Returns
163:     -------
164:     delta : float
165:         Amout that should be added to the element (k, k) of the
166:         leading k by k submatrix of ``A`` to make it singular.
167:     v : ndarray
168:         A vector such that ``v.T B v = 0``. Where B is the matrix A after
169:         ``delta`` is added to its element (k, k).
170:     '''
171: 
172:     # Compute delta
173:     delta = np.sum(U[:k-1, k-1]**2) - A[k-1, k-1]
174: 
175:     n = len(A)
176: 
177:     # Inicialize v
178:     v = np.zeros(n)
179:     v[k-1] = 1
180: 
181:     # Compute the remaining values of v by solving a triangular system.
182:     if k != 1:
183:         v[:k-1] = solve_triangular(U[:k-1, :k-1], -U[:k-1, k-1])
184: 
185:     return delta, v
186: 
187: 
188: class IterativeSubproblem(BaseQuadraticSubproblem):
189:     '''Quadratic subproblem solved by nearly exact iterative method.
190: 
191:     Notes
192:     -----
193:     This subproblem solver was based on [1]_, [2]_ and [3]_,
194:     which implement similar algorithms. The algorithm is basically
195:     that of [1]_ but ideas from [2]_ and [3]_ were also used.
196: 
197:     References
198:     ----------
199:     .. [1] A.R. Conn, N.I. Gould, and P.L. Toint, "Trust region methods",
200:            Siam, pp. 169-200, 2000.
201:     .. [2] J. Nocedal and  S. Wright, "Numerical optimization",
202:            Springer Science & Business Media. pp. 83-91, 2006.
203:     .. [3] J.J. More and D.C. Sorensen, "Computing a trust region step",
204:            SIAM Journal on Scientific and Statistical Computing, vol. 4(3),
205:            pp. 553-572, 1983.
206:     '''
207: 
208:     # UPDATE_COEFF appears in reference [1]_
209:     # in formula 7.3.14 (p. 190) named as "theta".
210:     # As recommended there it value is fixed in 0.01.
211:     UPDATE_COEFF = 0.01
212: 
213:     EPS = np.finfo(float).eps
214: 
215:     def __init__(self, x, fun, jac, hess, hessp=None,
216:                  k_easy=0.1, k_hard=0.2):
217: 
218:         super(IterativeSubproblem, self).__init__(x, fun, jac, hess)
219: 
220:         # When the trust-region shrinks in two consecutive
221:         # calculations (``tr_radius < previous_tr_radius``)
222:         # the lower bound ``lambda_lb`` may be reused,
223:         # facilitating  the convergence.  To indicate no
224:         # previous value is known at first ``previous_tr_radius``
225:         # is set to -1  and ``lambda_lb`` to None.
226:         self.previous_tr_radius = -1
227:         self.lambda_lb = None
228: 
229:         self.niter = 0
230: 
231:         # ``k_easy`` and ``k_hard`` are parameters used
232:         # to detemine the stop criteria to the iterative
233:         # subproblem solver. Take a look at pp. 194-197
234:         # from reference _[1] for a more detailed description.
235:         self.k_easy = k_easy
236:         self.k_hard = k_hard
237: 
238:         # Get Lapack function for cholesky decomposition.
239:         # The implemented Scipy wrapper does not return
240:         # the incomplete factorization needed by the method.
241:         self.cholesky, = get_lapack_funcs(('potrf',), (self.hess,))
242: 
243:         # Get info about Hessian
244:         self.dimension = len(self.hess)
245:         self.hess_gershgorin_lb,\
246:             self.hess_gershgorin_ub = gershgorin_bounds(self.hess)
247:         self.hess_inf = norm(self.hess, np.Inf)
248:         self.hess_fro = norm(self.hess, 'fro')
249: 
250:         # A constant such that for vectors smaler than that
251:         # backward substituition is not reliable. It was stabilished
252:         # based on Golub, G. H., Van Loan, C. F. (2013).
253:         # "Matrix computations". Forth Edition. JHU press., p.165.
254:         self.CLOSE_TO_ZERO = self.dimension * self.EPS * self.hess_inf
255: 
256:     def _initial_values(self, tr_radius):
257:         '''Given a trust radius, return a good initial guess for
258:         the damping factor, the lower bound and the upper bound.
259:         The values were chosen accordingly to the guidelines on
260:         section 7.3.8 (p. 192) from [1]_.
261:         '''
262: 
263:         # Upper bound for the damping factor
264:         lambda_ub = max(0, self.jac_mag/tr_radius + min(-self.hess_gershgorin_lb,
265:                                                         self.hess_fro,
266:                                                         self.hess_inf))
267: 
268:         # Lower bound for the damping factor
269:         lambda_lb = max(0, -min(self.hess.diagonal()),
270:                         self.jac_mag/tr_radius - min(self.hess_gershgorin_ub,
271:                                                      self.hess_fro,
272:                                                      self.hess_inf))
273: 
274:         # Improve bounds with previous info
275:         if tr_radius < self.previous_tr_radius:
276:             lambda_lb = max(self.lambda_lb, lambda_lb)
277: 
278:         # Initial guess for the damping factor
279:         if lambda_lb == 0:
280:             lambda_initial = 0
281:         else:
282:             lambda_initial = max(np.sqrt(lambda_lb * lambda_ub),
283:                                  lambda_lb + self.UPDATE_COEFF*(lambda_ub-lambda_lb))
284: 
285:         return lambda_initial, lambda_lb, lambda_ub
286: 
287:     def solve(self, tr_radius):
288:         '''Solve quadratic subproblem'''
289: 
290:         lambda_current, lambda_lb, lambda_ub = self._initial_values(tr_radius)
291:         n = self.dimension
292:         hits_boundary = True
293:         already_factorized = False
294:         self.niter = 0
295: 
296:         while True:
297: 
298:             # Compute Cholesky factorization
299:             if already_factorized:
300:                 already_factorized = False
301:             else:
302:                 H = self.hess+lambda_current*np.eye(n)
303:                 U, info = self.cholesky(H, lower=False,
304:                                         overwrite_a=False,
305:                                         clean=True)
306: 
307:             self.niter += 1
308: 
309:             # Check if factorization succeded
310:             if info == 0 and self.jac_mag > self.CLOSE_TO_ZERO:
311:                 # Successfull factorization
312: 
313:                 # Solve `U.T U p = s`
314:                 p = cho_solve((U, False), -self.jac)
315: 
316:                 p_norm = norm(p)
317: 
318:                 # Check for interior convergence
319:                 if p_norm <= tr_radius and lambda_current == 0:
320:                     hits_boundary = False
321:                     break
322: 
323:                 # Solve `U.T w = p`
324:                 w = solve_triangular(U, p, trans='T')
325: 
326:                 w_norm = norm(w)
327: 
328:                 # Compute Newton step accordingly to
329:                 # formula (4.44) p.87 from ref [2]_.
330:                 delta_lambda = (p_norm/w_norm)**2 * (p_norm-tr_radius)/tr_radius
331:                 lambda_new = lambda_current + delta_lambda
332: 
333:                 if p_norm < tr_radius:  # Inside boundary
334:                     s_min, z_min = estimate_smallest_singular_value(U)
335: 
336:                     ta, tb = self.get_boundaries_intersections(p, z_min,
337:                                                                tr_radius)
338: 
339:                     # Choose `step_len` with the smallest magnitude.
340:                     # The reason for this choice is explained at
341:                     # ref [3]_, p. 6 (Immediately before the formula
342:                     # for `tau`).
343:                     step_len = min([ta, tb], key=abs)
344: 
345:                     # Compute the quadratic term  (p.T*H*p)
346:                     quadratic_term = np.dot(p, np.dot(H, p))
347: 
348:                     # Check stop criteria
349:                     relative_error = (step_len**2 * s_min**2) / (quadratic_term + lambda_current*tr_radius**2)
350:                     if relative_error <= self.k_hard:
351:                         p += step_len * z_min
352:                         break
353: 
354:                     # Update uncertanty bounds
355:                     lambda_ub = lambda_current
356:                     lambda_lb = max(lambda_lb, lambda_current - s_min**2)
357: 
358:                     # Compute Cholesky factorization
359:                     H = self.hess + lambda_new*np.eye(n)
360:                     c, info = self.cholesky(H, lower=False,
361:                                             overwrite_a=False,
362:                                             clean=True)
363: 
364:                     # Check if the factorization have succeded
365:                     #
366:                     if info == 0:  # Successfull factorization
367:                         # Update damping factor
368:                         lambda_current = lambda_new
369:                         already_factorized = True
370:                     else:  # Unsuccessfull factorization
371:                         # Update uncertanty bounds
372:                         lambda_lb = max(lambda_lb, lambda_new)
373: 
374:                         # Update damping factor
375:                         lambda_current = max(np.sqrt(lambda_lb * lambda_ub),
376:                                              lambda_lb + self.UPDATE_COEFF*(lambda_ub-lambda_lb))
377: 
378:                 else:  # Outside boundary
379:                     # Check stop criteria
380:                     relative_error = abs(p_norm - tr_radius) / tr_radius
381:                     if relative_error <= self.k_easy:
382:                         break
383: 
384:                     # Update uncertanty bounds
385:                     lambda_lb = lambda_current
386: 
387:                     # Update damping factor
388:                     lambda_current = lambda_new
389: 
390:             elif info == 0 and self.jac_mag <= self.CLOSE_TO_ZERO:
391:                 # jac_mag very close to zero
392: 
393:                 # Check for interior convergence
394:                 if lambda_current == 0:
395:                     p = np.zeros(n)
396:                     hits_boundary = False
397:                     break
398: 
399:                 s_min, z_min = estimate_smallest_singular_value(U)
400:                 step_len = tr_radius
401: 
402:                 # Check stop criteria
403:                 if step_len**2 * s_min**2 <= self.k_hard * lambda_current * tr_radius**2:
404:                     p = step_len * z_min
405:                     break
406: 
407:                 # Update uncertanty bounds
408:                 lambda_ub = lambda_current
409:                 lambda_lb = max(lambda_lb, lambda_current - s_min**2)
410: 
411:                 # Update damping factor
412:                 lambda_current = max(np.sqrt(lambda_lb * lambda_ub),
413:                                      lambda_lb + self.UPDATE_COEFF*(lambda_ub-lambda_lb))
414: 
415:             else:  # Unsuccessfull factorization
416: 
417:                 # Compute auxiliar terms
418:                 delta, v = singular_leading_submatrix(H, U, info)
419:                 v_norm = norm(v)
420: 
421:                 # Update uncertanty interval
422:                 lambda_lb = max(lambda_lb, lambda_current + delta/v_norm**2)
423: 
424:                 # Update damping factor
425:                 lambda_current = max(np.sqrt(lambda_lb * lambda_ub),
426:                                      lambda_lb + self.UPDATE_COEFF*(lambda_ub-lambda_lb))
427: 
428:         self.lambda_lb = lambda_lb
429:         self.lambda_current = lambda_current
430:         self.previous_tr_radius = tr_radius
431: 
432:         return p, hits_boundary
433: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_203300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Nearly exact trust-region optimization subproblem.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_203301 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_203301) is not StypyTypeError):

    if (import_203301 != 'pyd_module'):
        __import__(import_203301)
        sys_modules_203302 = sys.modules[import_203301]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_203302.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_203301)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.linalg import norm, get_lapack_funcs, solve_triangular, cho_solve' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_203303 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.linalg')

if (type(import_203303) is not StypyTypeError):

    if (import_203303 != 'pyd_module'):
        __import__(import_203303)
        sys_modules_203304 = sys.modules[import_203303]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.linalg', sys_modules_203304.module_type_store, module_type_store, ['norm', 'get_lapack_funcs', 'solve_triangular', 'cho_solve'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_203304, sys_modules_203304.module_type_store, module_type_store)
    else:
        from scipy.linalg import norm, get_lapack_funcs, solve_triangular, cho_solve

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.linalg', None, module_type_store, ['norm', 'get_lapack_funcs', 'solve_triangular', 'cho_solve'], [norm, get_lapack_funcs, solve_triangular, cho_solve])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.linalg', import_203303)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.optimize._trustregion import _minimize_trust_region, BaseQuadraticSubproblem' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_203305 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.optimize._trustregion')

if (type(import_203305) is not StypyTypeError):

    if (import_203305 != 'pyd_module'):
        __import__(import_203305)
        sys_modules_203306 = sys.modules[import_203305]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.optimize._trustregion', sys_modules_203306.module_type_store, module_type_store, ['_minimize_trust_region', 'BaseQuadraticSubproblem'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_203306, sys_modules_203306.module_type_store, module_type_store)
    else:
        from scipy.optimize._trustregion import _minimize_trust_region, BaseQuadraticSubproblem

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.optimize._trustregion', None, module_type_store, ['_minimize_trust_region', 'BaseQuadraticSubproblem'], [_minimize_trust_region, BaseQuadraticSubproblem])

else:
    # Assigning a type to the variable 'scipy.optimize._trustregion' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.optimize._trustregion', import_203305)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')


# Assigning a List to a Name (line 9):

# Assigning a List to a Name (line 9):
__all__ = ['_minimize_trustregion_exact', 'estimate_smallest_singular_value', 'singular_leading_submatrix', 'IterativeSubproblem']
module_type_store.set_exportable_members(['_minimize_trustregion_exact', 'estimate_smallest_singular_value', 'singular_leading_submatrix', 'IterativeSubproblem'])

# Obtaining an instance of the builtin type 'list' (line 9)
list_203307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
str_203308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 11), 'str', '_minimize_trustregion_exact')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_203307, str_203308)
# Adding element type (line 9)
str_203309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 11), 'str', 'estimate_smallest_singular_value')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_203307, str_203309)
# Adding element type (line 9)
str_203310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 11), 'str', 'singular_leading_submatrix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_203307, str_203310)
# Adding element type (line 9)
str_203311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 11), 'str', 'IterativeSubproblem')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_203307, str_203311)

# Assigning a type to the variable '__all__' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), '__all__', list_203307)

@norecursion
def _minimize_trustregion_exact(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 15)
    tuple_203312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 15)
    
    # Getting the type of 'None' (line 15)
    None_203313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 54), 'None')
    # Getting the type of 'None' (line 15)
    None_203314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 65), 'None')
    defaults = [tuple_203312, None_203313, None_203314]
    # Create a new context for function '_minimize_trustregion_exact'
    module_type_store = module_type_store.open_function_context('_minimize_trustregion_exact', 15, 0, False)
    
    # Passed parameters checking function
    _minimize_trustregion_exact.stypy_localization = localization
    _minimize_trustregion_exact.stypy_type_of_self = None
    _minimize_trustregion_exact.stypy_type_store = module_type_store
    _minimize_trustregion_exact.stypy_function_name = '_minimize_trustregion_exact'
    _minimize_trustregion_exact.stypy_param_names_list = ['fun', 'x0', 'args', 'jac', 'hess']
    _minimize_trustregion_exact.stypy_varargs_param_name = None
    _minimize_trustregion_exact.stypy_kwargs_param_name = 'trust_region_options'
    _minimize_trustregion_exact.stypy_call_defaults = defaults
    _minimize_trustregion_exact.stypy_call_varargs = varargs
    _minimize_trustregion_exact.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_minimize_trustregion_exact', ['fun', 'x0', 'args', 'jac', 'hess'], None, 'trust_region_options', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_minimize_trustregion_exact', localization, ['fun', 'x0', 'args', 'jac', 'hess'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_minimize_trustregion_exact(...)' code ##################

    str_203315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, (-1)), 'str', '\n    Minimization of scalar function of one or more variables using\n    a nearly exact trust-region algorithm.\n\n    Options\n    -------\n    initial_tr_radius : float\n        Initial trust-region radius.\n    max_tr_radius : float\n        Maximum value of the trust-region radius. No steps that are longer\n        than this value will be proposed.\n    eta : float\n        Trust region related acceptance stringency for proposed steps.\n    gtol : float\n        Gradient norm must be less than ``gtol`` before successful\n        termination.\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 35)
    # Getting the type of 'jac' (line 35)
    jac_203316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 7), 'jac')
    # Getting the type of 'None' (line 35)
    None_203317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 14), 'None')
    
    (may_be_203318, more_types_in_union_203319) = may_be_none(jac_203316, None_203317)

    if may_be_203318:

        if more_types_in_union_203319:
            # Runtime conditional SSA (line 35)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to ValueError(...): (line 36)
        # Processing the call arguments (line 36)
        str_203321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 25), 'str', 'Jacobian is required for trust region exact minimization.')
        # Processing the call keyword arguments (line 36)
        kwargs_203322 = {}
        # Getting the type of 'ValueError' (line 36)
        ValueError_203320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 36)
        ValueError_call_result_203323 = invoke(stypy.reporting.localization.Localization(__file__, 36, 14), ValueError_203320, *[str_203321], **kwargs_203322)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 36, 8), ValueError_call_result_203323, 'raise parameter', BaseException)

        if more_types_in_union_203319:
            # SSA join for if statement (line 35)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 38)
    # Getting the type of 'hess' (line 38)
    hess_203324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 7), 'hess')
    # Getting the type of 'None' (line 38)
    None_203325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'None')
    
    (may_be_203326, more_types_in_union_203327) = may_be_none(hess_203324, None_203325)

    if may_be_203326:

        if more_types_in_union_203327:
            # Runtime conditional SSA (line 38)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to ValueError(...): (line 39)
        # Processing the call arguments (line 39)
        str_203329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 25), 'str', 'Hessian matrix is required for trust region exact minimization.')
        # Processing the call keyword arguments (line 39)
        kwargs_203330 = {}
        # Getting the type of 'ValueError' (line 39)
        ValueError_203328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 39)
        ValueError_call_result_203331 = invoke(stypy.reporting.localization.Localization(__file__, 39, 14), ValueError_203328, *[str_203329], **kwargs_203330)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 39, 8), ValueError_call_result_203331, 'raise parameter', BaseException)

        if more_types_in_union_203327:
            # SSA join for if statement (line 38)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to _minimize_trust_region(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'fun' (line 41)
    fun_203333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 34), 'fun', False)
    # Getting the type of 'x0' (line 41)
    x0_203334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 39), 'x0', False)
    # Processing the call keyword arguments (line 41)
    # Getting the type of 'args' (line 41)
    args_203335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 48), 'args', False)
    keyword_203336 = args_203335
    # Getting the type of 'jac' (line 41)
    jac_203337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 58), 'jac', False)
    keyword_203338 = jac_203337
    # Getting the type of 'hess' (line 41)
    hess_203339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 68), 'hess', False)
    keyword_203340 = hess_203339
    # Getting the type of 'IterativeSubproblem' (line 42)
    IterativeSubproblem_203341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 45), 'IterativeSubproblem', False)
    keyword_203342 = IterativeSubproblem_203341
    # Getting the type of 'trust_region_options' (line 43)
    trust_region_options_203343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 36), 'trust_region_options', False)
    kwargs_203344 = {'hess': keyword_203340, 'subproblem': keyword_203342, 'args': keyword_203336, 'jac': keyword_203338, 'trust_region_options_203343': trust_region_options_203343}
    # Getting the type of '_minimize_trust_region' (line 41)
    _minimize_trust_region_203332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 11), '_minimize_trust_region', False)
    # Calling _minimize_trust_region(args, kwargs) (line 41)
    _minimize_trust_region_call_result_203345 = invoke(stypy.reporting.localization.Localization(__file__, 41, 11), _minimize_trust_region_203332, *[fun_203333, x0_203334], **kwargs_203344)
    
    # Assigning a type to the variable 'stypy_return_type' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type', _minimize_trust_region_call_result_203345)
    
    # ################# End of '_minimize_trustregion_exact(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_minimize_trustregion_exact' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_203346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_203346)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_minimize_trustregion_exact'
    return stypy_return_type_203346

# Assigning a type to the variable '_minimize_trustregion_exact' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), '_minimize_trustregion_exact', _minimize_trustregion_exact)

@norecursion
def estimate_smallest_singular_value(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'estimate_smallest_singular_value'
    module_type_store = module_type_store.open_function_context('estimate_smallest_singular_value', 46, 0, False)
    
    # Passed parameters checking function
    estimate_smallest_singular_value.stypy_localization = localization
    estimate_smallest_singular_value.stypy_type_of_self = None
    estimate_smallest_singular_value.stypy_type_store = module_type_store
    estimate_smallest_singular_value.stypy_function_name = 'estimate_smallest_singular_value'
    estimate_smallest_singular_value.stypy_param_names_list = ['U']
    estimate_smallest_singular_value.stypy_varargs_param_name = None
    estimate_smallest_singular_value.stypy_kwargs_param_name = None
    estimate_smallest_singular_value.stypy_call_defaults = defaults
    estimate_smallest_singular_value.stypy_call_varargs = varargs
    estimate_smallest_singular_value.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'estimate_smallest_singular_value', ['U'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'estimate_smallest_singular_value', localization, ['U'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'estimate_smallest_singular_value(...)' code ##################

    str_203347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, (-1)), 'str', 'Given upper triangular matrix ``U`` estimate the smallest singular\n    value and the correspondent right singular vector in O(n**2) operations.\n\n    Parameters\n    ----------\n    U : ndarray\n        Square upper triangular matrix.\n\n    Returns\n    -------\n    s_min : float\n        Estimated smallest singular value of the provided matrix.\n    z_min : ndarray\n        Estimatied right singular vector.\n\n    Notes\n    -----\n    The procedure is based on [1]_ and is done in two steps. First it finds\n    a vector ``e`` with components selected from {+1, -1} such that the\n    solution ``w`` from the system ``U.T w = e`` is as large as possible.\n    Next it estimate ``U v = w``. The smallest singular value is close\n    to ``norm(w)/norm(v)`` and the right singular vector is close\n    to ``v/norm(v)``.\n\n    The estimation will be better more ill-conditioned is the matrix.\n\n    References\n    ----------\n    .. [1] Cline, A. K., Moler, C. B., Stewart, G. W., Wilkinson, J. H.\n           An estimate for the condition number of a matrix.  1979.\n           SIAM Journal on Numerical Analysis, 16(2), 368-375.\n    ')
    
    # Assigning a Call to a Name (line 80):
    
    # Assigning a Call to a Name (line 80):
    
    # Call to atleast_2d(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'U' (line 80)
    U_203350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 22), 'U', False)
    # Processing the call keyword arguments (line 80)
    kwargs_203351 = {}
    # Getting the type of 'np' (line 80)
    np_203348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'np', False)
    # Obtaining the member 'atleast_2d' of a type (line 80)
    atleast_2d_203349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), np_203348, 'atleast_2d')
    # Calling atleast_2d(args, kwargs) (line 80)
    atleast_2d_call_result_203352 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), atleast_2d_203349, *[U_203350], **kwargs_203351)
    
    # Assigning a type to the variable 'U' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'U', atleast_2d_call_result_203352)
    
    # Assigning a Attribute to a Tuple (line 81):
    
    # Assigning a Subscript to a Name (line 81):
    
    # Obtaining the type of the subscript
    int_203353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 4), 'int')
    # Getting the type of 'U' (line 81)
    U_203354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'U')
    # Obtaining the member 'shape' of a type (line 81)
    shape_203355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 11), U_203354, 'shape')
    # Obtaining the member '__getitem__' of a type (line 81)
    getitem___203356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 4), shape_203355, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 81)
    subscript_call_result_203357 = invoke(stypy.reporting.localization.Localization(__file__, 81, 4), getitem___203356, int_203353)
    
    # Assigning a type to the variable 'tuple_var_assignment_203280' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'tuple_var_assignment_203280', subscript_call_result_203357)
    
    # Assigning a Subscript to a Name (line 81):
    
    # Obtaining the type of the subscript
    int_203358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 4), 'int')
    # Getting the type of 'U' (line 81)
    U_203359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'U')
    # Obtaining the member 'shape' of a type (line 81)
    shape_203360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 11), U_203359, 'shape')
    # Obtaining the member '__getitem__' of a type (line 81)
    getitem___203361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 4), shape_203360, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 81)
    subscript_call_result_203362 = invoke(stypy.reporting.localization.Localization(__file__, 81, 4), getitem___203361, int_203358)
    
    # Assigning a type to the variable 'tuple_var_assignment_203281' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'tuple_var_assignment_203281', subscript_call_result_203362)
    
    # Assigning a Name to a Name (line 81):
    # Getting the type of 'tuple_var_assignment_203280' (line 81)
    tuple_var_assignment_203280_203363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'tuple_var_assignment_203280')
    # Assigning a type to the variable 'm' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'm', tuple_var_assignment_203280_203363)
    
    # Assigning a Name to a Name (line 81):
    # Getting the type of 'tuple_var_assignment_203281' (line 81)
    tuple_var_assignment_203281_203364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'tuple_var_assignment_203281')
    # Assigning a type to the variable 'n' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 7), 'n', tuple_var_assignment_203281_203364)
    
    
    # Getting the type of 'm' (line 83)
    m_203365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 7), 'm')
    # Getting the type of 'n' (line 83)
    n_203366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'n')
    # Applying the binary operator '!=' (line 83)
    result_ne_203367 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 7), '!=', m_203365, n_203366)
    
    # Testing the type of an if condition (line 83)
    if_condition_203368 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 4), result_ne_203367)
    # Assigning a type to the variable 'if_condition_203368' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'if_condition_203368', if_condition_203368)
    # SSA begins for if statement (line 83)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 84)
    # Processing the call arguments (line 84)
    str_203370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 25), 'str', 'A square triangular matrix should be provided.')
    # Processing the call keyword arguments (line 84)
    kwargs_203371 = {}
    # Getting the type of 'ValueError' (line 84)
    ValueError_203369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 84)
    ValueError_call_result_203372 = invoke(stypy.reporting.localization.Localization(__file__, 84, 14), ValueError_203369, *[str_203370], **kwargs_203371)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 84, 8), ValueError_call_result_203372, 'raise parameter', BaseException)
    # SSA join for if statement (line 83)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 92):
    
    # Assigning a Call to a Name (line 92):
    
    # Call to zeros(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'n' (line 92)
    n_203375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 17), 'n', False)
    # Processing the call keyword arguments (line 92)
    kwargs_203376 = {}
    # Getting the type of 'np' (line 92)
    np_203373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 92)
    zeros_203374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), np_203373, 'zeros')
    # Calling zeros(args, kwargs) (line 92)
    zeros_call_result_203377 = invoke(stypy.reporting.localization.Localization(__file__, 92, 8), zeros_203374, *[n_203375], **kwargs_203376)
    
    # Assigning a type to the variable 'p' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'p', zeros_call_result_203377)
    
    # Assigning a Call to a Name (line 93):
    
    # Assigning a Call to a Name (line 93):
    
    # Call to empty(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'n' (line 93)
    n_203380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 17), 'n', False)
    # Processing the call keyword arguments (line 93)
    kwargs_203381 = {}
    # Getting the type of 'np' (line 93)
    np_203378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'np', False)
    # Obtaining the member 'empty' of a type (line 93)
    empty_203379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), np_203378, 'empty')
    # Calling empty(args, kwargs) (line 93)
    empty_call_result_203382 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), empty_203379, *[n_203380], **kwargs_203381)
    
    # Assigning a type to the variable 'w' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'w', empty_call_result_203382)
    
    
    # Call to range(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'n' (line 97)
    n_203384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 19), 'n', False)
    # Processing the call keyword arguments (line 97)
    kwargs_203385 = {}
    # Getting the type of 'range' (line 97)
    range_203383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 13), 'range', False)
    # Calling range(args, kwargs) (line 97)
    range_call_result_203386 = invoke(stypy.reporting.localization.Localization(__file__, 97, 13), range_203383, *[n_203384], **kwargs_203385)
    
    # Testing the type of a for loop iterable (line 97)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 97, 4), range_call_result_203386)
    # Getting the type of the for loop variable (line 97)
    for_loop_var_203387 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 97, 4), range_call_result_203386)
    # Assigning a type to the variable 'k' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'k', for_loop_var_203387)
    # SSA begins for a for statement (line 97)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 98):
    
    # Assigning a BinOp to a Name (line 98):
    int_203388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 14), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 98)
    k_203389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 18), 'k')
    # Getting the type of 'p' (line 98)
    p_203390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 'p')
    # Obtaining the member '__getitem__' of a type (line 98)
    getitem___203391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 16), p_203390, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 98)
    subscript_call_result_203392 = invoke(stypy.reporting.localization.Localization(__file__, 98, 16), getitem___203391, k_203389)
    
    # Applying the binary operator '-' (line 98)
    result_sub_203393 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 14), '-', int_203388, subscript_call_result_203392)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 98)
    tuple_203394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 98)
    # Adding element type (line 98)
    # Getting the type of 'k' (line 98)
    k_203395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 28), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 28), tuple_203394, k_203395)
    # Adding element type (line 98)
    # Getting the type of 'k' (line 98)
    k_203396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 31), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 28), tuple_203394, k_203396)
    
    # Getting the type of 'U' (line 98)
    U_203397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 24), 'U')
    # Obtaining the member 'T' of a type (line 98)
    T_203398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 24), U_203397, 'T')
    # Obtaining the member '__getitem__' of a type (line 98)
    getitem___203399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 24), T_203398, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 98)
    subscript_call_result_203400 = invoke(stypy.reporting.localization.Localization(__file__, 98, 24), getitem___203399, tuple_203394)
    
    # Applying the binary operator 'div' (line 98)
    result_div_203401 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 13), 'div', result_sub_203393, subscript_call_result_203400)
    
    # Assigning a type to the variable 'wp' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'wp', result_div_203401)
    
    # Assigning a BinOp to a Name (line 99):
    
    # Assigning a BinOp to a Name (line 99):
    int_203402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 14), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 99)
    k_203403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 19), 'k')
    # Getting the type of 'p' (line 99)
    p_203404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 17), 'p')
    # Obtaining the member '__getitem__' of a type (line 99)
    getitem___203405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 17), p_203404, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 99)
    subscript_call_result_203406 = invoke(stypy.reporting.localization.Localization(__file__, 99, 17), getitem___203405, k_203403)
    
    # Applying the binary operator '-' (line 99)
    result_sub_203407 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 14), '-', int_203402, subscript_call_result_203406)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 99)
    tuple_203408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 99)
    # Adding element type (line 99)
    # Getting the type of 'k' (line 99)
    k_203409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 29), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 29), tuple_203408, k_203409)
    # Adding element type (line 99)
    # Getting the type of 'k' (line 99)
    k_203410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 32), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 29), tuple_203408, k_203410)
    
    # Getting the type of 'U' (line 99)
    U_203411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 25), 'U')
    # Obtaining the member 'T' of a type (line 99)
    T_203412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 25), U_203411, 'T')
    # Obtaining the member '__getitem__' of a type (line 99)
    getitem___203413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 25), T_203412, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 99)
    subscript_call_result_203414 = invoke(stypy.reporting.localization.Localization(__file__, 99, 25), getitem___203413, tuple_203408)
    
    # Applying the binary operator 'div' (line 99)
    result_div_203415 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 13), 'div', result_sub_203407, subscript_call_result_203414)
    
    # Assigning a type to the variable 'wm' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'wm', result_div_203415)
    
    # Assigning a BinOp to a Name (line 100):
    
    # Assigning a BinOp to a Name (line 100):
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 100)
    k_203416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'k')
    int_203417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 17), 'int')
    # Applying the binary operator '+' (line 100)
    result_add_203418 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 15), '+', k_203416, int_203417)
    
    slice_203419 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 100, 13), result_add_203418, None, None)
    # Getting the type of 'p' (line 100)
    p_203420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 13), 'p')
    # Obtaining the member '__getitem__' of a type (line 100)
    getitem___203421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 13), p_203420, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 100)
    subscript_call_result_203422 = invoke(stypy.reporting.localization.Localization(__file__, 100, 13), getitem___203421, slice_203419)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 100)
    k_203423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'k')
    int_203424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 29), 'int')
    # Applying the binary operator '+' (line 100)
    result_add_203425 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 27), '+', k_203423, int_203424)
    
    slice_203426 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 100, 23), result_add_203425, None, None)
    # Getting the type of 'k' (line 100)
    k_203427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 33), 'k')
    # Getting the type of 'U' (line 100)
    U_203428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 23), 'U')
    # Obtaining the member 'T' of a type (line 100)
    T_203429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 23), U_203428, 'T')
    # Obtaining the member '__getitem__' of a type (line 100)
    getitem___203430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 23), T_203429, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 100)
    subscript_call_result_203431 = invoke(stypy.reporting.localization.Localization(__file__, 100, 23), getitem___203430, (slice_203426, k_203427))
    
    # Getting the type of 'wp' (line 100)
    wp_203432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 36), 'wp')
    # Applying the binary operator '*' (line 100)
    result_mul_203433 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 23), '*', subscript_call_result_203431, wp_203432)
    
    # Applying the binary operator '+' (line 100)
    result_add_203434 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 13), '+', subscript_call_result_203422, result_mul_203433)
    
    # Assigning a type to the variable 'pp' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'pp', result_add_203434)
    
    # Assigning a BinOp to a Name (line 101):
    
    # Assigning a BinOp to a Name (line 101):
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 101)
    k_203435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'k')
    int_203436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 17), 'int')
    # Applying the binary operator '+' (line 101)
    result_add_203437 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 15), '+', k_203435, int_203436)
    
    slice_203438 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 101, 13), result_add_203437, None, None)
    # Getting the type of 'p' (line 101)
    p_203439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 13), 'p')
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___203440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 13), p_203439, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_203441 = invoke(stypy.reporting.localization.Localization(__file__, 101, 13), getitem___203440, slice_203438)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 101)
    k_203442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 27), 'k')
    int_203443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 29), 'int')
    # Applying the binary operator '+' (line 101)
    result_add_203444 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 27), '+', k_203442, int_203443)
    
    slice_203445 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 101, 23), result_add_203444, None, None)
    # Getting the type of 'k' (line 101)
    k_203446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 33), 'k')
    # Getting the type of 'U' (line 101)
    U_203447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 23), 'U')
    # Obtaining the member 'T' of a type (line 101)
    T_203448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 23), U_203447, 'T')
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___203449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 23), T_203448, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_203450 = invoke(stypy.reporting.localization.Localization(__file__, 101, 23), getitem___203449, (slice_203445, k_203446))
    
    # Getting the type of 'wm' (line 101)
    wm_203451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 36), 'wm')
    # Applying the binary operator '*' (line 101)
    result_mul_203452 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 23), '*', subscript_call_result_203450, wm_203451)
    
    # Applying the binary operator '+' (line 101)
    result_add_203453 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 13), '+', subscript_call_result_203441, result_mul_203452)
    
    # Assigning a type to the variable 'pm' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'pm', result_add_203453)
    
    
    
    # Call to abs(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'wp' (line 103)
    wp_203455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'wp', False)
    # Processing the call keyword arguments (line 103)
    kwargs_203456 = {}
    # Getting the type of 'abs' (line 103)
    abs_203454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 11), 'abs', False)
    # Calling abs(args, kwargs) (line 103)
    abs_call_result_203457 = invoke(stypy.reporting.localization.Localization(__file__, 103, 11), abs_203454, *[wp_203455], **kwargs_203456)
    
    
    # Call to norm(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'pp' (line 103)
    pp_203459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 26), 'pp', False)
    int_203460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 30), 'int')
    # Processing the call keyword arguments (line 103)
    kwargs_203461 = {}
    # Getting the type of 'norm' (line 103)
    norm_203458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 21), 'norm', False)
    # Calling norm(args, kwargs) (line 103)
    norm_call_result_203462 = invoke(stypy.reporting.localization.Localization(__file__, 103, 21), norm_203458, *[pp_203459, int_203460], **kwargs_203461)
    
    # Applying the binary operator '+' (line 103)
    result_add_203463 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 11), '+', abs_call_result_203457, norm_call_result_203462)
    
    
    # Call to abs(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'wm' (line 103)
    wm_203465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 40), 'wm', False)
    # Processing the call keyword arguments (line 103)
    kwargs_203466 = {}
    # Getting the type of 'abs' (line 103)
    abs_203464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 36), 'abs', False)
    # Calling abs(args, kwargs) (line 103)
    abs_call_result_203467 = invoke(stypy.reporting.localization.Localization(__file__, 103, 36), abs_203464, *[wm_203465], **kwargs_203466)
    
    
    # Call to norm(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'pm' (line 103)
    pm_203469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 51), 'pm', False)
    int_203470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 55), 'int')
    # Processing the call keyword arguments (line 103)
    kwargs_203471 = {}
    # Getting the type of 'norm' (line 103)
    norm_203468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 46), 'norm', False)
    # Calling norm(args, kwargs) (line 103)
    norm_call_result_203472 = invoke(stypy.reporting.localization.Localization(__file__, 103, 46), norm_203468, *[pm_203469, int_203470], **kwargs_203471)
    
    # Applying the binary operator '+' (line 103)
    result_add_203473 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 36), '+', abs_call_result_203467, norm_call_result_203472)
    
    # Applying the binary operator '>=' (line 103)
    result_ge_203474 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 11), '>=', result_add_203463, result_add_203473)
    
    # Testing the type of an if condition (line 103)
    if_condition_203475 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 8), result_ge_203474)
    # Assigning a type to the variable 'if_condition_203475' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'if_condition_203475', if_condition_203475)
    # SSA begins for if statement (line 103)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 104):
    
    # Assigning a Name to a Subscript (line 104):
    # Getting the type of 'wp' (line 104)
    wp_203476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 19), 'wp')
    # Getting the type of 'w' (line 104)
    w_203477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'w')
    # Getting the type of 'k' (line 104)
    k_203478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 14), 'k')
    # Storing an element on a container (line 104)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 12), w_203477, (k_203478, wp_203476))
    
    # Assigning a Name to a Subscript (line 105):
    
    # Assigning a Name to a Subscript (line 105):
    # Getting the type of 'pp' (line 105)
    pp_203479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 22), 'pp')
    # Getting the type of 'p' (line 105)
    p_203480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'p')
    # Getting the type of 'k' (line 105)
    k_203481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 14), 'k')
    int_203482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 16), 'int')
    # Applying the binary operator '+' (line 105)
    result_add_203483 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 14), '+', k_203481, int_203482)
    
    slice_203484 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 105, 12), result_add_203483, None, None)
    # Storing an element on a container (line 105)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), p_203480, (slice_203484, pp_203479))
    # SSA branch for the else part of an if statement (line 103)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Subscript (line 107):
    
    # Assigning a Name to a Subscript (line 107):
    # Getting the type of 'wm' (line 107)
    wm_203485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'wm')
    # Getting the type of 'w' (line 107)
    w_203486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'w')
    # Getting the type of 'k' (line 107)
    k_203487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 14), 'k')
    # Storing an element on a container (line 107)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 12), w_203486, (k_203487, wm_203485))
    
    # Assigning a Name to a Subscript (line 108):
    
    # Assigning a Name to a Subscript (line 108):
    # Getting the type of 'pm' (line 108)
    pm_203488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 22), 'pm')
    # Getting the type of 'p' (line 108)
    p_203489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'p')
    # Getting the type of 'k' (line 108)
    k_203490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 'k')
    int_203491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 16), 'int')
    # Applying the binary operator '+' (line 108)
    result_add_203492 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), '+', k_203490, int_203491)
    
    slice_203493 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 108, 12), result_add_203492, None, None)
    # Storing an element on a container (line 108)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 12), p_203489, (slice_203493, pm_203488))
    # SSA join for if statement (line 103)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 111):
    
    # Assigning a Call to a Name (line 111):
    
    # Call to solve_triangular(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'U' (line 111)
    U_203495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 25), 'U', False)
    # Getting the type of 'w' (line 111)
    w_203496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 28), 'w', False)
    # Processing the call keyword arguments (line 111)
    kwargs_203497 = {}
    # Getting the type of 'solve_triangular' (line 111)
    solve_triangular_203494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'solve_triangular', False)
    # Calling solve_triangular(args, kwargs) (line 111)
    solve_triangular_call_result_203498 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), solve_triangular_203494, *[U_203495, w_203496], **kwargs_203497)
    
    # Assigning a type to the variable 'v' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'v', solve_triangular_call_result_203498)
    
    # Assigning a Call to a Name (line 113):
    
    # Assigning a Call to a Name (line 113):
    
    # Call to norm(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'v' (line 113)
    v_203500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 18), 'v', False)
    # Processing the call keyword arguments (line 113)
    kwargs_203501 = {}
    # Getting the type of 'norm' (line 113)
    norm_203499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 13), 'norm', False)
    # Calling norm(args, kwargs) (line 113)
    norm_call_result_203502 = invoke(stypy.reporting.localization.Localization(__file__, 113, 13), norm_203499, *[v_203500], **kwargs_203501)
    
    # Assigning a type to the variable 'v_norm' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'v_norm', norm_call_result_203502)
    
    # Assigning a Call to a Name (line 114):
    
    # Assigning a Call to a Name (line 114):
    
    # Call to norm(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'w' (line 114)
    w_203504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 18), 'w', False)
    # Processing the call keyword arguments (line 114)
    kwargs_203505 = {}
    # Getting the type of 'norm' (line 114)
    norm_203503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 13), 'norm', False)
    # Calling norm(args, kwargs) (line 114)
    norm_call_result_203506 = invoke(stypy.reporting.localization.Localization(__file__, 114, 13), norm_203503, *[w_203504], **kwargs_203505)
    
    # Assigning a type to the variable 'w_norm' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'w_norm', norm_call_result_203506)
    
    # Assigning a BinOp to a Name (line 117):
    
    # Assigning a BinOp to a Name (line 117):
    # Getting the type of 'w_norm' (line 117)
    w_norm_203507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'w_norm')
    # Getting the type of 'v_norm' (line 117)
    v_norm_203508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 21), 'v_norm')
    # Applying the binary operator 'div' (line 117)
    result_div_203509 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 12), 'div', w_norm_203507, v_norm_203508)
    
    # Assigning a type to the variable 's_min' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 's_min', result_div_203509)
    
    # Assigning a BinOp to a Name (line 120):
    
    # Assigning a BinOp to a Name (line 120):
    # Getting the type of 'v' (line 120)
    v_203510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'v')
    # Getting the type of 'v_norm' (line 120)
    v_norm_203511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'v_norm')
    # Applying the binary operator 'div' (line 120)
    result_div_203512 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 12), 'div', v_203510, v_norm_203511)
    
    # Assigning a type to the variable 'z_min' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'z_min', result_div_203512)
    
    # Obtaining an instance of the builtin type 'tuple' (line 122)
    tuple_203513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 122)
    # Adding element type (line 122)
    # Getting the type of 's_min' (line 122)
    s_min_203514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 11), 's_min')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 11), tuple_203513, s_min_203514)
    # Adding element type (line 122)
    # Getting the type of 'z_min' (line 122)
    z_min_203515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 18), 'z_min')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 11), tuple_203513, z_min_203515)
    
    # Assigning a type to the variable 'stypy_return_type' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type', tuple_203513)
    
    # ################# End of 'estimate_smallest_singular_value(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'estimate_smallest_singular_value' in the type store
    # Getting the type of 'stypy_return_type' (line 46)
    stypy_return_type_203516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_203516)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'estimate_smallest_singular_value'
    return stypy_return_type_203516

# Assigning a type to the variable 'estimate_smallest_singular_value' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'estimate_smallest_singular_value', estimate_smallest_singular_value)

@norecursion
def gershgorin_bounds(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'gershgorin_bounds'
    module_type_store = module_type_store.open_function_context('gershgorin_bounds', 125, 0, False)
    
    # Passed parameters checking function
    gershgorin_bounds.stypy_localization = localization
    gershgorin_bounds.stypy_type_of_self = None
    gershgorin_bounds.stypy_type_store = module_type_store
    gershgorin_bounds.stypy_function_name = 'gershgorin_bounds'
    gershgorin_bounds.stypy_param_names_list = ['H']
    gershgorin_bounds.stypy_varargs_param_name = None
    gershgorin_bounds.stypy_kwargs_param_name = None
    gershgorin_bounds.stypy_call_defaults = defaults
    gershgorin_bounds.stypy_call_varargs = varargs
    gershgorin_bounds.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gershgorin_bounds', ['H'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gershgorin_bounds', localization, ['H'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gershgorin_bounds(...)' code ##################

    str_203517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, (-1)), 'str', '\n    Given a square matrix ``H`` compute upper\n    and lower bounds for its eigenvalues (Gregoshgorin Bounds).\n    Defined ref. [1].\n\n    References\n    ----------\n    .. [1] Conn, A. R., Gould, N. I., & Toint, P. L.\n           Trust region methods. 2000. Siam. pp. 19.\n    ')
    
    # Assigning a Call to a Name (line 137):
    
    # Assigning a Call to a Name (line 137):
    
    # Call to diag(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'H' (line 137)
    H_203520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 21), 'H', False)
    # Processing the call keyword arguments (line 137)
    kwargs_203521 = {}
    # Getting the type of 'np' (line 137)
    np_203518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 13), 'np', False)
    # Obtaining the member 'diag' of a type (line 137)
    diag_203519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 13), np_203518, 'diag')
    # Calling diag(args, kwargs) (line 137)
    diag_call_result_203522 = invoke(stypy.reporting.localization.Localization(__file__, 137, 13), diag_203519, *[H_203520], **kwargs_203521)
    
    # Assigning a type to the variable 'H_diag' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'H_diag', diag_call_result_203522)
    
    # Assigning a Call to a Name (line 138):
    
    # Assigning a Call to a Name (line 138):
    
    # Call to abs(...): (line 138)
    # Processing the call arguments (line 138)
    # Getting the type of 'H_diag' (line 138)
    H_diag_203525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 24), 'H_diag', False)
    # Processing the call keyword arguments (line 138)
    kwargs_203526 = {}
    # Getting the type of 'np' (line 138)
    np_203523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 17), 'np', False)
    # Obtaining the member 'abs' of a type (line 138)
    abs_203524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 17), np_203523, 'abs')
    # Calling abs(args, kwargs) (line 138)
    abs_call_result_203527 = invoke(stypy.reporting.localization.Localization(__file__, 138, 17), abs_203524, *[H_diag_203525], **kwargs_203526)
    
    # Assigning a type to the variable 'H_diag_abs' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'H_diag_abs', abs_call_result_203527)
    
    # Assigning a Call to a Name (line 139):
    
    # Assigning a Call to a Name (line 139):
    
    # Call to sum(...): (line 139)
    # Processing the call arguments (line 139)
    
    # Call to abs(...): (line 139)
    # Processing the call arguments (line 139)
    # Getting the type of 'H' (line 139)
    H_203532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 31), 'H', False)
    # Processing the call keyword arguments (line 139)
    kwargs_203533 = {}
    # Getting the type of 'np' (line 139)
    np_203530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 24), 'np', False)
    # Obtaining the member 'abs' of a type (line 139)
    abs_203531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 24), np_203530, 'abs')
    # Calling abs(args, kwargs) (line 139)
    abs_call_result_203534 = invoke(stypy.reporting.localization.Localization(__file__, 139, 24), abs_203531, *[H_203532], **kwargs_203533)
    
    # Processing the call keyword arguments (line 139)
    int_203535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 40), 'int')
    keyword_203536 = int_203535
    kwargs_203537 = {'axis': keyword_203536}
    # Getting the type of 'np' (line 139)
    np_203528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 17), 'np', False)
    # Obtaining the member 'sum' of a type (line 139)
    sum_203529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 17), np_203528, 'sum')
    # Calling sum(args, kwargs) (line 139)
    sum_call_result_203538 = invoke(stypy.reporting.localization.Localization(__file__, 139, 17), sum_203529, *[abs_call_result_203534], **kwargs_203537)
    
    # Assigning a type to the variable 'H_row_sums' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'H_row_sums', sum_call_result_203538)
    
    # Assigning a Call to a Name (line 140):
    
    # Assigning a Call to a Name (line 140):
    
    # Call to min(...): (line 140)
    # Processing the call arguments (line 140)
    # Getting the type of 'H_diag' (line 140)
    H_diag_203541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 16), 'H_diag', False)
    # Getting the type of 'H_diag_abs' (line 140)
    H_diag_abs_203542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 25), 'H_diag_abs', False)
    # Applying the binary operator '+' (line 140)
    result_add_203543 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 16), '+', H_diag_203541, H_diag_abs_203542)
    
    # Getting the type of 'H_row_sums' (line 140)
    H_row_sums_203544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 38), 'H_row_sums', False)
    # Applying the binary operator '-' (line 140)
    result_sub_203545 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 36), '-', result_add_203543, H_row_sums_203544)
    
    # Processing the call keyword arguments (line 140)
    kwargs_203546 = {}
    # Getting the type of 'np' (line 140)
    np_203539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 9), 'np', False)
    # Obtaining the member 'min' of a type (line 140)
    min_203540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 9), np_203539, 'min')
    # Calling min(args, kwargs) (line 140)
    min_call_result_203547 = invoke(stypy.reporting.localization.Localization(__file__, 140, 9), min_203540, *[result_sub_203545], **kwargs_203546)
    
    # Assigning a type to the variable 'lb' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'lb', min_call_result_203547)
    
    # Assigning a Call to a Name (line 141):
    
    # Assigning a Call to a Name (line 141):
    
    # Call to max(...): (line 141)
    # Processing the call arguments (line 141)
    # Getting the type of 'H_diag' (line 141)
    H_diag_203550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'H_diag', False)
    # Getting the type of 'H_diag_abs' (line 141)
    H_diag_abs_203551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 25), 'H_diag_abs', False)
    # Applying the binary operator '-' (line 141)
    result_sub_203552 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 16), '-', H_diag_203550, H_diag_abs_203551)
    
    # Getting the type of 'H_row_sums' (line 141)
    H_row_sums_203553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 38), 'H_row_sums', False)
    # Applying the binary operator '+' (line 141)
    result_add_203554 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 36), '+', result_sub_203552, H_row_sums_203553)
    
    # Processing the call keyword arguments (line 141)
    kwargs_203555 = {}
    # Getting the type of 'np' (line 141)
    np_203548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 9), 'np', False)
    # Obtaining the member 'max' of a type (line 141)
    max_203549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 9), np_203548, 'max')
    # Calling max(args, kwargs) (line 141)
    max_call_result_203556 = invoke(stypy.reporting.localization.Localization(__file__, 141, 9), max_203549, *[result_add_203554], **kwargs_203555)
    
    # Assigning a type to the variable 'ub' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'ub', max_call_result_203556)
    
    # Obtaining an instance of the builtin type 'tuple' (line 143)
    tuple_203557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 143)
    # Adding element type (line 143)
    # Getting the type of 'lb' (line 143)
    lb_203558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 11), 'lb')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 11), tuple_203557, lb_203558)
    # Adding element type (line 143)
    # Getting the type of 'ub' (line 143)
    ub_203559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 15), 'ub')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 11), tuple_203557, ub_203559)
    
    # Assigning a type to the variable 'stypy_return_type' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type', tuple_203557)
    
    # ################# End of 'gershgorin_bounds(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gershgorin_bounds' in the type store
    # Getting the type of 'stypy_return_type' (line 125)
    stypy_return_type_203560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_203560)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gershgorin_bounds'
    return stypy_return_type_203560

# Assigning a type to the variable 'gershgorin_bounds' (line 125)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 0), 'gershgorin_bounds', gershgorin_bounds)

@norecursion
def singular_leading_submatrix(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'singular_leading_submatrix'
    module_type_store = module_type_store.open_function_context('singular_leading_submatrix', 146, 0, False)
    
    # Passed parameters checking function
    singular_leading_submatrix.stypy_localization = localization
    singular_leading_submatrix.stypy_type_of_self = None
    singular_leading_submatrix.stypy_type_store = module_type_store
    singular_leading_submatrix.stypy_function_name = 'singular_leading_submatrix'
    singular_leading_submatrix.stypy_param_names_list = ['A', 'U', 'k']
    singular_leading_submatrix.stypy_varargs_param_name = None
    singular_leading_submatrix.stypy_kwargs_param_name = None
    singular_leading_submatrix.stypy_call_defaults = defaults
    singular_leading_submatrix.stypy_call_varargs = varargs
    singular_leading_submatrix.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'singular_leading_submatrix', ['A', 'U', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'singular_leading_submatrix', localization, ['A', 'U', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'singular_leading_submatrix(...)' code ##################

    str_203561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, (-1)), 'str', '\n    Compute term that makes the leading ``k`` by ``k``\n    submatrix from ``A`` singular.\n\n    Parameters\n    ----------\n    A : ndarray\n        Symmetric matrix that is not positive definite.\n    U : ndarray\n        Upper triangular matrix resulting of an incomplete\n        Cholesky decomposition of matrix ``A``.\n    k : int\n        Positive integer such that the leading k by k submatrix from\n        `A` is the first non-positive definite leading submatrix.\n\n    Returns\n    -------\n    delta : float\n        Amout that should be added to the element (k, k) of the\n        leading k by k submatrix of ``A`` to make it singular.\n    v : ndarray\n        A vector such that ``v.T B v = 0``. Where B is the matrix A after\n        ``delta`` is added to its element (k, k).\n    ')
    
    # Assigning a BinOp to a Name (line 173):
    
    # Assigning a BinOp to a Name (line 173):
    
    # Call to sum(...): (line 173)
    # Processing the call arguments (line 173)
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 173)
    k_203564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 22), 'k', False)
    int_203565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 24), 'int')
    # Applying the binary operator '-' (line 173)
    result_sub_203566 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 22), '-', k_203564, int_203565)
    
    slice_203567 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 173, 19), None, result_sub_203566, None)
    # Getting the type of 'k' (line 173)
    k_203568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 27), 'k', False)
    int_203569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 29), 'int')
    # Applying the binary operator '-' (line 173)
    result_sub_203570 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 27), '-', k_203568, int_203569)
    
    # Getting the type of 'U' (line 173)
    U_203571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 19), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 173)
    getitem___203572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 19), U_203571, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 173)
    subscript_call_result_203573 = invoke(stypy.reporting.localization.Localization(__file__, 173, 19), getitem___203572, (slice_203567, result_sub_203570))
    
    int_203574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 33), 'int')
    # Applying the binary operator '**' (line 173)
    result_pow_203575 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 19), '**', subscript_call_result_203573, int_203574)
    
    # Processing the call keyword arguments (line 173)
    kwargs_203576 = {}
    # Getting the type of 'np' (line 173)
    np_203562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'np', False)
    # Obtaining the member 'sum' of a type (line 173)
    sum_203563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 12), np_203562, 'sum')
    # Calling sum(args, kwargs) (line 173)
    sum_call_result_203577 = invoke(stypy.reporting.localization.Localization(__file__, 173, 12), sum_203563, *[result_pow_203575], **kwargs_203576)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 173)
    tuple_203578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 173)
    # Adding element type (line 173)
    # Getting the type of 'k' (line 173)
    k_203579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 40), 'k')
    int_203580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 42), 'int')
    # Applying the binary operator '-' (line 173)
    result_sub_203581 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 40), '-', k_203579, int_203580)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 40), tuple_203578, result_sub_203581)
    # Adding element type (line 173)
    # Getting the type of 'k' (line 173)
    k_203582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 45), 'k')
    int_203583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 47), 'int')
    # Applying the binary operator '-' (line 173)
    result_sub_203584 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 45), '-', k_203582, int_203583)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 40), tuple_203578, result_sub_203584)
    
    # Getting the type of 'A' (line 173)
    A_203585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 38), 'A')
    # Obtaining the member '__getitem__' of a type (line 173)
    getitem___203586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 38), A_203585, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 173)
    subscript_call_result_203587 = invoke(stypy.reporting.localization.Localization(__file__, 173, 38), getitem___203586, tuple_203578)
    
    # Applying the binary operator '-' (line 173)
    result_sub_203588 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 12), '-', sum_call_result_203577, subscript_call_result_203587)
    
    # Assigning a type to the variable 'delta' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'delta', result_sub_203588)
    
    # Assigning a Call to a Name (line 175):
    
    # Assigning a Call to a Name (line 175):
    
    # Call to len(...): (line 175)
    # Processing the call arguments (line 175)
    # Getting the type of 'A' (line 175)
    A_203590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'A', False)
    # Processing the call keyword arguments (line 175)
    kwargs_203591 = {}
    # Getting the type of 'len' (line 175)
    len_203589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'len', False)
    # Calling len(args, kwargs) (line 175)
    len_call_result_203592 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), len_203589, *[A_203590], **kwargs_203591)
    
    # Assigning a type to the variable 'n' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'n', len_call_result_203592)
    
    # Assigning a Call to a Name (line 178):
    
    # Assigning a Call to a Name (line 178):
    
    # Call to zeros(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'n' (line 178)
    n_203595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 17), 'n', False)
    # Processing the call keyword arguments (line 178)
    kwargs_203596 = {}
    # Getting the type of 'np' (line 178)
    np_203593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 178)
    zeros_203594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), np_203593, 'zeros')
    # Calling zeros(args, kwargs) (line 178)
    zeros_call_result_203597 = invoke(stypy.reporting.localization.Localization(__file__, 178, 8), zeros_203594, *[n_203595], **kwargs_203596)
    
    # Assigning a type to the variable 'v' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'v', zeros_call_result_203597)
    
    # Assigning a Num to a Subscript (line 179):
    
    # Assigning a Num to a Subscript (line 179):
    int_203598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 13), 'int')
    # Getting the type of 'v' (line 179)
    v_203599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'v')
    # Getting the type of 'k' (line 179)
    k_203600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 6), 'k')
    int_203601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 8), 'int')
    # Applying the binary operator '-' (line 179)
    result_sub_203602 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 6), '-', k_203600, int_203601)
    
    # Storing an element on a container (line 179)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 4), v_203599, (result_sub_203602, int_203598))
    
    
    # Getting the type of 'k' (line 182)
    k_203603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 7), 'k')
    int_203604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 12), 'int')
    # Applying the binary operator '!=' (line 182)
    result_ne_203605 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 7), '!=', k_203603, int_203604)
    
    # Testing the type of an if condition (line 182)
    if_condition_203606 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 182, 4), result_ne_203605)
    # Assigning a type to the variable 'if_condition_203606' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'if_condition_203606', if_condition_203606)
    # SSA begins for if statement (line 182)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 183):
    
    # Assigning a Call to a Subscript (line 183):
    
    # Call to solve_triangular(...): (line 183)
    # Processing the call arguments (line 183)
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 183)
    k_203608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 38), 'k', False)
    int_203609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 40), 'int')
    # Applying the binary operator '-' (line 183)
    result_sub_203610 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 38), '-', k_203608, int_203609)
    
    slice_203611 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 183, 35), None, result_sub_203610, None)
    # Getting the type of 'k' (line 183)
    k_203612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 44), 'k', False)
    int_203613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 46), 'int')
    # Applying the binary operator '-' (line 183)
    result_sub_203614 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 44), '-', k_203612, int_203613)
    
    slice_203615 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 183, 35), None, result_sub_203614, None)
    # Getting the type of 'U' (line 183)
    U_203616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 35), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 183)
    getitem___203617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 35), U_203616, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 183)
    subscript_call_result_203618 = invoke(stypy.reporting.localization.Localization(__file__, 183, 35), getitem___203617, (slice_203611, slice_203615))
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 183)
    k_203619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 54), 'k', False)
    int_203620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 56), 'int')
    # Applying the binary operator '-' (line 183)
    result_sub_203621 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 54), '-', k_203619, int_203620)
    
    slice_203622 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 183, 51), None, result_sub_203621, None)
    # Getting the type of 'k' (line 183)
    k_203623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 59), 'k', False)
    int_203624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 61), 'int')
    # Applying the binary operator '-' (line 183)
    result_sub_203625 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 59), '-', k_203623, int_203624)
    
    # Getting the type of 'U' (line 183)
    U_203626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 51), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 183)
    getitem___203627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 51), U_203626, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 183)
    subscript_call_result_203628 = invoke(stypy.reporting.localization.Localization(__file__, 183, 51), getitem___203627, (slice_203622, result_sub_203625))
    
    # Applying the 'usub' unary operator (line 183)
    result___neg___203629 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 50), 'usub', subscript_call_result_203628)
    
    # Processing the call keyword arguments (line 183)
    kwargs_203630 = {}
    # Getting the type of 'solve_triangular' (line 183)
    solve_triangular_203607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 18), 'solve_triangular', False)
    # Calling solve_triangular(args, kwargs) (line 183)
    solve_triangular_call_result_203631 = invoke(stypy.reporting.localization.Localization(__file__, 183, 18), solve_triangular_203607, *[subscript_call_result_203618, result___neg___203629], **kwargs_203630)
    
    # Getting the type of 'v' (line 183)
    v_203632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'v')
    # Getting the type of 'k' (line 183)
    k_203633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 11), 'k')
    int_203634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 13), 'int')
    # Applying the binary operator '-' (line 183)
    result_sub_203635 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 11), '-', k_203633, int_203634)
    
    slice_203636 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 183, 8), None, result_sub_203635, None)
    # Storing an element on a container (line 183)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 8), v_203632, (slice_203636, solve_triangular_call_result_203631))
    # SSA join for if statement (line 182)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 185)
    tuple_203637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 185)
    # Adding element type (line 185)
    # Getting the type of 'delta' (line 185)
    delta_203638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 11), 'delta')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 11), tuple_203637, delta_203638)
    # Adding element type (line 185)
    # Getting the type of 'v' (line 185)
    v_203639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 18), 'v')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 11), tuple_203637, v_203639)
    
    # Assigning a type to the variable 'stypy_return_type' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'stypy_return_type', tuple_203637)
    
    # ################# End of 'singular_leading_submatrix(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'singular_leading_submatrix' in the type store
    # Getting the type of 'stypy_return_type' (line 146)
    stypy_return_type_203640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_203640)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'singular_leading_submatrix'
    return stypy_return_type_203640

# Assigning a type to the variable 'singular_leading_submatrix' (line 146)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 0), 'singular_leading_submatrix', singular_leading_submatrix)
# Declaration of the 'IterativeSubproblem' class
# Getting the type of 'BaseQuadraticSubproblem' (line 188)
BaseQuadraticSubproblem_203641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 26), 'BaseQuadraticSubproblem')

class IterativeSubproblem(BaseQuadraticSubproblem_203641, ):
    str_203642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, (-1)), 'str', 'Quadratic subproblem solved by nearly exact iterative method.\n\n    Notes\n    -----\n    This subproblem solver was based on [1]_, [2]_ and [3]_,\n    which implement similar algorithms. The algorithm is basically\n    that of [1]_ but ideas from [2]_ and [3]_ were also used.\n\n    References\n    ----------\n    .. [1] A.R. Conn, N.I. Gould, and P.L. Toint, "Trust region methods",\n           Siam, pp. 169-200, 2000.\n    .. [2] J. Nocedal and  S. Wright, "Numerical optimization",\n           Springer Science & Business Media. pp. 83-91, 2006.\n    .. [3] J.J. More and D.C. Sorensen, "Computing a trust region step",\n           SIAM Journal on Scientific and Statistical Computing, vol. 4(3),\n           pp. 553-572, 1983.\n    ')
    
    # Assigning a Num to a Name (line 211):
    
    # Assigning a Attribute to a Name (line 213):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 215)
        None_203643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 48), 'None')
        float_203644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 24), 'float')
        float_203645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 36), 'float')
        defaults = [None_203643, float_203644, float_203645]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 215, 4, False)
        # Assigning a type to the variable 'self' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IterativeSubproblem.__init__', ['x', 'fun', 'jac', 'hess', 'hessp', 'k_easy', 'k_hard'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['x', 'fun', 'jac', 'hess', 'hessp', 'k_easy', 'k_hard'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'x' (line 218)
        x_203652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 50), 'x', False)
        # Getting the type of 'fun' (line 218)
        fun_203653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 53), 'fun', False)
        # Getting the type of 'jac' (line 218)
        jac_203654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 58), 'jac', False)
        # Getting the type of 'hess' (line 218)
        hess_203655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 63), 'hess', False)
        # Processing the call keyword arguments (line 218)
        kwargs_203656 = {}
        
        # Call to super(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'IterativeSubproblem' (line 218)
        IterativeSubproblem_203647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 14), 'IterativeSubproblem', False)
        # Getting the type of 'self' (line 218)
        self_203648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 35), 'self', False)
        # Processing the call keyword arguments (line 218)
        kwargs_203649 = {}
        # Getting the type of 'super' (line 218)
        super_203646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'super', False)
        # Calling super(args, kwargs) (line 218)
        super_call_result_203650 = invoke(stypy.reporting.localization.Localization(__file__, 218, 8), super_203646, *[IterativeSubproblem_203647, self_203648], **kwargs_203649)
        
        # Obtaining the member '__init__' of a type (line 218)
        init___203651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), super_call_result_203650, '__init__')
        # Calling __init__(args, kwargs) (line 218)
        init___call_result_203657 = invoke(stypy.reporting.localization.Localization(__file__, 218, 8), init___203651, *[x_203652, fun_203653, jac_203654, hess_203655], **kwargs_203656)
        
        
        # Assigning a Num to a Attribute (line 226):
        
        # Assigning a Num to a Attribute (line 226):
        int_203658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 34), 'int')
        # Getting the type of 'self' (line 226)
        self_203659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'self')
        # Setting the type of the member 'previous_tr_radius' of a type (line 226)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), self_203659, 'previous_tr_radius', int_203658)
        
        # Assigning a Name to a Attribute (line 227):
        
        # Assigning a Name to a Attribute (line 227):
        # Getting the type of 'None' (line 227)
        None_203660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 25), 'None')
        # Getting the type of 'self' (line 227)
        self_203661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'self')
        # Setting the type of the member 'lambda_lb' of a type (line 227)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), self_203661, 'lambda_lb', None_203660)
        
        # Assigning a Num to a Attribute (line 229):
        
        # Assigning a Num to a Attribute (line 229):
        int_203662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 21), 'int')
        # Getting the type of 'self' (line 229)
        self_203663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'self')
        # Setting the type of the member 'niter' of a type (line 229)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), self_203663, 'niter', int_203662)
        
        # Assigning a Name to a Attribute (line 235):
        
        # Assigning a Name to a Attribute (line 235):
        # Getting the type of 'k_easy' (line 235)
        k_easy_203664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 22), 'k_easy')
        # Getting the type of 'self' (line 235)
        self_203665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'self')
        # Setting the type of the member 'k_easy' of a type (line 235)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), self_203665, 'k_easy', k_easy_203664)
        
        # Assigning a Name to a Attribute (line 236):
        
        # Assigning a Name to a Attribute (line 236):
        # Getting the type of 'k_hard' (line 236)
        k_hard_203666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 22), 'k_hard')
        # Getting the type of 'self' (line 236)
        self_203667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'self')
        # Setting the type of the member 'k_hard' of a type (line 236)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 8), self_203667, 'k_hard', k_hard_203666)
        
        # Assigning a Call to a Tuple (line 241):
        
        # Assigning a Subscript to a Name (line 241):
        
        # Obtaining the type of the subscript
        int_203668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 8), 'int')
        
        # Call to get_lapack_funcs(...): (line 241)
        # Processing the call arguments (line 241)
        
        # Obtaining an instance of the builtin type 'tuple' (line 241)
        tuple_203670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 241)
        # Adding element type (line 241)
        str_203671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 43), 'str', 'potrf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 43), tuple_203670, str_203671)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 241)
        tuple_203672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 241)
        # Adding element type (line 241)
        # Getting the type of 'self' (line 241)
        self_203673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 55), 'self', False)
        # Obtaining the member 'hess' of a type (line 241)
        hess_203674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 55), self_203673, 'hess')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 55), tuple_203672, hess_203674)
        
        # Processing the call keyword arguments (line 241)
        kwargs_203675 = {}
        # Getting the type of 'get_lapack_funcs' (line 241)
        get_lapack_funcs_203669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 25), 'get_lapack_funcs', False)
        # Calling get_lapack_funcs(args, kwargs) (line 241)
        get_lapack_funcs_call_result_203676 = invoke(stypy.reporting.localization.Localization(__file__, 241, 25), get_lapack_funcs_203669, *[tuple_203670, tuple_203672], **kwargs_203675)
        
        # Obtaining the member '__getitem__' of a type (line 241)
        getitem___203677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 8), get_lapack_funcs_call_result_203676, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 241)
        subscript_call_result_203678 = invoke(stypy.reporting.localization.Localization(__file__, 241, 8), getitem___203677, int_203668)
        
        # Assigning a type to the variable 'tuple_var_assignment_203282' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'tuple_var_assignment_203282', subscript_call_result_203678)
        
        # Assigning a Name to a Attribute (line 241):
        # Getting the type of 'tuple_var_assignment_203282' (line 241)
        tuple_var_assignment_203282_203679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'tuple_var_assignment_203282')
        # Getting the type of 'self' (line 241)
        self_203680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'self')
        # Setting the type of the member 'cholesky' of a type (line 241)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 8), self_203680, 'cholesky', tuple_var_assignment_203282_203679)
        
        # Assigning a Call to a Attribute (line 244):
        
        # Assigning a Call to a Attribute (line 244):
        
        # Call to len(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 'self' (line 244)
        self_203682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 29), 'self', False)
        # Obtaining the member 'hess' of a type (line 244)
        hess_203683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 29), self_203682, 'hess')
        # Processing the call keyword arguments (line 244)
        kwargs_203684 = {}
        # Getting the type of 'len' (line 244)
        len_203681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 25), 'len', False)
        # Calling len(args, kwargs) (line 244)
        len_call_result_203685 = invoke(stypy.reporting.localization.Localization(__file__, 244, 25), len_203681, *[hess_203683], **kwargs_203684)
        
        # Getting the type of 'self' (line 244)
        self_203686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'self')
        # Setting the type of the member 'dimension' of a type (line 244)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), self_203686, 'dimension', len_call_result_203685)
        
        # Assigning a Call to a Tuple (line 245):
        
        # Assigning a Subscript to a Name (line 245):
        
        # Obtaining the type of the subscript
        int_203687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 8), 'int')
        
        # Call to gershgorin_bounds(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'self' (line 246)
        self_203689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 56), 'self', False)
        # Obtaining the member 'hess' of a type (line 246)
        hess_203690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 56), self_203689, 'hess')
        # Processing the call keyword arguments (line 246)
        kwargs_203691 = {}
        # Getting the type of 'gershgorin_bounds' (line 246)
        gershgorin_bounds_203688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 38), 'gershgorin_bounds', False)
        # Calling gershgorin_bounds(args, kwargs) (line 246)
        gershgorin_bounds_call_result_203692 = invoke(stypy.reporting.localization.Localization(__file__, 246, 38), gershgorin_bounds_203688, *[hess_203690], **kwargs_203691)
        
        # Obtaining the member '__getitem__' of a type (line 245)
        getitem___203693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 8), gershgorin_bounds_call_result_203692, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 245)
        subscript_call_result_203694 = invoke(stypy.reporting.localization.Localization(__file__, 245, 8), getitem___203693, int_203687)
        
        # Assigning a type to the variable 'tuple_var_assignment_203283' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'tuple_var_assignment_203283', subscript_call_result_203694)
        
        # Assigning a Subscript to a Name (line 245):
        
        # Obtaining the type of the subscript
        int_203695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 8), 'int')
        
        # Call to gershgorin_bounds(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'self' (line 246)
        self_203697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 56), 'self', False)
        # Obtaining the member 'hess' of a type (line 246)
        hess_203698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 56), self_203697, 'hess')
        # Processing the call keyword arguments (line 246)
        kwargs_203699 = {}
        # Getting the type of 'gershgorin_bounds' (line 246)
        gershgorin_bounds_203696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 38), 'gershgorin_bounds', False)
        # Calling gershgorin_bounds(args, kwargs) (line 246)
        gershgorin_bounds_call_result_203700 = invoke(stypy.reporting.localization.Localization(__file__, 246, 38), gershgorin_bounds_203696, *[hess_203698], **kwargs_203699)
        
        # Obtaining the member '__getitem__' of a type (line 245)
        getitem___203701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 8), gershgorin_bounds_call_result_203700, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 245)
        subscript_call_result_203702 = invoke(stypy.reporting.localization.Localization(__file__, 245, 8), getitem___203701, int_203695)
        
        # Assigning a type to the variable 'tuple_var_assignment_203284' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'tuple_var_assignment_203284', subscript_call_result_203702)
        
        # Assigning a Name to a Attribute (line 245):
        # Getting the type of 'tuple_var_assignment_203283' (line 245)
        tuple_var_assignment_203283_203703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'tuple_var_assignment_203283')
        # Getting the type of 'self' (line 245)
        self_203704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'self')
        # Setting the type of the member 'hess_gershgorin_lb' of a type (line 245)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 8), self_203704, 'hess_gershgorin_lb', tuple_var_assignment_203283_203703)
        
        # Assigning a Name to a Attribute (line 245):
        # Getting the type of 'tuple_var_assignment_203284' (line 245)
        tuple_var_assignment_203284_203705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'tuple_var_assignment_203284')
        # Getting the type of 'self' (line 246)
        self_203706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'self')
        # Setting the type of the member 'hess_gershgorin_ub' of a type (line 246)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 12), self_203706, 'hess_gershgorin_ub', tuple_var_assignment_203284_203705)
        
        # Assigning a Call to a Attribute (line 247):
        
        # Assigning a Call to a Attribute (line 247):
        
        # Call to norm(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'self' (line 247)
        self_203708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 29), 'self', False)
        # Obtaining the member 'hess' of a type (line 247)
        hess_203709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 29), self_203708, 'hess')
        # Getting the type of 'np' (line 247)
        np_203710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 40), 'np', False)
        # Obtaining the member 'Inf' of a type (line 247)
        Inf_203711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 40), np_203710, 'Inf')
        # Processing the call keyword arguments (line 247)
        kwargs_203712 = {}
        # Getting the type of 'norm' (line 247)
        norm_203707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 24), 'norm', False)
        # Calling norm(args, kwargs) (line 247)
        norm_call_result_203713 = invoke(stypy.reporting.localization.Localization(__file__, 247, 24), norm_203707, *[hess_203709, Inf_203711], **kwargs_203712)
        
        # Getting the type of 'self' (line 247)
        self_203714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'self')
        # Setting the type of the member 'hess_inf' of a type (line 247)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), self_203714, 'hess_inf', norm_call_result_203713)
        
        # Assigning a Call to a Attribute (line 248):
        
        # Assigning a Call to a Attribute (line 248):
        
        # Call to norm(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'self' (line 248)
        self_203716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 29), 'self', False)
        # Obtaining the member 'hess' of a type (line 248)
        hess_203717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 29), self_203716, 'hess')
        str_203718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 40), 'str', 'fro')
        # Processing the call keyword arguments (line 248)
        kwargs_203719 = {}
        # Getting the type of 'norm' (line 248)
        norm_203715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 24), 'norm', False)
        # Calling norm(args, kwargs) (line 248)
        norm_call_result_203720 = invoke(stypy.reporting.localization.Localization(__file__, 248, 24), norm_203715, *[hess_203717, str_203718], **kwargs_203719)
        
        # Getting the type of 'self' (line 248)
        self_203721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'self')
        # Setting the type of the member 'hess_fro' of a type (line 248)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), self_203721, 'hess_fro', norm_call_result_203720)
        
        # Assigning a BinOp to a Attribute (line 254):
        
        # Assigning a BinOp to a Attribute (line 254):
        # Getting the type of 'self' (line 254)
        self_203722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 29), 'self')
        # Obtaining the member 'dimension' of a type (line 254)
        dimension_203723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 29), self_203722, 'dimension')
        # Getting the type of 'self' (line 254)
        self_203724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 46), 'self')
        # Obtaining the member 'EPS' of a type (line 254)
        EPS_203725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 46), self_203724, 'EPS')
        # Applying the binary operator '*' (line 254)
        result_mul_203726 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 29), '*', dimension_203723, EPS_203725)
        
        # Getting the type of 'self' (line 254)
        self_203727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 57), 'self')
        # Obtaining the member 'hess_inf' of a type (line 254)
        hess_inf_203728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 57), self_203727, 'hess_inf')
        # Applying the binary operator '*' (line 254)
        result_mul_203729 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 55), '*', result_mul_203726, hess_inf_203728)
        
        # Getting the type of 'self' (line 254)
        self_203730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'self')
        # Setting the type of the member 'CLOSE_TO_ZERO' of a type (line 254)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 8), self_203730, 'CLOSE_TO_ZERO', result_mul_203729)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _initial_values(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_initial_values'
        module_type_store = module_type_store.open_function_context('_initial_values', 256, 4, False)
        # Assigning a type to the variable 'self' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IterativeSubproblem._initial_values.__dict__.__setitem__('stypy_localization', localization)
        IterativeSubproblem._initial_values.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IterativeSubproblem._initial_values.__dict__.__setitem__('stypy_type_store', module_type_store)
        IterativeSubproblem._initial_values.__dict__.__setitem__('stypy_function_name', 'IterativeSubproblem._initial_values')
        IterativeSubproblem._initial_values.__dict__.__setitem__('stypy_param_names_list', ['tr_radius'])
        IterativeSubproblem._initial_values.__dict__.__setitem__('stypy_varargs_param_name', None)
        IterativeSubproblem._initial_values.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IterativeSubproblem._initial_values.__dict__.__setitem__('stypy_call_defaults', defaults)
        IterativeSubproblem._initial_values.__dict__.__setitem__('stypy_call_varargs', varargs)
        IterativeSubproblem._initial_values.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IterativeSubproblem._initial_values.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IterativeSubproblem._initial_values', ['tr_radius'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_initial_values', localization, ['tr_radius'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_initial_values(...)' code ##################

        str_203731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, (-1)), 'str', 'Given a trust radius, return a good initial guess for\n        the damping factor, the lower bound and the upper bound.\n        The values were chosen accordingly to the guidelines on\n        section 7.3.8 (p. 192) from [1]_.\n        ')
        
        # Assigning a Call to a Name (line 264):
        
        # Assigning a Call to a Name (line 264):
        
        # Call to max(...): (line 264)
        # Processing the call arguments (line 264)
        int_203733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 24), 'int')
        # Getting the type of 'self' (line 264)
        self_203734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 27), 'self', False)
        # Obtaining the member 'jac_mag' of a type (line 264)
        jac_mag_203735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 27), self_203734, 'jac_mag')
        # Getting the type of 'tr_radius' (line 264)
        tr_radius_203736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 40), 'tr_radius', False)
        # Applying the binary operator 'div' (line 264)
        result_div_203737 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 27), 'div', jac_mag_203735, tr_radius_203736)
        
        
        # Call to min(...): (line 264)
        # Processing the call arguments (line 264)
        
        # Getting the type of 'self' (line 264)
        self_203739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 57), 'self', False)
        # Obtaining the member 'hess_gershgorin_lb' of a type (line 264)
        hess_gershgorin_lb_203740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 57), self_203739, 'hess_gershgorin_lb')
        # Applying the 'usub' unary operator (line 264)
        result___neg___203741 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 56), 'usub', hess_gershgorin_lb_203740)
        
        # Getting the type of 'self' (line 265)
        self_203742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 56), 'self', False)
        # Obtaining the member 'hess_fro' of a type (line 265)
        hess_fro_203743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 56), self_203742, 'hess_fro')
        # Getting the type of 'self' (line 266)
        self_203744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 56), 'self', False)
        # Obtaining the member 'hess_inf' of a type (line 266)
        hess_inf_203745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 56), self_203744, 'hess_inf')
        # Processing the call keyword arguments (line 264)
        kwargs_203746 = {}
        # Getting the type of 'min' (line 264)
        min_203738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 52), 'min', False)
        # Calling min(args, kwargs) (line 264)
        min_call_result_203747 = invoke(stypy.reporting.localization.Localization(__file__, 264, 52), min_203738, *[result___neg___203741, hess_fro_203743, hess_inf_203745], **kwargs_203746)
        
        # Applying the binary operator '+' (line 264)
        result_add_203748 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 27), '+', result_div_203737, min_call_result_203747)
        
        # Processing the call keyword arguments (line 264)
        kwargs_203749 = {}
        # Getting the type of 'max' (line 264)
        max_203732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 20), 'max', False)
        # Calling max(args, kwargs) (line 264)
        max_call_result_203750 = invoke(stypy.reporting.localization.Localization(__file__, 264, 20), max_203732, *[int_203733, result_add_203748], **kwargs_203749)
        
        # Assigning a type to the variable 'lambda_ub' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'lambda_ub', max_call_result_203750)
        
        # Assigning a Call to a Name (line 269):
        
        # Assigning a Call to a Name (line 269):
        
        # Call to max(...): (line 269)
        # Processing the call arguments (line 269)
        int_203752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 24), 'int')
        
        
        # Call to min(...): (line 269)
        # Processing the call arguments (line 269)
        
        # Call to diagonal(...): (line 269)
        # Processing the call keyword arguments (line 269)
        kwargs_203757 = {}
        # Getting the type of 'self' (line 269)
        self_203754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 32), 'self', False)
        # Obtaining the member 'hess' of a type (line 269)
        hess_203755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 32), self_203754, 'hess')
        # Obtaining the member 'diagonal' of a type (line 269)
        diagonal_203756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 32), hess_203755, 'diagonal')
        # Calling diagonal(args, kwargs) (line 269)
        diagonal_call_result_203758 = invoke(stypy.reporting.localization.Localization(__file__, 269, 32), diagonal_203756, *[], **kwargs_203757)
        
        # Processing the call keyword arguments (line 269)
        kwargs_203759 = {}
        # Getting the type of 'min' (line 269)
        min_203753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 28), 'min', False)
        # Calling min(args, kwargs) (line 269)
        min_call_result_203760 = invoke(stypy.reporting.localization.Localization(__file__, 269, 28), min_203753, *[diagonal_call_result_203758], **kwargs_203759)
        
        # Applying the 'usub' unary operator (line 269)
        result___neg___203761 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 27), 'usub', min_call_result_203760)
        
        # Getting the type of 'self' (line 270)
        self_203762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 24), 'self', False)
        # Obtaining the member 'jac_mag' of a type (line 270)
        jac_mag_203763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 24), self_203762, 'jac_mag')
        # Getting the type of 'tr_radius' (line 270)
        tr_radius_203764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 37), 'tr_radius', False)
        # Applying the binary operator 'div' (line 270)
        result_div_203765 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 24), 'div', jac_mag_203763, tr_radius_203764)
        
        
        # Call to min(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'self' (line 270)
        self_203767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 53), 'self', False)
        # Obtaining the member 'hess_gershgorin_ub' of a type (line 270)
        hess_gershgorin_ub_203768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 53), self_203767, 'hess_gershgorin_ub')
        # Getting the type of 'self' (line 271)
        self_203769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 53), 'self', False)
        # Obtaining the member 'hess_fro' of a type (line 271)
        hess_fro_203770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 53), self_203769, 'hess_fro')
        # Getting the type of 'self' (line 272)
        self_203771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 53), 'self', False)
        # Obtaining the member 'hess_inf' of a type (line 272)
        hess_inf_203772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 53), self_203771, 'hess_inf')
        # Processing the call keyword arguments (line 270)
        kwargs_203773 = {}
        # Getting the type of 'min' (line 270)
        min_203766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 49), 'min', False)
        # Calling min(args, kwargs) (line 270)
        min_call_result_203774 = invoke(stypy.reporting.localization.Localization(__file__, 270, 49), min_203766, *[hess_gershgorin_ub_203768, hess_fro_203770, hess_inf_203772], **kwargs_203773)
        
        # Applying the binary operator '-' (line 270)
        result_sub_203775 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 24), '-', result_div_203765, min_call_result_203774)
        
        # Processing the call keyword arguments (line 269)
        kwargs_203776 = {}
        # Getting the type of 'max' (line 269)
        max_203751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 20), 'max', False)
        # Calling max(args, kwargs) (line 269)
        max_call_result_203777 = invoke(stypy.reporting.localization.Localization(__file__, 269, 20), max_203751, *[int_203752, result___neg___203761, result_sub_203775], **kwargs_203776)
        
        # Assigning a type to the variable 'lambda_lb' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'lambda_lb', max_call_result_203777)
        
        
        # Getting the type of 'tr_radius' (line 275)
        tr_radius_203778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 11), 'tr_radius')
        # Getting the type of 'self' (line 275)
        self_203779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 23), 'self')
        # Obtaining the member 'previous_tr_radius' of a type (line 275)
        previous_tr_radius_203780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 23), self_203779, 'previous_tr_radius')
        # Applying the binary operator '<' (line 275)
        result_lt_203781 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 11), '<', tr_radius_203778, previous_tr_radius_203780)
        
        # Testing the type of an if condition (line 275)
        if_condition_203782 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 275, 8), result_lt_203781)
        # Assigning a type to the variable 'if_condition_203782' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'if_condition_203782', if_condition_203782)
        # SSA begins for if statement (line 275)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 276):
        
        # Assigning a Call to a Name (line 276):
        
        # Call to max(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'self' (line 276)
        self_203784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 28), 'self', False)
        # Obtaining the member 'lambda_lb' of a type (line 276)
        lambda_lb_203785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 28), self_203784, 'lambda_lb')
        # Getting the type of 'lambda_lb' (line 276)
        lambda_lb_203786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 44), 'lambda_lb', False)
        # Processing the call keyword arguments (line 276)
        kwargs_203787 = {}
        # Getting the type of 'max' (line 276)
        max_203783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 24), 'max', False)
        # Calling max(args, kwargs) (line 276)
        max_call_result_203788 = invoke(stypy.reporting.localization.Localization(__file__, 276, 24), max_203783, *[lambda_lb_203785, lambda_lb_203786], **kwargs_203787)
        
        # Assigning a type to the variable 'lambda_lb' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'lambda_lb', max_call_result_203788)
        # SSA join for if statement (line 275)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'lambda_lb' (line 279)
        lambda_lb_203789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 11), 'lambda_lb')
        int_203790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 24), 'int')
        # Applying the binary operator '==' (line 279)
        result_eq_203791 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 11), '==', lambda_lb_203789, int_203790)
        
        # Testing the type of an if condition (line 279)
        if_condition_203792 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 8), result_eq_203791)
        # Assigning a type to the variable 'if_condition_203792' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'if_condition_203792', if_condition_203792)
        # SSA begins for if statement (line 279)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 280):
        
        # Assigning a Num to a Name (line 280):
        int_203793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 29), 'int')
        # Assigning a type to the variable 'lambda_initial' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'lambda_initial', int_203793)
        # SSA branch for the else part of an if statement (line 279)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 282):
        
        # Assigning a Call to a Name (line 282):
        
        # Call to max(...): (line 282)
        # Processing the call arguments (line 282)
        
        # Call to sqrt(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'lambda_lb' (line 282)
        lambda_lb_203797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 41), 'lambda_lb', False)
        # Getting the type of 'lambda_ub' (line 282)
        lambda_ub_203798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 53), 'lambda_ub', False)
        # Applying the binary operator '*' (line 282)
        result_mul_203799 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 41), '*', lambda_lb_203797, lambda_ub_203798)
        
        # Processing the call keyword arguments (line 282)
        kwargs_203800 = {}
        # Getting the type of 'np' (line 282)
        np_203795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 33), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 282)
        sqrt_203796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 33), np_203795, 'sqrt')
        # Calling sqrt(args, kwargs) (line 282)
        sqrt_call_result_203801 = invoke(stypy.reporting.localization.Localization(__file__, 282, 33), sqrt_203796, *[result_mul_203799], **kwargs_203800)
        
        # Getting the type of 'lambda_lb' (line 283)
        lambda_lb_203802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 33), 'lambda_lb', False)
        # Getting the type of 'self' (line 283)
        self_203803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 45), 'self', False)
        # Obtaining the member 'UPDATE_COEFF' of a type (line 283)
        UPDATE_COEFF_203804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 45), self_203803, 'UPDATE_COEFF')
        # Getting the type of 'lambda_ub' (line 283)
        lambda_ub_203805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 64), 'lambda_ub', False)
        # Getting the type of 'lambda_lb' (line 283)
        lambda_lb_203806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 74), 'lambda_lb', False)
        # Applying the binary operator '-' (line 283)
        result_sub_203807 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 64), '-', lambda_ub_203805, lambda_lb_203806)
        
        # Applying the binary operator '*' (line 283)
        result_mul_203808 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 45), '*', UPDATE_COEFF_203804, result_sub_203807)
        
        # Applying the binary operator '+' (line 283)
        result_add_203809 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 33), '+', lambda_lb_203802, result_mul_203808)
        
        # Processing the call keyword arguments (line 282)
        kwargs_203810 = {}
        # Getting the type of 'max' (line 282)
        max_203794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 29), 'max', False)
        # Calling max(args, kwargs) (line 282)
        max_call_result_203811 = invoke(stypy.reporting.localization.Localization(__file__, 282, 29), max_203794, *[sqrt_call_result_203801, result_add_203809], **kwargs_203810)
        
        # Assigning a type to the variable 'lambda_initial' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'lambda_initial', max_call_result_203811)
        # SSA join for if statement (line 279)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 285)
        tuple_203812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 285)
        # Adding element type (line 285)
        # Getting the type of 'lambda_initial' (line 285)
        lambda_initial_203813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 15), 'lambda_initial')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 15), tuple_203812, lambda_initial_203813)
        # Adding element type (line 285)
        # Getting the type of 'lambda_lb' (line 285)
        lambda_lb_203814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 31), 'lambda_lb')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 15), tuple_203812, lambda_lb_203814)
        # Adding element type (line 285)
        # Getting the type of 'lambda_ub' (line 285)
        lambda_ub_203815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 42), 'lambda_ub')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 15), tuple_203812, lambda_ub_203815)
        
        # Assigning a type to the variable 'stypy_return_type' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'stypy_return_type', tuple_203812)
        
        # ################# End of '_initial_values(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_initial_values' in the type store
        # Getting the type of 'stypy_return_type' (line 256)
        stypy_return_type_203816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_203816)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_initial_values'
        return stypy_return_type_203816


    @norecursion
    def solve(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'solve'
        module_type_store = module_type_store.open_function_context('solve', 287, 4, False)
        # Assigning a type to the variable 'self' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IterativeSubproblem.solve.__dict__.__setitem__('stypy_localization', localization)
        IterativeSubproblem.solve.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IterativeSubproblem.solve.__dict__.__setitem__('stypy_type_store', module_type_store)
        IterativeSubproblem.solve.__dict__.__setitem__('stypy_function_name', 'IterativeSubproblem.solve')
        IterativeSubproblem.solve.__dict__.__setitem__('stypy_param_names_list', ['tr_radius'])
        IterativeSubproblem.solve.__dict__.__setitem__('stypy_varargs_param_name', None)
        IterativeSubproblem.solve.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IterativeSubproblem.solve.__dict__.__setitem__('stypy_call_defaults', defaults)
        IterativeSubproblem.solve.__dict__.__setitem__('stypy_call_varargs', varargs)
        IterativeSubproblem.solve.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IterativeSubproblem.solve.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IterativeSubproblem.solve', ['tr_radius'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'solve', localization, ['tr_radius'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'solve(...)' code ##################

        str_203817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 8), 'str', 'Solve quadratic subproblem')
        
        # Assigning a Call to a Tuple (line 290):
        
        # Assigning a Subscript to a Name (line 290):
        
        # Obtaining the type of the subscript
        int_203818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 8), 'int')
        
        # Call to _initial_values(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'tr_radius' (line 290)
        tr_radius_203821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 68), 'tr_radius', False)
        # Processing the call keyword arguments (line 290)
        kwargs_203822 = {}
        # Getting the type of 'self' (line 290)
        self_203819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 47), 'self', False)
        # Obtaining the member '_initial_values' of a type (line 290)
        _initial_values_203820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 47), self_203819, '_initial_values')
        # Calling _initial_values(args, kwargs) (line 290)
        _initial_values_call_result_203823 = invoke(stypy.reporting.localization.Localization(__file__, 290, 47), _initial_values_203820, *[tr_radius_203821], **kwargs_203822)
        
        # Obtaining the member '__getitem__' of a type (line 290)
        getitem___203824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 8), _initial_values_call_result_203823, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 290)
        subscript_call_result_203825 = invoke(stypy.reporting.localization.Localization(__file__, 290, 8), getitem___203824, int_203818)
        
        # Assigning a type to the variable 'tuple_var_assignment_203285' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'tuple_var_assignment_203285', subscript_call_result_203825)
        
        # Assigning a Subscript to a Name (line 290):
        
        # Obtaining the type of the subscript
        int_203826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 8), 'int')
        
        # Call to _initial_values(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'tr_radius' (line 290)
        tr_radius_203829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 68), 'tr_radius', False)
        # Processing the call keyword arguments (line 290)
        kwargs_203830 = {}
        # Getting the type of 'self' (line 290)
        self_203827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 47), 'self', False)
        # Obtaining the member '_initial_values' of a type (line 290)
        _initial_values_203828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 47), self_203827, '_initial_values')
        # Calling _initial_values(args, kwargs) (line 290)
        _initial_values_call_result_203831 = invoke(stypy.reporting.localization.Localization(__file__, 290, 47), _initial_values_203828, *[tr_radius_203829], **kwargs_203830)
        
        # Obtaining the member '__getitem__' of a type (line 290)
        getitem___203832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 8), _initial_values_call_result_203831, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 290)
        subscript_call_result_203833 = invoke(stypy.reporting.localization.Localization(__file__, 290, 8), getitem___203832, int_203826)
        
        # Assigning a type to the variable 'tuple_var_assignment_203286' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'tuple_var_assignment_203286', subscript_call_result_203833)
        
        # Assigning a Subscript to a Name (line 290):
        
        # Obtaining the type of the subscript
        int_203834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 8), 'int')
        
        # Call to _initial_values(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'tr_radius' (line 290)
        tr_radius_203837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 68), 'tr_radius', False)
        # Processing the call keyword arguments (line 290)
        kwargs_203838 = {}
        # Getting the type of 'self' (line 290)
        self_203835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 47), 'self', False)
        # Obtaining the member '_initial_values' of a type (line 290)
        _initial_values_203836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 47), self_203835, '_initial_values')
        # Calling _initial_values(args, kwargs) (line 290)
        _initial_values_call_result_203839 = invoke(stypy.reporting.localization.Localization(__file__, 290, 47), _initial_values_203836, *[tr_radius_203837], **kwargs_203838)
        
        # Obtaining the member '__getitem__' of a type (line 290)
        getitem___203840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 8), _initial_values_call_result_203839, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 290)
        subscript_call_result_203841 = invoke(stypy.reporting.localization.Localization(__file__, 290, 8), getitem___203840, int_203834)
        
        # Assigning a type to the variable 'tuple_var_assignment_203287' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'tuple_var_assignment_203287', subscript_call_result_203841)
        
        # Assigning a Name to a Name (line 290):
        # Getting the type of 'tuple_var_assignment_203285' (line 290)
        tuple_var_assignment_203285_203842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'tuple_var_assignment_203285')
        # Assigning a type to the variable 'lambda_current' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'lambda_current', tuple_var_assignment_203285_203842)
        
        # Assigning a Name to a Name (line 290):
        # Getting the type of 'tuple_var_assignment_203286' (line 290)
        tuple_var_assignment_203286_203843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'tuple_var_assignment_203286')
        # Assigning a type to the variable 'lambda_lb' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 24), 'lambda_lb', tuple_var_assignment_203286_203843)
        
        # Assigning a Name to a Name (line 290):
        # Getting the type of 'tuple_var_assignment_203287' (line 290)
        tuple_var_assignment_203287_203844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'tuple_var_assignment_203287')
        # Assigning a type to the variable 'lambda_ub' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 35), 'lambda_ub', tuple_var_assignment_203287_203844)
        
        # Assigning a Attribute to a Name (line 291):
        
        # Assigning a Attribute to a Name (line 291):
        # Getting the type of 'self' (line 291)
        self_203845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'self')
        # Obtaining the member 'dimension' of a type (line 291)
        dimension_203846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 12), self_203845, 'dimension')
        # Assigning a type to the variable 'n' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'n', dimension_203846)
        
        # Assigning a Name to a Name (line 292):
        
        # Assigning a Name to a Name (line 292):
        # Getting the type of 'True' (line 292)
        True_203847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 24), 'True')
        # Assigning a type to the variable 'hits_boundary' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'hits_boundary', True_203847)
        
        # Assigning a Name to a Name (line 293):
        
        # Assigning a Name to a Name (line 293):
        # Getting the type of 'False' (line 293)
        False_203848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 29), 'False')
        # Assigning a type to the variable 'already_factorized' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'already_factorized', False_203848)
        
        # Assigning a Num to a Attribute (line 294):
        
        # Assigning a Num to a Attribute (line 294):
        int_203849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 21), 'int')
        # Getting the type of 'self' (line 294)
        self_203850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'self')
        # Setting the type of the member 'niter' of a type (line 294)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), self_203850, 'niter', int_203849)
        
        # Getting the type of 'True' (line 296)
        True_203851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 14), 'True')
        # Testing the type of an if condition (line 296)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 296, 8), True_203851)
        # SSA begins for while statement (line 296)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Getting the type of 'already_factorized' (line 299)
        already_factorized_203852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 15), 'already_factorized')
        # Testing the type of an if condition (line 299)
        if_condition_203853 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 299, 12), already_factorized_203852)
        # Assigning a type to the variable 'if_condition_203853' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'if_condition_203853', if_condition_203853)
        # SSA begins for if statement (line 299)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 300):
        
        # Assigning a Name to a Name (line 300):
        # Getting the type of 'False' (line 300)
        False_203854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 37), 'False')
        # Assigning a type to the variable 'already_factorized' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'already_factorized', False_203854)
        # SSA branch for the else part of an if statement (line 299)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 302):
        
        # Assigning a BinOp to a Name (line 302):
        # Getting the type of 'self' (line 302)
        self_203855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 20), 'self')
        # Obtaining the member 'hess' of a type (line 302)
        hess_203856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 20), self_203855, 'hess')
        # Getting the type of 'lambda_current' (line 302)
        lambda_current_203857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 30), 'lambda_current')
        
        # Call to eye(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'n' (line 302)
        n_203860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 52), 'n', False)
        # Processing the call keyword arguments (line 302)
        kwargs_203861 = {}
        # Getting the type of 'np' (line 302)
        np_203858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 45), 'np', False)
        # Obtaining the member 'eye' of a type (line 302)
        eye_203859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 45), np_203858, 'eye')
        # Calling eye(args, kwargs) (line 302)
        eye_call_result_203862 = invoke(stypy.reporting.localization.Localization(__file__, 302, 45), eye_203859, *[n_203860], **kwargs_203861)
        
        # Applying the binary operator '*' (line 302)
        result_mul_203863 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 30), '*', lambda_current_203857, eye_call_result_203862)
        
        # Applying the binary operator '+' (line 302)
        result_add_203864 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 20), '+', hess_203856, result_mul_203863)
        
        # Assigning a type to the variable 'H' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 16), 'H', result_add_203864)
        
        # Assigning a Call to a Tuple (line 303):
        
        # Assigning a Subscript to a Name (line 303):
        
        # Obtaining the type of the subscript
        int_203865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 16), 'int')
        
        # Call to cholesky(...): (line 303)
        # Processing the call arguments (line 303)
        # Getting the type of 'H' (line 303)
        H_203868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 40), 'H', False)
        # Processing the call keyword arguments (line 303)
        # Getting the type of 'False' (line 303)
        False_203869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 49), 'False', False)
        keyword_203870 = False_203869
        # Getting the type of 'False' (line 304)
        False_203871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 52), 'False', False)
        keyword_203872 = False_203871
        # Getting the type of 'True' (line 305)
        True_203873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 46), 'True', False)
        keyword_203874 = True_203873
        kwargs_203875 = {'lower': keyword_203870, 'overwrite_a': keyword_203872, 'clean': keyword_203874}
        # Getting the type of 'self' (line 303)
        self_203866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 26), 'self', False)
        # Obtaining the member 'cholesky' of a type (line 303)
        cholesky_203867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 26), self_203866, 'cholesky')
        # Calling cholesky(args, kwargs) (line 303)
        cholesky_call_result_203876 = invoke(stypy.reporting.localization.Localization(__file__, 303, 26), cholesky_203867, *[H_203868], **kwargs_203875)
        
        # Obtaining the member '__getitem__' of a type (line 303)
        getitem___203877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 16), cholesky_call_result_203876, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 303)
        subscript_call_result_203878 = invoke(stypy.reporting.localization.Localization(__file__, 303, 16), getitem___203877, int_203865)
        
        # Assigning a type to the variable 'tuple_var_assignment_203288' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'tuple_var_assignment_203288', subscript_call_result_203878)
        
        # Assigning a Subscript to a Name (line 303):
        
        # Obtaining the type of the subscript
        int_203879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 16), 'int')
        
        # Call to cholesky(...): (line 303)
        # Processing the call arguments (line 303)
        # Getting the type of 'H' (line 303)
        H_203882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 40), 'H', False)
        # Processing the call keyword arguments (line 303)
        # Getting the type of 'False' (line 303)
        False_203883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 49), 'False', False)
        keyword_203884 = False_203883
        # Getting the type of 'False' (line 304)
        False_203885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 52), 'False', False)
        keyword_203886 = False_203885
        # Getting the type of 'True' (line 305)
        True_203887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 46), 'True', False)
        keyword_203888 = True_203887
        kwargs_203889 = {'lower': keyword_203884, 'overwrite_a': keyword_203886, 'clean': keyword_203888}
        # Getting the type of 'self' (line 303)
        self_203880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 26), 'self', False)
        # Obtaining the member 'cholesky' of a type (line 303)
        cholesky_203881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 26), self_203880, 'cholesky')
        # Calling cholesky(args, kwargs) (line 303)
        cholesky_call_result_203890 = invoke(stypy.reporting.localization.Localization(__file__, 303, 26), cholesky_203881, *[H_203882], **kwargs_203889)
        
        # Obtaining the member '__getitem__' of a type (line 303)
        getitem___203891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 16), cholesky_call_result_203890, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 303)
        subscript_call_result_203892 = invoke(stypy.reporting.localization.Localization(__file__, 303, 16), getitem___203891, int_203879)
        
        # Assigning a type to the variable 'tuple_var_assignment_203289' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'tuple_var_assignment_203289', subscript_call_result_203892)
        
        # Assigning a Name to a Name (line 303):
        # Getting the type of 'tuple_var_assignment_203288' (line 303)
        tuple_var_assignment_203288_203893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'tuple_var_assignment_203288')
        # Assigning a type to the variable 'U' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'U', tuple_var_assignment_203288_203893)
        
        # Assigning a Name to a Name (line 303):
        # Getting the type of 'tuple_var_assignment_203289' (line 303)
        tuple_var_assignment_203289_203894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'tuple_var_assignment_203289')
        # Assigning a type to the variable 'info' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 19), 'info', tuple_var_assignment_203289_203894)
        # SSA join for if statement (line 299)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 307)
        self_203895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'self')
        # Obtaining the member 'niter' of a type (line 307)
        niter_203896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 12), self_203895, 'niter')
        int_203897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 26), 'int')
        # Applying the binary operator '+=' (line 307)
        result_iadd_203898 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 12), '+=', niter_203896, int_203897)
        # Getting the type of 'self' (line 307)
        self_203899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'self')
        # Setting the type of the member 'niter' of a type (line 307)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 12), self_203899, 'niter', result_iadd_203898)
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'info' (line 310)
        info_203900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 15), 'info')
        int_203901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 23), 'int')
        # Applying the binary operator '==' (line 310)
        result_eq_203902 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 15), '==', info_203900, int_203901)
        
        
        # Getting the type of 'self' (line 310)
        self_203903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 29), 'self')
        # Obtaining the member 'jac_mag' of a type (line 310)
        jac_mag_203904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 29), self_203903, 'jac_mag')
        # Getting the type of 'self' (line 310)
        self_203905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 44), 'self')
        # Obtaining the member 'CLOSE_TO_ZERO' of a type (line 310)
        CLOSE_TO_ZERO_203906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 44), self_203905, 'CLOSE_TO_ZERO')
        # Applying the binary operator '>' (line 310)
        result_gt_203907 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 29), '>', jac_mag_203904, CLOSE_TO_ZERO_203906)
        
        # Applying the binary operator 'and' (line 310)
        result_and_keyword_203908 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 15), 'and', result_eq_203902, result_gt_203907)
        
        # Testing the type of an if condition (line 310)
        if_condition_203909 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 310, 12), result_and_keyword_203908)
        # Assigning a type to the variable 'if_condition_203909' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'if_condition_203909', if_condition_203909)
        # SSA begins for if statement (line 310)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 314):
        
        # Assigning a Call to a Name (line 314):
        
        # Call to cho_solve(...): (line 314)
        # Processing the call arguments (line 314)
        
        # Obtaining an instance of the builtin type 'tuple' (line 314)
        tuple_203911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 314)
        # Adding element type (line 314)
        # Getting the type of 'U' (line 314)
        U_203912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 31), 'U', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 31), tuple_203911, U_203912)
        # Adding element type (line 314)
        # Getting the type of 'False' (line 314)
        False_203913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 34), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 31), tuple_203911, False_203913)
        
        
        # Getting the type of 'self' (line 314)
        self_203914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 43), 'self', False)
        # Obtaining the member 'jac' of a type (line 314)
        jac_203915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 43), self_203914, 'jac')
        # Applying the 'usub' unary operator (line 314)
        result___neg___203916 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 42), 'usub', jac_203915)
        
        # Processing the call keyword arguments (line 314)
        kwargs_203917 = {}
        # Getting the type of 'cho_solve' (line 314)
        cho_solve_203910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 20), 'cho_solve', False)
        # Calling cho_solve(args, kwargs) (line 314)
        cho_solve_call_result_203918 = invoke(stypy.reporting.localization.Localization(__file__, 314, 20), cho_solve_203910, *[tuple_203911, result___neg___203916], **kwargs_203917)
        
        # Assigning a type to the variable 'p' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 16), 'p', cho_solve_call_result_203918)
        
        # Assigning a Call to a Name (line 316):
        
        # Assigning a Call to a Name (line 316):
        
        # Call to norm(...): (line 316)
        # Processing the call arguments (line 316)
        # Getting the type of 'p' (line 316)
        p_203920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 30), 'p', False)
        # Processing the call keyword arguments (line 316)
        kwargs_203921 = {}
        # Getting the type of 'norm' (line 316)
        norm_203919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 25), 'norm', False)
        # Calling norm(args, kwargs) (line 316)
        norm_call_result_203922 = invoke(stypy.reporting.localization.Localization(__file__, 316, 25), norm_203919, *[p_203920], **kwargs_203921)
        
        # Assigning a type to the variable 'p_norm' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 16), 'p_norm', norm_call_result_203922)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'p_norm' (line 319)
        p_norm_203923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 19), 'p_norm')
        # Getting the type of 'tr_radius' (line 319)
        tr_radius_203924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 29), 'tr_radius')
        # Applying the binary operator '<=' (line 319)
        result_le_203925 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 19), '<=', p_norm_203923, tr_radius_203924)
        
        
        # Getting the type of 'lambda_current' (line 319)
        lambda_current_203926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 43), 'lambda_current')
        int_203927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 61), 'int')
        # Applying the binary operator '==' (line 319)
        result_eq_203928 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 43), '==', lambda_current_203926, int_203927)
        
        # Applying the binary operator 'and' (line 319)
        result_and_keyword_203929 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 19), 'and', result_le_203925, result_eq_203928)
        
        # Testing the type of an if condition (line 319)
        if_condition_203930 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 319, 16), result_and_keyword_203929)
        # Assigning a type to the variable 'if_condition_203930' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 16), 'if_condition_203930', if_condition_203930)
        # SSA begins for if statement (line 319)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 320):
        
        # Assigning a Name to a Name (line 320):
        # Getting the type of 'False' (line 320)
        False_203931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 36), 'False')
        # Assigning a type to the variable 'hits_boundary' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 20), 'hits_boundary', False_203931)
        # SSA join for if statement (line 319)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 324):
        
        # Assigning a Call to a Name (line 324):
        
        # Call to solve_triangular(...): (line 324)
        # Processing the call arguments (line 324)
        # Getting the type of 'U' (line 324)
        U_203933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 37), 'U', False)
        # Getting the type of 'p' (line 324)
        p_203934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 40), 'p', False)
        # Processing the call keyword arguments (line 324)
        str_203935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 49), 'str', 'T')
        keyword_203936 = str_203935
        kwargs_203937 = {'trans': keyword_203936}
        # Getting the type of 'solve_triangular' (line 324)
        solve_triangular_203932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 20), 'solve_triangular', False)
        # Calling solve_triangular(args, kwargs) (line 324)
        solve_triangular_call_result_203938 = invoke(stypy.reporting.localization.Localization(__file__, 324, 20), solve_triangular_203932, *[U_203933, p_203934], **kwargs_203937)
        
        # Assigning a type to the variable 'w' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 16), 'w', solve_triangular_call_result_203938)
        
        # Assigning a Call to a Name (line 326):
        
        # Assigning a Call to a Name (line 326):
        
        # Call to norm(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'w' (line 326)
        w_203940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 30), 'w', False)
        # Processing the call keyword arguments (line 326)
        kwargs_203941 = {}
        # Getting the type of 'norm' (line 326)
        norm_203939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 25), 'norm', False)
        # Calling norm(args, kwargs) (line 326)
        norm_call_result_203942 = invoke(stypy.reporting.localization.Localization(__file__, 326, 25), norm_203939, *[w_203940], **kwargs_203941)
        
        # Assigning a type to the variable 'w_norm' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 16), 'w_norm', norm_call_result_203942)
        
        # Assigning a BinOp to a Name (line 330):
        
        # Assigning a BinOp to a Name (line 330):
        # Getting the type of 'p_norm' (line 330)
        p_norm_203943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 32), 'p_norm')
        # Getting the type of 'w_norm' (line 330)
        w_norm_203944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 39), 'w_norm')
        # Applying the binary operator 'div' (line 330)
        result_div_203945 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 32), 'div', p_norm_203943, w_norm_203944)
        
        int_203946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 48), 'int')
        # Applying the binary operator '**' (line 330)
        result_pow_203947 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 31), '**', result_div_203945, int_203946)
        
        # Getting the type of 'p_norm' (line 330)
        p_norm_203948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 53), 'p_norm')
        # Getting the type of 'tr_radius' (line 330)
        tr_radius_203949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 60), 'tr_radius')
        # Applying the binary operator '-' (line 330)
        result_sub_203950 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 53), '-', p_norm_203948, tr_radius_203949)
        
        # Applying the binary operator '*' (line 330)
        result_mul_203951 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 31), '*', result_pow_203947, result_sub_203950)
        
        # Getting the type of 'tr_radius' (line 330)
        tr_radius_203952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 71), 'tr_radius')
        # Applying the binary operator 'div' (line 330)
        result_div_203953 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 70), 'div', result_mul_203951, tr_radius_203952)
        
        # Assigning a type to the variable 'delta_lambda' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'delta_lambda', result_div_203953)
        
        # Assigning a BinOp to a Name (line 331):
        
        # Assigning a BinOp to a Name (line 331):
        # Getting the type of 'lambda_current' (line 331)
        lambda_current_203954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 29), 'lambda_current')
        # Getting the type of 'delta_lambda' (line 331)
        delta_lambda_203955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 46), 'delta_lambda')
        # Applying the binary operator '+' (line 331)
        result_add_203956 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 29), '+', lambda_current_203954, delta_lambda_203955)
        
        # Assigning a type to the variable 'lambda_new' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'lambda_new', result_add_203956)
        
        
        # Getting the type of 'p_norm' (line 333)
        p_norm_203957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 19), 'p_norm')
        # Getting the type of 'tr_radius' (line 333)
        tr_radius_203958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 28), 'tr_radius')
        # Applying the binary operator '<' (line 333)
        result_lt_203959 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 19), '<', p_norm_203957, tr_radius_203958)
        
        # Testing the type of an if condition (line 333)
        if_condition_203960 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 333, 16), result_lt_203959)
        # Assigning a type to the variable 'if_condition_203960' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 16), 'if_condition_203960', if_condition_203960)
        # SSA begins for if statement (line 333)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 334):
        
        # Assigning a Subscript to a Name (line 334):
        
        # Obtaining the type of the subscript
        int_203961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 20), 'int')
        
        # Call to estimate_smallest_singular_value(...): (line 334)
        # Processing the call arguments (line 334)
        # Getting the type of 'U' (line 334)
        U_203963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 68), 'U', False)
        # Processing the call keyword arguments (line 334)
        kwargs_203964 = {}
        # Getting the type of 'estimate_smallest_singular_value' (line 334)
        estimate_smallest_singular_value_203962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 35), 'estimate_smallest_singular_value', False)
        # Calling estimate_smallest_singular_value(args, kwargs) (line 334)
        estimate_smallest_singular_value_call_result_203965 = invoke(stypy.reporting.localization.Localization(__file__, 334, 35), estimate_smallest_singular_value_203962, *[U_203963], **kwargs_203964)
        
        # Obtaining the member '__getitem__' of a type (line 334)
        getitem___203966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 20), estimate_smallest_singular_value_call_result_203965, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 334)
        subscript_call_result_203967 = invoke(stypy.reporting.localization.Localization(__file__, 334, 20), getitem___203966, int_203961)
        
        # Assigning a type to the variable 'tuple_var_assignment_203290' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 20), 'tuple_var_assignment_203290', subscript_call_result_203967)
        
        # Assigning a Subscript to a Name (line 334):
        
        # Obtaining the type of the subscript
        int_203968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 20), 'int')
        
        # Call to estimate_smallest_singular_value(...): (line 334)
        # Processing the call arguments (line 334)
        # Getting the type of 'U' (line 334)
        U_203970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 68), 'U', False)
        # Processing the call keyword arguments (line 334)
        kwargs_203971 = {}
        # Getting the type of 'estimate_smallest_singular_value' (line 334)
        estimate_smallest_singular_value_203969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 35), 'estimate_smallest_singular_value', False)
        # Calling estimate_smallest_singular_value(args, kwargs) (line 334)
        estimate_smallest_singular_value_call_result_203972 = invoke(stypy.reporting.localization.Localization(__file__, 334, 35), estimate_smallest_singular_value_203969, *[U_203970], **kwargs_203971)
        
        # Obtaining the member '__getitem__' of a type (line 334)
        getitem___203973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 20), estimate_smallest_singular_value_call_result_203972, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 334)
        subscript_call_result_203974 = invoke(stypy.reporting.localization.Localization(__file__, 334, 20), getitem___203973, int_203968)
        
        # Assigning a type to the variable 'tuple_var_assignment_203291' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 20), 'tuple_var_assignment_203291', subscript_call_result_203974)
        
        # Assigning a Name to a Name (line 334):
        # Getting the type of 'tuple_var_assignment_203290' (line 334)
        tuple_var_assignment_203290_203975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 20), 'tuple_var_assignment_203290')
        # Assigning a type to the variable 's_min' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 20), 's_min', tuple_var_assignment_203290_203975)
        
        # Assigning a Name to a Name (line 334):
        # Getting the type of 'tuple_var_assignment_203291' (line 334)
        tuple_var_assignment_203291_203976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 20), 'tuple_var_assignment_203291')
        # Assigning a type to the variable 'z_min' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 27), 'z_min', tuple_var_assignment_203291_203976)
        
        # Assigning a Call to a Tuple (line 336):
        
        # Assigning a Subscript to a Name (line 336):
        
        # Obtaining the type of the subscript
        int_203977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 20), 'int')
        
        # Call to get_boundaries_intersections(...): (line 336)
        # Processing the call arguments (line 336)
        # Getting the type of 'p' (line 336)
        p_203980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 63), 'p', False)
        # Getting the type of 'z_min' (line 336)
        z_min_203981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 66), 'z_min', False)
        # Getting the type of 'tr_radius' (line 337)
        tr_radius_203982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 63), 'tr_radius', False)
        # Processing the call keyword arguments (line 336)
        kwargs_203983 = {}
        # Getting the type of 'self' (line 336)
        self_203978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 29), 'self', False)
        # Obtaining the member 'get_boundaries_intersections' of a type (line 336)
        get_boundaries_intersections_203979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 29), self_203978, 'get_boundaries_intersections')
        # Calling get_boundaries_intersections(args, kwargs) (line 336)
        get_boundaries_intersections_call_result_203984 = invoke(stypy.reporting.localization.Localization(__file__, 336, 29), get_boundaries_intersections_203979, *[p_203980, z_min_203981, tr_radius_203982], **kwargs_203983)
        
        # Obtaining the member '__getitem__' of a type (line 336)
        getitem___203985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 20), get_boundaries_intersections_call_result_203984, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 336)
        subscript_call_result_203986 = invoke(stypy.reporting.localization.Localization(__file__, 336, 20), getitem___203985, int_203977)
        
        # Assigning a type to the variable 'tuple_var_assignment_203292' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 20), 'tuple_var_assignment_203292', subscript_call_result_203986)
        
        # Assigning a Subscript to a Name (line 336):
        
        # Obtaining the type of the subscript
        int_203987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 20), 'int')
        
        # Call to get_boundaries_intersections(...): (line 336)
        # Processing the call arguments (line 336)
        # Getting the type of 'p' (line 336)
        p_203990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 63), 'p', False)
        # Getting the type of 'z_min' (line 336)
        z_min_203991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 66), 'z_min', False)
        # Getting the type of 'tr_radius' (line 337)
        tr_radius_203992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 63), 'tr_radius', False)
        # Processing the call keyword arguments (line 336)
        kwargs_203993 = {}
        # Getting the type of 'self' (line 336)
        self_203988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 29), 'self', False)
        # Obtaining the member 'get_boundaries_intersections' of a type (line 336)
        get_boundaries_intersections_203989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 29), self_203988, 'get_boundaries_intersections')
        # Calling get_boundaries_intersections(args, kwargs) (line 336)
        get_boundaries_intersections_call_result_203994 = invoke(stypy.reporting.localization.Localization(__file__, 336, 29), get_boundaries_intersections_203989, *[p_203990, z_min_203991, tr_radius_203992], **kwargs_203993)
        
        # Obtaining the member '__getitem__' of a type (line 336)
        getitem___203995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 20), get_boundaries_intersections_call_result_203994, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 336)
        subscript_call_result_203996 = invoke(stypy.reporting.localization.Localization(__file__, 336, 20), getitem___203995, int_203987)
        
        # Assigning a type to the variable 'tuple_var_assignment_203293' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 20), 'tuple_var_assignment_203293', subscript_call_result_203996)
        
        # Assigning a Name to a Name (line 336):
        # Getting the type of 'tuple_var_assignment_203292' (line 336)
        tuple_var_assignment_203292_203997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 20), 'tuple_var_assignment_203292')
        # Assigning a type to the variable 'ta' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 20), 'ta', tuple_var_assignment_203292_203997)
        
        # Assigning a Name to a Name (line 336):
        # Getting the type of 'tuple_var_assignment_203293' (line 336)
        tuple_var_assignment_203293_203998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 20), 'tuple_var_assignment_203293')
        # Assigning a type to the variable 'tb' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 24), 'tb', tuple_var_assignment_203293_203998)
        
        # Assigning a Call to a Name (line 343):
        
        # Assigning a Call to a Name (line 343):
        
        # Call to min(...): (line 343)
        # Processing the call arguments (line 343)
        
        # Obtaining an instance of the builtin type 'list' (line 343)
        list_204000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 343)
        # Adding element type (line 343)
        # Getting the type of 'ta' (line 343)
        ta_204001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 36), 'ta', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 35), list_204000, ta_204001)
        # Adding element type (line 343)
        # Getting the type of 'tb' (line 343)
        tb_204002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 40), 'tb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 35), list_204000, tb_204002)
        
        # Processing the call keyword arguments (line 343)
        # Getting the type of 'abs' (line 343)
        abs_204003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 49), 'abs', False)
        keyword_204004 = abs_204003
        kwargs_204005 = {'key': keyword_204004}
        # Getting the type of 'min' (line 343)
        min_203999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 31), 'min', False)
        # Calling min(args, kwargs) (line 343)
        min_call_result_204006 = invoke(stypy.reporting.localization.Localization(__file__, 343, 31), min_203999, *[list_204000], **kwargs_204005)
        
        # Assigning a type to the variable 'step_len' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 20), 'step_len', min_call_result_204006)
        
        # Assigning a Call to a Name (line 346):
        
        # Assigning a Call to a Name (line 346):
        
        # Call to dot(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'p' (line 346)
        p_204009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 44), 'p', False)
        
        # Call to dot(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'H' (line 346)
        H_204012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 54), 'H', False)
        # Getting the type of 'p' (line 346)
        p_204013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 57), 'p', False)
        # Processing the call keyword arguments (line 346)
        kwargs_204014 = {}
        # Getting the type of 'np' (line 346)
        np_204010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 47), 'np', False)
        # Obtaining the member 'dot' of a type (line 346)
        dot_204011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 47), np_204010, 'dot')
        # Calling dot(args, kwargs) (line 346)
        dot_call_result_204015 = invoke(stypy.reporting.localization.Localization(__file__, 346, 47), dot_204011, *[H_204012, p_204013], **kwargs_204014)
        
        # Processing the call keyword arguments (line 346)
        kwargs_204016 = {}
        # Getting the type of 'np' (line 346)
        np_204007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 37), 'np', False)
        # Obtaining the member 'dot' of a type (line 346)
        dot_204008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 37), np_204007, 'dot')
        # Calling dot(args, kwargs) (line 346)
        dot_call_result_204017 = invoke(stypy.reporting.localization.Localization(__file__, 346, 37), dot_204008, *[p_204009, dot_call_result_204015], **kwargs_204016)
        
        # Assigning a type to the variable 'quadratic_term' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 20), 'quadratic_term', dot_call_result_204017)
        
        # Assigning a BinOp to a Name (line 349):
        
        # Assigning a BinOp to a Name (line 349):
        # Getting the type of 'step_len' (line 349)
        step_len_204018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 38), 'step_len')
        int_204019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 48), 'int')
        # Applying the binary operator '**' (line 349)
        result_pow_204020 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 38), '**', step_len_204018, int_204019)
        
        # Getting the type of 's_min' (line 349)
        s_min_204021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 52), 's_min')
        int_204022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 59), 'int')
        # Applying the binary operator '**' (line 349)
        result_pow_204023 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 52), '**', s_min_204021, int_204022)
        
        # Applying the binary operator '*' (line 349)
        result_mul_204024 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 38), '*', result_pow_204020, result_pow_204023)
        
        # Getting the type of 'quadratic_term' (line 349)
        quadratic_term_204025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 65), 'quadratic_term')
        # Getting the type of 'lambda_current' (line 349)
        lambda_current_204026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 82), 'lambda_current')
        # Getting the type of 'tr_radius' (line 349)
        tr_radius_204027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 97), 'tr_radius')
        int_204028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 108), 'int')
        # Applying the binary operator '**' (line 349)
        result_pow_204029 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 97), '**', tr_radius_204027, int_204028)
        
        # Applying the binary operator '*' (line 349)
        result_mul_204030 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 82), '*', lambda_current_204026, result_pow_204029)
        
        # Applying the binary operator '+' (line 349)
        result_add_204031 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 65), '+', quadratic_term_204025, result_mul_204030)
        
        # Applying the binary operator 'div' (line 349)
        result_div_204032 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 37), 'div', result_mul_204024, result_add_204031)
        
        # Assigning a type to the variable 'relative_error' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 20), 'relative_error', result_div_204032)
        
        
        # Getting the type of 'relative_error' (line 350)
        relative_error_204033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 23), 'relative_error')
        # Getting the type of 'self' (line 350)
        self_204034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 41), 'self')
        # Obtaining the member 'k_hard' of a type (line 350)
        k_hard_204035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 41), self_204034, 'k_hard')
        # Applying the binary operator '<=' (line 350)
        result_le_204036 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 23), '<=', relative_error_204033, k_hard_204035)
        
        # Testing the type of an if condition (line 350)
        if_condition_204037 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 350, 20), result_le_204036)
        # Assigning a type to the variable 'if_condition_204037' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 20), 'if_condition_204037', if_condition_204037)
        # SSA begins for if statement (line 350)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'p' (line 351)
        p_204038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 24), 'p')
        # Getting the type of 'step_len' (line 351)
        step_len_204039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 29), 'step_len')
        # Getting the type of 'z_min' (line 351)
        z_min_204040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 40), 'z_min')
        # Applying the binary operator '*' (line 351)
        result_mul_204041 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 29), '*', step_len_204039, z_min_204040)
        
        # Applying the binary operator '+=' (line 351)
        result_iadd_204042 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 24), '+=', p_204038, result_mul_204041)
        # Assigning a type to the variable 'p' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 24), 'p', result_iadd_204042)
        
        # SSA join for if statement (line 350)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 355):
        
        # Assigning a Name to a Name (line 355):
        # Getting the type of 'lambda_current' (line 355)
        lambda_current_204043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 32), 'lambda_current')
        # Assigning a type to the variable 'lambda_ub' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 20), 'lambda_ub', lambda_current_204043)
        
        # Assigning a Call to a Name (line 356):
        
        # Assigning a Call to a Name (line 356):
        
        # Call to max(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'lambda_lb' (line 356)
        lambda_lb_204045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 36), 'lambda_lb', False)
        # Getting the type of 'lambda_current' (line 356)
        lambda_current_204046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 47), 'lambda_current', False)
        # Getting the type of 's_min' (line 356)
        s_min_204047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 64), 's_min', False)
        int_204048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 71), 'int')
        # Applying the binary operator '**' (line 356)
        result_pow_204049 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 64), '**', s_min_204047, int_204048)
        
        # Applying the binary operator '-' (line 356)
        result_sub_204050 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 47), '-', lambda_current_204046, result_pow_204049)
        
        # Processing the call keyword arguments (line 356)
        kwargs_204051 = {}
        # Getting the type of 'max' (line 356)
        max_204044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 32), 'max', False)
        # Calling max(args, kwargs) (line 356)
        max_call_result_204052 = invoke(stypy.reporting.localization.Localization(__file__, 356, 32), max_204044, *[lambda_lb_204045, result_sub_204050], **kwargs_204051)
        
        # Assigning a type to the variable 'lambda_lb' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 20), 'lambda_lb', max_call_result_204052)
        
        # Assigning a BinOp to a Name (line 359):
        
        # Assigning a BinOp to a Name (line 359):
        # Getting the type of 'self' (line 359)
        self_204053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 24), 'self')
        # Obtaining the member 'hess' of a type (line 359)
        hess_204054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 24), self_204053, 'hess')
        # Getting the type of 'lambda_new' (line 359)
        lambda_new_204055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 36), 'lambda_new')
        
        # Call to eye(...): (line 359)
        # Processing the call arguments (line 359)
        # Getting the type of 'n' (line 359)
        n_204058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 54), 'n', False)
        # Processing the call keyword arguments (line 359)
        kwargs_204059 = {}
        # Getting the type of 'np' (line 359)
        np_204056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 47), 'np', False)
        # Obtaining the member 'eye' of a type (line 359)
        eye_204057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 47), np_204056, 'eye')
        # Calling eye(args, kwargs) (line 359)
        eye_call_result_204060 = invoke(stypy.reporting.localization.Localization(__file__, 359, 47), eye_204057, *[n_204058], **kwargs_204059)
        
        # Applying the binary operator '*' (line 359)
        result_mul_204061 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 36), '*', lambda_new_204055, eye_call_result_204060)
        
        # Applying the binary operator '+' (line 359)
        result_add_204062 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 24), '+', hess_204054, result_mul_204061)
        
        # Assigning a type to the variable 'H' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 20), 'H', result_add_204062)
        
        # Assigning a Call to a Tuple (line 360):
        
        # Assigning a Subscript to a Name (line 360):
        
        # Obtaining the type of the subscript
        int_204063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 20), 'int')
        
        # Call to cholesky(...): (line 360)
        # Processing the call arguments (line 360)
        # Getting the type of 'H' (line 360)
        H_204066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 44), 'H', False)
        # Processing the call keyword arguments (line 360)
        # Getting the type of 'False' (line 360)
        False_204067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 53), 'False', False)
        keyword_204068 = False_204067
        # Getting the type of 'False' (line 361)
        False_204069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 56), 'False', False)
        keyword_204070 = False_204069
        # Getting the type of 'True' (line 362)
        True_204071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 50), 'True', False)
        keyword_204072 = True_204071
        kwargs_204073 = {'lower': keyword_204068, 'overwrite_a': keyword_204070, 'clean': keyword_204072}
        # Getting the type of 'self' (line 360)
        self_204064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 30), 'self', False)
        # Obtaining the member 'cholesky' of a type (line 360)
        cholesky_204065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 30), self_204064, 'cholesky')
        # Calling cholesky(args, kwargs) (line 360)
        cholesky_call_result_204074 = invoke(stypy.reporting.localization.Localization(__file__, 360, 30), cholesky_204065, *[H_204066], **kwargs_204073)
        
        # Obtaining the member '__getitem__' of a type (line 360)
        getitem___204075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 20), cholesky_call_result_204074, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 360)
        subscript_call_result_204076 = invoke(stypy.reporting.localization.Localization(__file__, 360, 20), getitem___204075, int_204063)
        
        # Assigning a type to the variable 'tuple_var_assignment_203294' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 20), 'tuple_var_assignment_203294', subscript_call_result_204076)
        
        # Assigning a Subscript to a Name (line 360):
        
        # Obtaining the type of the subscript
        int_204077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 20), 'int')
        
        # Call to cholesky(...): (line 360)
        # Processing the call arguments (line 360)
        # Getting the type of 'H' (line 360)
        H_204080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 44), 'H', False)
        # Processing the call keyword arguments (line 360)
        # Getting the type of 'False' (line 360)
        False_204081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 53), 'False', False)
        keyword_204082 = False_204081
        # Getting the type of 'False' (line 361)
        False_204083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 56), 'False', False)
        keyword_204084 = False_204083
        # Getting the type of 'True' (line 362)
        True_204085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 50), 'True', False)
        keyword_204086 = True_204085
        kwargs_204087 = {'lower': keyword_204082, 'overwrite_a': keyword_204084, 'clean': keyword_204086}
        # Getting the type of 'self' (line 360)
        self_204078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 30), 'self', False)
        # Obtaining the member 'cholesky' of a type (line 360)
        cholesky_204079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 30), self_204078, 'cholesky')
        # Calling cholesky(args, kwargs) (line 360)
        cholesky_call_result_204088 = invoke(stypy.reporting.localization.Localization(__file__, 360, 30), cholesky_204079, *[H_204080], **kwargs_204087)
        
        # Obtaining the member '__getitem__' of a type (line 360)
        getitem___204089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 20), cholesky_call_result_204088, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 360)
        subscript_call_result_204090 = invoke(stypy.reporting.localization.Localization(__file__, 360, 20), getitem___204089, int_204077)
        
        # Assigning a type to the variable 'tuple_var_assignment_203295' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 20), 'tuple_var_assignment_203295', subscript_call_result_204090)
        
        # Assigning a Name to a Name (line 360):
        # Getting the type of 'tuple_var_assignment_203294' (line 360)
        tuple_var_assignment_203294_204091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 20), 'tuple_var_assignment_203294')
        # Assigning a type to the variable 'c' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 20), 'c', tuple_var_assignment_203294_204091)
        
        # Assigning a Name to a Name (line 360):
        # Getting the type of 'tuple_var_assignment_203295' (line 360)
        tuple_var_assignment_203295_204092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 20), 'tuple_var_assignment_203295')
        # Assigning a type to the variable 'info' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 23), 'info', tuple_var_assignment_203295_204092)
        
        
        # Getting the type of 'info' (line 366)
        info_204093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 23), 'info')
        int_204094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 31), 'int')
        # Applying the binary operator '==' (line 366)
        result_eq_204095 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 23), '==', info_204093, int_204094)
        
        # Testing the type of an if condition (line 366)
        if_condition_204096 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 366, 20), result_eq_204095)
        # Assigning a type to the variable 'if_condition_204096' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 20), 'if_condition_204096', if_condition_204096)
        # SSA begins for if statement (line 366)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 368):
        
        # Assigning a Name to a Name (line 368):
        # Getting the type of 'lambda_new' (line 368)
        lambda_new_204097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 41), 'lambda_new')
        # Assigning a type to the variable 'lambda_current' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 24), 'lambda_current', lambda_new_204097)
        
        # Assigning a Name to a Name (line 369):
        
        # Assigning a Name to a Name (line 369):
        # Getting the type of 'True' (line 369)
        True_204098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 45), 'True')
        # Assigning a type to the variable 'already_factorized' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 24), 'already_factorized', True_204098)
        # SSA branch for the else part of an if statement (line 366)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 372):
        
        # Assigning a Call to a Name (line 372):
        
        # Call to max(...): (line 372)
        # Processing the call arguments (line 372)
        # Getting the type of 'lambda_lb' (line 372)
        lambda_lb_204100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 40), 'lambda_lb', False)
        # Getting the type of 'lambda_new' (line 372)
        lambda_new_204101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 51), 'lambda_new', False)
        # Processing the call keyword arguments (line 372)
        kwargs_204102 = {}
        # Getting the type of 'max' (line 372)
        max_204099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 36), 'max', False)
        # Calling max(args, kwargs) (line 372)
        max_call_result_204103 = invoke(stypy.reporting.localization.Localization(__file__, 372, 36), max_204099, *[lambda_lb_204100, lambda_new_204101], **kwargs_204102)
        
        # Assigning a type to the variable 'lambda_lb' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 24), 'lambda_lb', max_call_result_204103)
        
        # Assigning a Call to a Name (line 375):
        
        # Assigning a Call to a Name (line 375):
        
        # Call to max(...): (line 375)
        # Processing the call arguments (line 375)
        
        # Call to sqrt(...): (line 375)
        # Processing the call arguments (line 375)
        # Getting the type of 'lambda_lb' (line 375)
        lambda_lb_204107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 53), 'lambda_lb', False)
        # Getting the type of 'lambda_ub' (line 375)
        lambda_ub_204108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 65), 'lambda_ub', False)
        # Applying the binary operator '*' (line 375)
        result_mul_204109 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 53), '*', lambda_lb_204107, lambda_ub_204108)
        
        # Processing the call keyword arguments (line 375)
        kwargs_204110 = {}
        # Getting the type of 'np' (line 375)
        np_204105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 45), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 375)
        sqrt_204106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 45), np_204105, 'sqrt')
        # Calling sqrt(args, kwargs) (line 375)
        sqrt_call_result_204111 = invoke(stypy.reporting.localization.Localization(__file__, 375, 45), sqrt_204106, *[result_mul_204109], **kwargs_204110)
        
        # Getting the type of 'lambda_lb' (line 376)
        lambda_lb_204112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 45), 'lambda_lb', False)
        # Getting the type of 'self' (line 376)
        self_204113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 57), 'self', False)
        # Obtaining the member 'UPDATE_COEFF' of a type (line 376)
        UPDATE_COEFF_204114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 57), self_204113, 'UPDATE_COEFF')
        # Getting the type of 'lambda_ub' (line 376)
        lambda_ub_204115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 76), 'lambda_ub', False)
        # Getting the type of 'lambda_lb' (line 376)
        lambda_lb_204116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 86), 'lambda_lb', False)
        # Applying the binary operator '-' (line 376)
        result_sub_204117 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 76), '-', lambda_ub_204115, lambda_lb_204116)
        
        # Applying the binary operator '*' (line 376)
        result_mul_204118 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 57), '*', UPDATE_COEFF_204114, result_sub_204117)
        
        # Applying the binary operator '+' (line 376)
        result_add_204119 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 45), '+', lambda_lb_204112, result_mul_204118)
        
        # Processing the call keyword arguments (line 375)
        kwargs_204120 = {}
        # Getting the type of 'max' (line 375)
        max_204104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 41), 'max', False)
        # Calling max(args, kwargs) (line 375)
        max_call_result_204121 = invoke(stypy.reporting.localization.Localization(__file__, 375, 41), max_204104, *[sqrt_call_result_204111, result_add_204119], **kwargs_204120)
        
        # Assigning a type to the variable 'lambda_current' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 24), 'lambda_current', max_call_result_204121)
        # SSA join for if statement (line 366)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 333)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 380):
        
        # Assigning a BinOp to a Name (line 380):
        
        # Call to abs(...): (line 380)
        # Processing the call arguments (line 380)
        # Getting the type of 'p_norm' (line 380)
        p_norm_204123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 41), 'p_norm', False)
        # Getting the type of 'tr_radius' (line 380)
        tr_radius_204124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 50), 'tr_radius', False)
        # Applying the binary operator '-' (line 380)
        result_sub_204125 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 41), '-', p_norm_204123, tr_radius_204124)
        
        # Processing the call keyword arguments (line 380)
        kwargs_204126 = {}
        # Getting the type of 'abs' (line 380)
        abs_204122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 37), 'abs', False)
        # Calling abs(args, kwargs) (line 380)
        abs_call_result_204127 = invoke(stypy.reporting.localization.Localization(__file__, 380, 37), abs_204122, *[result_sub_204125], **kwargs_204126)
        
        # Getting the type of 'tr_radius' (line 380)
        tr_radius_204128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 63), 'tr_radius')
        # Applying the binary operator 'div' (line 380)
        result_div_204129 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 37), 'div', abs_call_result_204127, tr_radius_204128)
        
        # Assigning a type to the variable 'relative_error' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 20), 'relative_error', result_div_204129)
        
        
        # Getting the type of 'relative_error' (line 381)
        relative_error_204130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 23), 'relative_error')
        # Getting the type of 'self' (line 381)
        self_204131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 41), 'self')
        # Obtaining the member 'k_easy' of a type (line 381)
        k_easy_204132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 41), self_204131, 'k_easy')
        # Applying the binary operator '<=' (line 381)
        result_le_204133 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 23), '<=', relative_error_204130, k_easy_204132)
        
        # Testing the type of an if condition (line 381)
        if_condition_204134 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 381, 20), result_le_204133)
        # Assigning a type to the variable 'if_condition_204134' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 20), 'if_condition_204134', if_condition_204134)
        # SSA begins for if statement (line 381)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 381)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 385):
        
        # Assigning a Name to a Name (line 385):
        # Getting the type of 'lambda_current' (line 385)
        lambda_current_204135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 32), 'lambda_current')
        # Assigning a type to the variable 'lambda_lb' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 20), 'lambda_lb', lambda_current_204135)
        
        # Assigning a Name to a Name (line 388):
        
        # Assigning a Name to a Name (line 388):
        # Getting the type of 'lambda_new' (line 388)
        lambda_new_204136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 37), 'lambda_new')
        # Assigning a type to the variable 'lambda_current' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 20), 'lambda_current', lambda_new_204136)
        # SSA join for if statement (line 333)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 310)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'info' (line 390)
        info_204137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 17), 'info')
        int_204138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 25), 'int')
        # Applying the binary operator '==' (line 390)
        result_eq_204139 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 17), '==', info_204137, int_204138)
        
        
        # Getting the type of 'self' (line 390)
        self_204140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 31), 'self')
        # Obtaining the member 'jac_mag' of a type (line 390)
        jac_mag_204141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 31), self_204140, 'jac_mag')
        # Getting the type of 'self' (line 390)
        self_204142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 47), 'self')
        # Obtaining the member 'CLOSE_TO_ZERO' of a type (line 390)
        CLOSE_TO_ZERO_204143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 47), self_204142, 'CLOSE_TO_ZERO')
        # Applying the binary operator '<=' (line 390)
        result_le_204144 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 31), '<=', jac_mag_204141, CLOSE_TO_ZERO_204143)
        
        # Applying the binary operator 'and' (line 390)
        result_and_keyword_204145 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 17), 'and', result_eq_204139, result_le_204144)
        
        # Testing the type of an if condition (line 390)
        if_condition_204146 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 390, 17), result_and_keyword_204145)
        # Assigning a type to the variable 'if_condition_204146' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 17), 'if_condition_204146', if_condition_204146)
        # SSA begins for if statement (line 390)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'lambda_current' (line 394)
        lambda_current_204147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 19), 'lambda_current')
        int_204148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 37), 'int')
        # Applying the binary operator '==' (line 394)
        result_eq_204149 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 19), '==', lambda_current_204147, int_204148)
        
        # Testing the type of an if condition (line 394)
        if_condition_204150 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 394, 16), result_eq_204149)
        # Assigning a type to the variable 'if_condition_204150' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 16), 'if_condition_204150', if_condition_204150)
        # SSA begins for if statement (line 394)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 395):
        
        # Assigning a Call to a Name (line 395):
        
        # Call to zeros(...): (line 395)
        # Processing the call arguments (line 395)
        # Getting the type of 'n' (line 395)
        n_204153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 33), 'n', False)
        # Processing the call keyword arguments (line 395)
        kwargs_204154 = {}
        # Getting the type of 'np' (line 395)
        np_204151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 24), 'np', False)
        # Obtaining the member 'zeros' of a type (line 395)
        zeros_204152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 24), np_204151, 'zeros')
        # Calling zeros(args, kwargs) (line 395)
        zeros_call_result_204155 = invoke(stypy.reporting.localization.Localization(__file__, 395, 24), zeros_204152, *[n_204153], **kwargs_204154)
        
        # Assigning a type to the variable 'p' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 20), 'p', zeros_call_result_204155)
        
        # Assigning a Name to a Name (line 396):
        
        # Assigning a Name to a Name (line 396):
        # Getting the type of 'False' (line 396)
        False_204156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 36), 'False')
        # Assigning a type to the variable 'hits_boundary' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 20), 'hits_boundary', False_204156)
        # SSA join for if statement (line 394)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 399):
        
        # Assigning a Subscript to a Name (line 399):
        
        # Obtaining the type of the subscript
        int_204157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 16), 'int')
        
        # Call to estimate_smallest_singular_value(...): (line 399)
        # Processing the call arguments (line 399)
        # Getting the type of 'U' (line 399)
        U_204159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 64), 'U', False)
        # Processing the call keyword arguments (line 399)
        kwargs_204160 = {}
        # Getting the type of 'estimate_smallest_singular_value' (line 399)
        estimate_smallest_singular_value_204158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 31), 'estimate_smallest_singular_value', False)
        # Calling estimate_smallest_singular_value(args, kwargs) (line 399)
        estimate_smallest_singular_value_call_result_204161 = invoke(stypy.reporting.localization.Localization(__file__, 399, 31), estimate_smallest_singular_value_204158, *[U_204159], **kwargs_204160)
        
        # Obtaining the member '__getitem__' of a type (line 399)
        getitem___204162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 16), estimate_smallest_singular_value_call_result_204161, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 399)
        subscript_call_result_204163 = invoke(stypy.reporting.localization.Localization(__file__, 399, 16), getitem___204162, int_204157)
        
        # Assigning a type to the variable 'tuple_var_assignment_203296' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 16), 'tuple_var_assignment_203296', subscript_call_result_204163)
        
        # Assigning a Subscript to a Name (line 399):
        
        # Obtaining the type of the subscript
        int_204164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 16), 'int')
        
        # Call to estimate_smallest_singular_value(...): (line 399)
        # Processing the call arguments (line 399)
        # Getting the type of 'U' (line 399)
        U_204166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 64), 'U', False)
        # Processing the call keyword arguments (line 399)
        kwargs_204167 = {}
        # Getting the type of 'estimate_smallest_singular_value' (line 399)
        estimate_smallest_singular_value_204165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 31), 'estimate_smallest_singular_value', False)
        # Calling estimate_smallest_singular_value(args, kwargs) (line 399)
        estimate_smallest_singular_value_call_result_204168 = invoke(stypy.reporting.localization.Localization(__file__, 399, 31), estimate_smallest_singular_value_204165, *[U_204166], **kwargs_204167)
        
        # Obtaining the member '__getitem__' of a type (line 399)
        getitem___204169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 16), estimate_smallest_singular_value_call_result_204168, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 399)
        subscript_call_result_204170 = invoke(stypy.reporting.localization.Localization(__file__, 399, 16), getitem___204169, int_204164)
        
        # Assigning a type to the variable 'tuple_var_assignment_203297' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 16), 'tuple_var_assignment_203297', subscript_call_result_204170)
        
        # Assigning a Name to a Name (line 399):
        # Getting the type of 'tuple_var_assignment_203296' (line 399)
        tuple_var_assignment_203296_204171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 16), 'tuple_var_assignment_203296')
        # Assigning a type to the variable 's_min' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 16), 's_min', tuple_var_assignment_203296_204171)
        
        # Assigning a Name to a Name (line 399):
        # Getting the type of 'tuple_var_assignment_203297' (line 399)
        tuple_var_assignment_203297_204172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 16), 'tuple_var_assignment_203297')
        # Assigning a type to the variable 'z_min' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 23), 'z_min', tuple_var_assignment_203297_204172)
        
        # Assigning a Name to a Name (line 400):
        
        # Assigning a Name to a Name (line 400):
        # Getting the type of 'tr_radius' (line 400)
        tr_radius_204173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 27), 'tr_radius')
        # Assigning a type to the variable 'step_len' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 16), 'step_len', tr_radius_204173)
        
        
        # Getting the type of 'step_len' (line 403)
        step_len_204174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 19), 'step_len')
        int_204175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 29), 'int')
        # Applying the binary operator '**' (line 403)
        result_pow_204176 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 19), '**', step_len_204174, int_204175)
        
        # Getting the type of 's_min' (line 403)
        s_min_204177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 33), 's_min')
        int_204178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 40), 'int')
        # Applying the binary operator '**' (line 403)
        result_pow_204179 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 33), '**', s_min_204177, int_204178)
        
        # Applying the binary operator '*' (line 403)
        result_mul_204180 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 19), '*', result_pow_204176, result_pow_204179)
        
        # Getting the type of 'self' (line 403)
        self_204181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 45), 'self')
        # Obtaining the member 'k_hard' of a type (line 403)
        k_hard_204182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 45), self_204181, 'k_hard')
        # Getting the type of 'lambda_current' (line 403)
        lambda_current_204183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 59), 'lambda_current')
        # Applying the binary operator '*' (line 403)
        result_mul_204184 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 45), '*', k_hard_204182, lambda_current_204183)
        
        # Getting the type of 'tr_radius' (line 403)
        tr_radius_204185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 76), 'tr_radius')
        int_204186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 87), 'int')
        # Applying the binary operator '**' (line 403)
        result_pow_204187 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 76), '**', tr_radius_204185, int_204186)
        
        # Applying the binary operator '*' (line 403)
        result_mul_204188 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 74), '*', result_mul_204184, result_pow_204187)
        
        # Applying the binary operator '<=' (line 403)
        result_le_204189 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 19), '<=', result_mul_204180, result_mul_204188)
        
        # Testing the type of an if condition (line 403)
        if_condition_204190 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 403, 16), result_le_204189)
        # Assigning a type to the variable 'if_condition_204190' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 16), 'if_condition_204190', if_condition_204190)
        # SSA begins for if statement (line 403)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 404):
        
        # Assigning a BinOp to a Name (line 404):
        # Getting the type of 'step_len' (line 404)
        step_len_204191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 24), 'step_len')
        # Getting the type of 'z_min' (line 404)
        z_min_204192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 35), 'z_min')
        # Applying the binary operator '*' (line 404)
        result_mul_204193 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 24), '*', step_len_204191, z_min_204192)
        
        # Assigning a type to the variable 'p' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 20), 'p', result_mul_204193)
        # SSA join for if statement (line 403)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 408):
        
        # Assigning a Name to a Name (line 408):
        # Getting the type of 'lambda_current' (line 408)
        lambda_current_204194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 28), 'lambda_current')
        # Assigning a type to the variable 'lambda_ub' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 16), 'lambda_ub', lambda_current_204194)
        
        # Assigning a Call to a Name (line 409):
        
        # Assigning a Call to a Name (line 409):
        
        # Call to max(...): (line 409)
        # Processing the call arguments (line 409)
        # Getting the type of 'lambda_lb' (line 409)
        lambda_lb_204196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 32), 'lambda_lb', False)
        # Getting the type of 'lambda_current' (line 409)
        lambda_current_204197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 43), 'lambda_current', False)
        # Getting the type of 's_min' (line 409)
        s_min_204198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 60), 's_min', False)
        int_204199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 67), 'int')
        # Applying the binary operator '**' (line 409)
        result_pow_204200 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 60), '**', s_min_204198, int_204199)
        
        # Applying the binary operator '-' (line 409)
        result_sub_204201 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 43), '-', lambda_current_204197, result_pow_204200)
        
        # Processing the call keyword arguments (line 409)
        kwargs_204202 = {}
        # Getting the type of 'max' (line 409)
        max_204195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 28), 'max', False)
        # Calling max(args, kwargs) (line 409)
        max_call_result_204203 = invoke(stypy.reporting.localization.Localization(__file__, 409, 28), max_204195, *[lambda_lb_204196, result_sub_204201], **kwargs_204202)
        
        # Assigning a type to the variable 'lambda_lb' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 16), 'lambda_lb', max_call_result_204203)
        
        # Assigning a Call to a Name (line 412):
        
        # Assigning a Call to a Name (line 412):
        
        # Call to max(...): (line 412)
        # Processing the call arguments (line 412)
        
        # Call to sqrt(...): (line 412)
        # Processing the call arguments (line 412)
        # Getting the type of 'lambda_lb' (line 412)
        lambda_lb_204207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 45), 'lambda_lb', False)
        # Getting the type of 'lambda_ub' (line 412)
        lambda_ub_204208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 57), 'lambda_ub', False)
        # Applying the binary operator '*' (line 412)
        result_mul_204209 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 45), '*', lambda_lb_204207, lambda_ub_204208)
        
        # Processing the call keyword arguments (line 412)
        kwargs_204210 = {}
        # Getting the type of 'np' (line 412)
        np_204205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 37), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 412)
        sqrt_204206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 37), np_204205, 'sqrt')
        # Calling sqrt(args, kwargs) (line 412)
        sqrt_call_result_204211 = invoke(stypy.reporting.localization.Localization(__file__, 412, 37), sqrt_204206, *[result_mul_204209], **kwargs_204210)
        
        # Getting the type of 'lambda_lb' (line 413)
        lambda_lb_204212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 37), 'lambda_lb', False)
        # Getting the type of 'self' (line 413)
        self_204213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 49), 'self', False)
        # Obtaining the member 'UPDATE_COEFF' of a type (line 413)
        UPDATE_COEFF_204214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 49), self_204213, 'UPDATE_COEFF')
        # Getting the type of 'lambda_ub' (line 413)
        lambda_ub_204215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 68), 'lambda_ub', False)
        # Getting the type of 'lambda_lb' (line 413)
        lambda_lb_204216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 78), 'lambda_lb', False)
        # Applying the binary operator '-' (line 413)
        result_sub_204217 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 68), '-', lambda_ub_204215, lambda_lb_204216)
        
        # Applying the binary operator '*' (line 413)
        result_mul_204218 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 49), '*', UPDATE_COEFF_204214, result_sub_204217)
        
        # Applying the binary operator '+' (line 413)
        result_add_204219 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 37), '+', lambda_lb_204212, result_mul_204218)
        
        # Processing the call keyword arguments (line 412)
        kwargs_204220 = {}
        # Getting the type of 'max' (line 412)
        max_204204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 33), 'max', False)
        # Calling max(args, kwargs) (line 412)
        max_call_result_204221 = invoke(stypy.reporting.localization.Localization(__file__, 412, 33), max_204204, *[sqrt_call_result_204211, result_add_204219], **kwargs_204220)
        
        # Assigning a type to the variable 'lambda_current' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 16), 'lambda_current', max_call_result_204221)
        # SSA branch for the else part of an if statement (line 390)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Tuple (line 418):
        
        # Assigning a Subscript to a Name (line 418):
        
        # Obtaining the type of the subscript
        int_204222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 16), 'int')
        
        # Call to singular_leading_submatrix(...): (line 418)
        # Processing the call arguments (line 418)
        # Getting the type of 'H' (line 418)
        H_204224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 54), 'H', False)
        # Getting the type of 'U' (line 418)
        U_204225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 57), 'U', False)
        # Getting the type of 'info' (line 418)
        info_204226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 60), 'info', False)
        # Processing the call keyword arguments (line 418)
        kwargs_204227 = {}
        # Getting the type of 'singular_leading_submatrix' (line 418)
        singular_leading_submatrix_204223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 27), 'singular_leading_submatrix', False)
        # Calling singular_leading_submatrix(args, kwargs) (line 418)
        singular_leading_submatrix_call_result_204228 = invoke(stypy.reporting.localization.Localization(__file__, 418, 27), singular_leading_submatrix_204223, *[H_204224, U_204225, info_204226], **kwargs_204227)
        
        # Obtaining the member '__getitem__' of a type (line 418)
        getitem___204229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 16), singular_leading_submatrix_call_result_204228, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 418)
        subscript_call_result_204230 = invoke(stypy.reporting.localization.Localization(__file__, 418, 16), getitem___204229, int_204222)
        
        # Assigning a type to the variable 'tuple_var_assignment_203298' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 16), 'tuple_var_assignment_203298', subscript_call_result_204230)
        
        # Assigning a Subscript to a Name (line 418):
        
        # Obtaining the type of the subscript
        int_204231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 16), 'int')
        
        # Call to singular_leading_submatrix(...): (line 418)
        # Processing the call arguments (line 418)
        # Getting the type of 'H' (line 418)
        H_204233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 54), 'H', False)
        # Getting the type of 'U' (line 418)
        U_204234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 57), 'U', False)
        # Getting the type of 'info' (line 418)
        info_204235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 60), 'info', False)
        # Processing the call keyword arguments (line 418)
        kwargs_204236 = {}
        # Getting the type of 'singular_leading_submatrix' (line 418)
        singular_leading_submatrix_204232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 27), 'singular_leading_submatrix', False)
        # Calling singular_leading_submatrix(args, kwargs) (line 418)
        singular_leading_submatrix_call_result_204237 = invoke(stypy.reporting.localization.Localization(__file__, 418, 27), singular_leading_submatrix_204232, *[H_204233, U_204234, info_204235], **kwargs_204236)
        
        # Obtaining the member '__getitem__' of a type (line 418)
        getitem___204238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 16), singular_leading_submatrix_call_result_204237, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 418)
        subscript_call_result_204239 = invoke(stypy.reporting.localization.Localization(__file__, 418, 16), getitem___204238, int_204231)
        
        # Assigning a type to the variable 'tuple_var_assignment_203299' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 16), 'tuple_var_assignment_203299', subscript_call_result_204239)
        
        # Assigning a Name to a Name (line 418):
        # Getting the type of 'tuple_var_assignment_203298' (line 418)
        tuple_var_assignment_203298_204240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 16), 'tuple_var_assignment_203298')
        # Assigning a type to the variable 'delta' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 16), 'delta', tuple_var_assignment_203298_204240)
        
        # Assigning a Name to a Name (line 418):
        # Getting the type of 'tuple_var_assignment_203299' (line 418)
        tuple_var_assignment_203299_204241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 16), 'tuple_var_assignment_203299')
        # Assigning a type to the variable 'v' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 23), 'v', tuple_var_assignment_203299_204241)
        
        # Assigning a Call to a Name (line 419):
        
        # Assigning a Call to a Name (line 419):
        
        # Call to norm(...): (line 419)
        # Processing the call arguments (line 419)
        # Getting the type of 'v' (line 419)
        v_204243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 30), 'v', False)
        # Processing the call keyword arguments (line 419)
        kwargs_204244 = {}
        # Getting the type of 'norm' (line 419)
        norm_204242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 25), 'norm', False)
        # Calling norm(args, kwargs) (line 419)
        norm_call_result_204245 = invoke(stypy.reporting.localization.Localization(__file__, 419, 25), norm_204242, *[v_204243], **kwargs_204244)
        
        # Assigning a type to the variable 'v_norm' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 16), 'v_norm', norm_call_result_204245)
        
        # Assigning a Call to a Name (line 422):
        
        # Assigning a Call to a Name (line 422):
        
        # Call to max(...): (line 422)
        # Processing the call arguments (line 422)
        # Getting the type of 'lambda_lb' (line 422)
        lambda_lb_204247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 32), 'lambda_lb', False)
        # Getting the type of 'lambda_current' (line 422)
        lambda_current_204248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 43), 'lambda_current', False)
        # Getting the type of 'delta' (line 422)
        delta_204249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 60), 'delta', False)
        # Getting the type of 'v_norm' (line 422)
        v_norm_204250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 66), 'v_norm', False)
        int_204251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 74), 'int')
        # Applying the binary operator '**' (line 422)
        result_pow_204252 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 66), '**', v_norm_204250, int_204251)
        
        # Applying the binary operator 'div' (line 422)
        result_div_204253 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 60), 'div', delta_204249, result_pow_204252)
        
        # Applying the binary operator '+' (line 422)
        result_add_204254 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 43), '+', lambda_current_204248, result_div_204253)
        
        # Processing the call keyword arguments (line 422)
        kwargs_204255 = {}
        # Getting the type of 'max' (line 422)
        max_204246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 28), 'max', False)
        # Calling max(args, kwargs) (line 422)
        max_call_result_204256 = invoke(stypy.reporting.localization.Localization(__file__, 422, 28), max_204246, *[lambda_lb_204247, result_add_204254], **kwargs_204255)
        
        # Assigning a type to the variable 'lambda_lb' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 16), 'lambda_lb', max_call_result_204256)
        
        # Assigning a Call to a Name (line 425):
        
        # Assigning a Call to a Name (line 425):
        
        # Call to max(...): (line 425)
        # Processing the call arguments (line 425)
        
        # Call to sqrt(...): (line 425)
        # Processing the call arguments (line 425)
        # Getting the type of 'lambda_lb' (line 425)
        lambda_lb_204260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 45), 'lambda_lb', False)
        # Getting the type of 'lambda_ub' (line 425)
        lambda_ub_204261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 57), 'lambda_ub', False)
        # Applying the binary operator '*' (line 425)
        result_mul_204262 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 45), '*', lambda_lb_204260, lambda_ub_204261)
        
        # Processing the call keyword arguments (line 425)
        kwargs_204263 = {}
        # Getting the type of 'np' (line 425)
        np_204258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 37), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 425)
        sqrt_204259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 37), np_204258, 'sqrt')
        # Calling sqrt(args, kwargs) (line 425)
        sqrt_call_result_204264 = invoke(stypy.reporting.localization.Localization(__file__, 425, 37), sqrt_204259, *[result_mul_204262], **kwargs_204263)
        
        # Getting the type of 'lambda_lb' (line 426)
        lambda_lb_204265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 37), 'lambda_lb', False)
        # Getting the type of 'self' (line 426)
        self_204266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 49), 'self', False)
        # Obtaining the member 'UPDATE_COEFF' of a type (line 426)
        UPDATE_COEFF_204267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 49), self_204266, 'UPDATE_COEFF')
        # Getting the type of 'lambda_ub' (line 426)
        lambda_ub_204268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 68), 'lambda_ub', False)
        # Getting the type of 'lambda_lb' (line 426)
        lambda_lb_204269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 78), 'lambda_lb', False)
        # Applying the binary operator '-' (line 426)
        result_sub_204270 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 68), '-', lambda_ub_204268, lambda_lb_204269)
        
        # Applying the binary operator '*' (line 426)
        result_mul_204271 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 49), '*', UPDATE_COEFF_204267, result_sub_204270)
        
        # Applying the binary operator '+' (line 426)
        result_add_204272 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 37), '+', lambda_lb_204265, result_mul_204271)
        
        # Processing the call keyword arguments (line 425)
        kwargs_204273 = {}
        # Getting the type of 'max' (line 425)
        max_204257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 33), 'max', False)
        # Calling max(args, kwargs) (line 425)
        max_call_result_204274 = invoke(stypy.reporting.localization.Localization(__file__, 425, 33), max_204257, *[sqrt_call_result_204264, result_add_204272], **kwargs_204273)
        
        # Assigning a type to the variable 'lambda_current' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 16), 'lambda_current', max_call_result_204274)
        # SSA join for if statement (line 390)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 310)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 296)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 428):
        
        # Assigning a Name to a Attribute (line 428):
        # Getting the type of 'lambda_lb' (line 428)
        lambda_lb_204275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 25), 'lambda_lb')
        # Getting the type of 'self' (line 428)
        self_204276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'self')
        # Setting the type of the member 'lambda_lb' of a type (line 428)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 8), self_204276, 'lambda_lb', lambda_lb_204275)
        
        # Assigning a Name to a Attribute (line 429):
        
        # Assigning a Name to a Attribute (line 429):
        # Getting the type of 'lambda_current' (line 429)
        lambda_current_204277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 30), 'lambda_current')
        # Getting the type of 'self' (line 429)
        self_204278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'self')
        # Setting the type of the member 'lambda_current' of a type (line 429)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 8), self_204278, 'lambda_current', lambda_current_204277)
        
        # Assigning a Name to a Attribute (line 430):
        
        # Assigning a Name to a Attribute (line 430):
        # Getting the type of 'tr_radius' (line 430)
        tr_radius_204279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 34), 'tr_radius')
        # Getting the type of 'self' (line 430)
        self_204280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'self')
        # Setting the type of the member 'previous_tr_radius' of a type (line 430)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 8), self_204280, 'previous_tr_radius', tr_radius_204279)
        
        # Obtaining an instance of the builtin type 'tuple' (line 432)
        tuple_204281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 432)
        # Adding element type (line 432)
        # Getting the type of 'p' (line 432)
        p_204282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 15), 'p')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 15), tuple_204281, p_204282)
        # Adding element type (line 432)
        # Getting the type of 'hits_boundary' (line 432)
        hits_boundary_204283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 18), 'hits_boundary')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 15), tuple_204281, hits_boundary_204283)
        
        # Assigning a type to the variable 'stypy_return_type' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'stypy_return_type', tuple_204281)
        
        # ################# End of 'solve(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'solve' in the type store
        # Getting the type of 'stypy_return_type' (line 287)
        stypy_return_type_204284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204284)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'solve'
        return stypy_return_type_204284


# Assigning a type to the variable 'IterativeSubproblem' (line 188)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 0), 'IterativeSubproblem', IterativeSubproblem)

# Assigning a Num to a Name (line 211):
float_204285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 19), 'float')
# Getting the type of 'IterativeSubproblem'
IterativeSubproblem_204286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IterativeSubproblem')
# Setting the type of the member 'UPDATE_COEFF' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IterativeSubproblem_204286, 'UPDATE_COEFF', float_204285)

# Assigning a Attribute to a Name (line 213):

# Call to finfo(...): (line 213)
# Processing the call arguments (line 213)
# Getting the type of 'float' (line 213)
float_204289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 19), 'float', False)
# Processing the call keyword arguments (line 213)
kwargs_204290 = {}
# Getting the type of 'np' (line 213)
np_204287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 10), 'np', False)
# Obtaining the member 'finfo' of a type (line 213)
finfo_204288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 10), np_204287, 'finfo')
# Calling finfo(args, kwargs) (line 213)
finfo_call_result_204291 = invoke(stypy.reporting.localization.Localization(__file__, 213, 10), finfo_204288, *[float_204289], **kwargs_204290)

# Obtaining the member 'eps' of a type (line 213)
eps_204292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 10), finfo_call_result_204291, 'eps')
# Getting the type of 'IterativeSubproblem'
IterativeSubproblem_204293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IterativeSubproblem')
# Setting the type of the member 'EPS' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IterativeSubproblem_204293, 'EPS', eps_204292)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
