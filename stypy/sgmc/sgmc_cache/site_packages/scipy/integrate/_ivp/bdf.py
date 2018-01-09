
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: import numpy as np
3: from scipy.linalg import lu_factor, lu_solve
4: from scipy.sparse import issparse, csc_matrix, eye
5: from scipy.sparse.linalg import splu
6: from scipy.optimize._numdiff import group_columns
7: from .common import (validate_max_step, validate_tol, select_initial_step,
8:                      norm, EPS, num_jac, warn_extraneous)
9: from .base import OdeSolver, DenseOutput
10: 
11: 
12: MAX_ORDER = 5
13: NEWTON_MAXITER = 4
14: MIN_FACTOR = 0.2
15: MAX_FACTOR = 10
16: 
17: 
18: def compute_R(order, factor):
19:     '''Compute the matrix for changing the differences array.'''
20:     I = np.arange(1, order + 1)[:, None]
21:     J = np.arange(1, order + 1)
22:     M = np.zeros((order + 1, order + 1))
23:     M[1:, 1:] = (I - 1 - factor * J) / I
24:     M[0] = 1
25:     return np.cumprod(M, axis=0)
26: 
27: 
28: def change_D(D, order, factor):
29:     '''Change differences array in-place when step size is changed.'''
30:     R = compute_R(order, factor)
31:     U = compute_R(order, 1)
32:     RU = R.dot(U)
33:     D[:order + 1] = np.dot(RU.T, D[:order + 1])
34: 
35: 
36: def solve_bdf_system(fun, t_new, y_predict, c, psi, LU, solve_lu, scale, tol):
37:     '''Solve the algebraic system resulting from BDF method.'''
38:     d = 0
39:     y = y_predict.copy()
40:     dy_norm_old = None
41:     converged = False
42:     for k in range(NEWTON_MAXITER):
43:         f = fun(t_new, y)
44:         if not np.all(np.isfinite(f)):
45:             break
46: 
47:         dy = solve_lu(LU, c * f - psi - d)
48:         dy_norm = norm(dy / scale)
49: 
50:         if dy_norm_old is None:
51:             rate = None
52:         else:
53:             rate = dy_norm / dy_norm_old
54: 
55:         if (rate is not None and (rate >= 1 or
56:                 rate ** (NEWTON_MAXITER - k) / (1 - rate) * dy_norm > tol)):
57:             break
58: 
59:         y += dy
60:         d += dy
61: 
62:         if (dy_norm == 0 or
63:                 rate is not None and rate / (1 - rate) * dy_norm < tol):
64:             converged = True
65:             break
66: 
67:         dy_norm_old = dy_norm
68: 
69:     return converged, k + 1, y, d
70: 
71: 
72: class BDF(OdeSolver):
73:     '''Implicit method based on Backward Differentiation Formulas.
74: 
75:     This is a variable order method with the order varying automatically from
76:     1 to 5. The general framework of the BDF algorithm is described in [1]_.
77:     This class implements a quasi-constant step size approach as explained
78:     in [2]_. The error estimation strategy for the constant step BDF is derived
79:     in [3]_. An accuracy enhancement using modified formulas (NDF) [2]_ is also
80:     implemented.
81: 
82:     Can be applied in a complex domain.
83: 
84:     Parameters
85:     ----------
86:     fun : callable
87:         Right-hand side of the system. The calling signature is ``fun(t, y)``.
88:         Here ``t`` is a scalar and there are two options for ndarray ``y``.
89:         It can either have shape (n,), then ``fun`` must return array_like with
90:         shape (n,). Or alternatively it can have shape (n, k), then ``fun``
91:         must return array_like with shape (n, k), i.e. each column
92:         corresponds to a single column in ``y``. The choice between the two
93:         options is determined by `vectorized` argument (see below). The
94:         vectorized implementation allows faster approximation of the Jacobian
95:         by finite differences.
96:     t0 : float
97:         Initial time.
98:     y0 : array_like, shape (n,)
99:         Initial state.
100:     t_bound : float
101:         Boundary time --- the integration won't continue beyond it. It also
102:         determines the direction of the integration.
103:     max_step : float, optional
104:         Maximum allowed step size. Default is np.inf, i.e. the step is not
105:         bounded and determined solely by the solver.
106:     rtol, atol : float and array_like, optional
107:         Relative and absolute tolerances. The solver keeps the local error
108:         estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
109:         relative accuracy (number of correct digits). But if a component of `y`
110:         is approximately below `atol` then the error only needs to fall within
111:         the same `atol` threshold, and the number of correct digits is not
112:         guaranteed. If components of y have different scales, it might be
113:         beneficial to set different `atol` values for different components by
114:         passing array_like with shape (n,) for `atol`. Default values are
115:         1e-3 for `rtol` and 1e-6 for `atol`.
116:     jac : {None, array_like, sparse_matrix, callable}, optional
117:         Jacobian matrix of the right-hand side of the system with respect to
118:         y, required only by 'Radau' and 'BDF' methods. The Jacobian matrix
119:         has shape (n, n) and its element (i, j) is equal to ``d f_i / d y_j``.
120:         There are 3 ways to define the Jacobian:
121: 
122:             * If array_like or sparse_matrix, then the Jacobian is assumed to
123:               be constant.
124:             * If callable, then the Jacobian is assumed to depend on both
125:               t and y, and will be called as ``jac(t, y)`` as necessary. The
126:               return value might be a sparse matrix.
127:             * If None (default), then the Jacobian will be approximated by
128:               finite differences.
129: 
130:         It is generally recommended to provide the Jacobian rather than
131:         relying on a finite difference approximation.
132:     jac_sparsity : {None, array_like, sparse matrix}, optional
133:         Defines a sparsity structure of the Jacobian matrix for a finite
134:         difference approximation, its shape must be (n, n). If the Jacobian has
135:         only few non-zero elements in *each* row, providing the sparsity
136:         structure will greatly speed up the computations [4]_. A zero
137:         entry means that a corresponding element in the Jacobian is identically
138:         zero. If None (default), the Jacobian is assumed to be dense.
139:     vectorized : bool, optional
140:         Whether `fun` is implemented in a vectorized fashion. Default is False.
141: 
142:     Attributes
143:     ----------
144:     n : int
145:         Number of equations.
146:     status : string
147:         Current status of the solver: 'running', 'finished' or 'failed'.
148:     t_bound : float
149:         Boundary time.
150:     direction : float
151:         Integration direction: +1 or -1.
152:     t : float
153:         Current time.
154:     y : ndarray
155:         Current state.
156:     t_old : float
157:         Previous time. None if no steps were made yet.
158:     step_size : float
159:         Size of the last successful step. None if no steps were made yet.
160:     nfev : int
161:         Number of the system's rhs evaluations.
162:     njev : int
163:         Number of the Jacobian evaluations.
164:     nlu : int
165:         Number of LU decompositions.
166: 
167:     References
168:     ----------
169:     .. [1] G. D. Byrne, A. C. Hindmarsh, "A Polyalgorithm for the Numerical
170:            Solution of Ordinary Differential Equations", ACM Transactions on
171:            Mathematical Software, Vol. 1, No. 1, pp. 71-96, March 1975.
172:     .. [2] L. F. Shampine, M. W. Reichelt, "THE MATLAB ODE SUITE", SIAM J. SCI.
173:            COMPUTE., Vol. 18, No. 1, pp. 1-22, January 1997.
174:     .. [3] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations I:
175:            Nonstiff Problems", Sec. III.2.
176:     .. [4] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
177:            sparse Jacobian matrices", Journal of the Institute of Mathematics
178:            and its Applications, 13, pp. 117-120, 1974.
179:     '''
180:     def __init__(self, fun, t0, y0, t_bound, max_step=np.inf,
181:                  rtol=1e-3, atol=1e-6, jac=None, jac_sparsity=None,
182:                  vectorized=False, **extraneous):
183:         warn_extraneous(extraneous)
184:         super(BDF, self).__init__(fun, t0, y0, t_bound, vectorized,
185:                                   support_complex=True)
186:         self.max_step = validate_max_step(max_step)
187:         self.rtol, self.atol = validate_tol(rtol, atol, self.n)
188:         f = self.fun(self.t, self.y)
189:         self.h_abs = select_initial_step(self.fun, self.t, self.y, f,
190:                                          self.direction, 1,
191:                                          self.rtol, self.atol)
192:         self.h_abs_old = None
193:         self.error_norm_old = None
194: 
195:         self.newton_tol = max(10 * EPS / rtol, min(0.03, rtol ** 0.5))
196: 
197:         self.jac_factor = None
198:         self.jac, self.J = self._validate_jac(jac, jac_sparsity)
199:         if issparse(self.J):
200:             def lu(A):
201:                 self.nlu += 1
202:                 return splu(A)
203: 
204:             def solve_lu(LU, b):
205:                 return LU.solve(b)
206: 
207:             I = eye(self.n, format='csc', dtype=self.y.dtype)
208:         else:
209:             def lu(A):
210:                 self.nlu += 1
211:                 return lu_factor(A, overwrite_a=True)
212: 
213:             def solve_lu(LU, b):
214:                 return lu_solve(LU, b, overwrite_b=True)
215: 
216:             I = np.identity(self.n, dtype=self.y.dtype)
217: 
218:         self.lu = lu
219:         self.solve_lu = solve_lu
220:         self.I = I
221: 
222:         kappa = np.array([0, -0.1850, -1/9, -0.0823, -0.0415, 0])
223:         self.gamma = np.hstack((0, np.cumsum(1 / np.arange(1, MAX_ORDER + 1))))
224:         self.alpha = (1 - kappa) * self.gamma
225:         self.error_const = kappa * self.gamma + 1 / np.arange(1, MAX_ORDER + 2)
226: 
227:         D = np.empty((MAX_ORDER + 3, self.n), dtype=self.y.dtype)
228:         D[0] = self.y
229:         D[1] = f * self.h_abs * self.direction
230:         self.D = D
231: 
232:         self.order = 1
233:         self.n_equal_steps = 0
234:         self.LU = None
235: 
236:     def _validate_jac(self, jac, sparsity):
237:         t0 = self.t
238:         y0 = self.y
239: 
240:         if jac is None:
241:             if sparsity is not None:
242:                 if issparse(sparsity):
243:                     sparsity = csc_matrix(sparsity)
244:                 groups = group_columns(sparsity)
245:                 sparsity = (sparsity, groups)
246: 
247:             def jac_wrapped(t, y):
248:                 self.njev += 1
249:                 f = self.fun_single(t, y)
250:                 J, self.jac_factor = num_jac(self.fun_vectorized, t, y, f,
251:                                              self.atol, self.jac_factor,
252:                                              sparsity)
253:                 return J
254:             J = jac_wrapped(t0, y0)
255:         elif callable(jac):
256:             J = jac(t0, y0)
257:             self.njev += 1
258:             if issparse(J):
259:                 J = csc_matrix(J, dtype=y0.dtype)
260: 
261:                 def jac_wrapped(t, y):
262:                     self.njev += 1
263:                     return csc_matrix(jac(t, y), dtype=y0.dtype)
264:             else:
265:                 J = np.asarray(J, dtype=y0.dtype)
266: 
267:                 def jac_wrapped(t, y):
268:                     self.njev += 1
269:                     return np.asarray(jac(t, y), dtype=y0.dtype)
270: 
271:             if J.shape != (self.n, self.n):
272:                 raise ValueError("`jac` is expected to have shape {}, but "
273:                                  "actually has {}."
274:                                  .format((self.n, self.n), J.shape))
275:         else:
276:             if issparse(jac):
277:                 J = csc_matrix(jac, dtype=y0.dtype)
278:             else:
279:                 J = np.asarray(jac, dtype=y0.dtype)
280: 
281:             if J.shape != (self.n, self.n):
282:                 raise ValueError("`jac` is expected to have shape {}, but "
283:                                  "actually has {}."
284:                                  .format((self.n, self.n), J.shape))
285:             jac_wrapped = None
286: 
287:         return jac_wrapped, J
288: 
289:     def _step_impl(self):
290:         t = self.t
291:         D = self.D
292: 
293:         max_step = self.max_step
294:         min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
295:         if self.h_abs > max_step:
296:             h_abs = max_step
297:             change_D(D, self.order, max_step / self.h_abs)
298:             self.n_equal_steps = 0
299:         elif self.h_abs < min_step:
300:             h_abs = min_step
301:             change_D(D, self.order, min_step / self.h_abs)
302:             self.n_equal_steps = 0
303:         else:
304:             h_abs = self.h_abs
305: 
306:         atol = self.atol
307:         rtol = self.rtol
308:         order = self.order
309: 
310:         alpha = self.alpha
311:         gamma = self.gamma
312:         error_const = self.error_const
313: 
314:         J = self.J
315:         LU = self.LU
316:         current_jac = self.jac is None
317: 
318:         step_accepted = False
319:         while not step_accepted:
320:             if h_abs < min_step:
321:                 return False, self.TOO_SMALL_STEP
322: 
323:             h = h_abs * self.direction
324:             t_new = t + h
325: 
326:             if self.direction * (t_new - self.t_bound) > 0:
327:                 t_new = self.t_bound
328:                 change_D(D, order, np.abs(t_new - t) / h_abs)
329:                 self.n_equal_steps = 0
330:                 LU = None
331: 
332:             h = t_new - t
333:             h_abs = np.abs(h)
334: 
335:             y_predict = np.sum(D[:order + 1], axis=0)
336: 
337:             scale = atol + rtol * np.abs(y_predict)
338:             psi = np.dot(D[1: order + 1].T, gamma[1: order + 1]) / alpha[order]
339: 
340:             converged = False
341:             c = h / alpha[order]
342:             while not converged:
343:                 if LU is None:
344:                     LU = self.lu(self.I - c * J)
345: 
346:                 converged, n_iter, y_new, d = solve_bdf_system(
347:                     self.fun, t_new, y_predict, c, psi, LU, self.solve_lu,
348:                     scale, self.newton_tol)
349: 
350:                 if not converged:
351:                     if current_jac:
352:                         break
353:                     J = self.jac(t_new, y_predict)
354:                     LU = None
355:                     current_jac = True
356: 
357:             if not converged:
358:                 factor = 0.5
359:                 h_abs *= factor
360:                 change_D(D, order, factor)
361:                 self.n_equal_steps = 0
362:                 LU = None
363:                 continue
364: 
365:             safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER
366:                                                        + n_iter)
367: 
368:             scale = atol + rtol * np.abs(y_new)
369:             error = error_const[order] * d
370:             error_norm = norm(error / scale)
371: 
372:             if error_norm > 1:
373:                 factor = max(MIN_FACTOR,
374:                              safety * error_norm ** (-1 / (order + 1)))
375:                 h_abs *= factor
376:                 change_D(D, order, factor)
377:                 self.n_equal_steps = 0
378:                 # As we didn't have problems with convergence, we don't
379:                 # reset LU here.
380:             else:
381:                 step_accepted = True
382: 
383:         self.n_equal_steps += 1
384: 
385:         self.t = t_new
386:         self.y = y_new
387: 
388:         self.h_abs = h_abs
389:         self.J = J
390:         self.LU = LU
391: 
392:         # Update differences. The principal relation here is
393:         # D^{j + 1} y_n = D^{j} y_n - D^{j} y_{n - 1}. Keep in mind that D
394:         # contained difference for previous interpolating polynomial and
395:         # d = D^{k + 1} y_n. Thus this elegant code follows.
396:         D[order + 2] = d - D[order + 1]
397:         D[order + 1] = d
398:         for i in reversed(range(order + 1)):
399:             D[i] += D[i + 1]
400: 
401:         if self.n_equal_steps < order + 1:
402:             return True, None
403: 
404:         if order > 1:
405:             error_m = error_const[order - 1] * D[order]
406:             error_m_norm = norm(error_m / scale)
407:         else:
408:             error_m_norm = np.inf
409: 
410:         if order < MAX_ORDER:
411:             error_p = error_const[order + 1] * D[order + 2]
412:             error_p_norm = norm(error_p / scale)
413:         else:
414:             error_p_norm = np.inf
415: 
416:         error_norms = np.array([error_m_norm, error_norm, error_p_norm])
417:         factors = error_norms ** (-1 / np.arange(order, order + 3))
418: 
419:         delta_order = np.argmax(factors) - 1
420:         order += delta_order
421:         self.order = order
422: 
423:         factor = min(MAX_FACTOR, safety * np.max(factors))
424:         self.h_abs *= factor
425:         change_D(D, order, factor)
426:         self.n_equal_steps = 0
427:         self.LU = None
428: 
429:         return True, None
430: 
431:     def _dense_output_impl(self):
432:         return BdfDenseOutput(self.t_old, self.t, self.h_abs * self.direction,
433:                               self.order, self.D[:self.order + 1].copy())
434: 
435: 
436: class BdfDenseOutput(DenseOutput):
437:     def __init__(self, t_old, t, h, order, D):
438:         super(BdfDenseOutput, self).__init__(t_old, t)
439:         self.order = order
440:         self.t_shift = self.t - h * np.arange(self.order)
441:         self.denom = h * (1 + np.arange(self.order))
442:         self.D = D
443: 
444:     def _call_impl(self, t):
445:         if t.ndim == 0:
446:             x = (t - self.t_shift) / self.denom
447:             p = np.cumprod(x)
448:         else:
449:             x = (t - self.t_shift[:, None]) / self.denom[:, None]
450:             p = np.cumprod(x, axis=0)
451: 
452:         y = np.dot(self.D[1:].T, p)
453:         if y.ndim == 1:
454:             y += self.D[0]
455:         else:
456:             y += self.D[0, :, None]
457: 
458:         return y
459: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import numpy' statement (line 2)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_52587 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy')

if (type(import_52587) is not StypyTypeError):

    if (import_52587 != 'pyd_module'):
        __import__(import_52587)
        sys_modules_52588 = sys.modules[import_52587]
        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'np', sys_modules_52588.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy', import_52587)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from scipy.linalg import lu_factor, lu_solve' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_52589 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.linalg')

if (type(import_52589) is not StypyTypeError):

    if (import_52589 != 'pyd_module'):
        __import__(import_52589)
        sys_modules_52590 = sys.modules[import_52589]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.linalg', sys_modules_52590.module_type_store, module_type_store, ['lu_factor', 'lu_solve'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_52590, sys_modules_52590.module_type_store, module_type_store)
    else:
        from scipy.linalg import lu_factor, lu_solve

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.linalg', None, module_type_store, ['lu_factor', 'lu_solve'], [lu_factor, lu_solve])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.linalg', import_52589)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from scipy.sparse import issparse, csc_matrix, eye' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_52591 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.sparse')

if (type(import_52591) is not StypyTypeError):

    if (import_52591 != 'pyd_module'):
        __import__(import_52591)
        sys_modules_52592 = sys.modules[import_52591]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.sparse', sys_modules_52592.module_type_store, module_type_store, ['issparse', 'csc_matrix', 'eye'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_52592, sys_modules_52592.module_type_store, module_type_store)
    else:
        from scipy.sparse import issparse, csc_matrix, eye

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.sparse', None, module_type_store, ['issparse', 'csc_matrix', 'eye'], [issparse, csc_matrix, eye])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.sparse', import_52591)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.sparse.linalg import splu' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_52593 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse.linalg')

if (type(import_52593) is not StypyTypeError):

    if (import_52593 != 'pyd_module'):
        __import__(import_52593)
        sys_modules_52594 = sys.modules[import_52593]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse.linalg', sys_modules_52594.module_type_store, module_type_store, ['splu'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_52594, sys_modules_52594.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import splu

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse.linalg', None, module_type_store, ['splu'], [splu])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse.linalg', import_52593)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.optimize._numdiff import group_columns' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_52595 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize._numdiff')

if (type(import_52595) is not StypyTypeError):

    if (import_52595 != 'pyd_module'):
        __import__(import_52595)
        sys_modules_52596 = sys.modules[import_52595]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize._numdiff', sys_modules_52596.module_type_store, module_type_store, ['group_columns'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_52596, sys_modules_52596.module_type_store, module_type_store)
    else:
        from scipy.optimize._numdiff import group_columns

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize._numdiff', None, module_type_store, ['group_columns'], [group_columns])

else:
    # Assigning a type to the variable 'scipy.optimize._numdiff' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize._numdiff', import_52595)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.integrate._ivp.common import validate_max_step, validate_tol, select_initial_step, norm, EPS, num_jac, warn_extraneous' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_52597 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.integrate._ivp.common')

if (type(import_52597) is not StypyTypeError):

    if (import_52597 != 'pyd_module'):
        __import__(import_52597)
        sys_modules_52598 = sys.modules[import_52597]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.integrate._ivp.common', sys_modules_52598.module_type_store, module_type_store, ['validate_max_step', 'validate_tol', 'select_initial_step', 'norm', 'EPS', 'num_jac', 'warn_extraneous'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_52598, sys_modules_52598.module_type_store, module_type_store)
    else:
        from scipy.integrate._ivp.common import validate_max_step, validate_tol, select_initial_step, norm, EPS, num_jac, warn_extraneous

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.integrate._ivp.common', None, module_type_store, ['validate_max_step', 'validate_tol', 'select_initial_step', 'norm', 'EPS', 'num_jac', 'warn_extraneous'], [validate_max_step, validate_tol, select_initial_step, norm, EPS, num_jac, warn_extraneous])

else:
    # Assigning a type to the variable 'scipy.integrate._ivp.common' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.integrate._ivp.common', import_52597)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.integrate._ivp.base import OdeSolver, DenseOutput' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_52599 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.integrate._ivp.base')

if (type(import_52599) is not StypyTypeError):

    if (import_52599 != 'pyd_module'):
        __import__(import_52599)
        sys_modules_52600 = sys.modules[import_52599]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.integrate._ivp.base', sys_modules_52600.module_type_store, module_type_store, ['OdeSolver', 'DenseOutput'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_52600, sys_modules_52600.module_type_store, module_type_store)
    else:
        from scipy.integrate._ivp.base import OdeSolver, DenseOutput

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.integrate._ivp.base', None, module_type_store, ['OdeSolver', 'DenseOutput'], [OdeSolver, DenseOutput])

else:
    # Assigning a type to the variable 'scipy.integrate._ivp.base' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.integrate._ivp.base', import_52599)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')


# Assigning a Num to a Name (line 12):

# Assigning a Num to a Name (line 12):
int_52601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 12), 'int')
# Assigning a type to the variable 'MAX_ORDER' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'MAX_ORDER', int_52601)

# Assigning a Num to a Name (line 13):

# Assigning a Num to a Name (line 13):
int_52602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 17), 'int')
# Assigning a type to the variable 'NEWTON_MAXITER' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'NEWTON_MAXITER', int_52602)

# Assigning a Num to a Name (line 14):

# Assigning a Num to a Name (line 14):
float_52603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 13), 'float')
# Assigning a type to the variable 'MIN_FACTOR' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'MIN_FACTOR', float_52603)

# Assigning a Num to a Name (line 15):

# Assigning a Num to a Name (line 15):
int_52604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 13), 'int')
# Assigning a type to the variable 'MAX_FACTOR' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'MAX_FACTOR', int_52604)

@norecursion
def compute_R(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'compute_R'
    module_type_store = module_type_store.open_function_context('compute_R', 18, 0, False)
    
    # Passed parameters checking function
    compute_R.stypy_localization = localization
    compute_R.stypy_type_of_self = None
    compute_R.stypy_type_store = module_type_store
    compute_R.stypy_function_name = 'compute_R'
    compute_R.stypy_param_names_list = ['order', 'factor']
    compute_R.stypy_varargs_param_name = None
    compute_R.stypy_kwargs_param_name = None
    compute_R.stypy_call_defaults = defaults
    compute_R.stypy_call_varargs = varargs
    compute_R.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'compute_R', ['order', 'factor'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'compute_R', localization, ['order', 'factor'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'compute_R(...)' code ##################

    str_52605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 4), 'str', 'Compute the matrix for changing the differences array.')
    
    # Assigning a Subscript to a Name (line 20):
    
    # Assigning a Subscript to a Name (line 20):
    
    # Obtaining the type of the subscript
    slice_52606 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 20, 8), None, None, None)
    # Getting the type of 'None' (line 20)
    None_52607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 35), 'None')
    
    # Call to arange(...): (line 20)
    # Processing the call arguments (line 20)
    int_52610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 18), 'int')
    # Getting the type of 'order' (line 20)
    order_52611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 21), 'order', False)
    int_52612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 29), 'int')
    # Applying the binary operator '+' (line 20)
    result_add_52613 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 21), '+', order_52611, int_52612)
    
    # Processing the call keyword arguments (line 20)
    kwargs_52614 = {}
    # Getting the type of 'np' (line 20)
    np_52608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 20)
    arange_52609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), np_52608, 'arange')
    # Calling arange(args, kwargs) (line 20)
    arange_call_result_52615 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), arange_52609, *[int_52610, result_add_52613], **kwargs_52614)
    
    # Obtaining the member '__getitem__' of a type (line 20)
    getitem___52616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), arange_call_result_52615, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 20)
    subscript_call_result_52617 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), getitem___52616, (slice_52606, None_52607))
    
    # Assigning a type to the variable 'I' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'I', subscript_call_result_52617)
    
    # Assigning a Call to a Name (line 21):
    
    # Assigning a Call to a Name (line 21):
    
    # Call to arange(...): (line 21)
    # Processing the call arguments (line 21)
    int_52620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 18), 'int')
    # Getting the type of 'order' (line 21)
    order_52621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 21), 'order', False)
    int_52622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 29), 'int')
    # Applying the binary operator '+' (line 21)
    result_add_52623 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 21), '+', order_52621, int_52622)
    
    # Processing the call keyword arguments (line 21)
    kwargs_52624 = {}
    # Getting the type of 'np' (line 21)
    np_52618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 21)
    arange_52619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), np_52618, 'arange')
    # Calling arange(args, kwargs) (line 21)
    arange_call_result_52625 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), arange_52619, *[int_52620, result_add_52623], **kwargs_52624)
    
    # Assigning a type to the variable 'J' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'J', arange_call_result_52625)
    
    # Assigning a Call to a Name (line 22):
    
    # Assigning a Call to a Name (line 22):
    
    # Call to zeros(...): (line 22)
    # Processing the call arguments (line 22)
    
    # Obtaining an instance of the builtin type 'tuple' (line 22)
    tuple_52628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 22)
    # Adding element type (line 22)
    # Getting the type of 'order' (line 22)
    order_52629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 18), 'order', False)
    int_52630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 26), 'int')
    # Applying the binary operator '+' (line 22)
    result_add_52631 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 18), '+', order_52629, int_52630)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 18), tuple_52628, result_add_52631)
    # Adding element type (line 22)
    # Getting the type of 'order' (line 22)
    order_52632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 29), 'order', False)
    int_52633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 37), 'int')
    # Applying the binary operator '+' (line 22)
    result_add_52634 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 29), '+', order_52632, int_52633)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 18), tuple_52628, result_add_52634)
    
    # Processing the call keyword arguments (line 22)
    kwargs_52635 = {}
    # Getting the type of 'np' (line 22)
    np_52626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 22)
    zeros_52627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), np_52626, 'zeros')
    # Calling zeros(args, kwargs) (line 22)
    zeros_call_result_52636 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), zeros_52627, *[tuple_52628], **kwargs_52635)
    
    # Assigning a type to the variable 'M' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'M', zeros_call_result_52636)
    
    # Assigning a BinOp to a Subscript (line 23):
    
    # Assigning a BinOp to a Subscript (line 23):
    # Getting the type of 'I' (line 23)
    I_52637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 17), 'I')
    int_52638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 21), 'int')
    # Applying the binary operator '-' (line 23)
    result_sub_52639 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 17), '-', I_52637, int_52638)
    
    # Getting the type of 'factor' (line 23)
    factor_52640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 25), 'factor')
    # Getting the type of 'J' (line 23)
    J_52641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 34), 'J')
    # Applying the binary operator '*' (line 23)
    result_mul_52642 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 25), '*', factor_52640, J_52641)
    
    # Applying the binary operator '-' (line 23)
    result_sub_52643 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 23), '-', result_sub_52639, result_mul_52642)
    
    # Getting the type of 'I' (line 23)
    I_52644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 39), 'I')
    # Applying the binary operator 'div' (line 23)
    result_div_52645 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 16), 'div', result_sub_52643, I_52644)
    
    # Getting the type of 'M' (line 23)
    M_52646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'M')
    int_52647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 6), 'int')
    slice_52648 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 23, 4), int_52647, None, None)
    int_52649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 10), 'int')
    slice_52650 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 23, 4), int_52649, None, None)
    # Storing an element on a container (line 23)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 4), M_52646, ((slice_52648, slice_52650), result_div_52645))
    
    # Assigning a Num to a Subscript (line 24):
    
    # Assigning a Num to a Subscript (line 24):
    int_52651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 11), 'int')
    # Getting the type of 'M' (line 24)
    M_52652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'M')
    int_52653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 6), 'int')
    # Storing an element on a container (line 24)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 4), M_52652, (int_52653, int_52651))
    
    # Call to cumprod(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'M' (line 25)
    M_52656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 22), 'M', False)
    # Processing the call keyword arguments (line 25)
    int_52657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 30), 'int')
    keyword_52658 = int_52657
    kwargs_52659 = {'axis': keyword_52658}
    # Getting the type of 'np' (line 25)
    np_52654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 11), 'np', False)
    # Obtaining the member 'cumprod' of a type (line 25)
    cumprod_52655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 11), np_52654, 'cumprod')
    # Calling cumprod(args, kwargs) (line 25)
    cumprod_call_result_52660 = invoke(stypy.reporting.localization.Localization(__file__, 25, 11), cumprod_52655, *[M_52656], **kwargs_52659)
    
    # Assigning a type to the variable 'stypy_return_type' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type', cumprod_call_result_52660)
    
    # ################# End of 'compute_R(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'compute_R' in the type store
    # Getting the type of 'stypy_return_type' (line 18)
    stypy_return_type_52661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_52661)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'compute_R'
    return stypy_return_type_52661

# Assigning a type to the variable 'compute_R' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'compute_R', compute_R)

@norecursion
def change_D(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'change_D'
    module_type_store = module_type_store.open_function_context('change_D', 28, 0, False)
    
    # Passed parameters checking function
    change_D.stypy_localization = localization
    change_D.stypy_type_of_self = None
    change_D.stypy_type_store = module_type_store
    change_D.stypy_function_name = 'change_D'
    change_D.stypy_param_names_list = ['D', 'order', 'factor']
    change_D.stypy_varargs_param_name = None
    change_D.stypy_kwargs_param_name = None
    change_D.stypy_call_defaults = defaults
    change_D.stypy_call_varargs = varargs
    change_D.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'change_D', ['D', 'order', 'factor'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'change_D', localization, ['D', 'order', 'factor'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'change_D(...)' code ##################

    str_52662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'str', 'Change differences array in-place when step size is changed.')
    
    # Assigning a Call to a Name (line 30):
    
    # Assigning a Call to a Name (line 30):
    
    # Call to compute_R(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'order' (line 30)
    order_52664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 18), 'order', False)
    # Getting the type of 'factor' (line 30)
    factor_52665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 25), 'factor', False)
    # Processing the call keyword arguments (line 30)
    kwargs_52666 = {}
    # Getting the type of 'compute_R' (line 30)
    compute_R_52663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'compute_R', False)
    # Calling compute_R(args, kwargs) (line 30)
    compute_R_call_result_52667 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), compute_R_52663, *[order_52664, factor_52665], **kwargs_52666)
    
    # Assigning a type to the variable 'R' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'R', compute_R_call_result_52667)
    
    # Assigning a Call to a Name (line 31):
    
    # Assigning a Call to a Name (line 31):
    
    # Call to compute_R(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'order' (line 31)
    order_52669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 18), 'order', False)
    int_52670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 25), 'int')
    # Processing the call keyword arguments (line 31)
    kwargs_52671 = {}
    # Getting the type of 'compute_R' (line 31)
    compute_R_52668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'compute_R', False)
    # Calling compute_R(args, kwargs) (line 31)
    compute_R_call_result_52672 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), compute_R_52668, *[order_52669, int_52670], **kwargs_52671)
    
    # Assigning a type to the variable 'U' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'U', compute_R_call_result_52672)
    
    # Assigning a Call to a Name (line 32):
    
    # Assigning a Call to a Name (line 32):
    
    # Call to dot(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'U' (line 32)
    U_52675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'U', False)
    # Processing the call keyword arguments (line 32)
    kwargs_52676 = {}
    # Getting the type of 'R' (line 32)
    R_52673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 9), 'R', False)
    # Obtaining the member 'dot' of a type (line 32)
    dot_52674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 9), R_52673, 'dot')
    # Calling dot(args, kwargs) (line 32)
    dot_call_result_52677 = invoke(stypy.reporting.localization.Localization(__file__, 32, 9), dot_52674, *[U_52675], **kwargs_52676)
    
    # Assigning a type to the variable 'RU' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'RU', dot_call_result_52677)
    
    # Assigning a Call to a Subscript (line 33):
    
    # Assigning a Call to a Subscript (line 33):
    
    # Call to dot(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'RU' (line 33)
    RU_52680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 27), 'RU', False)
    # Obtaining the member 'T' of a type (line 33)
    T_52681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 27), RU_52680, 'T')
    
    # Obtaining the type of the subscript
    # Getting the type of 'order' (line 33)
    order_52682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 36), 'order', False)
    int_52683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 44), 'int')
    # Applying the binary operator '+' (line 33)
    result_add_52684 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 36), '+', order_52682, int_52683)
    
    slice_52685 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 33, 33), None, result_add_52684, None)
    # Getting the type of 'D' (line 33)
    D_52686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 33), 'D', False)
    # Obtaining the member '__getitem__' of a type (line 33)
    getitem___52687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 33), D_52686, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 33)
    subscript_call_result_52688 = invoke(stypy.reporting.localization.Localization(__file__, 33, 33), getitem___52687, slice_52685)
    
    # Processing the call keyword arguments (line 33)
    kwargs_52689 = {}
    # Getting the type of 'np' (line 33)
    np_52678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 20), 'np', False)
    # Obtaining the member 'dot' of a type (line 33)
    dot_52679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 20), np_52678, 'dot')
    # Calling dot(args, kwargs) (line 33)
    dot_call_result_52690 = invoke(stypy.reporting.localization.Localization(__file__, 33, 20), dot_52679, *[T_52681, subscript_call_result_52688], **kwargs_52689)
    
    # Getting the type of 'D' (line 33)
    D_52691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'D')
    # Getting the type of 'order' (line 33)
    order_52692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 7), 'order')
    int_52693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 15), 'int')
    # Applying the binary operator '+' (line 33)
    result_add_52694 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 7), '+', order_52692, int_52693)
    
    slice_52695 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 33, 4), None, result_add_52694, None)
    # Storing an element on a container (line 33)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 4), D_52691, (slice_52695, dot_call_result_52690))
    
    # ################# End of 'change_D(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'change_D' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_52696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_52696)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'change_D'
    return stypy_return_type_52696

# Assigning a type to the variable 'change_D' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'change_D', change_D)

@norecursion
def solve_bdf_system(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'solve_bdf_system'
    module_type_store = module_type_store.open_function_context('solve_bdf_system', 36, 0, False)
    
    # Passed parameters checking function
    solve_bdf_system.stypy_localization = localization
    solve_bdf_system.stypy_type_of_self = None
    solve_bdf_system.stypy_type_store = module_type_store
    solve_bdf_system.stypy_function_name = 'solve_bdf_system'
    solve_bdf_system.stypy_param_names_list = ['fun', 't_new', 'y_predict', 'c', 'psi', 'LU', 'solve_lu', 'scale', 'tol']
    solve_bdf_system.stypy_varargs_param_name = None
    solve_bdf_system.stypy_kwargs_param_name = None
    solve_bdf_system.stypy_call_defaults = defaults
    solve_bdf_system.stypy_call_varargs = varargs
    solve_bdf_system.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'solve_bdf_system', ['fun', 't_new', 'y_predict', 'c', 'psi', 'LU', 'solve_lu', 'scale', 'tol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'solve_bdf_system', localization, ['fun', 't_new', 'y_predict', 'c', 'psi', 'LU', 'solve_lu', 'scale', 'tol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'solve_bdf_system(...)' code ##################

    str_52697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 4), 'str', 'Solve the algebraic system resulting from BDF method.')
    
    # Assigning a Num to a Name (line 38):
    
    # Assigning a Num to a Name (line 38):
    int_52698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 8), 'int')
    # Assigning a type to the variable 'd' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'd', int_52698)
    
    # Assigning a Call to a Name (line 39):
    
    # Assigning a Call to a Name (line 39):
    
    # Call to copy(...): (line 39)
    # Processing the call keyword arguments (line 39)
    kwargs_52701 = {}
    # Getting the type of 'y_predict' (line 39)
    y_predict_52699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'y_predict', False)
    # Obtaining the member 'copy' of a type (line 39)
    copy_52700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), y_predict_52699, 'copy')
    # Calling copy(args, kwargs) (line 39)
    copy_call_result_52702 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), copy_52700, *[], **kwargs_52701)
    
    # Assigning a type to the variable 'y' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'y', copy_call_result_52702)
    
    # Assigning a Name to a Name (line 40):
    
    # Assigning a Name to a Name (line 40):
    # Getting the type of 'None' (line 40)
    None_52703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 18), 'None')
    # Assigning a type to the variable 'dy_norm_old' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'dy_norm_old', None_52703)
    
    # Assigning a Name to a Name (line 41):
    
    # Assigning a Name to a Name (line 41):
    # Getting the type of 'False' (line 41)
    False_52704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'False')
    # Assigning a type to the variable 'converged' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'converged', False_52704)
    
    
    # Call to range(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'NEWTON_MAXITER' (line 42)
    NEWTON_MAXITER_52706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 19), 'NEWTON_MAXITER', False)
    # Processing the call keyword arguments (line 42)
    kwargs_52707 = {}
    # Getting the type of 'range' (line 42)
    range_52705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 13), 'range', False)
    # Calling range(args, kwargs) (line 42)
    range_call_result_52708 = invoke(stypy.reporting.localization.Localization(__file__, 42, 13), range_52705, *[NEWTON_MAXITER_52706], **kwargs_52707)
    
    # Testing the type of a for loop iterable (line 42)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 42, 4), range_call_result_52708)
    # Getting the type of the for loop variable (line 42)
    for_loop_var_52709 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 42, 4), range_call_result_52708)
    # Assigning a type to the variable 'k' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'k', for_loop_var_52709)
    # SSA begins for a for statement (line 42)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 43):
    
    # Assigning a Call to a Name (line 43):
    
    # Call to fun(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 't_new' (line 43)
    t_new_52711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 16), 't_new', False)
    # Getting the type of 'y' (line 43)
    y_52712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 23), 'y', False)
    # Processing the call keyword arguments (line 43)
    kwargs_52713 = {}
    # Getting the type of 'fun' (line 43)
    fun_52710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'fun', False)
    # Calling fun(args, kwargs) (line 43)
    fun_call_result_52714 = invoke(stypy.reporting.localization.Localization(__file__, 43, 12), fun_52710, *[t_new_52711, y_52712], **kwargs_52713)
    
    # Assigning a type to the variable 'f' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'f', fun_call_result_52714)
    
    
    
    # Call to all(...): (line 44)
    # Processing the call arguments (line 44)
    
    # Call to isfinite(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'f' (line 44)
    f_52719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 34), 'f', False)
    # Processing the call keyword arguments (line 44)
    kwargs_52720 = {}
    # Getting the type of 'np' (line 44)
    np_52717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 22), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 44)
    isfinite_52718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 22), np_52717, 'isfinite')
    # Calling isfinite(args, kwargs) (line 44)
    isfinite_call_result_52721 = invoke(stypy.reporting.localization.Localization(__file__, 44, 22), isfinite_52718, *[f_52719], **kwargs_52720)
    
    # Processing the call keyword arguments (line 44)
    kwargs_52722 = {}
    # Getting the type of 'np' (line 44)
    np_52715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'np', False)
    # Obtaining the member 'all' of a type (line 44)
    all_52716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 15), np_52715, 'all')
    # Calling all(args, kwargs) (line 44)
    all_call_result_52723 = invoke(stypy.reporting.localization.Localization(__file__, 44, 15), all_52716, *[isfinite_call_result_52721], **kwargs_52722)
    
    # Applying the 'not' unary operator (line 44)
    result_not__52724 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 11), 'not', all_call_result_52723)
    
    # Testing the type of an if condition (line 44)
    if_condition_52725 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 8), result_not__52724)
    # Assigning a type to the variable 'if_condition_52725' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'if_condition_52725', if_condition_52725)
    # SSA begins for if statement (line 44)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 44)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 47):
    
    # Assigning a Call to a Name (line 47):
    
    # Call to solve_lu(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'LU' (line 47)
    LU_52727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 22), 'LU', False)
    # Getting the type of 'c' (line 47)
    c_52728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 26), 'c', False)
    # Getting the type of 'f' (line 47)
    f_52729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 30), 'f', False)
    # Applying the binary operator '*' (line 47)
    result_mul_52730 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 26), '*', c_52728, f_52729)
    
    # Getting the type of 'psi' (line 47)
    psi_52731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 34), 'psi', False)
    # Applying the binary operator '-' (line 47)
    result_sub_52732 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 26), '-', result_mul_52730, psi_52731)
    
    # Getting the type of 'd' (line 47)
    d_52733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 40), 'd', False)
    # Applying the binary operator '-' (line 47)
    result_sub_52734 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 38), '-', result_sub_52732, d_52733)
    
    # Processing the call keyword arguments (line 47)
    kwargs_52735 = {}
    # Getting the type of 'solve_lu' (line 47)
    solve_lu_52726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 13), 'solve_lu', False)
    # Calling solve_lu(args, kwargs) (line 47)
    solve_lu_call_result_52736 = invoke(stypy.reporting.localization.Localization(__file__, 47, 13), solve_lu_52726, *[LU_52727, result_sub_52734], **kwargs_52735)
    
    # Assigning a type to the variable 'dy' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'dy', solve_lu_call_result_52736)
    
    # Assigning a Call to a Name (line 48):
    
    # Assigning a Call to a Name (line 48):
    
    # Call to norm(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'dy' (line 48)
    dy_52738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 23), 'dy', False)
    # Getting the type of 'scale' (line 48)
    scale_52739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 28), 'scale', False)
    # Applying the binary operator 'div' (line 48)
    result_div_52740 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 23), 'div', dy_52738, scale_52739)
    
    # Processing the call keyword arguments (line 48)
    kwargs_52741 = {}
    # Getting the type of 'norm' (line 48)
    norm_52737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 18), 'norm', False)
    # Calling norm(args, kwargs) (line 48)
    norm_call_result_52742 = invoke(stypy.reporting.localization.Localization(__file__, 48, 18), norm_52737, *[result_div_52740], **kwargs_52741)
    
    # Assigning a type to the variable 'dy_norm' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'dy_norm', norm_call_result_52742)
    
    # Type idiom detected: calculating its left and rigth part (line 50)
    # Getting the type of 'dy_norm_old' (line 50)
    dy_norm_old_52743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'dy_norm_old')
    # Getting the type of 'None' (line 50)
    None_52744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 26), 'None')
    
    (may_be_52745, more_types_in_union_52746) = may_be_none(dy_norm_old_52743, None_52744)

    if may_be_52745:

        if more_types_in_union_52746:
            # Runtime conditional SSA (line 50)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 51):
        
        # Assigning a Name to a Name (line 51):
        # Getting the type of 'None' (line 51)
        None_52747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), 'None')
        # Assigning a type to the variable 'rate' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'rate', None_52747)

        if more_types_in_union_52746:
            # Runtime conditional SSA for else branch (line 50)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_52745) or more_types_in_union_52746):
        
        # Assigning a BinOp to a Name (line 53):
        
        # Assigning a BinOp to a Name (line 53):
        # Getting the type of 'dy_norm' (line 53)
        dy_norm_52748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 19), 'dy_norm')
        # Getting the type of 'dy_norm_old' (line 53)
        dy_norm_old_52749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 29), 'dy_norm_old')
        # Applying the binary operator 'div' (line 53)
        result_div_52750 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 19), 'div', dy_norm_52748, dy_norm_old_52749)
        
        # Assigning a type to the variable 'rate' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'rate', result_div_52750)

        if (may_be_52745 and more_types_in_union_52746):
            # SSA join for if statement (line 50)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'rate' (line 55)
    rate_52751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'rate')
    # Getting the type of 'None' (line 55)
    None_52752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 24), 'None')
    # Applying the binary operator 'isnot' (line 55)
    result_is_not_52753 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 12), 'isnot', rate_52751, None_52752)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'rate' (line 55)
    rate_52754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 34), 'rate')
    int_52755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 42), 'int')
    # Applying the binary operator '>=' (line 55)
    result_ge_52756 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 34), '>=', rate_52754, int_52755)
    
    
    # Getting the type of 'rate' (line 56)
    rate_52757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'rate')
    # Getting the type of 'NEWTON_MAXITER' (line 56)
    NEWTON_MAXITER_52758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 25), 'NEWTON_MAXITER')
    # Getting the type of 'k' (line 56)
    k_52759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 42), 'k')
    # Applying the binary operator '-' (line 56)
    result_sub_52760 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 25), '-', NEWTON_MAXITER_52758, k_52759)
    
    # Applying the binary operator '**' (line 56)
    result_pow_52761 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 16), '**', rate_52757, result_sub_52760)
    
    int_52762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 48), 'int')
    # Getting the type of 'rate' (line 56)
    rate_52763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 52), 'rate')
    # Applying the binary operator '-' (line 56)
    result_sub_52764 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 48), '-', int_52762, rate_52763)
    
    # Applying the binary operator 'div' (line 56)
    result_div_52765 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 16), 'div', result_pow_52761, result_sub_52764)
    
    # Getting the type of 'dy_norm' (line 56)
    dy_norm_52766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 60), 'dy_norm')
    # Applying the binary operator '*' (line 56)
    result_mul_52767 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 58), '*', result_div_52765, dy_norm_52766)
    
    # Getting the type of 'tol' (line 56)
    tol_52768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 70), 'tol')
    # Applying the binary operator '>' (line 56)
    result_gt_52769 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 16), '>', result_mul_52767, tol_52768)
    
    # Applying the binary operator 'or' (line 55)
    result_or_keyword_52770 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 34), 'or', result_ge_52756, result_gt_52769)
    
    # Applying the binary operator 'and' (line 55)
    result_and_keyword_52771 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 12), 'and', result_is_not_52753, result_or_keyword_52770)
    
    # Testing the type of an if condition (line 55)
    if_condition_52772 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 55, 8), result_and_keyword_52771)
    # Assigning a type to the variable 'if_condition_52772' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'if_condition_52772', if_condition_52772)
    # SSA begins for if statement (line 55)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 55)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'y' (line 59)
    y_52773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'y')
    # Getting the type of 'dy' (line 59)
    dy_52774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 13), 'dy')
    # Applying the binary operator '+=' (line 59)
    result_iadd_52775 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 8), '+=', y_52773, dy_52774)
    # Assigning a type to the variable 'y' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'y', result_iadd_52775)
    
    
    # Getting the type of 'd' (line 60)
    d_52776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'd')
    # Getting the type of 'dy' (line 60)
    dy_52777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 13), 'dy')
    # Applying the binary operator '+=' (line 60)
    result_iadd_52778 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 8), '+=', d_52776, dy_52777)
    # Assigning a type to the variable 'd' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'd', result_iadd_52778)
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'dy_norm' (line 62)
    dy_norm_52779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'dy_norm')
    int_52780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 23), 'int')
    # Applying the binary operator '==' (line 62)
    result_eq_52781 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 12), '==', dy_norm_52779, int_52780)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'rate' (line 63)
    rate_52782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), 'rate')
    # Getting the type of 'None' (line 63)
    None_52783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 28), 'None')
    # Applying the binary operator 'isnot' (line 63)
    result_is_not_52784 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 16), 'isnot', rate_52782, None_52783)
    
    
    # Getting the type of 'rate' (line 63)
    rate_52785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 37), 'rate')
    int_52786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 45), 'int')
    # Getting the type of 'rate' (line 63)
    rate_52787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 49), 'rate')
    # Applying the binary operator '-' (line 63)
    result_sub_52788 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 45), '-', int_52786, rate_52787)
    
    # Applying the binary operator 'div' (line 63)
    result_div_52789 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 37), 'div', rate_52785, result_sub_52788)
    
    # Getting the type of 'dy_norm' (line 63)
    dy_norm_52790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 57), 'dy_norm')
    # Applying the binary operator '*' (line 63)
    result_mul_52791 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 55), '*', result_div_52789, dy_norm_52790)
    
    # Getting the type of 'tol' (line 63)
    tol_52792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 67), 'tol')
    # Applying the binary operator '<' (line 63)
    result_lt_52793 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 37), '<', result_mul_52791, tol_52792)
    
    # Applying the binary operator 'and' (line 63)
    result_and_keyword_52794 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 16), 'and', result_is_not_52784, result_lt_52793)
    
    # Applying the binary operator 'or' (line 62)
    result_or_keyword_52795 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 12), 'or', result_eq_52781, result_and_keyword_52794)
    
    # Testing the type of an if condition (line 62)
    if_condition_52796 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 8), result_or_keyword_52795)
    # Assigning a type to the variable 'if_condition_52796' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'if_condition_52796', if_condition_52796)
    # SSA begins for if statement (line 62)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 64):
    
    # Assigning a Name to a Name (line 64):
    # Getting the type of 'True' (line 64)
    True_52797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 24), 'True')
    # Assigning a type to the variable 'converged' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'converged', True_52797)
    # SSA join for if statement (line 62)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 67):
    
    # Assigning a Name to a Name (line 67):
    # Getting the type of 'dy_norm' (line 67)
    dy_norm_52798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 22), 'dy_norm')
    # Assigning a type to the variable 'dy_norm_old' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'dy_norm_old', dy_norm_52798)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 69)
    tuple_52799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 69)
    # Adding element type (line 69)
    # Getting the type of 'converged' (line 69)
    converged_52800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), 'converged')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 11), tuple_52799, converged_52800)
    # Adding element type (line 69)
    # Getting the type of 'k' (line 69)
    k_52801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 22), 'k')
    int_52802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 26), 'int')
    # Applying the binary operator '+' (line 69)
    result_add_52803 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 22), '+', k_52801, int_52802)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 11), tuple_52799, result_add_52803)
    # Adding element type (line 69)
    # Getting the type of 'y' (line 69)
    y_52804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 29), 'y')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 11), tuple_52799, y_52804)
    # Adding element type (line 69)
    # Getting the type of 'd' (line 69)
    d_52805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 32), 'd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 11), tuple_52799, d_52805)
    
    # Assigning a type to the variable 'stypy_return_type' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type', tuple_52799)
    
    # ################# End of 'solve_bdf_system(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'solve_bdf_system' in the type store
    # Getting the type of 'stypy_return_type' (line 36)
    stypy_return_type_52806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_52806)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'solve_bdf_system'
    return stypy_return_type_52806

# Assigning a type to the variable 'solve_bdf_system' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'solve_bdf_system', solve_bdf_system)
# Declaration of the 'BDF' class
# Getting the type of 'OdeSolver' (line 72)
OdeSolver_52807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 10), 'OdeSolver')

class BDF(OdeSolver_52807, ):
    str_52808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, (-1)), 'str', 'Implicit method based on Backward Differentiation Formulas.\n\n    This is a variable order method with the order varying automatically from\n    1 to 5. The general framework of the BDF algorithm is described in [1]_.\n    This class implements a quasi-constant step size approach as explained\n    in [2]_. The error estimation strategy for the constant step BDF is derived\n    in [3]_. An accuracy enhancement using modified formulas (NDF) [2]_ is also\n    implemented.\n\n    Can be applied in a complex domain.\n\n    Parameters\n    ----------\n    fun : callable\n        Right-hand side of the system. The calling signature is ``fun(t, y)``.\n        Here ``t`` is a scalar and there are two options for ndarray ``y``.\n        It can either have shape (n,), then ``fun`` must return array_like with\n        shape (n,). Or alternatively it can have shape (n, k), then ``fun``\n        must return array_like with shape (n, k), i.e. each column\n        corresponds to a single column in ``y``. The choice between the two\n        options is determined by `vectorized` argument (see below). The\n        vectorized implementation allows faster approximation of the Jacobian\n        by finite differences.\n    t0 : float\n        Initial time.\n    y0 : array_like, shape (n,)\n        Initial state.\n    t_bound : float\n        Boundary time --- the integration won\'t continue beyond it. It also\n        determines the direction of the integration.\n    max_step : float, optional\n        Maximum allowed step size. Default is np.inf, i.e. the step is not\n        bounded and determined solely by the solver.\n    rtol, atol : float and array_like, optional\n        Relative and absolute tolerances. The solver keeps the local error\n        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a\n        relative accuracy (number of correct digits). But if a component of `y`\n        is approximately below `atol` then the error only needs to fall within\n        the same `atol` threshold, and the number of correct digits is not\n        guaranteed. If components of y have different scales, it might be\n        beneficial to set different `atol` values for different components by\n        passing array_like with shape (n,) for `atol`. Default values are\n        1e-3 for `rtol` and 1e-6 for `atol`.\n    jac : {None, array_like, sparse_matrix, callable}, optional\n        Jacobian matrix of the right-hand side of the system with respect to\n        y, required only by \'Radau\' and \'BDF\' methods. The Jacobian matrix\n        has shape (n, n) and its element (i, j) is equal to ``d f_i / d y_j``.\n        There are 3 ways to define the Jacobian:\n\n            * If array_like or sparse_matrix, then the Jacobian is assumed to\n              be constant.\n            * If callable, then the Jacobian is assumed to depend on both\n              t and y, and will be called as ``jac(t, y)`` as necessary. The\n              return value might be a sparse matrix.\n            * If None (default), then the Jacobian will be approximated by\n              finite differences.\n\n        It is generally recommended to provide the Jacobian rather than\n        relying on a finite difference approximation.\n    jac_sparsity : {None, array_like, sparse matrix}, optional\n        Defines a sparsity structure of the Jacobian matrix for a finite\n        difference approximation, its shape must be (n, n). If the Jacobian has\n        only few non-zero elements in *each* row, providing the sparsity\n        structure will greatly speed up the computations [4]_. A zero\n        entry means that a corresponding element in the Jacobian is identically\n        zero. If None (default), the Jacobian is assumed to be dense.\n    vectorized : bool, optional\n        Whether `fun` is implemented in a vectorized fashion. Default is False.\n\n    Attributes\n    ----------\n    n : int\n        Number of equations.\n    status : string\n        Current status of the solver: \'running\', \'finished\' or \'failed\'.\n    t_bound : float\n        Boundary time.\n    direction : float\n        Integration direction: +1 or -1.\n    t : float\n        Current time.\n    y : ndarray\n        Current state.\n    t_old : float\n        Previous time. None if no steps were made yet.\n    step_size : float\n        Size of the last successful step. None if no steps were made yet.\n    nfev : int\n        Number of the system\'s rhs evaluations.\n    njev : int\n        Number of the Jacobian evaluations.\n    nlu : int\n        Number of LU decompositions.\n\n    References\n    ----------\n    .. [1] G. D. Byrne, A. C. Hindmarsh, "A Polyalgorithm for the Numerical\n           Solution of Ordinary Differential Equations", ACM Transactions on\n           Mathematical Software, Vol. 1, No. 1, pp. 71-96, March 1975.\n    .. [2] L. F. Shampine, M. W. Reichelt, "THE MATLAB ODE SUITE", SIAM J. SCI.\n           COMPUTE., Vol. 18, No. 1, pp. 1-22, January 1997.\n    .. [3] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations I:\n           Nonstiff Problems", Sec. III.2.\n    .. [4] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of\n           sparse Jacobian matrices", Journal of the Institute of Mathematics\n           and its Applications, 13, pp. 117-120, 1974.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'np' (line 180)
        np_52809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 54), 'np')
        # Obtaining the member 'inf' of a type (line 180)
        inf_52810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 54), np_52809, 'inf')
        float_52811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 22), 'float')
        float_52812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 33), 'float')
        # Getting the type of 'None' (line 181)
        None_52813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 43), 'None')
        # Getting the type of 'None' (line 181)
        None_52814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 62), 'None')
        # Getting the type of 'False' (line 182)
        False_52815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 28), 'False')
        defaults = [inf_52810, float_52811, float_52812, None_52813, None_52814, False_52815]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 180, 4, False)
        # Assigning a type to the variable 'self' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BDF.__init__', ['fun', 't0', 'y0', 't_bound', 'max_step', 'rtol', 'atol', 'jac', 'jac_sparsity', 'vectorized'], None, 'extraneous', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['fun', 't0', 'y0', 't_bound', 'max_step', 'rtol', 'atol', 'jac', 'jac_sparsity', 'vectorized'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to warn_extraneous(...): (line 183)
        # Processing the call arguments (line 183)
        # Getting the type of 'extraneous' (line 183)
        extraneous_52817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 24), 'extraneous', False)
        # Processing the call keyword arguments (line 183)
        kwargs_52818 = {}
        # Getting the type of 'warn_extraneous' (line 183)
        warn_extraneous_52816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'warn_extraneous', False)
        # Calling warn_extraneous(args, kwargs) (line 183)
        warn_extraneous_call_result_52819 = invoke(stypy.reporting.localization.Localization(__file__, 183, 8), warn_extraneous_52816, *[extraneous_52817], **kwargs_52818)
        
        
        # Call to __init__(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'fun' (line 184)
        fun_52826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 34), 'fun', False)
        # Getting the type of 't0' (line 184)
        t0_52827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 39), 't0', False)
        # Getting the type of 'y0' (line 184)
        y0_52828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 43), 'y0', False)
        # Getting the type of 't_bound' (line 184)
        t_bound_52829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 47), 't_bound', False)
        # Getting the type of 'vectorized' (line 184)
        vectorized_52830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 56), 'vectorized', False)
        # Processing the call keyword arguments (line 184)
        # Getting the type of 'True' (line 185)
        True_52831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 50), 'True', False)
        keyword_52832 = True_52831
        kwargs_52833 = {'support_complex': keyword_52832}
        
        # Call to super(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'BDF' (line 184)
        BDF_52821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 14), 'BDF', False)
        # Getting the type of 'self' (line 184)
        self_52822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 19), 'self', False)
        # Processing the call keyword arguments (line 184)
        kwargs_52823 = {}
        # Getting the type of 'super' (line 184)
        super_52820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'super', False)
        # Calling super(args, kwargs) (line 184)
        super_call_result_52824 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), super_52820, *[BDF_52821, self_52822], **kwargs_52823)
        
        # Obtaining the member '__init__' of a type (line 184)
        init___52825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), super_call_result_52824, '__init__')
        # Calling __init__(args, kwargs) (line 184)
        init___call_result_52834 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), init___52825, *[fun_52826, t0_52827, y0_52828, t_bound_52829, vectorized_52830], **kwargs_52833)
        
        
        # Assigning a Call to a Attribute (line 186):
        
        # Assigning a Call to a Attribute (line 186):
        
        # Call to validate_max_step(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'max_step' (line 186)
        max_step_52836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 42), 'max_step', False)
        # Processing the call keyword arguments (line 186)
        kwargs_52837 = {}
        # Getting the type of 'validate_max_step' (line 186)
        validate_max_step_52835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 24), 'validate_max_step', False)
        # Calling validate_max_step(args, kwargs) (line 186)
        validate_max_step_call_result_52838 = invoke(stypy.reporting.localization.Localization(__file__, 186, 24), validate_max_step_52835, *[max_step_52836], **kwargs_52837)
        
        # Getting the type of 'self' (line 186)
        self_52839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'self')
        # Setting the type of the member 'max_step' of a type (line 186)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), self_52839, 'max_step', validate_max_step_call_result_52838)
        
        # Assigning a Call to a Tuple (line 187):
        
        # Assigning a Subscript to a Name (line 187):
        
        # Obtaining the type of the subscript
        int_52840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 8), 'int')
        
        # Call to validate_tol(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'rtol' (line 187)
        rtol_52842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 44), 'rtol', False)
        # Getting the type of 'atol' (line 187)
        atol_52843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 50), 'atol', False)
        # Getting the type of 'self' (line 187)
        self_52844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 56), 'self', False)
        # Obtaining the member 'n' of a type (line 187)
        n_52845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 56), self_52844, 'n')
        # Processing the call keyword arguments (line 187)
        kwargs_52846 = {}
        # Getting the type of 'validate_tol' (line 187)
        validate_tol_52841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 31), 'validate_tol', False)
        # Calling validate_tol(args, kwargs) (line 187)
        validate_tol_call_result_52847 = invoke(stypy.reporting.localization.Localization(__file__, 187, 31), validate_tol_52841, *[rtol_52842, atol_52843, n_52845], **kwargs_52846)
        
        # Obtaining the member '__getitem__' of a type (line 187)
        getitem___52848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 8), validate_tol_call_result_52847, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 187)
        subscript_call_result_52849 = invoke(stypy.reporting.localization.Localization(__file__, 187, 8), getitem___52848, int_52840)
        
        # Assigning a type to the variable 'tuple_var_assignment_52577' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'tuple_var_assignment_52577', subscript_call_result_52849)
        
        # Assigning a Subscript to a Name (line 187):
        
        # Obtaining the type of the subscript
        int_52850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 8), 'int')
        
        # Call to validate_tol(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'rtol' (line 187)
        rtol_52852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 44), 'rtol', False)
        # Getting the type of 'atol' (line 187)
        atol_52853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 50), 'atol', False)
        # Getting the type of 'self' (line 187)
        self_52854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 56), 'self', False)
        # Obtaining the member 'n' of a type (line 187)
        n_52855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 56), self_52854, 'n')
        # Processing the call keyword arguments (line 187)
        kwargs_52856 = {}
        # Getting the type of 'validate_tol' (line 187)
        validate_tol_52851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 31), 'validate_tol', False)
        # Calling validate_tol(args, kwargs) (line 187)
        validate_tol_call_result_52857 = invoke(stypy.reporting.localization.Localization(__file__, 187, 31), validate_tol_52851, *[rtol_52852, atol_52853, n_52855], **kwargs_52856)
        
        # Obtaining the member '__getitem__' of a type (line 187)
        getitem___52858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 8), validate_tol_call_result_52857, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 187)
        subscript_call_result_52859 = invoke(stypy.reporting.localization.Localization(__file__, 187, 8), getitem___52858, int_52850)
        
        # Assigning a type to the variable 'tuple_var_assignment_52578' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'tuple_var_assignment_52578', subscript_call_result_52859)
        
        # Assigning a Name to a Attribute (line 187):
        # Getting the type of 'tuple_var_assignment_52577' (line 187)
        tuple_var_assignment_52577_52860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'tuple_var_assignment_52577')
        # Getting the type of 'self' (line 187)
        self_52861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'self')
        # Setting the type of the member 'rtol' of a type (line 187)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 8), self_52861, 'rtol', tuple_var_assignment_52577_52860)
        
        # Assigning a Name to a Attribute (line 187):
        # Getting the type of 'tuple_var_assignment_52578' (line 187)
        tuple_var_assignment_52578_52862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'tuple_var_assignment_52578')
        # Getting the type of 'self' (line 187)
        self_52863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 19), 'self')
        # Setting the type of the member 'atol' of a type (line 187)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 19), self_52863, 'atol', tuple_var_assignment_52578_52862)
        
        # Assigning a Call to a Name (line 188):
        
        # Assigning a Call to a Name (line 188):
        
        # Call to fun(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'self' (line 188)
        self_52866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 21), 'self', False)
        # Obtaining the member 't' of a type (line 188)
        t_52867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 21), self_52866, 't')
        # Getting the type of 'self' (line 188)
        self_52868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 29), 'self', False)
        # Obtaining the member 'y' of a type (line 188)
        y_52869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 29), self_52868, 'y')
        # Processing the call keyword arguments (line 188)
        kwargs_52870 = {}
        # Getting the type of 'self' (line 188)
        self_52864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'self', False)
        # Obtaining the member 'fun' of a type (line 188)
        fun_52865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 12), self_52864, 'fun')
        # Calling fun(args, kwargs) (line 188)
        fun_call_result_52871 = invoke(stypy.reporting.localization.Localization(__file__, 188, 12), fun_52865, *[t_52867, y_52869], **kwargs_52870)
        
        # Assigning a type to the variable 'f' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'f', fun_call_result_52871)
        
        # Assigning a Call to a Attribute (line 189):
        
        # Assigning a Call to a Attribute (line 189):
        
        # Call to select_initial_step(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'self' (line 189)
        self_52873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 41), 'self', False)
        # Obtaining the member 'fun' of a type (line 189)
        fun_52874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 41), self_52873, 'fun')
        # Getting the type of 'self' (line 189)
        self_52875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 51), 'self', False)
        # Obtaining the member 't' of a type (line 189)
        t_52876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 51), self_52875, 't')
        # Getting the type of 'self' (line 189)
        self_52877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 59), 'self', False)
        # Obtaining the member 'y' of a type (line 189)
        y_52878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 59), self_52877, 'y')
        # Getting the type of 'f' (line 189)
        f_52879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 67), 'f', False)
        # Getting the type of 'self' (line 190)
        self_52880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 41), 'self', False)
        # Obtaining the member 'direction' of a type (line 190)
        direction_52881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 41), self_52880, 'direction')
        int_52882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 57), 'int')
        # Getting the type of 'self' (line 191)
        self_52883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 41), 'self', False)
        # Obtaining the member 'rtol' of a type (line 191)
        rtol_52884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 41), self_52883, 'rtol')
        # Getting the type of 'self' (line 191)
        self_52885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 52), 'self', False)
        # Obtaining the member 'atol' of a type (line 191)
        atol_52886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 52), self_52885, 'atol')
        # Processing the call keyword arguments (line 189)
        kwargs_52887 = {}
        # Getting the type of 'select_initial_step' (line 189)
        select_initial_step_52872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 21), 'select_initial_step', False)
        # Calling select_initial_step(args, kwargs) (line 189)
        select_initial_step_call_result_52888 = invoke(stypy.reporting.localization.Localization(__file__, 189, 21), select_initial_step_52872, *[fun_52874, t_52876, y_52878, f_52879, direction_52881, int_52882, rtol_52884, atol_52886], **kwargs_52887)
        
        # Getting the type of 'self' (line 189)
        self_52889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'self')
        # Setting the type of the member 'h_abs' of a type (line 189)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), self_52889, 'h_abs', select_initial_step_call_result_52888)
        
        # Assigning a Name to a Attribute (line 192):
        
        # Assigning a Name to a Attribute (line 192):
        # Getting the type of 'None' (line 192)
        None_52890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 25), 'None')
        # Getting the type of 'self' (line 192)
        self_52891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'self')
        # Setting the type of the member 'h_abs_old' of a type (line 192)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), self_52891, 'h_abs_old', None_52890)
        
        # Assigning a Name to a Attribute (line 193):
        
        # Assigning a Name to a Attribute (line 193):
        # Getting the type of 'None' (line 193)
        None_52892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 30), 'None')
        # Getting the type of 'self' (line 193)
        self_52893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'self')
        # Setting the type of the member 'error_norm_old' of a type (line 193)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 8), self_52893, 'error_norm_old', None_52892)
        
        # Assigning a Call to a Attribute (line 195):
        
        # Assigning a Call to a Attribute (line 195):
        
        # Call to max(...): (line 195)
        # Processing the call arguments (line 195)
        int_52895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 30), 'int')
        # Getting the type of 'EPS' (line 195)
        EPS_52896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 35), 'EPS', False)
        # Applying the binary operator '*' (line 195)
        result_mul_52897 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 30), '*', int_52895, EPS_52896)
        
        # Getting the type of 'rtol' (line 195)
        rtol_52898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 41), 'rtol', False)
        # Applying the binary operator 'div' (line 195)
        result_div_52899 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 39), 'div', result_mul_52897, rtol_52898)
        
        
        # Call to min(...): (line 195)
        # Processing the call arguments (line 195)
        float_52901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 51), 'float')
        # Getting the type of 'rtol' (line 195)
        rtol_52902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 57), 'rtol', False)
        float_52903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 65), 'float')
        # Applying the binary operator '**' (line 195)
        result_pow_52904 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 57), '**', rtol_52902, float_52903)
        
        # Processing the call keyword arguments (line 195)
        kwargs_52905 = {}
        # Getting the type of 'min' (line 195)
        min_52900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 47), 'min', False)
        # Calling min(args, kwargs) (line 195)
        min_call_result_52906 = invoke(stypy.reporting.localization.Localization(__file__, 195, 47), min_52900, *[float_52901, result_pow_52904], **kwargs_52905)
        
        # Processing the call keyword arguments (line 195)
        kwargs_52907 = {}
        # Getting the type of 'max' (line 195)
        max_52894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 26), 'max', False)
        # Calling max(args, kwargs) (line 195)
        max_call_result_52908 = invoke(stypy.reporting.localization.Localization(__file__, 195, 26), max_52894, *[result_div_52899, min_call_result_52906], **kwargs_52907)
        
        # Getting the type of 'self' (line 195)
        self_52909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'self')
        # Setting the type of the member 'newton_tol' of a type (line 195)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 8), self_52909, 'newton_tol', max_call_result_52908)
        
        # Assigning a Name to a Attribute (line 197):
        
        # Assigning a Name to a Attribute (line 197):
        # Getting the type of 'None' (line 197)
        None_52910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 26), 'None')
        # Getting the type of 'self' (line 197)
        self_52911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'self')
        # Setting the type of the member 'jac_factor' of a type (line 197)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), self_52911, 'jac_factor', None_52910)
        
        # Assigning a Call to a Tuple (line 198):
        
        # Assigning a Subscript to a Name (line 198):
        
        # Obtaining the type of the subscript
        int_52912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 8), 'int')
        
        # Call to _validate_jac(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'jac' (line 198)
        jac_52915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 46), 'jac', False)
        # Getting the type of 'jac_sparsity' (line 198)
        jac_sparsity_52916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 51), 'jac_sparsity', False)
        # Processing the call keyword arguments (line 198)
        kwargs_52917 = {}
        # Getting the type of 'self' (line 198)
        self_52913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 27), 'self', False)
        # Obtaining the member '_validate_jac' of a type (line 198)
        _validate_jac_52914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 27), self_52913, '_validate_jac')
        # Calling _validate_jac(args, kwargs) (line 198)
        _validate_jac_call_result_52918 = invoke(stypy.reporting.localization.Localization(__file__, 198, 27), _validate_jac_52914, *[jac_52915, jac_sparsity_52916], **kwargs_52917)
        
        # Obtaining the member '__getitem__' of a type (line 198)
        getitem___52919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), _validate_jac_call_result_52918, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 198)
        subscript_call_result_52920 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), getitem___52919, int_52912)
        
        # Assigning a type to the variable 'tuple_var_assignment_52579' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'tuple_var_assignment_52579', subscript_call_result_52920)
        
        # Assigning a Subscript to a Name (line 198):
        
        # Obtaining the type of the subscript
        int_52921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 8), 'int')
        
        # Call to _validate_jac(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'jac' (line 198)
        jac_52924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 46), 'jac', False)
        # Getting the type of 'jac_sparsity' (line 198)
        jac_sparsity_52925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 51), 'jac_sparsity', False)
        # Processing the call keyword arguments (line 198)
        kwargs_52926 = {}
        # Getting the type of 'self' (line 198)
        self_52922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 27), 'self', False)
        # Obtaining the member '_validate_jac' of a type (line 198)
        _validate_jac_52923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 27), self_52922, '_validate_jac')
        # Calling _validate_jac(args, kwargs) (line 198)
        _validate_jac_call_result_52927 = invoke(stypy.reporting.localization.Localization(__file__, 198, 27), _validate_jac_52923, *[jac_52924, jac_sparsity_52925], **kwargs_52926)
        
        # Obtaining the member '__getitem__' of a type (line 198)
        getitem___52928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), _validate_jac_call_result_52927, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 198)
        subscript_call_result_52929 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), getitem___52928, int_52921)
        
        # Assigning a type to the variable 'tuple_var_assignment_52580' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'tuple_var_assignment_52580', subscript_call_result_52929)
        
        # Assigning a Name to a Attribute (line 198):
        # Getting the type of 'tuple_var_assignment_52579' (line 198)
        tuple_var_assignment_52579_52930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'tuple_var_assignment_52579')
        # Getting the type of 'self' (line 198)
        self_52931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'self')
        # Setting the type of the member 'jac' of a type (line 198)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), self_52931, 'jac', tuple_var_assignment_52579_52930)
        
        # Assigning a Name to a Attribute (line 198):
        # Getting the type of 'tuple_var_assignment_52580' (line 198)
        tuple_var_assignment_52580_52932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'tuple_var_assignment_52580')
        # Getting the type of 'self' (line 198)
        self_52933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 18), 'self')
        # Setting the type of the member 'J' of a type (line 198)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 18), self_52933, 'J', tuple_var_assignment_52580_52932)
        
        
        # Call to issparse(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'self' (line 199)
        self_52935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'self', False)
        # Obtaining the member 'J' of a type (line 199)
        J_52936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 20), self_52935, 'J')
        # Processing the call keyword arguments (line 199)
        kwargs_52937 = {}
        # Getting the type of 'issparse' (line 199)
        issparse_52934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 11), 'issparse', False)
        # Calling issparse(args, kwargs) (line 199)
        issparse_call_result_52938 = invoke(stypy.reporting.localization.Localization(__file__, 199, 11), issparse_52934, *[J_52936], **kwargs_52937)
        
        # Testing the type of an if condition (line 199)
        if_condition_52939 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 199, 8), issparse_call_result_52938)
        # Assigning a type to the variable 'if_condition_52939' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'if_condition_52939', if_condition_52939)
        # SSA begins for if statement (line 199)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

        @norecursion
        def lu(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'lu'
            module_type_store = module_type_store.open_function_context('lu', 200, 12, False)
            
            # Passed parameters checking function
            lu.stypy_localization = localization
            lu.stypy_type_of_self = None
            lu.stypy_type_store = module_type_store
            lu.stypy_function_name = 'lu'
            lu.stypy_param_names_list = ['A']
            lu.stypy_varargs_param_name = None
            lu.stypy_kwargs_param_name = None
            lu.stypy_call_defaults = defaults
            lu.stypy_call_varargs = varargs
            lu.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'lu', ['A'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'lu', localization, ['A'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'lu(...)' code ##################

            
            # Getting the type of 'self' (line 201)
            self_52940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'self')
            # Obtaining the member 'nlu' of a type (line 201)
            nlu_52941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 16), self_52940, 'nlu')
            int_52942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 28), 'int')
            # Applying the binary operator '+=' (line 201)
            result_iadd_52943 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 16), '+=', nlu_52941, int_52942)
            # Getting the type of 'self' (line 201)
            self_52944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'self')
            # Setting the type of the member 'nlu' of a type (line 201)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 16), self_52944, 'nlu', result_iadd_52943)
            
            
            # Call to splu(...): (line 202)
            # Processing the call arguments (line 202)
            # Getting the type of 'A' (line 202)
            A_52946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 28), 'A', False)
            # Processing the call keyword arguments (line 202)
            kwargs_52947 = {}
            # Getting the type of 'splu' (line 202)
            splu_52945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 23), 'splu', False)
            # Calling splu(args, kwargs) (line 202)
            splu_call_result_52948 = invoke(stypy.reporting.localization.Localization(__file__, 202, 23), splu_52945, *[A_52946], **kwargs_52947)
            
            # Assigning a type to the variable 'stypy_return_type' (line 202)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'stypy_return_type', splu_call_result_52948)
            
            # ################# End of 'lu(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'lu' in the type store
            # Getting the type of 'stypy_return_type' (line 200)
            stypy_return_type_52949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_52949)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'lu'
            return stypy_return_type_52949

        # Assigning a type to the variable 'lu' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'lu', lu)

        @norecursion
        def solve_lu(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'solve_lu'
            module_type_store = module_type_store.open_function_context('solve_lu', 204, 12, False)
            
            # Passed parameters checking function
            solve_lu.stypy_localization = localization
            solve_lu.stypy_type_of_self = None
            solve_lu.stypy_type_store = module_type_store
            solve_lu.stypy_function_name = 'solve_lu'
            solve_lu.stypy_param_names_list = ['LU', 'b']
            solve_lu.stypy_varargs_param_name = None
            solve_lu.stypy_kwargs_param_name = None
            solve_lu.stypy_call_defaults = defaults
            solve_lu.stypy_call_varargs = varargs
            solve_lu.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'solve_lu', ['LU', 'b'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'solve_lu', localization, ['LU', 'b'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'solve_lu(...)' code ##################

            
            # Call to solve(...): (line 205)
            # Processing the call arguments (line 205)
            # Getting the type of 'b' (line 205)
            b_52952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 32), 'b', False)
            # Processing the call keyword arguments (line 205)
            kwargs_52953 = {}
            # Getting the type of 'LU' (line 205)
            LU_52950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 23), 'LU', False)
            # Obtaining the member 'solve' of a type (line 205)
            solve_52951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 23), LU_52950, 'solve')
            # Calling solve(args, kwargs) (line 205)
            solve_call_result_52954 = invoke(stypy.reporting.localization.Localization(__file__, 205, 23), solve_52951, *[b_52952], **kwargs_52953)
            
            # Assigning a type to the variable 'stypy_return_type' (line 205)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 16), 'stypy_return_type', solve_call_result_52954)
            
            # ################# End of 'solve_lu(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'solve_lu' in the type store
            # Getting the type of 'stypy_return_type' (line 204)
            stypy_return_type_52955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_52955)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'solve_lu'
            return stypy_return_type_52955

        # Assigning a type to the variable 'solve_lu' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'solve_lu', solve_lu)
        
        # Assigning a Call to a Name (line 207):
        
        # Assigning a Call to a Name (line 207):
        
        # Call to eye(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'self' (line 207)
        self_52957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 20), 'self', False)
        # Obtaining the member 'n' of a type (line 207)
        n_52958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 20), self_52957, 'n')
        # Processing the call keyword arguments (line 207)
        str_52959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 35), 'str', 'csc')
        keyword_52960 = str_52959
        # Getting the type of 'self' (line 207)
        self_52961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 48), 'self', False)
        # Obtaining the member 'y' of a type (line 207)
        y_52962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 48), self_52961, 'y')
        # Obtaining the member 'dtype' of a type (line 207)
        dtype_52963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 48), y_52962, 'dtype')
        keyword_52964 = dtype_52963
        kwargs_52965 = {'dtype': keyword_52964, 'format': keyword_52960}
        # Getting the type of 'eye' (line 207)
        eye_52956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 16), 'eye', False)
        # Calling eye(args, kwargs) (line 207)
        eye_call_result_52966 = invoke(stypy.reporting.localization.Localization(__file__, 207, 16), eye_52956, *[n_52958], **kwargs_52965)
        
        # Assigning a type to the variable 'I' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'I', eye_call_result_52966)
        # SSA branch for the else part of an if statement (line 199)
        module_type_store.open_ssa_branch('else')

        @norecursion
        def lu(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'lu'
            module_type_store = module_type_store.open_function_context('lu', 209, 12, False)
            
            # Passed parameters checking function
            lu.stypy_localization = localization
            lu.stypy_type_of_self = None
            lu.stypy_type_store = module_type_store
            lu.stypy_function_name = 'lu'
            lu.stypy_param_names_list = ['A']
            lu.stypy_varargs_param_name = None
            lu.stypy_kwargs_param_name = None
            lu.stypy_call_defaults = defaults
            lu.stypy_call_varargs = varargs
            lu.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'lu', ['A'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'lu', localization, ['A'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'lu(...)' code ##################

            
            # Getting the type of 'self' (line 210)
            self_52967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'self')
            # Obtaining the member 'nlu' of a type (line 210)
            nlu_52968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 16), self_52967, 'nlu')
            int_52969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 28), 'int')
            # Applying the binary operator '+=' (line 210)
            result_iadd_52970 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 16), '+=', nlu_52968, int_52969)
            # Getting the type of 'self' (line 210)
            self_52971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'self')
            # Setting the type of the member 'nlu' of a type (line 210)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 16), self_52971, 'nlu', result_iadd_52970)
            
            
            # Call to lu_factor(...): (line 211)
            # Processing the call arguments (line 211)
            # Getting the type of 'A' (line 211)
            A_52973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 33), 'A', False)
            # Processing the call keyword arguments (line 211)
            # Getting the type of 'True' (line 211)
            True_52974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 48), 'True', False)
            keyword_52975 = True_52974
            kwargs_52976 = {'overwrite_a': keyword_52975}
            # Getting the type of 'lu_factor' (line 211)
            lu_factor_52972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 23), 'lu_factor', False)
            # Calling lu_factor(args, kwargs) (line 211)
            lu_factor_call_result_52977 = invoke(stypy.reporting.localization.Localization(__file__, 211, 23), lu_factor_52972, *[A_52973], **kwargs_52976)
            
            # Assigning a type to the variable 'stypy_return_type' (line 211)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 16), 'stypy_return_type', lu_factor_call_result_52977)
            
            # ################# End of 'lu(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'lu' in the type store
            # Getting the type of 'stypy_return_type' (line 209)
            stypy_return_type_52978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_52978)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'lu'
            return stypy_return_type_52978

        # Assigning a type to the variable 'lu' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'lu', lu)

        @norecursion
        def solve_lu(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'solve_lu'
            module_type_store = module_type_store.open_function_context('solve_lu', 213, 12, False)
            
            # Passed parameters checking function
            solve_lu.stypy_localization = localization
            solve_lu.stypy_type_of_self = None
            solve_lu.stypy_type_store = module_type_store
            solve_lu.stypy_function_name = 'solve_lu'
            solve_lu.stypy_param_names_list = ['LU', 'b']
            solve_lu.stypy_varargs_param_name = None
            solve_lu.stypy_kwargs_param_name = None
            solve_lu.stypy_call_defaults = defaults
            solve_lu.stypy_call_varargs = varargs
            solve_lu.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'solve_lu', ['LU', 'b'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'solve_lu', localization, ['LU', 'b'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'solve_lu(...)' code ##################

            
            # Call to lu_solve(...): (line 214)
            # Processing the call arguments (line 214)
            # Getting the type of 'LU' (line 214)
            LU_52980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 32), 'LU', False)
            # Getting the type of 'b' (line 214)
            b_52981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 36), 'b', False)
            # Processing the call keyword arguments (line 214)
            # Getting the type of 'True' (line 214)
            True_52982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 51), 'True', False)
            keyword_52983 = True_52982
            kwargs_52984 = {'overwrite_b': keyword_52983}
            # Getting the type of 'lu_solve' (line 214)
            lu_solve_52979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 23), 'lu_solve', False)
            # Calling lu_solve(args, kwargs) (line 214)
            lu_solve_call_result_52985 = invoke(stypy.reporting.localization.Localization(__file__, 214, 23), lu_solve_52979, *[LU_52980, b_52981], **kwargs_52984)
            
            # Assigning a type to the variable 'stypy_return_type' (line 214)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 16), 'stypy_return_type', lu_solve_call_result_52985)
            
            # ################# End of 'solve_lu(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'solve_lu' in the type store
            # Getting the type of 'stypy_return_type' (line 213)
            stypy_return_type_52986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_52986)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'solve_lu'
            return stypy_return_type_52986

        # Assigning a type to the variable 'solve_lu' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'solve_lu', solve_lu)
        
        # Assigning a Call to a Name (line 216):
        
        # Assigning a Call to a Name (line 216):
        
        # Call to identity(...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 'self' (line 216)
        self_52989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 28), 'self', False)
        # Obtaining the member 'n' of a type (line 216)
        n_52990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 28), self_52989, 'n')
        # Processing the call keyword arguments (line 216)
        # Getting the type of 'self' (line 216)
        self_52991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 42), 'self', False)
        # Obtaining the member 'y' of a type (line 216)
        y_52992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 42), self_52991, 'y')
        # Obtaining the member 'dtype' of a type (line 216)
        dtype_52993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 42), y_52992, 'dtype')
        keyword_52994 = dtype_52993
        kwargs_52995 = {'dtype': keyword_52994}
        # Getting the type of 'np' (line 216)
        np_52987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'np', False)
        # Obtaining the member 'identity' of a type (line 216)
        identity_52988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 16), np_52987, 'identity')
        # Calling identity(args, kwargs) (line 216)
        identity_call_result_52996 = invoke(stypy.reporting.localization.Localization(__file__, 216, 16), identity_52988, *[n_52990], **kwargs_52995)
        
        # Assigning a type to the variable 'I' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'I', identity_call_result_52996)
        # SSA join for if statement (line 199)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 218):
        
        # Assigning a Name to a Attribute (line 218):
        # Getting the type of 'lu' (line 218)
        lu_52997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 18), 'lu')
        # Getting the type of 'self' (line 218)
        self_52998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'self')
        # Setting the type of the member 'lu' of a type (line 218)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), self_52998, 'lu', lu_52997)
        
        # Assigning a Name to a Attribute (line 219):
        
        # Assigning a Name to a Attribute (line 219):
        # Getting the type of 'solve_lu' (line 219)
        solve_lu_52999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 24), 'solve_lu')
        # Getting the type of 'self' (line 219)
        self_53000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'self')
        # Setting the type of the member 'solve_lu' of a type (line 219)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 8), self_53000, 'solve_lu', solve_lu_52999)
        
        # Assigning a Name to a Attribute (line 220):
        
        # Assigning a Name to a Attribute (line 220):
        # Getting the type of 'I' (line 220)
        I_53001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 17), 'I')
        # Getting the type of 'self' (line 220)
        self_53002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'self')
        # Setting the type of the member 'I' of a type (line 220)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 8), self_53002, 'I', I_53001)
        
        # Assigning a Call to a Name (line 222):
        
        # Assigning a Call to a Name (line 222):
        
        # Call to array(...): (line 222)
        # Processing the call arguments (line 222)
        
        # Obtaining an instance of the builtin type 'list' (line 222)
        list_53005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 222)
        # Adding element type (line 222)
        int_53006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 25), list_53005, int_53006)
        # Adding element type (line 222)
        float_53007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 25), list_53005, float_53007)
        # Adding element type (line 222)
        int_53008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 38), 'int')
        int_53009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 41), 'int')
        # Applying the binary operator 'div' (line 222)
        result_div_53010 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 38), 'div', int_53008, int_53009)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 25), list_53005, result_div_53010)
        # Adding element type (line 222)
        float_53011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 25), list_53005, float_53011)
        # Adding element type (line 222)
        float_53012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 25), list_53005, float_53012)
        # Adding element type (line 222)
        int_53013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 25), list_53005, int_53013)
        
        # Processing the call keyword arguments (line 222)
        kwargs_53014 = {}
        # Getting the type of 'np' (line 222)
        np_53003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 222)
        array_53004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 16), np_53003, 'array')
        # Calling array(args, kwargs) (line 222)
        array_call_result_53015 = invoke(stypy.reporting.localization.Localization(__file__, 222, 16), array_53004, *[list_53005], **kwargs_53014)
        
        # Assigning a type to the variable 'kappa' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'kappa', array_call_result_53015)
        
        # Assigning a Call to a Attribute (line 223):
        
        # Assigning a Call to a Attribute (line 223):
        
        # Call to hstack(...): (line 223)
        # Processing the call arguments (line 223)
        
        # Obtaining an instance of the builtin type 'tuple' (line 223)
        tuple_53018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 223)
        # Adding element type (line 223)
        int_53019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 32), tuple_53018, int_53019)
        # Adding element type (line 223)
        
        # Call to cumsum(...): (line 223)
        # Processing the call arguments (line 223)
        int_53022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 45), 'int')
        
        # Call to arange(...): (line 223)
        # Processing the call arguments (line 223)
        int_53025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 59), 'int')
        # Getting the type of 'MAX_ORDER' (line 223)
        MAX_ORDER_53026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 62), 'MAX_ORDER', False)
        int_53027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 74), 'int')
        # Applying the binary operator '+' (line 223)
        result_add_53028 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 62), '+', MAX_ORDER_53026, int_53027)
        
        # Processing the call keyword arguments (line 223)
        kwargs_53029 = {}
        # Getting the type of 'np' (line 223)
        np_53023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 49), 'np', False)
        # Obtaining the member 'arange' of a type (line 223)
        arange_53024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 49), np_53023, 'arange')
        # Calling arange(args, kwargs) (line 223)
        arange_call_result_53030 = invoke(stypy.reporting.localization.Localization(__file__, 223, 49), arange_53024, *[int_53025, result_add_53028], **kwargs_53029)
        
        # Applying the binary operator 'div' (line 223)
        result_div_53031 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 45), 'div', int_53022, arange_call_result_53030)
        
        # Processing the call keyword arguments (line 223)
        kwargs_53032 = {}
        # Getting the type of 'np' (line 223)
        np_53020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 35), 'np', False)
        # Obtaining the member 'cumsum' of a type (line 223)
        cumsum_53021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 35), np_53020, 'cumsum')
        # Calling cumsum(args, kwargs) (line 223)
        cumsum_call_result_53033 = invoke(stypy.reporting.localization.Localization(__file__, 223, 35), cumsum_53021, *[result_div_53031], **kwargs_53032)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 32), tuple_53018, cumsum_call_result_53033)
        
        # Processing the call keyword arguments (line 223)
        kwargs_53034 = {}
        # Getting the type of 'np' (line 223)
        np_53016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 21), 'np', False)
        # Obtaining the member 'hstack' of a type (line 223)
        hstack_53017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 21), np_53016, 'hstack')
        # Calling hstack(args, kwargs) (line 223)
        hstack_call_result_53035 = invoke(stypy.reporting.localization.Localization(__file__, 223, 21), hstack_53017, *[tuple_53018], **kwargs_53034)
        
        # Getting the type of 'self' (line 223)
        self_53036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'self')
        # Setting the type of the member 'gamma' of a type (line 223)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), self_53036, 'gamma', hstack_call_result_53035)
        
        # Assigning a BinOp to a Attribute (line 224):
        
        # Assigning a BinOp to a Attribute (line 224):
        int_53037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 22), 'int')
        # Getting the type of 'kappa' (line 224)
        kappa_53038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 26), 'kappa')
        # Applying the binary operator '-' (line 224)
        result_sub_53039 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 22), '-', int_53037, kappa_53038)
        
        # Getting the type of 'self' (line 224)
        self_53040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 35), 'self')
        # Obtaining the member 'gamma' of a type (line 224)
        gamma_53041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 35), self_53040, 'gamma')
        # Applying the binary operator '*' (line 224)
        result_mul_53042 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 21), '*', result_sub_53039, gamma_53041)
        
        # Getting the type of 'self' (line 224)
        self_53043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'self')
        # Setting the type of the member 'alpha' of a type (line 224)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), self_53043, 'alpha', result_mul_53042)
        
        # Assigning a BinOp to a Attribute (line 225):
        
        # Assigning a BinOp to a Attribute (line 225):
        # Getting the type of 'kappa' (line 225)
        kappa_53044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 27), 'kappa')
        # Getting the type of 'self' (line 225)
        self_53045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 35), 'self')
        # Obtaining the member 'gamma' of a type (line 225)
        gamma_53046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 35), self_53045, 'gamma')
        # Applying the binary operator '*' (line 225)
        result_mul_53047 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 27), '*', kappa_53044, gamma_53046)
        
        int_53048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 48), 'int')
        
        # Call to arange(...): (line 225)
        # Processing the call arguments (line 225)
        int_53051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 62), 'int')
        # Getting the type of 'MAX_ORDER' (line 225)
        MAX_ORDER_53052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 65), 'MAX_ORDER', False)
        int_53053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 77), 'int')
        # Applying the binary operator '+' (line 225)
        result_add_53054 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 65), '+', MAX_ORDER_53052, int_53053)
        
        # Processing the call keyword arguments (line 225)
        kwargs_53055 = {}
        # Getting the type of 'np' (line 225)
        np_53049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 52), 'np', False)
        # Obtaining the member 'arange' of a type (line 225)
        arange_53050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 52), np_53049, 'arange')
        # Calling arange(args, kwargs) (line 225)
        arange_call_result_53056 = invoke(stypy.reporting.localization.Localization(__file__, 225, 52), arange_53050, *[int_53051, result_add_53054], **kwargs_53055)
        
        # Applying the binary operator 'div' (line 225)
        result_div_53057 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 48), 'div', int_53048, arange_call_result_53056)
        
        # Applying the binary operator '+' (line 225)
        result_add_53058 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 27), '+', result_mul_53047, result_div_53057)
        
        # Getting the type of 'self' (line 225)
        self_53059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'self')
        # Setting the type of the member 'error_const' of a type (line 225)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), self_53059, 'error_const', result_add_53058)
        
        # Assigning a Call to a Name (line 227):
        
        # Assigning a Call to a Name (line 227):
        
        # Call to empty(...): (line 227)
        # Processing the call arguments (line 227)
        
        # Obtaining an instance of the builtin type 'tuple' (line 227)
        tuple_53062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 227)
        # Adding element type (line 227)
        # Getting the type of 'MAX_ORDER' (line 227)
        MAX_ORDER_53063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 22), 'MAX_ORDER', False)
        int_53064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 34), 'int')
        # Applying the binary operator '+' (line 227)
        result_add_53065 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 22), '+', MAX_ORDER_53063, int_53064)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 22), tuple_53062, result_add_53065)
        # Adding element type (line 227)
        # Getting the type of 'self' (line 227)
        self_53066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 37), 'self', False)
        # Obtaining the member 'n' of a type (line 227)
        n_53067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 37), self_53066, 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 22), tuple_53062, n_53067)
        
        # Processing the call keyword arguments (line 227)
        # Getting the type of 'self' (line 227)
        self_53068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 52), 'self', False)
        # Obtaining the member 'y' of a type (line 227)
        y_53069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 52), self_53068, 'y')
        # Obtaining the member 'dtype' of a type (line 227)
        dtype_53070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 52), y_53069, 'dtype')
        keyword_53071 = dtype_53070
        kwargs_53072 = {'dtype': keyword_53071}
        # Getting the type of 'np' (line 227)
        np_53060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'np', False)
        # Obtaining the member 'empty' of a type (line 227)
        empty_53061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 12), np_53060, 'empty')
        # Calling empty(args, kwargs) (line 227)
        empty_call_result_53073 = invoke(stypy.reporting.localization.Localization(__file__, 227, 12), empty_53061, *[tuple_53062], **kwargs_53072)
        
        # Assigning a type to the variable 'D' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'D', empty_call_result_53073)
        
        # Assigning a Attribute to a Subscript (line 228):
        
        # Assigning a Attribute to a Subscript (line 228):
        # Getting the type of 'self' (line 228)
        self_53074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 15), 'self')
        # Obtaining the member 'y' of a type (line 228)
        y_53075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 15), self_53074, 'y')
        # Getting the type of 'D' (line 228)
        D_53076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'D')
        int_53077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 10), 'int')
        # Storing an element on a container (line 228)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 8), D_53076, (int_53077, y_53075))
        
        # Assigning a BinOp to a Subscript (line 229):
        
        # Assigning a BinOp to a Subscript (line 229):
        # Getting the type of 'f' (line 229)
        f_53078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 15), 'f')
        # Getting the type of 'self' (line 229)
        self_53079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 19), 'self')
        # Obtaining the member 'h_abs' of a type (line 229)
        h_abs_53080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 19), self_53079, 'h_abs')
        # Applying the binary operator '*' (line 229)
        result_mul_53081 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 15), '*', f_53078, h_abs_53080)
        
        # Getting the type of 'self' (line 229)
        self_53082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 32), 'self')
        # Obtaining the member 'direction' of a type (line 229)
        direction_53083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 32), self_53082, 'direction')
        # Applying the binary operator '*' (line 229)
        result_mul_53084 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 30), '*', result_mul_53081, direction_53083)
        
        # Getting the type of 'D' (line 229)
        D_53085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'D')
        int_53086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 10), 'int')
        # Storing an element on a container (line 229)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 8), D_53085, (int_53086, result_mul_53084))
        
        # Assigning a Name to a Attribute (line 230):
        
        # Assigning a Name to a Attribute (line 230):
        # Getting the type of 'D' (line 230)
        D_53087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 17), 'D')
        # Getting the type of 'self' (line 230)
        self_53088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'self')
        # Setting the type of the member 'D' of a type (line 230)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), self_53088, 'D', D_53087)
        
        # Assigning a Num to a Attribute (line 232):
        
        # Assigning a Num to a Attribute (line 232):
        int_53089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 21), 'int')
        # Getting the type of 'self' (line 232)
        self_53090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'self')
        # Setting the type of the member 'order' of a type (line 232)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 8), self_53090, 'order', int_53089)
        
        # Assigning a Num to a Attribute (line 233):
        
        # Assigning a Num to a Attribute (line 233):
        int_53091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 29), 'int')
        # Getting the type of 'self' (line 233)
        self_53092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'self')
        # Setting the type of the member 'n_equal_steps' of a type (line 233)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 8), self_53092, 'n_equal_steps', int_53091)
        
        # Assigning a Name to a Attribute (line 234):
        
        # Assigning a Name to a Attribute (line 234):
        # Getting the type of 'None' (line 234)
        None_53093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 18), 'None')
        # Getting the type of 'self' (line 234)
        self_53094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'self')
        # Setting the type of the member 'LU' of a type (line 234)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 8), self_53094, 'LU', None_53093)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _validate_jac(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_validate_jac'
        module_type_store = module_type_store.open_function_context('_validate_jac', 236, 4, False)
        # Assigning a type to the variable 'self' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BDF._validate_jac.__dict__.__setitem__('stypy_localization', localization)
        BDF._validate_jac.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BDF._validate_jac.__dict__.__setitem__('stypy_type_store', module_type_store)
        BDF._validate_jac.__dict__.__setitem__('stypy_function_name', 'BDF._validate_jac')
        BDF._validate_jac.__dict__.__setitem__('stypy_param_names_list', ['jac', 'sparsity'])
        BDF._validate_jac.__dict__.__setitem__('stypy_varargs_param_name', None)
        BDF._validate_jac.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BDF._validate_jac.__dict__.__setitem__('stypy_call_defaults', defaults)
        BDF._validate_jac.__dict__.__setitem__('stypy_call_varargs', varargs)
        BDF._validate_jac.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BDF._validate_jac.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BDF._validate_jac', ['jac', 'sparsity'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_validate_jac', localization, ['jac', 'sparsity'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_validate_jac(...)' code ##################

        
        # Assigning a Attribute to a Name (line 237):
        
        # Assigning a Attribute to a Name (line 237):
        # Getting the type of 'self' (line 237)
        self_53095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 13), 'self')
        # Obtaining the member 't' of a type (line 237)
        t_53096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 13), self_53095, 't')
        # Assigning a type to the variable 't0' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 't0', t_53096)
        
        # Assigning a Attribute to a Name (line 238):
        
        # Assigning a Attribute to a Name (line 238):
        # Getting the type of 'self' (line 238)
        self_53097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 13), 'self')
        # Obtaining the member 'y' of a type (line 238)
        y_53098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 13), self_53097, 'y')
        # Assigning a type to the variable 'y0' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'y0', y_53098)
        
        # Type idiom detected: calculating its left and rigth part (line 240)
        # Getting the type of 'jac' (line 240)
        jac_53099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 'jac')
        # Getting the type of 'None' (line 240)
        None_53100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 18), 'None')
        
        (may_be_53101, more_types_in_union_53102) = may_be_none(jac_53099, None_53100)

        if may_be_53101:

            if more_types_in_union_53102:
                # Runtime conditional SSA (line 240)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Type idiom detected: calculating its left and rigth part (line 241)
            # Getting the type of 'sparsity' (line 241)
            sparsity_53103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'sparsity')
            # Getting the type of 'None' (line 241)
            None_53104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 31), 'None')
            
            (may_be_53105, more_types_in_union_53106) = may_not_be_none(sparsity_53103, None_53104)

            if may_be_53105:

                if more_types_in_union_53106:
                    # Runtime conditional SSA (line 241)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                
                # Call to issparse(...): (line 242)
                # Processing the call arguments (line 242)
                # Getting the type of 'sparsity' (line 242)
                sparsity_53108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 28), 'sparsity', False)
                # Processing the call keyword arguments (line 242)
                kwargs_53109 = {}
                # Getting the type of 'issparse' (line 242)
                issparse_53107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 19), 'issparse', False)
                # Calling issparse(args, kwargs) (line 242)
                issparse_call_result_53110 = invoke(stypy.reporting.localization.Localization(__file__, 242, 19), issparse_53107, *[sparsity_53108], **kwargs_53109)
                
                # Testing the type of an if condition (line 242)
                if_condition_53111 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 16), issparse_call_result_53110)
                # Assigning a type to the variable 'if_condition_53111' (line 242)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'if_condition_53111', if_condition_53111)
                # SSA begins for if statement (line 242)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 243):
                
                # Assigning a Call to a Name (line 243):
                
                # Call to csc_matrix(...): (line 243)
                # Processing the call arguments (line 243)
                # Getting the type of 'sparsity' (line 243)
                sparsity_53113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 42), 'sparsity', False)
                # Processing the call keyword arguments (line 243)
                kwargs_53114 = {}
                # Getting the type of 'csc_matrix' (line 243)
                csc_matrix_53112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 31), 'csc_matrix', False)
                # Calling csc_matrix(args, kwargs) (line 243)
                csc_matrix_call_result_53115 = invoke(stypy.reporting.localization.Localization(__file__, 243, 31), csc_matrix_53112, *[sparsity_53113], **kwargs_53114)
                
                # Assigning a type to the variable 'sparsity' (line 243)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 20), 'sparsity', csc_matrix_call_result_53115)
                # SSA join for if statement (line 242)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # Assigning a Call to a Name (line 244):
                
                # Assigning a Call to a Name (line 244):
                
                # Call to group_columns(...): (line 244)
                # Processing the call arguments (line 244)
                # Getting the type of 'sparsity' (line 244)
                sparsity_53117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 39), 'sparsity', False)
                # Processing the call keyword arguments (line 244)
                kwargs_53118 = {}
                # Getting the type of 'group_columns' (line 244)
                group_columns_53116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 25), 'group_columns', False)
                # Calling group_columns(args, kwargs) (line 244)
                group_columns_call_result_53119 = invoke(stypy.reporting.localization.Localization(__file__, 244, 25), group_columns_53116, *[sparsity_53117], **kwargs_53118)
                
                # Assigning a type to the variable 'groups' (line 244)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 16), 'groups', group_columns_call_result_53119)
                
                # Assigning a Tuple to a Name (line 245):
                
                # Assigning a Tuple to a Name (line 245):
                
                # Obtaining an instance of the builtin type 'tuple' (line 245)
                tuple_53120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 28), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 245)
                # Adding element type (line 245)
                # Getting the type of 'sparsity' (line 245)
                sparsity_53121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 28), 'sparsity')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 28), tuple_53120, sparsity_53121)
                # Adding element type (line 245)
                # Getting the type of 'groups' (line 245)
                groups_53122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 38), 'groups')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 28), tuple_53120, groups_53122)
                
                # Assigning a type to the variable 'sparsity' (line 245)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 16), 'sparsity', tuple_53120)

                if more_types_in_union_53106:
                    # SSA join for if statement (line 241)
                    module_type_store = module_type_store.join_ssa_context()


            

            @norecursion
            def jac_wrapped(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'jac_wrapped'
                module_type_store = module_type_store.open_function_context('jac_wrapped', 247, 12, False)
                
                # Passed parameters checking function
                jac_wrapped.stypy_localization = localization
                jac_wrapped.stypy_type_of_self = None
                jac_wrapped.stypy_type_store = module_type_store
                jac_wrapped.stypy_function_name = 'jac_wrapped'
                jac_wrapped.stypy_param_names_list = ['t', 'y']
                jac_wrapped.stypy_varargs_param_name = None
                jac_wrapped.stypy_kwargs_param_name = None
                jac_wrapped.stypy_call_defaults = defaults
                jac_wrapped.stypy_call_varargs = varargs
                jac_wrapped.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, 'jac_wrapped', ['t', 'y'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'jac_wrapped', localization, ['t', 'y'], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'jac_wrapped(...)' code ##################

                
                # Getting the type of 'self' (line 248)
                self_53123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'self')
                # Obtaining the member 'njev' of a type (line 248)
                njev_53124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 16), self_53123, 'njev')
                int_53125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 29), 'int')
                # Applying the binary operator '+=' (line 248)
                result_iadd_53126 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 16), '+=', njev_53124, int_53125)
                # Getting the type of 'self' (line 248)
                self_53127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'self')
                # Setting the type of the member 'njev' of a type (line 248)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 16), self_53127, 'njev', result_iadd_53126)
                
                
                # Assigning a Call to a Name (line 249):
                
                # Assigning a Call to a Name (line 249):
                
                # Call to fun_single(...): (line 249)
                # Processing the call arguments (line 249)
                # Getting the type of 't' (line 249)
                t_53130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 36), 't', False)
                # Getting the type of 'y' (line 249)
                y_53131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 39), 'y', False)
                # Processing the call keyword arguments (line 249)
                kwargs_53132 = {}
                # Getting the type of 'self' (line 249)
                self_53128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 20), 'self', False)
                # Obtaining the member 'fun_single' of a type (line 249)
                fun_single_53129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 20), self_53128, 'fun_single')
                # Calling fun_single(args, kwargs) (line 249)
                fun_single_call_result_53133 = invoke(stypy.reporting.localization.Localization(__file__, 249, 20), fun_single_53129, *[t_53130, y_53131], **kwargs_53132)
                
                # Assigning a type to the variable 'f' (line 249)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 16), 'f', fun_single_call_result_53133)
                
                # Assigning a Call to a Tuple (line 250):
                
                # Assigning a Subscript to a Name (line 250):
                
                # Obtaining the type of the subscript
                int_53134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 16), 'int')
                
                # Call to num_jac(...): (line 250)
                # Processing the call arguments (line 250)
                # Getting the type of 'self' (line 250)
                self_53136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 45), 'self', False)
                # Obtaining the member 'fun_vectorized' of a type (line 250)
                fun_vectorized_53137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 45), self_53136, 'fun_vectorized')
                # Getting the type of 't' (line 250)
                t_53138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 66), 't', False)
                # Getting the type of 'y' (line 250)
                y_53139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 69), 'y', False)
                # Getting the type of 'f' (line 250)
                f_53140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 72), 'f', False)
                # Getting the type of 'self' (line 251)
                self_53141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 45), 'self', False)
                # Obtaining the member 'atol' of a type (line 251)
                atol_53142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 45), self_53141, 'atol')
                # Getting the type of 'self' (line 251)
                self_53143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 56), 'self', False)
                # Obtaining the member 'jac_factor' of a type (line 251)
                jac_factor_53144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 56), self_53143, 'jac_factor')
                # Getting the type of 'sparsity' (line 252)
                sparsity_53145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 45), 'sparsity', False)
                # Processing the call keyword arguments (line 250)
                kwargs_53146 = {}
                # Getting the type of 'num_jac' (line 250)
                num_jac_53135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 37), 'num_jac', False)
                # Calling num_jac(args, kwargs) (line 250)
                num_jac_call_result_53147 = invoke(stypy.reporting.localization.Localization(__file__, 250, 37), num_jac_53135, *[fun_vectorized_53137, t_53138, y_53139, f_53140, atol_53142, jac_factor_53144, sparsity_53145], **kwargs_53146)
                
                # Obtaining the member '__getitem__' of a type (line 250)
                getitem___53148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 16), num_jac_call_result_53147, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 250)
                subscript_call_result_53149 = invoke(stypy.reporting.localization.Localization(__file__, 250, 16), getitem___53148, int_53134)
                
                # Assigning a type to the variable 'tuple_var_assignment_52581' (line 250)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'tuple_var_assignment_52581', subscript_call_result_53149)
                
                # Assigning a Subscript to a Name (line 250):
                
                # Obtaining the type of the subscript
                int_53150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 16), 'int')
                
                # Call to num_jac(...): (line 250)
                # Processing the call arguments (line 250)
                # Getting the type of 'self' (line 250)
                self_53152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 45), 'self', False)
                # Obtaining the member 'fun_vectorized' of a type (line 250)
                fun_vectorized_53153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 45), self_53152, 'fun_vectorized')
                # Getting the type of 't' (line 250)
                t_53154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 66), 't', False)
                # Getting the type of 'y' (line 250)
                y_53155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 69), 'y', False)
                # Getting the type of 'f' (line 250)
                f_53156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 72), 'f', False)
                # Getting the type of 'self' (line 251)
                self_53157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 45), 'self', False)
                # Obtaining the member 'atol' of a type (line 251)
                atol_53158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 45), self_53157, 'atol')
                # Getting the type of 'self' (line 251)
                self_53159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 56), 'self', False)
                # Obtaining the member 'jac_factor' of a type (line 251)
                jac_factor_53160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 56), self_53159, 'jac_factor')
                # Getting the type of 'sparsity' (line 252)
                sparsity_53161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 45), 'sparsity', False)
                # Processing the call keyword arguments (line 250)
                kwargs_53162 = {}
                # Getting the type of 'num_jac' (line 250)
                num_jac_53151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 37), 'num_jac', False)
                # Calling num_jac(args, kwargs) (line 250)
                num_jac_call_result_53163 = invoke(stypy.reporting.localization.Localization(__file__, 250, 37), num_jac_53151, *[fun_vectorized_53153, t_53154, y_53155, f_53156, atol_53158, jac_factor_53160, sparsity_53161], **kwargs_53162)
                
                # Obtaining the member '__getitem__' of a type (line 250)
                getitem___53164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 16), num_jac_call_result_53163, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 250)
                subscript_call_result_53165 = invoke(stypy.reporting.localization.Localization(__file__, 250, 16), getitem___53164, int_53150)
                
                # Assigning a type to the variable 'tuple_var_assignment_52582' (line 250)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'tuple_var_assignment_52582', subscript_call_result_53165)
                
                # Assigning a Name to a Name (line 250):
                # Getting the type of 'tuple_var_assignment_52581' (line 250)
                tuple_var_assignment_52581_53166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'tuple_var_assignment_52581')
                # Assigning a type to the variable 'J' (line 250)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'J', tuple_var_assignment_52581_53166)
                
                # Assigning a Name to a Attribute (line 250):
                # Getting the type of 'tuple_var_assignment_52582' (line 250)
                tuple_var_assignment_52582_53167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'tuple_var_assignment_52582')
                # Getting the type of 'self' (line 250)
                self_53168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 19), 'self')
                # Setting the type of the member 'jac_factor' of a type (line 250)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 19), self_53168, 'jac_factor', tuple_var_assignment_52582_53167)
                # Getting the type of 'J' (line 253)
                J_53169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 23), 'J')
                # Assigning a type to the variable 'stypy_return_type' (line 253)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 16), 'stypy_return_type', J_53169)
                
                # ################# End of 'jac_wrapped(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'jac_wrapped' in the type store
                # Getting the type of 'stypy_return_type' (line 247)
                stypy_return_type_53170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_53170)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'jac_wrapped'
                return stypy_return_type_53170

            # Assigning a type to the variable 'jac_wrapped' (line 247)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'jac_wrapped', jac_wrapped)
            
            # Assigning a Call to a Name (line 254):
            
            # Assigning a Call to a Name (line 254):
            
            # Call to jac_wrapped(...): (line 254)
            # Processing the call arguments (line 254)
            # Getting the type of 't0' (line 254)
            t0_53172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 28), 't0', False)
            # Getting the type of 'y0' (line 254)
            y0_53173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 32), 'y0', False)
            # Processing the call keyword arguments (line 254)
            kwargs_53174 = {}
            # Getting the type of 'jac_wrapped' (line 254)
            jac_wrapped_53171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 16), 'jac_wrapped', False)
            # Calling jac_wrapped(args, kwargs) (line 254)
            jac_wrapped_call_result_53175 = invoke(stypy.reporting.localization.Localization(__file__, 254, 16), jac_wrapped_53171, *[t0_53172, y0_53173], **kwargs_53174)
            
            # Assigning a type to the variable 'J' (line 254)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'J', jac_wrapped_call_result_53175)

            if more_types_in_union_53102:
                # Runtime conditional SSA for else branch (line 240)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_53101) or more_types_in_union_53102):
            
            
            # Call to callable(...): (line 255)
            # Processing the call arguments (line 255)
            # Getting the type of 'jac' (line 255)
            jac_53177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 22), 'jac', False)
            # Processing the call keyword arguments (line 255)
            kwargs_53178 = {}
            # Getting the type of 'callable' (line 255)
            callable_53176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 13), 'callable', False)
            # Calling callable(args, kwargs) (line 255)
            callable_call_result_53179 = invoke(stypy.reporting.localization.Localization(__file__, 255, 13), callable_53176, *[jac_53177], **kwargs_53178)
            
            # Testing the type of an if condition (line 255)
            if_condition_53180 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 13), callable_call_result_53179)
            # Assigning a type to the variable 'if_condition_53180' (line 255)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 13), 'if_condition_53180', if_condition_53180)
            # SSA begins for if statement (line 255)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 256):
            
            # Assigning a Call to a Name (line 256):
            
            # Call to jac(...): (line 256)
            # Processing the call arguments (line 256)
            # Getting the type of 't0' (line 256)
            t0_53182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 20), 't0', False)
            # Getting the type of 'y0' (line 256)
            y0_53183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 24), 'y0', False)
            # Processing the call keyword arguments (line 256)
            kwargs_53184 = {}
            # Getting the type of 'jac' (line 256)
            jac_53181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 16), 'jac', False)
            # Calling jac(args, kwargs) (line 256)
            jac_call_result_53185 = invoke(stypy.reporting.localization.Localization(__file__, 256, 16), jac_53181, *[t0_53182, y0_53183], **kwargs_53184)
            
            # Assigning a type to the variable 'J' (line 256)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'J', jac_call_result_53185)
            
            # Getting the type of 'self' (line 257)
            self_53186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'self')
            # Obtaining the member 'njev' of a type (line 257)
            njev_53187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 12), self_53186, 'njev')
            int_53188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 25), 'int')
            # Applying the binary operator '+=' (line 257)
            result_iadd_53189 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 12), '+=', njev_53187, int_53188)
            # Getting the type of 'self' (line 257)
            self_53190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'self')
            # Setting the type of the member 'njev' of a type (line 257)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 12), self_53190, 'njev', result_iadd_53189)
            
            
            
            # Call to issparse(...): (line 258)
            # Processing the call arguments (line 258)
            # Getting the type of 'J' (line 258)
            J_53192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 24), 'J', False)
            # Processing the call keyword arguments (line 258)
            kwargs_53193 = {}
            # Getting the type of 'issparse' (line 258)
            issparse_53191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 15), 'issparse', False)
            # Calling issparse(args, kwargs) (line 258)
            issparse_call_result_53194 = invoke(stypy.reporting.localization.Localization(__file__, 258, 15), issparse_53191, *[J_53192], **kwargs_53193)
            
            # Testing the type of an if condition (line 258)
            if_condition_53195 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 258, 12), issparse_call_result_53194)
            # Assigning a type to the variable 'if_condition_53195' (line 258)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'if_condition_53195', if_condition_53195)
            # SSA begins for if statement (line 258)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 259):
            
            # Assigning a Call to a Name (line 259):
            
            # Call to csc_matrix(...): (line 259)
            # Processing the call arguments (line 259)
            # Getting the type of 'J' (line 259)
            J_53197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 31), 'J', False)
            # Processing the call keyword arguments (line 259)
            # Getting the type of 'y0' (line 259)
            y0_53198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 40), 'y0', False)
            # Obtaining the member 'dtype' of a type (line 259)
            dtype_53199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 40), y0_53198, 'dtype')
            keyword_53200 = dtype_53199
            kwargs_53201 = {'dtype': keyword_53200}
            # Getting the type of 'csc_matrix' (line 259)
            csc_matrix_53196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 20), 'csc_matrix', False)
            # Calling csc_matrix(args, kwargs) (line 259)
            csc_matrix_call_result_53202 = invoke(stypy.reporting.localization.Localization(__file__, 259, 20), csc_matrix_53196, *[J_53197], **kwargs_53201)
            
            # Assigning a type to the variable 'J' (line 259)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 16), 'J', csc_matrix_call_result_53202)

            @norecursion
            def jac_wrapped(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'jac_wrapped'
                module_type_store = module_type_store.open_function_context('jac_wrapped', 261, 16, False)
                
                # Passed parameters checking function
                jac_wrapped.stypy_localization = localization
                jac_wrapped.stypy_type_of_self = None
                jac_wrapped.stypy_type_store = module_type_store
                jac_wrapped.stypy_function_name = 'jac_wrapped'
                jac_wrapped.stypy_param_names_list = ['t', 'y']
                jac_wrapped.stypy_varargs_param_name = None
                jac_wrapped.stypy_kwargs_param_name = None
                jac_wrapped.stypy_call_defaults = defaults
                jac_wrapped.stypy_call_varargs = varargs
                jac_wrapped.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, 'jac_wrapped', ['t', 'y'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'jac_wrapped', localization, ['t', 'y'], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'jac_wrapped(...)' code ##################

                
                # Getting the type of 'self' (line 262)
                self_53203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 20), 'self')
                # Obtaining the member 'njev' of a type (line 262)
                njev_53204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 20), self_53203, 'njev')
                int_53205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 33), 'int')
                # Applying the binary operator '+=' (line 262)
                result_iadd_53206 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 20), '+=', njev_53204, int_53205)
                # Getting the type of 'self' (line 262)
                self_53207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 20), 'self')
                # Setting the type of the member 'njev' of a type (line 262)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 20), self_53207, 'njev', result_iadd_53206)
                
                
                # Call to csc_matrix(...): (line 263)
                # Processing the call arguments (line 263)
                
                # Call to jac(...): (line 263)
                # Processing the call arguments (line 263)
                # Getting the type of 't' (line 263)
                t_53210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 42), 't', False)
                # Getting the type of 'y' (line 263)
                y_53211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 45), 'y', False)
                # Processing the call keyword arguments (line 263)
                kwargs_53212 = {}
                # Getting the type of 'jac' (line 263)
                jac_53209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 38), 'jac', False)
                # Calling jac(args, kwargs) (line 263)
                jac_call_result_53213 = invoke(stypy.reporting.localization.Localization(__file__, 263, 38), jac_53209, *[t_53210, y_53211], **kwargs_53212)
                
                # Processing the call keyword arguments (line 263)
                # Getting the type of 'y0' (line 263)
                y0_53214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 55), 'y0', False)
                # Obtaining the member 'dtype' of a type (line 263)
                dtype_53215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 55), y0_53214, 'dtype')
                keyword_53216 = dtype_53215
                kwargs_53217 = {'dtype': keyword_53216}
                # Getting the type of 'csc_matrix' (line 263)
                csc_matrix_53208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 27), 'csc_matrix', False)
                # Calling csc_matrix(args, kwargs) (line 263)
                csc_matrix_call_result_53218 = invoke(stypy.reporting.localization.Localization(__file__, 263, 27), csc_matrix_53208, *[jac_call_result_53213], **kwargs_53217)
                
                # Assigning a type to the variable 'stypy_return_type' (line 263)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 20), 'stypy_return_type', csc_matrix_call_result_53218)
                
                # ################# End of 'jac_wrapped(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'jac_wrapped' in the type store
                # Getting the type of 'stypy_return_type' (line 261)
                stypy_return_type_53219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 16), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_53219)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'jac_wrapped'
                return stypy_return_type_53219

            # Assigning a type to the variable 'jac_wrapped' (line 261)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 16), 'jac_wrapped', jac_wrapped)
            # SSA branch for the else part of an if statement (line 258)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 265):
            
            # Assigning a Call to a Name (line 265):
            
            # Call to asarray(...): (line 265)
            # Processing the call arguments (line 265)
            # Getting the type of 'J' (line 265)
            J_53222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 31), 'J', False)
            # Processing the call keyword arguments (line 265)
            # Getting the type of 'y0' (line 265)
            y0_53223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 40), 'y0', False)
            # Obtaining the member 'dtype' of a type (line 265)
            dtype_53224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 40), y0_53223, 'dtype')
            keyword_53225 = dtype_53224
            kwargs_53226 = {'dtype': keyword_53225}
            # Getting the type of 'np' (line 265)
            np_53220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 20), 'np', False)
            # Obtaining the member 'asarray' of a type (line 265)
            asarray_53221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 20), np_53220, 'asarray')
            # Calling asarray(args, kwargs) (line 265)
            asarray_call_result_53227 = invoke(stypy.reporting.localization.Localization(__file__, 265, 20), asarray_53221, *[J_53222], **kwargs_53226)
            
            # Assigning a type to the variable 'J' (line 265)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 16), 'J', asarray_call_result_53227)

            @norecursion
            def jac_wrapped(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'jac_wrapped'
                module_type_store = module_type_store.open_function_context('jac_wrapped', 267, 16, False)
                
                # Passed parameters checking function
                jac_wrapped.stypy_localization = localization
                jac_wrapped.stypy_type_of_self = None
                jac_wrapped.stypy_type_store = module_type_store
                jac_wrapped.stypy_function_name = 'jac_wrapped'
                jac_wrapped.stypy_param_names_list = ['t', 'y']
                jac_wrapped.stypy_varargs_param_name = None
                jac_wrapped.stypy_kwargs_param_name = None
                jac_wrapped.stypy_call_defaults = defaults
                jac_wrapped.stypy_call_varargs = varargs
                jac_wrapped.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, 'jac_wrapped', ['t', 'y'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'jac_wrapped', localization, ['t', 'y'], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'jac_wrapped(...)' code ##################

                
                # Getting the type of 'self' (line 268)
                self_53228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 20), 'self')
                # Obtaining the member 'njev' of a type (line 268)
                njev_53229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 20), self_53228, 'njev')
                int_53230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 33), 'int')
                # Applying the binary operator '+=' (line 268)
                result_iadd_53231 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 20), '+=', njev_53229, int_53230)
                # Getting the type of 'self' (line 268)
                self_53232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 20), 'self')
                # Setting the type of the member 'njev' of a type (line 268)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 20), self_53232, 'njev', result_iadd_53231)
                
                
                # Call to asarray(...): (line 269)
                # Processing the call arguments (line 269)
                
                # Call to jac(...): (line 269)
                # Processing the call arguments (line 269)
                # Getting the type of 't' (line 269)
                t_53236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 42), 't', False)
                # Getting the type of 'y' (line 269)
                y_53237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 45), 'y', False)
                # Processing the call keyword arguments (line 269)
                kwargs_53238 = {}
                # Getting the type of 'jac' (line 269)
                jac_53235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 38), 'jac', False)
                # Calling jac(args, kwargs) (line 269)
                jac_call_result_53239 = invoke(stypy.reporting.localization.Localization(__file__, 269, 38), jac_53235, *[t_53236, y_53237], **kwargs_53238)
                
                # Processing the call keyword arguments (line 269)
                # Getting the type of 'y0' (line 269)
                y0_53240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 55), 'y0', False)
                # Obtaining the member 'dtype' of a type (line 269)
                dtype_53241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 55), y0_53240, 'dtype')
                keyword_53242 = dtype_53241
                kwargs_53243 = {'dtype': keyword_53242}
                # Getting the type of 'np' (line 269)
                np_53233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 27), 'np', False)
                # Obtaining the member 'asarray' of a type (line 269)
                asarray_53234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 27), np_53233, 'asarray')
                # Calling asarray(args, kwargs) (line 269)
                asarray_call_result_53244 = invoke(stypy.reporting.localization.Localization(__file__, 269, 27), asarray_53234, *[jac_call_result_53239], **kwargs_53243)
                
                # Assigning a type to the variable 'stypy_return_type' (line 269)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 20), 'stypy_return_type', asarray_call_result_53244)
                
                # ################# End of 'jac_wrapped(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'jac_wrapped' in the type store
                # Getting the type of 'stypy_return_type' (line 267)
                stypy_return_type_53245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 16), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_53245)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'jac_wrapped'
                return stypy_return_type_53245

            # Assigning a type to the variable 'jac_wrapped' (line 267)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 16), 'jac_wrapped', jac_wrapped)
            # SSA join for if statement (line 258)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Getting the type of 'J' (line 271)
            J_53246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 15), 'J')
            # Obtaining the member 'shape' of a type (line 271)
            shape_53247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 15), J_53246, 'shape')
            
            # Obtaining an instance of the builtin type 'tuple' (line 271)
            tuple_53248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 27), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 271)
            # Adding element type (line 271)
            # Getting the type of 'self' (line 271)
            self_53249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 27), 'self')
            # Obtaining the member 'n' of a type (line 271)
            n_53250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 27), self_53249, 'n')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 27), tuple_53248, n_53250)
            # Adding element type (line 271)
            # Getting the type of 'self' (line 271)
            self_53251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 35), 'self')
            # Obtaining the member 'n' of a type (line 271)
            n_53252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 35), self_53251, 'n')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 27), tuple_53248, n_53252)
            
            # Applying the binary operator '!=' (line 271)
            result_ne_53253 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 15), '!=', shape_53247, tuple_53248)
            
            # Testing the type of an if condition (line 271)
            if_condition_53254 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 271, 12), result_ne_53253)
            # Assigning a type to the variable 'if_condition_53254' (line 271)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'if_condition_53254', if_condition_53254)
            # SSA begins for if statement (line 271)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 272)
            # Processing the call arguments (line 272)
            
            # Call to format(...): (line 272)
            # Processing the call arguments (line 272)
            
            # Obtaining an instance of the builtin type 'tuple' (line 274)
            tuple_53258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 42), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 274)
            # Adding element type (line 274)
            # Getting the type of 'self' (line 274)
            self_53259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 42), 'self', False)
            # Obtaining the member 'n' of a type (line 274)
            n_53260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 42), self_53259, 'n')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 42), tuple_53258, n_53260)
            # Adding element type (line 274)
            # Getting the type of 'self' (line 274)
            self_53261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 50), 'self', False)
            # Obtaining the member 'n' of a type (line 274)
            n_53262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 50), self_53261, 'n')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 42), tuple_53258, n_53262)
            
            # Getting the type of 'J' (line 274)
            J_53263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 59), 'J', False)
            # Obtaining the member 'shape' of a type (line 274)
            shape_53264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 59), J_53263, 'shape')
            # Processing the call keyword arguments (line 272)
            kwargs_53265 = {}
            str_53256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 33), 'str', '`jac` is expected to have shape {}, but actually has {}.')
            # Obtaining the member 'format' of a type (line 272)
            format_53257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 33), str_53256, 'format')
            # Calling format(args, kwargs) (line 272)
            format_call_result_53266 = invoke(stypy.reporting.localization.Localization(__file__, 272, 33), format_53257, *[tuple_53258, shape_53264], **kwargs_53265)
            
            # Processing the call keyword arguments (line 272)
            kwargs_53267 = {}
            # Getting the type of 'ValueError' (line 272)
            ValueError_53255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 272)
            ValueError_call_result_53268 = invoke(stypy.reporting.localization.Localization(__file__, 272, 22), ValueError_53255, *[format_call_result_53266], **kwargs_53267)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 272, 16), ValueError_call_result_53268, 'raise parameter', BaseException)
            # SSA join for if statement (line 271)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 255)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to issparse(...): (line 276)
            # Processing the call arguments (line 276)
            # Getting the type of 'jac' (line 276)
            jac_53270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 24), 'jac', False)
            # Processing the call keyword arguments (line 276)
            kwargs_53271 = {}
            # Getting the type of 'issparse' (line 276)
            issparse_53269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 15), 'issparse', False)
            # Calling issparse(args, kwargs) (line 276)
            issparse_call_result_53272 = invoke(stypy.reporting.localization.Localization(__file__, 276, 15), issparse_53269, *[jac_53270], **kwargs_53271)
            
            # Testing the type of an if condition (line 276)
            if_condition_53273 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 276, 12), issparse_call_result_53272)
            # Assigning a type to the variable 'if_condition_53273' (line 276)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'if_condition_53273', if_condition_53273)
            # SSA begins for if statement (line 276)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 277):
            
            # Assigning a Call to a Name (line 277):
            
            # Call to csc_matrix(...): (line 277)
            # Processing the call arguments (line 277)
            # Getting the type of 'jac' (line 277)
            jac_53275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 31), 'jac', False)
            # Processing the call keyword arguments (line 277)
            # Getting the type of 'y0' (line 277)
            y0_53276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 42), 'y0', False)
            # Obtaining the member 'dtype' of a type (line 277)
            dtype_53277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 42), y0_53276, 'dtype')
            keyword_53278 = dtype_53277
            kwargs_53279 = {'dtype': keyword_53278}
            # Getting the type of 'csc_matrix' (line 277)
            csc_matrix_53274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 20), 'csc_matrix', False)
            # Calling csc_matrix(args, kwargs) (line 277)
            csc_matrix_call_result_53280 = invoke(stypy.reporting.localization.Localization(__file__, 277, 20), csc_matrix_53274, *[jac_53275], **kwargs_53279)
            
            # Assigning a type to the variable 'J' (line 277)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 16), 'J', csc_matrix_call_result_53280)
            # SSA branch for the else part of an if statement (line 276)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 279):
            
            # Assigning a Call to a Name (line 279):
            
            # Call to asarray(...): (line 279)
            # Processing the call arguments (line 279)
            # Getting the type of 'jac' (line 279)
            jac_53283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 31), 'jac', False)
            # Processing the call keyword arguments (line 279)
            # Getting the type of 'y0' (line 279)
            y0_53284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 42), 'y0', False)
            # Obtaining the member 'dtype' of a type (line 279)
            dtype_53285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 42), y0_53284, 'dtype')
            keyword_53286 = dtype_53285
            kwargs_53287 = {'dtype': keyword_53286}
            # Getting the type of 'np' (line 279)
            np_53281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 20), 'np', False)
            # Obtaining the member 'asarray' of a type (line 279)
            asarray_53282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 20), np_53281, 'asarray')
            # Calling asarray(args, kwargs) (line 279)
            asarray_call_result_53288 = invoke(stypy.reporting.localization.Localization(__file__, 279, 20), asarray_53282, *[jac_53283], **kwargs_53287)
            
            # Assigning a type to the variable 'J' (line 279)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 16), 'J', asarray_call_result_53288)
            # SSA join for if statement (line 276)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Getting the type of 'J' (line 281)
            J_53289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 15), 'J')
            # Obtaining the member 'shape' of a type (line 281)
            shape_53290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 15), J_53289, 'shape')
            
            # Obtaining an instance of the builtin type 'tuple' (line 281)
            tuple_53291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 27), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 281)
            # Adding element type (line 281)
            # Getting the type of 'self' (line 281)
            self_53292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 27), 'self')
            # Obtaining the member 'n' of a type (line 281)
            n_53293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 27), self_53292, 'n')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 27), tuple_53291, n_53293)
            # Adding element type (line 281)
            # Getting the type of 'self' (line 281)
            self_53294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 35), 'self')
            # Obtaining the member 'n' of a type (line 281)
            n_53295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 35), self_53294, 'n')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 27), tuple_53291, n_53295)
            
            # Applying the binary operator '!=' (line 281)
            result_ne_53296 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 15), '!=', shape_53290, tuple_53291)
            
            # Testing the type of an if condition (line 281)
            if_condition_53297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 281, 12), result_ne_53296)
            # Assigning a type to the variable 'if_condition_53297' (line 281)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'if_condition_53297', if_condition_53297)
            # SSA begins for if statement (line 281)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 282)
            # Processing the call arguments (line 282)
            
            # Call to format(...): (line 282)
            # Processing the call arguments (line 282)
            
            # Obtaining an instance of the builtin type 'tuple' (line 284)
            tuple_53301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 42), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 284)
            # Adding element type (line 284)
            # Getting the type of 'self' (line 284)
            self_53302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 42), 'self', False)
            # Obtaining the member 'n' of a type (line 284)
            n_53303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 42), self_53302, 'n')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 42), tuple_53301, n_53303)
            # Adding element type (line 284)
            # Getting the type of 'self' (line 284)
            self_53304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 50), 'self', False)
            # Obtaining the member 'n' of a type (line 284)
            n_53305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 50), self_53304, 'n')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 42), tuple_53301, n_53305)
            
            # Getting the type of 'J' (line 284)
            J_53306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 59), 'J', False)
            # Obtaining the member 'shape' of a type (line 284)
            shape_53307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 59), J_53306, 'shape')
            # Processing the call keyword arguments (line 282)
            kwargs_53308 = {}
            str_53299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 33), 'str', '`jac` is expected to have shape {}, but actually has {}.')
            # Obtaining the member 'format' of a type (line 282)
            format_53300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 33), str_53299, 'format')
            # Calling format(args, kwargs) (line 282)
            format_call_result_53309 = invoke(stypy.reporting.localization.Localization(__file__, 282, 33), format_53300, *[tuple_53301, shape_53307], **kwargs_53308)
            
            # Processing the call keyword arguments (line 282)
            kwargs_53310 = {}
            # Getting the type of 'ValueError' (line 282)
            ValueError_53298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 282)
            ValueError_call_result_53311 = invoke(stypy.reporting.localization.Localization(__file__, 282, 22), ValueError_53298, *[format_call_result_53309], **kwargs_53310)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 282, 16), ValueError_call_result_53311, 'raise parameter', BaseException)
            # SSA join for if statement (line 281)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Name to a Name (line 285):
            
            # Assigning a Name to a Name (line 285):
            # Getting the type of 'None' (line 285)
            None_53312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 26), 'None')
            # Assigning a type to the variable 'jac_wrapped' (line 285)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'jac_wrapped', None_53312)
            # SSA join for if statement (line 255)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_53101 and more_types_in_union_53102):
                # SSA join for if statement (line 240)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Obtaining an instance of the builtin type 'tuple' (line 287)
        tuple_53313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 287)
        # Adding element type (line 287)
        # Getting the type of 'jac_wrapped' (line 287)
        jac_wrapped_53314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 15), 'jac_wrapped')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 15), tuple_53313, jac_wrapped_53314)
        # Adding element type (line 287)
        # Getting the type of 'J' (line 287)
        J_53315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 28), 'J')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 15), tuple_53313, J_53315)
        
        # Assigning a type to the variable 'stypy_return_type' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'stypy_return_type', tuple_53313)
        
        # ################# End of '_validate_jac(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_validate_jac' in the type store
        # Getting the type of 'stypy_return_type' (line 236)
        stypy_return_type_53316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_53316)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_validate_jac'
        return stypy_return_type_53316


    @norecursion
    def _step_impl(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_step_impl'
        module_type_store = module_type_store.open_function_context('_step_impl', 289, 4, False)
        # Assigning a type to the variable 'self' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BDF._step_impl.__dict__.__setitem__('stypy_localization', localization)
        BDF._step_impl.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BDF._step_impl.__dict__.__setitem__('stypy_type_store', module_type_store)
        BDF._step_impl.__dict__.__setitem__('stypy_function_name', 'BDF._step_impl')
        BDF._step_impl.__dict__.__setitem__('stypy_param_names_list', [])
        BDF._step_impl.__dict__.__setitem__('stypy_varargs_param_name', None)
        BDF._step_impl.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BDF._step_impl.__dict__.__setitem__('stypy_call_defaults', defaults)
        BDF._step_impl.__dict__.__setitem__('stypy_call_varargs', varargs)
        BDF._step_impl.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BDF._step_impl.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BDF._step_impl', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_step_impl', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_step_impl(...)' code ##################

        
        # Assigning a Attribute to a Name (line 290):
        
        # Assigning a Attribute to a Name (line 290):
        # Getting the type of 'self' (line 290)
        self_53317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'self')
        # Obtaining the member 't' of a type (line 290)
        t_53318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 12), self_53317, 't')
        # Assigning a type to the variable 't' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 't', t_53318)
        
        # Assigning a Attribute to a Name (line 291):
        
        # Assigning a Attribute to a Name (line 291):
        # Getting the type of 'self' (line 291)
        self_53319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'self')
        # Obtaining the member 'D' of a type (line 291)
        D_53320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 12), self_53319, 'D')
        # Assigning a type to the variable 'D' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'D', D_53320)
        
        # Assigning a Attribute to a Name (line 293):
        
        # Assigning a Attribute to a Name (line 293):
        # Getting the type of 'self' (line 293)
        self_53321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 19), 'self')
        # Obtaining the member 'max_step' of a type (line 293)
        max_step_53322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 19), self_53321, 'max_step')
        # Assigning a type to the variable 'max_step' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'max_step', max_step_53322)
        
        # Assigning a BinOp to a Name (line 294):
        
        # Assigning a BinOp to a Name (line 294):
        int_53323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 19), 'int')
        
        # Call to abs(...): (line 294)
        # Processing the call arguments (line 294)
        
        # Call to nextafter(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 't' (line 294)
        t_53328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 44), 't', False)
        # Getting the type of 'self' (line 294)
        self_53329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 47), 'self', False)
        # Obtaining the member 'direction' of a type (line 294)
        direction_53330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 47), self_53329, 'direction')
        # Getting the type of 'np' (line 294)
        np_53331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 64), 'np', False)
        # Obtaining the member 'inf' of a type (line 294)
        inf_53332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 64), np_53331, 'inf')
        # Applying the binary operator '*' (line 294)
        result_mul_53333 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 47), '*', direction_53330, inf_53332)
        
        # Processing the call keyword arguments (line 294)
        kwargs_53334 = {}
        # Getting the type of 'np' (line 294)
        np_53326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 31), 'np', False)
        # Obtaining the member 'nextafter' of a type (line 294)
        nextafter_53327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 31), np_53326, 'nextafter')
        # Calling nextafter(args, kwargs) (line 294)
        nextafter_call_result_53335 = invoke(stypy.reporting.localization.Localization(__file__, 294, 31), nextafter_53327, *[t_53328, result_mul_53333], **kwargs_53334)
        
        # Getting the type of 't' (line 294)
        t_53336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 74), 't', False)
        # Applying the binary operator '-' (line 294)
        result_sub_53337 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 31), '-', nextafter_call_result_53335, t_53336)
        
        # Processing the call keyword arguments (line 294)
        kwargs_53338 = {}
        # Getting the type of 'np' (line 294)
        np_53324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 24), 'np', False)
        # Obtaining the member 'abs' of a type (line 294)
        abs_53325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 24), np_53324, 'abs')
        # Calling abs(args, kwargs) (line 294)
        abs_call_result_53339 = invoke(stypy.reporting.localization.Localization(__file__, 294, 24), abs_53325, *[result_sub_53337], **kwargs_53338)
        
        # Applying the binary operator '*' (line 294)
        result_mul_53340 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 19), '*', int_53323, abs_call_result_53339)
        
        # Assigning a type to the variable 'min_step' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'min_step', result_mul_53340)
        
        
        # Getting the type of 'self' (line 295)
        self_53341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 11), 'self')
        # Obtaining the member 'h_abs' of a type (line 295)
        h_abs_53342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 11), self_53341, 'h_abs')
        # Getting the type of 'max_step' (line 295)
        max_step_53343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 24), 'max_step')
        # Applying the binary operator '>' (line 295)
        result_gt_53344 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 11), '>', h_abs_53342, max_step_53343)
        
        # Testing the type of an if condition (line 295)
        if_condition_53345 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 295, 8), result_gt_53344)
        # Assigning a type to the variable 'if_condition_53345' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'if_condition_53345', if_condition_53345)
        # SSA begins for if statement (line 295)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 296):
        
        # Assigning a Name to a Name (line 296):
        # Getting the type of 'max_step' (line 296)
        max_step_53346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 20), 'max_step')
        # Assigning a type to the variable 'h_abs' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'h_abs', max_step_53346)
        
        # Call to change_D(...): (line 297)
        # Processing the call arguments (line 297)
        # Getting the type of 'D' (line 297)
        D_53348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 21), 'D', False)
        # Getting the type of 'self' (line 297)
        self_53349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 24), 'self', False)
        # Obtaining the member 'order' of a type (line 297)
        order_53350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 24), self_53349, 'order')
        # Getting the type of 'max_step' (line 297)
        max_step_53351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 36), 'max_step', False)
        # Getting the type of 'self' (line 297)
        self_53352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 47), 'self', False)
        # Obtaining the member 'h_abs' of a type (line 297)
        h_abs_53353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 47), self_53352, 'h_abs')
        # Applying the binary operator 'div' (line 297)
        result_div_53354 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 36), 'div', max_step_53351, h_abs_53353)
        
        # Processing the call keyword arguments (line 297)
        kwargs_53355 = {}
        # Getting the type of 'change_D' (line 297)
        change_D_53347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'change_D', False)
        # Calling change_D(args, kwargs) (line 297)
        change_D_call_result_53356 = invoke(stypy.reporting.localization.Localization(__file__, 297, 12), change_D_53347, *[D_53348, order_53350, result_div_53354], **kwargs_53355)
        
        
        # Assigning a Num to a Attribute (line 298):
        
        # Assigning a Num to a Attribute (line 298):
        int_53357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 33), 'int')
        # Getting the type of 'self' (line 298)
        self_53358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'self')
        # Setting the type of the member 'n_equal_steps' of a type (line 298)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 12), self_53358, 'n_equal_steps', int_53357)
        # SSA branch for the else part of an if statement (line 295)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 299)
        self_53359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 13), 'self')
        # Obtaining the member 'h_abs' of a type (line 299)
        h_abs_53360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 13), self_53359, 'h_abs')
        # Getting the type of 'min_step' (line 299)
        min_step_53361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 26), 'min_step')
        # Applying the binary operator '<' (line 299)
        result_lt_53362 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 13), '<', h_abs_53360, min_step_53361)
        
        # Testing the type of an if condition (line 299)
        if_condition_53363 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 299, 13), result_lt_53362)
        # Assigning a type to the variable 'if_condition_53363' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 13), 'if_condition_53363', if_condition_53363)
        # SSA begins for if statement (line 299)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 300):
        
        # Assigning a Name to a Name (line 300):
        # Getting the type of 'min_step' (line 300)
        min_step_53364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 20), 'min_step')
        # Assigning a type to the variable 'h_abs' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'h_abs', min_step_53364)
        
        # Call to change_D(...): (line 301)
        # Processing the call arguments (line 301)
        # Getting the type of 'D' (line 301)
        D_53366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 21), 'D', False)
        # Getting the type of 'self' (line 301)
        self_53367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 24), 'self', False)
        # Obtaining the member 'order' of a type (line 301)
        order_53368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 24), self_53367, 'order')
        # Getting the type of 'min_step' (line 301)
        min_step_53369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 36), 'min_step', False)
        # Getting the type of 'self' (line 301)
        self_53370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 47), 'self', False)
        # Obtaining the member 'h_abs' of a type (line 301)
        h_abs_53371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 47), self_53370, 'h_abs')
        # Applying the binary operator 'div' (line 301)
        result_div_53372 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 36), 'div', min_step_53369, h_abs_53371)
        
        # Processing the call keyword arguments (line 301)
        kwargs_53373 = {}
        # Getting the type of 'change_D' (line 301)
        change_D_53365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'change_D', False)
        # Calling change_D(args, kwargs) (line 301)
        change_D_call_result_53374 = invoke(stypy.reporting.localization.Localization(__file__, 301, 12), change_D_53365, *[D_53366, order_53368, result_div_53372], **kwargs_53373)
        
        
        # Assigning a Num to a Attribute (line 302):
        
        # Assigning a Num to a Attribute (line 302):
        int_53375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 33), 'int')
        # Getting the type of 'self' (line 302)
        self_53376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'self')
        # Setting the type of the member 'n_equal_steps' of a type (line 302)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 12), self_53376, 'n_equal_steps', int_53375)
        # SSA branch for the else part of an if statement (line 299)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 304):
        
        # Assigning a Attribute to a Name (line 304):
        # Getting the type of 'self' (line 304)
        self_53377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 20), 'self')
        # Obtaining the member 'h_abs' of a type (line 304)
        h_abs_53378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 20), self_53377, 'h_abs')
        # Assigning a type to the variable 'h_abs' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'h_abs', h_abs_53378)
        # SSA join for if statement (line 299)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 295)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 306):
        
        # Assigning a Attribute to a Name (line 306):
        # Getting the type of 'self' (line 306)
        self_53379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 15), 'self')
        # Obtaining the member 'atol' of a type (line 306)
        atol_53380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 15), self_53379, 'atol')
        # Assigning a type to the variable 'atol' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'atol', atol_53380)
        
        # Assigning a Attribute to a Name (line 307):
        
        # Assigning a Attribute to a Name (line 307):
        # Getting the type of 'self' (line 307)
        self_53381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 15), 'self')
        # Obtaining the member 'rtol' of a type (line 307)
        rtol_53382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 15), self_53381, 'rtol')
        # Assigning a type to the variable 'rtol' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'rtol', rtol_53382)
        
        # Assigning a Attribute to a Name (line 308):
        
        # Assigning a Attribute to a Name (line 308):
        # Getting the type of 'self' (line 308)
        self_53383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 16), 'self')
        # Obtaining the member 'order' of a type (line 308)
        order_53384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 16), self_53383, 'order')
        # Assigning a type to the variable 'order' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'order', order_53384)
        
        # Assigning a Attribute to a Name (line 310):
        
        # Assigning a Attribute to a Name (line 310):
        # Getting the type of 'self' (line 310)
        self_53385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 16), 'self')
        # Obtaining the member 'alpha' of a type (line 310)
        alpha_53386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 16), self_53385, 'alpha')
        # Assigning a type to the variable 'alpha' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'alpha', alpha_53386)
        
        # Assigning a Attribute to a Name (line 311):
        
        # Assigning a Attribute to a Name (line 311):
        # Getting the type of 'self' (line 311)
        self_53387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 16), 'self')
        # Obtaining the member 'gamma' of a type (line 311)
        gamma_53388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 16), self_53387, 'gamma')
        # Assigning a type to the variable 'gamma' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'gamma', gamma_53388)
        
        # Assigning a Attribute to a Name (line 312):
        
        # Assigning a Attribute to a Name (line 312):
        # Getting the type of 'self' (line 312)
        self_53389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 22), 'self')
        # Obtaining the member 'error_const' of a type (line 312)
        error_const_53390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 22), self_53389, 'error_const')
        # Assigning a type to the variable 'error_const' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'error_const', error_const_53390)
        
        # Assigning a Attribute to a Name (line 314):
        
        # Assigning a Attribute to a Name (line 314):
        # Getting the type of 'self' (line 314)
        self_53391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'self')
        # Obtaining the member 'J' of a type (line 314)
        J_53392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 12), self_53391, 'J')
        # Assigning a type to the variable 'J' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'J', J_53392)
        
        # Assigning a Attribute to a Name (line 315):
        
        # Assigning a Attribute to a Name (line 315):
        # Getting the type of 'self' (line 315)
        self_53393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 13), 'self')
        # Obtaining the member 'LU' of a type (line 315)
        LU_53394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 13), self_53393, 'LU')
        # Assigning a type to the variable 'LU' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'LU', LU_53394)
        
        # Assigning a Compare to a Name (line 316):
        
        # Assigning a Compare to a Name (line 316):
        
        # Getting the type of 'self' (line 316)
        self_53395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 22), 'self')
        # Obtaining the member 'jac' of a type (line 316)
        jac_53396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 22), self_53395, 'jac')
        # Getting the type of 'None' (line 316)
        None_53397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 34), 'None')
        # Applying the binary operator 'is' (line 316)
        result_is__53398 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 22), 'is', jac_53396, None_53397)
        
        # Assigning a type to the variable 'current_jac' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'current_jac', result_is__53398)
        
        # Assigning a Name to a Name (line 318):
        
        # Assigning a Name to a Name (line 318):
        # Getting the type of 'False' (line 318)
        False_53399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 24), 'False')
        # Assigning a type to the variable 'step_accepted' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'step_accepted', False_53399)
        
        
        # Getting the type of 'step_accepted' (line 319)
        step_accepted_53400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 18), 'step_accepted')
        # Applying the 'not' unary operator (line 319)
        result_not__53401 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 14), 'not', step_accepted_53400)
        
        # Testing the type of an if condition (line 319)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 319, 8), result_not__53401)
        # SSA begins for while statement (line 319)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        
        # Getting the type of 'h_abs' (line 320)
        h_abs_53402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 15), 'h_abs')
        # Getting the type of 'min_step' (line 320)
        min_step_53403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 23), 'min_step')
        # Applying the binary operator '<' (line 320)
        result_lt_53404 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 15), '<', h_abs_53402, min_step_53403)
        
        # Testing the type of an if condition (line 320)
        if_condition_53405 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 320, 12), result_lt_53404)
        # Assigning a type to the variable 'if_condition_53405' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'if_condition_53405', if_condition_53405)
        # SSA begins for if statement (line 320)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 321)
        tuple_53406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 321)
        # Adding element type (line 321)
        # Getting the type of 'False' (line 321)
        False_53407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 23), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 23), tuple_53406, False_53407)
        # Adding element type (line 321)
        # Getting the type of 'self' (line 321)
        self_53408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 30), 'self')
        # Obtaining the member 'TOO_SMALL_STEP' of a type (line 321)
        TOO_SMALL_STEP_53409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 30), self_53408, 'TOO_SMALL_STEP')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 23), tuple_53406, TOO_SMALL_STEP_53409)
        
        # Assigning a type to the variable 'stypy_return_type' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 16), 'stypy_return_type', tuple_53406)
        # SSA join for if statement (line 320)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 323):
        
        # Assigning a BinOp to a Name (line 323):
        # Getting the type of 'h_abs' (line 323)
        h_abs_53410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 16), 'h_abs')
        # Getting the type of 'self' (line 323)
        self_53411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 24), 'self')
        # Obtaining the member 'direction' of a type (line 323)
        direction_53412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 24), self_53411, 'direction')
        # Applying the binary operator '*' (line 323)
        result_mul_53413 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 16), '*', h_abs_53410, direction_53412)
        
        # Assigning a type to the variable 'h' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'h', result_mul_53413)
        
        # Assigning a BinOp to a Name (line 324):
        
        # Assigning a BinOp to a Name (line 324):
        # Getting the type of 't' (line 324)
        t_53414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 20), 't')
        # Getting the type of 'h' (line 324)
        h_53415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 24), 'h')
        # Applying the binary operator '+' (line 324)
        result_add_53416 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 20), '+', t_53414, h_53415)
        
        # Assigning a type to the variable 't_new' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 't_new', result_add_53416)
        
        
        # Getting the type of 'self' (line 326)
        self_53417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 15), 'self')
        # Obtaining the member 'direction' of a type (line 326)
        direction_53418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 15), self_53417, 'direction')
        # Getting the type of 't_new' (line 326)
        t_new_53419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 33), 't_new')
        # Getting the type of 'self' (line 326)
        self_53420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 41), 'self')
        # Obtaining the member 't_bound' of a type (line 326)
        t_bound_53421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 41), self_53420, 't_bound')
        # Applying the binary operator '-' (line 326)
        result_sub_53422 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 33), '-', t_new_53419, t_bound_53421)
        
        # Applying the binary operator '*' (line 326)
        result_mul_53423 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 15), '*', direction_53418, result_sub_53422)
        
        int_53424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 57), 'int')
        # Applying the binary operator '>' (line 326)
        result_gt_53425 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 15), '>', result_mul_53423, int_53424)
        
        # Testing the type of an if condition (line 326)
        if_condition_53426 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 326, 12), result_gt_53425)
        # Assigning a type to the variable 'if_condition_53426' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'if_condition_53426', if_condition_53426)
        # SSA begins for if statement (line 326)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 327):
        
        # Assigning a Attribute to a Name (line 327):
        # Getting the type of 'self' (line 327)
        self_53427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 24), 'self')
        # Obtaining the member 't_bound' of a type (line 327)
        t_bound_53428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 24), self_53427, 't_bound')
        # Assigning a type to the variable 't_new' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 't_new', t_bound_53428)
        
        # Call to change_D(...): (line 328)
        # Processing the call arguments (line 328)
        # Getting the type of 'D' (line 328)
        D_53430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 25), 'D', False)
        # Getting the type of 'order' (line 328)
        order_53431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 28), 'order', False)
        
        # Call to abs(...): (line 328)
        # Processing the call arguments (line 328)
        # Getting the type of 't_new' (line 328)
        t_new_53434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 42), 't_new', False)
        # Getting the type of 't' (line 328)
        t_53435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 50), 't', False)
        # Applying the binary operator '-' (line 328)
        result_sub_53436 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 42), '-', t_new_53434, t_53435)
        
        # Processing the call keyword arguments (line 328)
        kwargs_53437 = {}
        # Getting the type of 'np' (line 328)
        np_53432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 35), 'np', False)
        # Obtaining the member 'abs' of a type (line 328)
        abs_53433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 35), np_53432, 'abs')
        # Calling abs(args, kwargs) (line 328)
        abs_call_result_53438 = invoke(stypy.reporting.localization.Localization(__file__, 328, 35), abs_53433, *[result_sub_53436], **kwargs_53437)
        
        # Getting the type of 'h_abs' (line 328)
        h_abs_53439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 55), 'h_abs', False)
        # Applying the binary operator 'div' (line 328)
        result_div_53440 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 35), 'div', abs_call_result_53438, h_abs_53439)
        
        # Processing the call keyword arguments (line 328)
        kwargs_53441 = {}
        # Getting the type of 'change_D' (line 328)
        change_D_53429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 16), 'change_D', False)
        # Calling change_D(args, kwargs) (line 328)
        change_D_call_result_53442 = invoke(stypy.reporting.localization.Localization(__file__, 328, 16), change_D_53429, *[D_53430, order_53431, result_div_53440], **kwargs_53441)
        
        
        # Assigning a Num to a Attribute (line 329):
        
        # Assigning a Num to a Attribute (line 329):
        int_53443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 37), 'int')
        # Getting the type of 'self' (line 329)
        self_53444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 16), 'self')
        # Setting the type of the member 'n_equal_steps' of a type (line 329)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 16), self_53444, 'n_equal_steps', int_53443)
        
        # Assigning a Name to a Name (line 330):
        
        # Assigning a Name to a Name (line 330):
        # Getting the type of 'None' (line 330)
        None_53445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 21), 'None')
        # Assigning a type to the variable 'LU' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'LU', None_53445)
        # SSA join for if statement (line 326)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 332):
        
        # Assigning a BinOp to a Name (line 332):
        # Getting the type of 't_new' (line 332)
        t_new_53446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 16), 't_new')
        # Getting the type of 't' (line 332)
        t_53447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 24), 't')
        # Applying the binary operator '-' (line 332)
        result_sub_53448 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 16), '-', t_new_53446, t_53447)
        
        # Assigning a type to the variable 'h' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'h', result_sub_53448)
        
        # Assigning a Call to a Name (line 333):
        
        # Assigning a Call to a Name (line 333):
        
        # Call to abs(...): (line 333)
        # Processing the call arguments (line 333)
        # Getting the type of 'h' (line 333)
        h_53451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 27), 'h', False)
        # Processing the call keyword arguments (line 333)
        kwargs_53452 = {}
        # Getting the type of 'np' (line 333)
        np_53449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 20), 'np', False)
        # Obtaining the member 'abs' of a type (line 333)
        abs_53450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 20), np_53449, 'abs')
        # Calling abs(args, kwargs) (line 333)
        abs_call_result_53453 = invoke(stypy.reporting.localization.Localization(__file__, 333, 20), abs_53450, *[h_53451], **kwargs_53452)
        
        # Assigning a type to the variable 'h_abs' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'h_abs', abs_call_result_53453)
        
        # Assigning a Call to a Name (line 335):
        
        # Assigning a Call to a Name (line 335):
        
        # Call to sum(...): (line 335)
        # Processing the call arguments (line 335)
        
        # Obtaining the type of the subscript
        # Getting the type of 'order' (line 335)
        order_53456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 34), 'order', False)
        int_53457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 42), 'int')
        # Applying the binary operator '+' (line 335)
        result_add_53458 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 34), '+', order_53456, int_53457)
        
        slice_53459 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 335, 31), None, result_add_53458, None)
        # Getting the type of 'D' (line 335)
        D_53460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 31), 'D', False)
        # Obtaining the member '__getitem__' of a type (line 335)
        getitem___53461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 31), D_53460, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 335)
        subscript_call_result_53462 = invoke(stypy.reporting.localization.Localization(__file__, 335, 31), getitem___53461, slice_53459)
        
        # Processing the call keyword arguments (line 335)
        int_53463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 51), 'int')
        keyword_53464 = int_53463
        kwargs_53465 = {'axis': keyword_53464}
        # Getting the type of 'np' (line 335)
        np_53454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 24), 'np', False)
        # Obtaining the member 'sum' of a type (line 335)
        sum_53455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 24), np_53454, 'sum')
        # Calling sum(args, kwargs) (line 335)
        sum_call_result_53466 = invoke(stypy.reporting.localization.Localization(__file__, 335, 24), sum_53455, *[subscript_call_result_53462], **kwargs_53465)
        
        # Assigning a type to the variable 'y_predict' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'y_predict', sum_call_result_53466)
        
        # Assigning a BinOp to a Name (line 337):
        
        # Assigning a BinOp to a Name (line 337):
        # Getting the type of 'atol' (line 337)
        atol_53467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 20), 'atol')
        # Getting the type of 'rtol' (line 337)
        rtol_53468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 27), 'rtol')
        
        # Call to abs(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'y_predict' (line 337)
        y_predict_53471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 41), 'y_predict', False)
        # Processing the call keyword arguments (line 337)
        kwargs_53472 = {}
        # Getting the type of 'np' (line 337)
        np_53469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 34), 'np', False)
        # Obtaining the member 'abs' of a type (line 337)
        abs_53470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 34), np_53469, 'abs')
        # Calling abs(args, kwargs) (line 337)
        abs_call_result_53473 = invoke(stypy.reporting.localization.Localization(__file__, 337, 34), abs_53470, *[y_predict_53471], **kwargs_53472)
        
        # Applying the binary operator '*' (line 337)
        result_mul_53474 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 27), '*', rtol_53468, abs_call_result_53473)
        
        # Applying the binary operator '+' (line 337)
        result_add_53475 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 20), '+', atol_53467, result_mul_53474)
        
        # Assigning a type to the variable 'scale' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'scale', result_add_53475)
        
        # Assigning a BinOp to a Name (line 338):
        
        # Assigning a BinOp to a Name (line 338):
        
        # Call to dot(...): (line 338)
        # Processing the call arguments (line 338)
        
        # Obtaining the type of the subscript
        int_53478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 27), 'int')
        # Getting the type of 'order' (line 338)
        order_53479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 30), 'order', False)
        int_53480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 38), 'int')
        # Applying the binary operator '+' (line 338)
        result_add_53481 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 30), '+', order_53479, int_53480)
        
        slice_53482 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 338, 25), int_53478, result_add_53481, None)
        # Getting the type of 'D' (line 338)
        D_53483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 25), 'D', False)
        # Obtaining the member '__getitem__' of a type (line 338)
        getitem___53484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 25), D_53483, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 338)
        subscript_call_result_53485 = invoke(stypy.reporting.localization.Localization(__file__, 338, 25), getitem___53484, slice_53482)
        
        # Obtaining the member 'T' of a type (line 338)
        T_53486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 25), subscript_call_result_53485, 'T')
        
        # Obtaining the type of the subscript
        int_53487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 50), 'int')
        # Getting the type of 'order' (line 338)
        order_53488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 53), 'order', False)
        int_53489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 61), 'int')
        # Applying the binary operator '+' (line 338)
        result_add_53490 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 53), '+', order_53488, int_53489)
        
        slice_53491 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 338, 44), int_53487, result_add_53490, None)
        # Getting the type of 'gamma' (line 338)
        gamma_53492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 44), 'gamma', False)
        # Obtaining the member '__getitem__' of a type (line 338)
        getitem___53493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 44), gamma_53492, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 338)
        subscript_call_result_53494 = invoke(stypy.reporting.localization.Localization(__file__, 338, 44), getitem___53493, slice_53491)
        
        # Processing the call keyword arguments (line 338)
        kwargs_53495 = {}
        # Getting the type of 'np' (line 338)
        np_53476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 18), 'np', False)
        # Obtaining the member 'dot' of a type (line 338)
        dot_53477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 18), np_53476, 'dot')
        # Calling dot(args, kwargs) (line 338)
        dot_call_result_53496 = invoke(stypy.reporting.localization.Localization(__file__, 338, 18), dot_53477, *[T_53486, subscript_call_result_53494], **kwargs_53495)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'order' (line 338)
        order_53497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 73), 'order')
        # Getting the type of 'alpha' (line 338)
        alpha_53498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 67), 'alpha')
        # Obtaining the member '__getitem__' of a type (line 338)
        getitem___53499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 67), alpha_53498, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 338)
        subscript_call_result_53500 = invoke(stypy.reporting.localization.Localization(__file__, 338, 67), getitem___53499, order_53497)
        
        # Applying the binary operator 'div' (line 338)
        result_div_53501 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 18), 'div', dot_call_result_53496, subscript_call_result_53500)
        
        # Assigning a type to the variable 'psi' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'psi', result_div_53501)
        
        # Assigning a Name to a Name (line 340):
        
        # Assigning a Name to a Name (line 340):
        # Getting the type of 'False' (line 340)
        False_53502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 24), 'False')
        # Assigning a type to the variable 'converged' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 12), 'converged', False_53502)
        
        # Assigning a BinOp to a Name (line 341):
        
        # Assigning a BinOp to a Name (line 341):
        # Getting the type of 'h' (line 341)
        h_53503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 16), 'h')
        
        # Obtaining the type of the subscript
        # Getting the type of 'order' (line 341)
        order_53504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 26), 'order')
        # Getting the type of 'alpha' (line 341)
        alpha_53505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 20), 'alpha')
        # Obtaining the member '__getitem__' of a type (line 341)
        getitem___53506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 20), alpha_53505, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 341)
        subscript_call_result_53507 = invoke(stypy.reporting.localization.Localization(__file__, 341, 20), getitem___53506, order_53504)
        
        # Applying the binary operator 'div' (line 341)
        result_div_53508 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 16), 'div', h_53503, subscript_call_result_53507)
        
        # Assigning a type to the variable 'c' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'c', result_div_53508)
        
        
        # Getting the type of 'converged' (line 342)
        converged_53509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 22), 'converged')
        # Applying the 'not' unary operator (line 342)
        result_not__53510 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 18), 'not', converged_53509)
        
        # Testing the type of an if condition (line 342)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 342, 12), result_not__53510)
        # SSA begins for while statement (line 342)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Type idiom detected: calculating its left and rigth part (line 343)
        # Getting the type of 'LU' (line 343)
        LU_53511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 19), 'LU')
        # Getting the type of 'None' (line 343)
        None_53512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 25), 'None')
        
        (may_be_53513, more_types_in_union_53514) = may_be_none(LU_53511, None_53512)

        if may_be_53513:

            if more_types_in_union_53514:
                # Runtime conditional SSA (line 343)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 344):
            
            # Assigning a Call to a Name (line 344):
            
            # Call to lu(...): (line 344)
            # Processing the call arguments (line 344)
            # Getting the type of 'self' (line 344)
            self_53517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 33), 'self', False)
            # Obtaining the member 'I' of a type (line 344)
            I_53518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 33), self_53517, 'I')
            # Getting the type of 'c' (line 344)
            c_53519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 42), 'c', False)
            # Getting the type of 'J' (line 344)
            J_53520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 46), 'J', False)
            # Applying the binary operator '*' (line 344)
            result_mul_53521 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 42), '*', c_53519, J_53520)
            
            # Applying the binary operator '-' (line 344)
            result_sub_53522 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 33), '-', I_53518, result_mul_53521)
            
            # Processing the call keyword arguments (line 344)
            kwargs_53523 = {}
            # Getting the type of 'self' (line 344)
            self_53515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 25), 'self', False)
            # Obtaining the member 'lu' of a type (line 344)
            lu_53516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 25), self_53515, 'lu')
            # Calling lu(args, kwargs) (line 344)
            lu_call_result_53524 = invoke(stypy.reporting.localization.Localization(__file__, 344, 25), lu_53516, *[result_sub_53522], **kwargs_53523)
            
            # Assigning a type to the variable 'LU' (line 344)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 20), 'LU', lu_call_result_53524)

            if more_types_in_union_53514:
                # SSA join for if statement (line 343)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Tuple (line 346):
        
        # Assigning a Subscript to a Name (line 346):
        
        # Obtaining the type of the subscript
        int_53525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 16), 'int')
        
        # Call to solve_bdf_system(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'self' (line 347)
        self_53527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 20), 'self', False)
        # Obtaining the member 'fun' of a type (line 347)
        fun_53528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 20), self_53527, 'fun')
        # Getting the type of 't_new' (line 347)
        t_new_53529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 30), 't_new', False)
        # Getting the type of 'y_predict' (line 347)
        y_predict_53530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 37), 'y_predict', False)
        # Getting the type of 'c' (line 347)
        c_53531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 48), 'c', False)
        # Getting the type of 'psi' (line 347)
        psi_53532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 51), 'psi', False)
        # Getting the type of 'LU' (line 347)
        LU_53533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 56), 'LU', False)
        # Getting the type of 'self' (line 347)
        self_53534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 60), 'self', False)
        # Obtaining the member 'solve_lu' of a type (line 347)
        solve_lu_53535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 60), self_53534, 'solve_lu')
        # Getting the type of 'scale' (line 348)
        scale_53536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 20), 'scale', False)
        # Getting the type of 'self' (line 348)
        self_53537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 27), 'self', False)
        # Obtaining the member 'newton_tol' of a type (line 348)
        newton_tol_53538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 27), self_53537, 'newton_tol')
        # Processing the call keyword arguments (line 346)
        kwargs_53539 = {}
        # Getting the type of 'solve_bdf_system' (line 346)
        solve_bdf_system_53526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 46), 'solve_bdf_system', False)
        # Calling solve_bdf_system(args, kwargs) (line 346)
        solve_bdf_system_call_result_53540 = invoke(stypy.reporting.localization.Localization(__file__, 346, 46), solve_bdf_system_53526, *[fun_53528, t_new_53529, y_predict_53530, c_53531, psi_53532, LU_53533, solve_lu_53535, scale_53536, newton_tol_53538], **kwargs_53539)
        
        # Obtaining the member '__getitem__' of a type (line 346)
        getitem___53541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 16), solve_bdf_system_call_result_53540, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 346)
        subscript_call_result_53542 = invoke(stypy.reporting.localization.Localization(__file__, 346, 16), getitem___53541, int_53525)
        
        # Assigning a type to the variable 'tuple_var_assignment_52583' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 16), 'tuple_var_assignment_52583', subscript_call_result_53542)
        
        # Assigning a Subscript to a Name (line 346):
        
        # Obtaining the type of the subscript
        int_53543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 16), 'int')
        
        # Call to solve_bdf_system(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'self' (line 347)
        self_53545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 20), 'self', False)
        # Obtaining the member 'fun' of a type (line 347)
        fun_53546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 20), self_53545, 'fun')
        # Getting the type of 't_new' (line 347)
        t_new_53547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 30), 't_new', False)
        # Getting the type of 'y_predict' (line 347)
        y_predict_53548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 37), 'y_predict', False)
        # Getting the type of 'c' (line 347)
        c_53549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 48), 'c', False)
        # Getting the type of 'psi' (line 347)
        psi_53550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 51), 'psi', False)
        # Getting the type of 'LU' (line 347)
        LU_53551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 56), 'LU', False)
        # Getting the type of 'self' (line 347)
        self_53552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 60), 'self', False)
        # Obtaining the member 'solve_lu' of a type (line 347)
        solve_lu_53553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 60), self_53552, 'solve_lu')
        # Getting the type of 'scale' (line 348)
        scale_53554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 20), 'scale', False)
        # Getting the type of 'self' (line 348)
        self_53555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 27), 'self', False)
        # Obtaining the member 'newton_tol' of a type (line 348)
        newton_tol_53556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 27), self_53555, 'newton_tol')
        # Processing the call keyword arguments (line 346)
        kwargs_53557 = {}
        # Getting the type of 'solve_bdf_system' (line 346)
        solve_bdf_system_53544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 46), 'solve_bdf_system', False)
        # Calling solve_bdf_system(args, kwargs) (line 346)
        solve_bdf_system_call_result_53558 = invoke(stypy.reporting.localization.Localization(__file__, 346, 46), solve_bdf_system_53544, *[fun_53546, t_new_53547, y_predict_53548, c_53549, psi_53550, LU_53551, solve_lu_53553, scale_53554, newton_tol_53556], **kwargs_53557)
        
        # Obtaining the member '__getitem__' of a type (line 346)
        getitem___53559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 16), solve_bdf_system_call_result_53558, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 346)
        subscript_call_result_53560 = invoke(stypy.reporting.localization.Localization(__file__, 346, 16), getitem___53559, int_53543)
        
        # Assigning a type to the variable 'tuple_var_assignment_52584' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 16), 'tuple_var_assignment_52584', subscript_call_result_53560)
        
        # Assigning a Subscript to a Name (line 346):
        
        # Obtaining the type of the subscript
        int_53561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 16), 'int')
        
        # Call to solve_bdf_system(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'self' (line 347)
        self_53563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 20), 'self', False)
        # Obtaining the member 'fun' of a type (line 347)
        fun_53564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 20), self_53563, 'fun')
        # Getting the type of 't_new' (line 347)
        t_new_53565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 30), 't_new', False)
        # Getting the type of 'y_predict' (line 347)
        y_predict_53566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 37), 'y_predict', False)
        # Getting the type of 'c' (line 347)
        c_53567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 48), 'c', False)
        # Getting the type of 'psi' (line 347)
        psi_53568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 51), 'psi', False)
        # Getting the type of 'LU' (line 347)
        LU_53569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 56), 'LU', False)
        # Getting the type of 'self' (line 347)
        self_53570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 60), 'self', False)
        # Obtaining the member 'solve_lu' of a type (line 347)
        solve_lu_53571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 60), self_53570, 'solve_lu')
        # Getting the type of 'scale' (line 348)
        scale_53572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 20), 'scale', False)
        # Getting the type of 'self' (line 348)
        self_53573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 27), 'self', False)
        # Obtaining the member 'newton_tol' of a type (line 348)
        newton_tol_53574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 27), self_53573, 'newton_tol')
        # Processing the call keyword arguments (line 346)
        kwargs_53575 = {}
        # Getting the type of 'solve_bdf_system' (line 346)
        solve_bdf_system_53562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 46), 'solve_bdf_system', False)
        # Calling solve_bdf_system(args, kwargs) (line 346)
        solve_bdf_system_call_result_53576 = invoke(stypy.reporting.localization.Localization(__file__, 346, 46), solve_bdf_system_53562, *[fun_53564, t_new_53565, y_predict_53566, c_53567, psi_53568, LU_53569, solve_lu_53571, scale_53572, newton_tol_53574], **kwargs_53575)
        
        # Obtaining the member '__getitem__' of a type (line 346)
        getitem___53577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 16), solve_bdf_system_call_result_53576, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 346)
        subscript_call_result_53578 = invoke(stypy.reporting.localization.Localization(__file__, 346, 16), getitem___53577, int_53561)
        
        # Assigning a type to the variable 'tuple_var_assignment_52585' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 16), 'tuple_var_assignment_52585', subscript_call_result_53578)
        
        # Assigning a Subscript to a Name (line 346):
        
        # Obtaining the type of the subscript
        int_53579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 16), 'int')
        
        # Call to solve_bdf_system(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'self' (line 347)
        self_53581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 20), 'self', False)
        # Obtaining the member 'fun' of a type (line 347)
        fun_53582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 20), self_53581, 'fun')
        # Getting the type of 't_new' (line 347)
        t_new_53583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 30), 't_new', False)
        # Getting the type of 'y_predict' (line 347)
        y_predict_53584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 37), 'y_predict', False)
        # Getting the type of 'c' (line 347)
        c_53585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 48), 'c', False)
        # Getting the type of 'psi' (line 347)
        psi_53586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 51), 'psi', False)
        # Getting the type of 'LU' (line 347)
        LU_53587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 56), 'LU', False)
        # Getting the type of 'self' (line 347)
        self_53588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 60), 'self', False)
        # Obtaining the member 'solve_lu' of a type (line 347)
        solve_lu_53589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 60), self_53588, 'solve_lu')
        # Getting the type of 'scale' (line 348)
        scale_53590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 20), 'scale', False)
        # Getting the type of 'self' (line 348)
        self_53591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 27), 'self', False)
        # Obtaining the member 'newton_tol' of a type (line 348)
        newton_tol_53592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 27), self_53591, 'newton_tol')
        # Processing the call keyword arguments (line 346)
        kwargs_53593 = {}
        # Getting the type of 'solve_bdf_system' (line 346)
        solve_bdf_system_53580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 46), 'solve_bdf_system', False)
        # Calling solve_bdf_system(args, kwargs) (line 346)
        solve_bdf_system_call_result_53594 = invoke(stypy.reporting.localization.Localization(__file__, 346, 46), solve_bdf_system_53580, *[fun_53582, t_new_53583, y_predict_53584, c_53585, psi_53586, LU_53587, solve_lu_53589, scale_53590, newton_tol_53592], **kwargs_53593)
        
        # Obtaining the member '__getitem__' of a type (line 346)
        getitem___53595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 16), solve_bdf_system_call_result_53594, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 346)
        subscript_call_result_53596 = invoke(stypy.reporting.localization.Localization(__file__, 346, 16), getitem___53595, int_53579)
        
        # Assigning a type to the variable 'tuple_var_assignment_52586' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 16), 'tuple_var_assignment_52586', subscript_call_result_53596)
        
        # Assigning a Name to a Name (line 346):
        # Getting the type of 'tuple_var_assignment_52583' (line 346)
        tuple_var_assignment_52583_53597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 16), 'tuple_var_assignment_52583')
        # Assigning a type to the variable 'converged' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 16), 'converged', tuple_var_assignment_52583_53597)
        
        # Assigning a Name to a Name (line 346):
        # Getting the type of 'tuple_var_assignment_52584' (line 346)
        tuple_var_assignment_52584_53598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 16), 'tuple_var_assignment_52584')
        # Assigning a type to the variable 'n_iter' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 27), 'n_iter', tuple_var_assignment_52584_53598)
        
        # Assigning a Name to a Name (line 346):
        # Getting the type of 'tuple_var_assignment_52585' (line 346)
        tuple_var_assignment_52585_53599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 16), 'tuple_var_assignment_52585')
        # Assigning a type to the variable 'y_new' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 35), 'y_new', tuple_var_assignment_52585_53599)
        
        # Assigning a Name to a Name (line 346):
        # Getting the type of 'tuple_var_assignment_52586' (line 346)
        tuple_var_assignment_52586_53600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 16), 'tuple_var_assignment_52586')
        # Assigning a type to the variable 'd' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 42), 'd', tuple_var_assignment_52586_53600)
        
        
        # Getting the type of 'converged' (line 350)
        converged_53601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 23), 'converged')
        # Applying the 'not' unary operator (line 350)
        result_not__53602 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 19), 'not', converged_53601)
        
        # Testing the type of an if condition (line 350)
        if_condition_53603 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 350, 16), result_not__53602)
        # Assigning a type to the variable 'if_condition_53603' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 16), 'if_condition_53603', if_condition_53603)
        # SSA begins for if statement (line 350)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'current_jac' (line 351)
        current_jac_53604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 23), 'current_jac')
        # Testing the type of an if condition (line 351)
        if_condition_53605 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 351, 20), current_jac_53604)
        # Assigning a type to the variable 'if_condition_53605' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 20), 'if_condition_53605', if_condition_53605)
        # SSA begins for if statement (line 351)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 351)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 353):
        
        # Assigning a Call to a Name (line 353):
        
        # Call to jac(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 't_new' (line 353)
        t_new_53608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 33), 't_new', False)
        # Getting the type of 'y_predict' (line 353)
        y_predict_53609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 40), 'y_predict', False)
        # Processing the call keyword arguments (line 353)
        kwargs_53610 = {}
        # Getting the type of 'self' (line 353)
        self_53606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 24), 'self', False)
        # Obtaining the member 'jac' of a type (line 353)
        jac_53607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 24), self_53606, 'jac')
        # Calling jac(args, kwargs) (line 353)
        jac_call_result_53611 = invoke(stypy.reporting.localization.Localization(__file__, 353, 24), jac_53607, *[t_new_53608, y_predict_53609], **kwargs_53610)
        
        # Assigning a type to the variable 'J' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 20), 'J', jac_call_result_53611)
        
        # Assigning a Name to a Name (line 354):
        
        # Assigning a Name to a Name (line 354):
        # Getting the type of 'None' (line 354)
        None_53612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 25), 'None')
        # Assigning a type to the variable 'LU' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 20), 'LU', None_53612)
        
        # Assigning a Name to a Name (line 355):
        
        # Assigning a Name to a Name (line 355):
        # Getting the type of 'True' (line 355)
        True_53613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 34), 'True')
        # Assigning a type to the variable 'current_jac' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 20), 'current_jac', True_53613)
        # SSA join for if statement (line 350)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 342)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'converged' (line 357)
        converged_53614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 19), 'converged')
        # Applying the 'not' unary operator (line 357)
        result_not__53615 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 15), 'not', converged_53614)
        
        # Testing the type of an if condition (line 357)
        if_condition_53616 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 357, 12), result_not__53615)
        # Assigning a type to the variable 'if_condition_53616' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'if_condition_53616', if_condition_53616)
        # SSA begins for if statement (line 357)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 358):
        
        # Assigning a Num to a Name (line 358):
        float_53617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 25), 'float')
        # Assigning a type to the variable 'factor' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 16), 'factor', float_53617)
        
        # Getting the type of 'h_abs' (line 359)
        h_abs_53618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 16), 'h_abs')
        # Getting the type of 'factor' (line 359)
        factor_53619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 25), 'factor')
        # Applying the binary operator '*=' (line 359)
        result_imul_53620 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 16), '*=', h_abs_53618, factor_53619)
        # Assigning a type to the variable 'h_abs' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 16), 'h_abs', result_imul_53620)
        
        
        # Call to change_D(...): (line 360)
        # Processing the call arguments (line 360)
        # Getting the type of 'D' (line 360)
        D_53622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 25), 'D', False)
        # Getting the type of 'order' (line 360)
        order_53623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 28), 'order', False)
        # Getting the type of 'factor' (line 360)
        factor_53624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 35), 'factor', False)
        # Processing the call keyword arguments (line 360)
        kwargs_53625 = {}
        # Getting the type of 'change_D' (line 360)
        change_D_53621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 16), 'change_D', False)
        # Calling change_D(args, kwargs) (line 360)
        change_D_call_result_53626 = invoke(stypy.reporting.localization.Localization(__file__, 360, 16), change_D_53621, *[D_53622, order_53623, factor_53624], **kwargs_53625)
        
        
        # Assigning a Num to a Attribute (line 361):
        
        # Assigning a Num to a Attribute (line 361):
        int_53627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 37), 'int')
        # Getting the type of 'self' (line 361)
        self_53628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 16), 'self')
        # Setting the type of the member 'n_equal_steps' of a type (line 361)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 16), self_53628, 'n_equal_steps', int_53627)
        
        # Assigning a Name to a Name (line 362):
        
        # Assigning a Name to a Name (line 362):
        # Getting the type of 'None' (line 362)
        None_53629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 21), 'None')
        # Assigning a type to the variable 'LU' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 16), 'LU', None_53629)
        # SSA join for if statement (line 357)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 365):
        
        # Assigning a BinOp to a Name (line 365):
        float_53630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 21), 'float')
        int_53631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 28), 'int')
        # Getting the type of 'NEWTON_MAXITER' (line 365)
        NEWTON_MAXITER_53632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 32), 'NEWTON_MAXITER')
        # Applying the binary operator '*' (line 365)
        result_mul_53633 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 28), '*', int_53631, NEWTON_MAXITER_53632)
        
        int_53634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 49), 'int')
        # Applying the binary operator '+' (line 365)
        result_add_53635 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 28), '+', result_mul_53633, int_53634)
        
        # Applying the binary operator '*' (line 365)
        result_mul_53636 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 21), '*', float_53630, result_add_53635)
        
        int_53637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 55), 'int')
        # Getting the type of 'NEWTON_MAXITER' (line 365)
        NEWTON_MAXITER_53638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 59), 'NEWTON_MAXITER')
        # Applying the binary operator '*' (line 365)
        result_mul_53639 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 55), '*', int_53637, NEWTON_MAXITER_53638)
        
        # Getting the type of 'n_iter' (line 366)
        n_iter_53640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 57), 'n_iter')
        # Applying the binary operator '+' (line 365)
        result_add_53641 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 55), '+', result_mul_53639, n_iter_53640)
        
        # Applying the binary operator 'div' (line 365)
        result_div_53642 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 52), 'div', result_mul_53636, result_add_53641)
        
        # Assigning a type to the variable 'safety' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'safety', result_div_53642)
        
        # Assigning a BinOp to a Name (line 368):
        
        # Assigning a BinOp to a Name (line 368):
        # Getting the type of 'atol' (line 368)
        atol_53643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 20), 'atol')
        # Getting the type of 'rtol' (line 368)
        rtol_53644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 27), 'rtol')
        
        # Call to abs(...): (line 368)
        # Processing the call arguments (line 368)
        # Getting the type of 'y_new' (line 368)
        y_new_53647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 41), 'y_new', False)
        # Processing the call keyword arguments (line 368)
        kwargs_53648 = {}
        # Getting the type of 'np' (line 368)
        np_53645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 34), 'np', False)
        # Obtaining the member 'abs' of a type (line 368)
        abs_53646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 34), np_53645, 'abs')
        # Calling abs(args, kwargs) (line 368)
        abs_call_result_53649 = invoke(stypy.reporting.localization.Localization(__file__, 368, 34), abs_53646, *[y_new_53647], **kwargs_53648)
        
        # Applying the binary operator '*' (line 368)
        result_mul_53650 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 27), '*', rtol_53644, abs_call_result_53649)
        
        # Applying the binary operator '+' (line 368)
        result_add_53651 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 20), '+', atol_53643, result_mul_53650)
        
        # Assigning a type to the variable 'scale' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'scale', result_add_53651)
        
        # Assigning a BinOp to a Name (line 369):
        
        # Assigning a BinOp to a Name (line 369):
        
        # Obtaining the type of the subscript
        # Getting the type of 'order' (line 369)
        order_53652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 32), 'order')
        # Getting the type of 'error_const' (line 369)
        error_const_53653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 20), 'error_const')
        # Obtaining the member '__getitem__' of a type (line 369)
        getitem___53654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 20), error_const_53653, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 369)
        subscript_call_result_53655 = invoke(stypy.reporting.localization.Localization(__file__, 369, 20), getitem___53654, order_53652)
        
        # Getting the type of 'd' (line 369)
        d_53656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 41), 'd')
        # Applying the binary operator '*' (line 369)
        result_mul_53657 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 20), '*', subscript_call_result_53655, d_53656)
        
        # Assigning a type to the variable 'error' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'error', result_mul_53657)
        
        # Assigning a Call to a Name (line 370):
        
        # Assigning a Call to a Name (line 370):
        
        # Call to norm(...): (line 370)
        # Processing the call arguments (line 370)
        # Getting the type of 'error' (line 370)
        error_53659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 30), 'error', False)
        # Getting the type of 'scale' (line 370)
        scale_53660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 38), 'scale', False)
        # Applying the binary operator 'div' (line 370)
        result_div_53661 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 30), 'div', error_53659, scale_53660)
        
        # Processing the call keyword arguments (line 370)
        kwargs_53662 = {}
        # Getting the type of 'norm' (line 370)
        norm_53658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 25), 'norm', False)
        # Calling norm(args, kwargs) (line 370)
        norm_call_result_53663 = invoke(stypy.reporting.localization.Localization(__file__, 370, 25), norm_53658, *[result_div_53661], **kwargs_53662)
        
        # Assigning a type to the variable 'error_norm' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'error_norm', norm_call_result_53663)
        
        
        # Getting the type of 'error_norm' (line 372)
        error_norm_53664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 15), 'error_norm')
        int_53665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 28), 'int')
        # Applying the binary operator '>' (line 372)
        result_gt_53666 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 15), '>', error_norm_53664, int_53665)
        
        # Testing the type of an if condition (line 372)
        if_condition_53667 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 372, 12), result_gt_53666)
        # Assigning a type to the variable 'if_condition_53667' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'if_condition_53667', if_condition_53667)
        # SSA begins for if statement (line 372)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 373):
        
        # Assigning a Call to a Name (line 373):
        
        # Call to max(...): (line 373)
        # Processing the call arguments (line 373)
        # Getting the type of 'MIN_FACTOR' (line 373)
        MIN_FACTOR_53669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 29), 'MIN_FACTOR', False)
        # Getting the type of 'safety' (line 374)
        safety_53670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 29), 'safety', False)
        # Getting the type of 'error_norm' (line 374)
        error_norm_53671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 38), 'error_norm', False)
        int_53672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 53), 'int')
        # Getting the type of 'order' (line 374)
        order_53673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 59), 'order', False)
        int_53674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 67), 'int')
        # Applying the binary operator '+' (line 374)
        result_add_53675 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 59), '+', order_53673, int_53674)
        
        # Applying the binary operator 'div' (line 374)
        result_div_53676 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 53), 'div', int_53672, result_add_53675)
        
        # Applying the binary operator '**' (line 374)
        result_pow_53677 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 38), '**', error_norm_53671, result_div_53676)
        
        # Applying the binary operator '*' (line 374)
        result_mul_53678 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 29), '*', safety_53670, result_pow_53677)
        
        # Processing the call keyword arguments (line 373)
        kwargs_53679 = {}
        # Getting the type of 'max' (line 373)
        max_53668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 25), 'max', False)
        # Calling max(args, kwargs) (line 373)
        max_call_result_53680 = invoke(stypy.reporting.localization.Localization(__file__, 373, 25), max_53668, *[MIN_FACTOR_53669, result_mul_53678], **kwargs_53679)
        
        # Assigning a type to the variable 'factor' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 16), 'factor', max_call_result_53680)
        
        # Getting the type of 'h_abs' (line 375)
        h_abs_53681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 16), 'h_abs')
        # Getting the type of 'factor' (line 375)
        factor_53682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 25), 'factor')
        # Applying the binary operator '*=' (line 375)
        result_imul_53683 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 16), '*=', h_abs_53681, factor_53682)
        # Assigning a type to the variable 'h_abs' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 16), 'h_abs', result_imul_53683)
        
        
        # Call to change_D(...): (line 376)
        # Processing the call arguments (line 376)
        # Getting the type of 'D' (line 376)
        D_53685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 25), 'D', False)
        # Getting the type of 'order' (line 376)
        order_53686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 28), 'order', False)
        # Getting the type of 'factor' (line 376)
        factor_53687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 35), 'factor', False)
        # Processing the call keyword arguments (line 376)
        kwargs_53688 = {}
        # Getting the type of 'change_D' (line 376)
        change_D_53684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 16), 'change_D', False)
        # Calling change_D(args, kwargs) (line 376)
        change_D_call_result_53689 = invoke(stypy.reporting.localization.Localization(__file__, 376, 16), change_D_53684, *[D_53685, order_53686, factor_53687], **kwargs_53688)
        
        
        # Assigning a Num to a Attribute (line 377):
        
        # Assigning a Num to a Attribute (line 377):
        int_53690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 37), 'int')
        # Getting the type of 'self' (line 377)
        self_53691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 16), 'self')
        # Setting the type of the member 'n_equal_steps' of a type (line 377)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 16), self_53691, 'n_equal_steps', int_53690)
        # SSA branch for the else part of an if statement (line 372)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 381):
        
        # Assigning a Name to a Name (line 381):
        # Getting the type of 'True' (line 381)
        True_53692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 32), 'True')
        # Assigning a type to the variable 'step_accepted' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 16), 'step_accepted', True_53692)
        # SSA join for if statement (line 372)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 319)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 383)
        self_53693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'self')
        # Obtaining the member 'n_equal_steps' of a type (line 383)
        n_equal_steps_53694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), self_53693, 'n_equal_steps')
        int_53695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 30), 'int')
        # Applying the binary operator '+=' (line 383)
        result_iadd_53696 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 8), '+=', n_equal_steps_53694, int_53695)
        # Getting the type of 'self' (line 383)
        self_53697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'self')
        # Setting the type of the member 'n_equal_steps' of a type (line 383)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), self_53697, 'n_equal_steps', result_iadd_53696)
        
        
        # Assigning a Name to a Attribute (line 385):
        
        # Assigning a Name to a Attribute (line 385):
        # Getting the type of 't_new' (line 385)
        t_new_53698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 17), 't_new')
        # Getting the type of 'self' (line 385)
        self_53699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'self')
        # Setting the type of the member 't' of a type (line 385)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 8), self_53699, 't', t_new_53698)
        
        # Assigning a Name to a Attribute (line 386):
        
        # Assigning a Name to a Attribute (line 386):
        # Getting the type of 'y_new' (line 386)
        y_new_53700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 17), 'y_new')
        # Getting the type of 'self' (line 386)
        self_53701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'self')
        # Setting the type of the member 'y' of a type (line 386)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 8), self_53701, 'y', y_new_53700)
        
        # Assigning a Name to a Attribute (line 388):
        
        # Assigning a Name to a Attribute (line 388):
        # Getting the type of 'h_abs' (line 388)
        h_abs_53702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 21), 'h_abs')
        # Getting the type of 'self' (line 388)
        self_53703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'self')
        # Setting the type of the member 'h_abs' of a type (line 388)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 8), self_53703, 'h_abs', h_abs_53702)
        
        # Assigning a Name to a Attribute (line 389):
        
        # Assigning a Name to a Attribute (line 389):
        # Getting the type of 'J' (line 389)
        J_53704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 17), 'J')
        # Getting the type of 'self' (line 389)
        self_53705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'self')
        # Setting the type of the member 'J' of a type (line 389)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 8), self_53705, 'J', J_53704)
        
        # Assigning a Name to a Attribute (line 390):
        
        # Assigning a Name to a Attribute (line 390):
        # Getting the type of 'LU' (line 390)
        LU_53706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 18), 'LU')
        # Getting the type of 'self' (line 390)
        self_53707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'self')
        # Setting the type of the member 'LU' of a type (line 390)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 8), self_53707, 'LU', LU_53706)
        
        # Assigning a BinOp to a Subscript (line 396):
        
        # Assigning a BinOp to a Subscript (line 396):
        # Getting the type of 'd' (line 396)
        d_53708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 23), 'd')
        
        # Obtaining the type of the subscript
        # Getting the type of 'order' (line 396)
        order_53709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 29), 'order')
        int_53710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 37), 'int')
        # Applying the binary operator '+' (line 396)
        result_add_53711 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 29), '+', order_53709, int_53710)
        
        # Getting the type of 'D' (line 396)
        D_53712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 27), 'D')
        # Obtaining the member '__getitem__' of a type (line 396)
        getitem___53713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 27), D_53712, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 396)
        subscript_call_result_53714 = invoke(stypy.reporting.localization.Localization(__file__, 396, 27), getitem___53713, result_add_53711)
        
        # Applying the binary operator '-' (line 396)
        result_sub_53715 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 23), '-', d_53708, subscript_call_result_53714)
        
        # Getting the type of 'D' (line 396)
        D_53716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'D')
        # Getting the type of 'order' (line 396)
        order_53717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 10), 'order')
        int_53718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 18), 'int')
        # Applying the binary operator '+' (line 396)
        result_add_53719 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 10), '+', order_53717, int_53718)
        
        # Storing an element on a container (line 396)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 8), D_53716, (result_add_53719, result_sub_53715))
        
        # Assigning a Name to a Subscript (line 397):
        
        # Assigning a Name to a Subscript (line 397):
        # Getting the type of 'd' (line 397)
        d_53720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 23), 'd')
        # Getting the type of 'D' (line 397)
        D_53721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'D')
        # Getting the type of 'order' (line 397)
        order_53722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 10), 'order')
        int_53723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 18), 'int')
        # Applying the binary operator '+' (line 397)
        result_add_53724 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 10), '+', order_53722, int_53723)
        
        # Storing an element on a container (line 397)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 8), D_53721, (result_add_53724, d_53720))
        
        
        # Call to reversed(...): (line 398)
        # Processing the call arguments (line 398)
        
        # Call to range(...): (line 398)
        # Processing the call arguments (line 398)
        # Getting the type of 'order' (line 398)
        order_53727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 32), 'order', False)
        int_53728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 40), 'int')
        # Applying the binary operator '+' (line 398)
        result_add_53729 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 32), '+', order_53727, int_53728)
        
        # Processing the call keyword arguments (line 398)
        kwargs_53730 = {}
        # Getting the type of 'range' (line 398)
        range_53726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 26), 'range', False)
        # Calling range(args, kwargs) (line 398)
        range_call_result_53731 = invoke(stypy.reporting.localization.Localization(__file__, 398, 26), range_53726, *[result_add_53729], **kwargs_53730)
        
        # Processing the call keyword arguments (line 398)
        kwargs_53732 = {}
        # Getting the type of 'reversed' (line 398)
        reversed_53725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 17), 'reversed', False)
        # Calling reversed(args, kwargs) (line 398)
        reversed_call_result_53733 = invoke(stypy.reporting.localization.Localization(__file__, 398, 17), reversed_53725, *[range_call_result_53731], **kwargs_53732)
        
        # Testing the type of a for loop iterable (line 398)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 398, 8), reversed_call_result_53733)
        # Getting the type of the for loop variable (line 398)
        for_loop_var_53734 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 398, 8), reversed_call_result_53733)
        # Assigning a type to the variable 'i' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'i', for_loop_var_53734)
        # SSA begins for a for statement (line 398)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'D' (line 399)
        D_53735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'D')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 399)
        i_53736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 14), 'i')
        # Getting the type of 'D' (line 399)
        D_53737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'D')
        # Obtaining the member '__getitem__' of a type (line 399)
        getitem___53738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 12), D_53737, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 399)
        subscript_call_result_53739 = invoke(stypy.reporting.localization.Localization(__file__, 399, 12), getitem___53738, i_53736)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 399)
        i_53740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 22), 'i')
        int_53741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 26), 'int')
        # Applying the binary operator '+' (line 399)
        result_add_53742 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 22), '+', i_53740, int_53741)
        
        # Getting the type of 'D' (line 399)
        D_53743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 20), 'D')
        # Obtaining the member '__getitem__' of a type (line 399)
        getitem___53744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 20), D_53743, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 399)
        subscript_call_result_53745 = invoke(stypy.reporting.localization.Localization(__file__, 399, 20), getitem___53744, result_add_53742)
        
        # Applying the binary operator '+=' (line 399)
        result_iadd_53746 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 12), '+=', subscript_call_result_53739, subscript_call_result_53745)
        # Getting the type of 'D' (line 399)
        D_53747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'D')
        # Getting the type of 'i' (line 399)
        i_53748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 14), 'i')
        # Storing an element on a container (line 399)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 12), D_53747, (i_53748, result_iadd_53746))
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 401)
        self_53749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 11), 'self')
        # Obtaining the member 'n_equal_steps' of a type (line 401)
        n_equal_steps_53750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 11), self_53749, 'n_equal_steps')
        # Getting the type of 'order' (line 401)
        order_53751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 32), 'order')
        int_53752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 40), 'int')
        # Applying the binary operator '+' (line 401)
        result_add_53753 = python_operator(stypy.reporting.localization.Localization(__file__, 401, 32), '+', order_53751, int_53752)
        
        # Applying the binary operator '<' (line 401)
        result_lt_53754 = python_operator(stypy.reporting.localization.Localization(__file__, 401, 11), '<', n_equal_steps_53750, result_add_53753)
        
        # Testing the type of an if condition (line 401)
        if_condition_53755 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 401, 8), result_lt_53754)
        # Assigning a type to the variable 'if_condition_53755' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'if_condition_53755', if_condition_53755)
        # SSA begins for if statement (line 401)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 402)
        tuple_53756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 402)
        # Adding element type (line 402)
        # Getting the type of 'True' (line 402)
        True_53757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 19), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 19), tuple_53756, True_53757)
        # Adding element type (line 402)
        # Getting the type of 'None' (line 402)
        None_53758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 25), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 19), tuple_53756, None_53758)
        
        # Assigning a type to the variable 'stypy_return_type' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'stypy_return_type', tuple_53756)
        # SSA join for if statement (line 401)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'order' (line 404)
        order_53759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 11), 'order')
        int_53760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 19), 'int')
        # Applying the binary operator '>' (line 404)
        result_gt_53761 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 11), '>', order_53759, int_53760)
        
        # Testing the type of an if condition (line 404)
        if_condition_53762 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 404, 8), result_gt_53761)
        # Assigning a type to the variable 'if_condition_53762' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'if_condition_53762', if_condition_53762)
        # SSA begins for if statement (line 404)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 405):
        
        # Assigning a BinOp to a Name (line 405):
        
        # Obtaining the type of the subscript
        # Getting the type of 'order' (line 405)
        order_53763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 34), 'order')
        int_53764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 42), 'int')
        # Applying the binary operator '-' (line 405)
        result_sub_53765 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 34), '-', order_53763, int_53764)
        
        # Getting the type of 'error_const' (line 405)
        error_const_53766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 22), 'error_const')
        # Obtaining the member '__getitem__' of a type (line 405)
        getitem___53767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 22), error_const_53766, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 405)
        subscript_call_result_53768 = invoke(stypy.reporting.localization.Localization(__file__, 405, 22), getitem___53767, result_sub_53765)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'order' (line 405)
        order_53769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 49), 'order')
        # Getting the type of 'D' (line 405)
        D_53770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 47), 'D')
        # Obtaining the member '__getitem__' of a type (line 405)
        getitem___53771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 47), D_53770, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 405)
        subscript_call_result_53772 = invoke(stypy.reporting.localization.Localization(__file__, 405, 47), getitem___53771, order_53769)
        
        # Applying the binary operator '*' (line 405)
        result_mul_53773 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 22), '*', subscript_call_result_53768, subscript_call_result_53772)
        
        # Assigning a type to the variable 'error_m' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'error_m', result_mul_53773)
        
        # Assigning a Call to a Name (line 406):
        
        # Assigning a Call to a Name (line 406):
        
        # Call to norm(...): (line 406)
        # Processing the call arguments (line 406)
        # Getting the type of 'error_m' (line 406)
        error_m_53775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 32), 'error_m', False)
        # Getting the type of 'scale' (line 406)
        scale_53776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 42), 'scale', False)
        # Applying the binary operator 'div' (line 406)
        result_div_53777 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 32), 'div', error_m_53775, scale_53776)
        
        # Processing the call keyword arguments (line 406)
        kwargs_53778 = {}
        # Getting the type of 'norm' (line 406)
        norm_53774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 27), 'norm', False)
        # Calling norm(args, kwargs) (line 406)
        norm_call_result_53779 = invoke(stypy.reporting.localization.Localization(__file__, 406, 27), norm_53774, *[result_div_53777], **kwargs_53778)
        
        # Assigning a type to the variable 'error_m_norm' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'error_m_norm', norm_call_result_53779)
        # SSA branch for the else part of an if statement (line 404)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 408):
        
        # Assigning a Attribute to a Name (line 408):
        # Getting the type of 'np' (line 408)
        np_53780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 27), 'np')
        # Obtaining the member 'inf' of a type (line 408)
        inf_53781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 27), np_53780, 'inf')
        # Assigning a type to the variable 'error_m_norm' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'error_m_norm', inf_53781)
        # SSA join for if statement (line 404)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'order' (line 410)
        order_53782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 11), 'order')
        # Getting the type of 'MAX_ORDER' (line 410)
        MAX_ORDER_53783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 19), 'MAX_ORDER')
        # Applying the binary operator '<' (line 410)
        result_lt_53784 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 11), '<', order_53782, MAX_ORDER_53783)
        
        # Testing the type of an if condition (line 410)
        if_condition_53785 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 410, 8), result_lt_53784)
        # Assigning a type to the variable 'if_condition_53785' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'if_condition_53785', if_condition_53785)
        # SSA begins for if statement (line 410)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 411):
        
        # Assigning a BinOp to a Name (line 411):
        
        # Obtaining the type of the subscript
        # Getting the type of 'order' (line 411)
        order_53786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 34), 'order')
        int_53787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 42), 'int')
        # Applying the binary operator '+' (line 411)
        result_add_53788 = python_operator(stypy.reporting.localization.Localization(__file__, 411, 34), '+', order_53786, int_53787)
        
        # Getting the type of 'error_const' (line 411)
        error_const_53789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 22), 'error_const')
        # Obtaining the member '__getitem__' of a type (line 411)
        getitem___53790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 22), error_const_53789, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 411)
        subscript_call_result_53791 = invoke(stypy.reporting.localization.Localization(__file__, 411, 22), getitem___53790, result_add_53788)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'order' (line 411)
        order_53792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 49), 'order')
        int_53793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 57), 'int')
        # Applying the binary operator '+' (line 411)
        result_add_53794 = python_operator(stypy.reporting.localization.Localization(__file__, 411, 49), '+', order_53792, int_53793)
        
        # Getting the type of 'D' (line 411)
        D_53795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 47), 'D')
        # Obtaining the member '__getitem__' of a type (line 411)
        getitem___53796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 47), D_53795, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 411)
        subscript_call_result_53797 = invoke(stypy.reporting.localization.Localization(__file__, 411, 47), getitem___53796, result_add_53794)
        
        # Applying the binary operator '*' (line 411)
        result_mul_53798 = python_operator(stypy.reporting.localization.Localization(__file__, 411, 22), '*', subscript_call_result_53791, subscript_call_result_53797)
        
        # Assigning a type to the variable 'error_p' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 12), 'error_p', result_mul_53798)
        
        # Assigning a Call to a Name (line 412):
        
        # Assigning a Call to a Name (line 412):
        
        # Call to norm(...): (line 412)
        # Processing the call arguments (line 412)
        # Getting the type of 'error_p' (line 412)
        error_p_53800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 32), 'error_p', False)
        # Getting the type of 'scale' (line 412)
        scale_53801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 42), 'scale', False)
        # Applying the binary operator 'div' (line 412)
        result_div_53802 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 32), 'div', error_p_53800, scale_53801)
        
        # Processing the call keyword arguments (line 412)
        kwargs_53803 = {}
        # Getting the type of 'norm' (line 412)
        norm_53799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 27), 'norm', False)
        # Calling norm(args, kwargs) (line 412)
        norm_call_result_53804 = invoke(stypy.reporting.localization.Localization(__file__, 412, 27), norm_53799, *[result_div_53802], **kwargs_53803)
        
        # Assigning a type to the variable 'error_p_norm' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 12), 'error_p_norm', norm_call_result_53804)
        # SSA branch for the else part of an if statement (line 410)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 414):
        
        # Assigning a Attribute to a Name (line 414):
        # Getting the type of 'np' (line 414)
        np_53805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 27), 'np')
        # Obtaining the member 'inf' of a type (line 414)
        inf_53806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 27), np_53805, 'inf')
        # Assigning a type to the variable 'error_p_norm' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'error_p_norm', inf_53806)
        # SSA join for if statement (line 410)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 416):
        
        # Assigning a Call to a Name (line 416):
        
        # Call to array(...): (line 416)
        # Processing the call arguments (line 416)
        
        # Obtaining an instance of the builtin type 'list' (line 416)
        list_53809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 416)
        # Adding element type (line 416)
        # Getting the type of 'error_m_norm' (line 416)
        error_m_norm_53810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 32), 'error_m_norm', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 31), list_53809, error_m_norm_53810)
        # Adding element type (line 416)
        # Getting the type of 'error_norm' (line 416)
        error_norm_53811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 46), 'error_norm', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 31), list_53809, error_norm_53811)
        # Adding element type (line 416)
        # Getting the type of 'error_p_norm' (line 416)
        error_p_norm_53812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 58), 'error_p_norm', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 31), list_53809, error_p_norm_53812)
        
        # Processing the call keyword arguments (line 416)
        kwargs_53813 = {}
        # Getting the type of 'np' (line 416)
        np_53807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 22), 'np', False)
        # Obtaining the member 'array' of a type (line 416)
        array_53808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 22), np_53807, 'array')
        # Calling array(args, kwargs) (line 416)
        array_call_result_53814 = invoke(stypy.reporting.localization.Localization(__file__, 416, 22), array_53808, *[list_53809], **kwargs_53813)
        
        # Assigning a type to the variable 'error_norms' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'error_norms', array_call_result_53814)
        
        # Assigning a BinOp to a Name (line 417):
        
        # Assigning a BinOp to a Name (line 417):
        # Getting the type of 'error_norms' (line 417)
        error_norms_53815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 18), 'error_norms')
        int_53816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 34), 'int')
        
        # Call to arange(...): (line 417)
        # Processing the call arguments (line 417)
        # Getting the type of 'order' (line 417)
        order_53819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 49), 'order', False)
        # Getting the type of 'order' (line 417)
        order_53820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 56), 'order', False)
        int_53821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 64), 'int')
        # Applying the binary operator '+' (line 417)
        result_add_53822 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 56), '+', order_53820, int_53821)
        
        # Processing the call keyword arguments (line 417)
        kwargs_53823 = {}
        # Getting the type of 'np' (line 417)
        np_53817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 39), 'np', False)
        # Obtaining the member 'arange' of a type (line 417)
        arange_53818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 39), np_53817, 'arange')
        # Calling arange(args, kwargs) (line 417)
        arange_call_result_53824 = invoke(stypy.reporting.localization.Localization(__file__, 417, 39), arange_53818, *[order_53819, result_add_53822], **kwargs_53823)
        
        # Applying the binary operator 'div' (line 417)
        result_div_53825 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 34), 'div', int_53816, arange_call_result_53824)
        
        # Applying the binary operator '**' (line 417)
        result_pow_53826 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 18), '**', error_norms_53815, result_div_53825)
        
        # Assigning a type to the variable 'factors' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'factors', result_pow_53826)
        
        # Assigning a BinOp to a Name (line 419):
        
        # Assigning a BinOp to a Name (line 419):
        
        # Call to argmax(...): (line 419)
        # Processing the call arguments (line 419)
        # Getting the type of 'factors' (line 419)
        factors_53829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 32), 'factors', False)
        # Processing the call keyword arguments (line 419)
        kwargs_53830 = {}
        # Getting the type of 'np' (line 419)
        np_53827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 22), 'np', False)
        # Obtaining the member 'argmax' of a type (line 419)
        argmax_53828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 22), np_53827, 'argmax')
        # Calling argmax(args, kwargs) (line 419)
        argmax_call_result_53831 = invoke(stypy.reporting.localization.Localization(__file__, 419, 22), argmax_53828, *[factors_53829], **kwargs_53830)
        
        int_53832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 43), 'int')
        # Applying the binary operator '-' (line 419)
        result_sub_53833 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 22), '-', argmax_call_result_53831, int_53832)
        
        # Assigning a type to the variable 'delta_order' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'delta_order', result_sub_53833)
        
        # Getting the type of 'order' (line 420)
        order_53834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'order')
        # Getting the type of 'delta_order' (line 420)
        delta_order_53835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 17), 'delta_order')
        # Applying the binary operator '+=' (line 420)
        result_iadd_53836 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 8), '+=', order_53834, delta_order_53835)
        # Assigning a type to the variable 'order' (line 420)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'order', result_iadd_53836)
        
        
        # Assigning a Name to a Attribute (line 421):
        
        # Assigning a Name to a Attribute (line 421):
        # Getting the type of 'order' (line 421)
        order_53837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 21), 'order')
        # Getting the type of 'self' (line 421)
        self_53838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'self')
        # Setting the type of the member 'order' of a type (line 421)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 8), self_53838, 'order', order_53837)
        
        # Assigning a Call to a Name (line 423):
        
        # Assigning a Call to a Name (line 423):
        
        # Call to min(...): (line 423)
        # Processing the call arguments (line 423)
        # Getting the type of 'MAX_FACTOR' (line 423)
        MAX_FACTOR_53840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 21), 'MAX_FACTOR', False)
        # Getting the type of 'safety' (line 423)
        safety_53841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 33), 'safety', False)
        
        # Call to max(...): (line 423)
        # Processing the call arguments (line 423)
        # Getting the type of 'factors' (line 423)
        factors_53844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 49), 'factors', False)
        # Processing the call keyword arguments (line 423)
        kwargs_53845 = {}
        # Getting the type of 'np' (line 423)
        np_53842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 42), 'np', False)
        # Obtaining the member 'max' of a type (line 423)
        max_53843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 42), np_53842, 'max')
        # Calling max(args, kwargs) (line 423)
        max_call_result_53846 = invoke(stypy.reporting.localization.Localization(__file__, 423, 42), max_53843, *[factors_53844], **kwargs_53845)
        
        # Applying the binary operator '*' (line 423)
        result_mul_53847 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 33), '*', safety_53841, max_call_result_53846)
        
        # Processing the call keyword arguments (line 423)
        kwargs_53848 = {}
        # Getting the type of 'min' (line 423)
        min_53839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 17), 'min', False)
        # Calling min(args, kwargs) (line 423)
        min_call_result_53849 = invoke(stypy.reporting.localization.Localization(__file__, 423, 17), min_53839, *[MAX_FACTOR_53840, result_mul_53847], **kwargs_53848)
        
        # Assigning a type to the variable 'factor' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'factor', min_call_result_53849)
        
        # Getting the type of 'self' (line 424)
        self_53850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'self')
        # Obtaining the member 'h_abs' of a type (line 424)
        h_abs_53851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 8), self_53850, 'h_abs')
        # Getting the type of 'factor' (line 424)
        factor_53852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 22), 'factor')
        # Applying the binary operator '*=' (line 424)
        result_imul_53853 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 8), '*=', h_abs_53851, factor_53852)
        # Getting the type of 'self' (line 424)
        self_53854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'self')
        # Setting the type of the member 'h_abs' of a type (line 424)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 8), self_53854, 'h_abs', result_imul_53853)
        
        
        # Call to change_D(...): (line 425)
        # Processing the call arguments (line 425)
        # Getting the type of 'D' (line 425)
        D_53856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 17), 'D', False)
        # Getting the type of 'order' (line 425)
        order_53857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 20), 'order', False)
        # Getting the type of 'factor' (line 425)
        factor_53858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 27), 'factor', False)
        # Processing the call keyword arguments (line 425)
        kwargs_53859 = {}
        # Getting the type of 'change_D' (line 425)
        change_D_53855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'change_D', False)
        # Calling change_D(args, kwargs) (line 425)
        change_D_call_result_53860 = invoke(stypy.reporting.localization.Localization(__file__, 425, 8), change_D_53855, *[D_53856, order_53857, factor_53858], **kwargs_53859)
        
        
        # Assigning a Num to a Attribute (line 426):
        
        # Assigning a Num to a Attribute (line 426):
        int_53861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 29), 'int')
        # Getting the type of 'self' (line 426)
        self_53862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'self')
        # Setting the type of the member 'n_equal_steps' of a type (line 426)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 8), self_53862, 'n_equal_steps', int_53861)
        
        # Assigning a Name to a Attribute (line 427):
        
        # Assigning a Name to a Attribute (line 427):
        # Getting the type of 'None' (line 427)
        None_53863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 18), 'None')
        # Getting the type of 'self' (line 427)
        self_53864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'self')
        # Setting the type of the member 'LU' of a type (line 427)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 8), self_53864, 'LU', None_53863)
        
        # Obtaining an instance of the builtin type 'tuple' (line 429)
        tuple_53865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 429)
        # Adding element type (line 429)
        # Getting the type of 'True' (line 429)
        True_53866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 15), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 15), tuple_53865, True_53866)
        # Adding element type (line 429)
        # Getting the type of 'None' (line 429)
        None_53867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 21), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 15), tuple_53865, None_53867)
        
        # Assigning a type to the variable 'stypy_return_type' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'stypy_return_type', tuple_53865)
        
        # ################# End of '_step_impl(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_step_impl' in the type store
        # Getting the type of 'stypy_return_type' (line 289)
        stypy_return_type_53868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_53868)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_step_impl'
        return stypy_return_type_53868


    @norecursion
    def _dense_output_impl(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_dense_output_impl'
        module_type_store = module_type_store.open_function_context('_dense_output_impl', 431, 4, False)
        # Assigning a type to the variable 'self' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BDF._dense_output_impl.__dict__.__setitem__('stypy_localization', localization)
        BDF._dense_output_impl.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BDF._dense_output_impl.__dict__.__setitem__('stypy_type_store', module_type_store)
        BDF._dense_output_impl.__dict__.__setitem__('stypy_function_name', 'BDF._dense_output_impl')
        BDF._dense_output_impl.__dict__.__setitem__('stypy_param_names_list', [])
        BDF._dense_output_impl.__dict__.__setitem__('stypy_varargs_param_name', None)
        BDF._dense_output_impl.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BDF._dense_output_impl.__dict__.__setitem__('stypy_call_defaults', defaults)
        BDF._dense_output_impl.__dict__.__setitem__('stypy_call_varargs', varargs)
        BDF._dense_output_impl.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BDF._dense_output_impl.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BDF._dense_output_impl', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_dense_output_impl', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_dense_output_impl(...)' code ##################

        
        # Call to BdfDenseOutput(...): (line 432)
        # Processing the call arguments (line 432)
        # Getting the type of 'self' (line 432)
        self_53870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 30), 'self', False)
        # Obtaining the member 't_old' of a type (line 432)
        t_old_53871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 30), self_53870, 't_old')
        # Getting the type of 'self' (line 432)
        self_53872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 42), 'self', False)
        # Obtaining the member 't' of a type (line 432)
        t_53873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 42), self_53872, 't')
        # Getting the type of 'self' (line 432)
        self_53874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 50), 'self', False)
        # Obtaining the member 'h_abs' of a type (line 432)
        h_abs_53875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 50), self_53874, 'h_abs')
        # Getting the type of 'self' (line 432)
        self_53876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 63), 'self', False)
        # Obtaining the member 'direction' of a type (line 432)
        direction_53877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 63), self_53876, 'direction')
        # Applying the binary operator '*' (line 432)
        result_mul_53878 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 50), '*', h_abs_53875, direction_53877)
        
        # Getting the type of 'self' (line 433)
        self_53879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 30), 'self', False)
        # Obtaining the member 'order' of a type (line 433)
        order_53880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 30), self_53879, 'order')
        
        # Call to copy(...): (line 433)
        # Processing the call keyword arguments (line 433)
        kwargs_53891 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 433)
        self_53881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 50), 'self', False)
        # Obtaining the member 'order' of a type (line 433)
        order_53882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 50), self_53881, 'order')
        int_53883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 63), 'int')
        # Applying the binary operator '+' (line 433)
        result_add_53884 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 50), '+', order_53882, int_53883)
        
        slice_53885 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 433, 42), None, result_add_53884, None)
        # Getting the type of 'self' (line 433)
        self_53886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 42), 'self', False)
        # Obtaining the member 'D' of a type (line 433)
        D_53887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 42), self_53886, 'D')
        # Obtaining the member '__getitem__' of a type (line 433)
        getitem___53888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 42), D_53887, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 433)
        subscript_call_result_53889 = invoke(stypy.reporting.localization.Localization(__file__, 433, 42), getitem___53888, slice_53885)
        
        # Obtaining the member 'copy' of a type (line 433)
        copy_53890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 42), subscript_call_result_53889, 'copy')
        # Calling copy(args, kwargs) (line 433)
        copy_call_result_53892 = invoke(stypy.reporting.localization.Localization(__file__, 433, 42), copy_53890, *[], **kwargs_53891)
        
        # Processing the call keyword arguments (line 432)
        kwargs_53893 = {}
        # Getting the type of 'BdfDenseOutput' (line 432)
        BdfDenseOutput_53869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 15), 'BdfDenseOutput', False)
        # Calling BdfDenseOutput(args, kwargs) (line 432)
        BdfDenseOutput_call_result_53894 = invoke(stypy.reporting.localization.Localization(__file__, 432, 15), BdfDenseOutput_53869, *[t_old_53871, t_53873, result_mul_53878, order_53880, copy_call_result_53892], **kwargs_53893)
        
        # Assigning a type to the variable 'stypy_return_type' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'stypy_return_type', BdfDenseOutput_call_result_53894)
        
        # ################# End of '_dense_output_impl(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_dense_output_impl' in the type store
        # Getting the type of 'stypy_return_type' (line 431)
        stypy_return_type_53895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_53895)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_dense_output_impl'
        return stypy_return_type_53895


# Assigning a type to the variable 'BDF' (line 72)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'BDF', BDF)
# Declaration of the 'BdfDenseOutput' class
# Getting the type of 'DenseOutput' (line 436)
DenseOutput_53896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 21), 'DenseOutput')

class BdfDenseOutput(DenseOutput_53896, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 437, 4, False)
        # Assigning a type to the variable 'self' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BdfDenseOutput.__init__', ['t_old', 't', 'h', 'order', 'D'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['t_old', 't', 'h', 'order', 'D'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 438)
        # Processing the call arguments (line 438)
        # Getting the type of 't_old' (line 438)
        t_old_53903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 45), 't_old', False)
        # Getting the type of 't' (line 438)
        t_53904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 52), 't', False)
        # Processing the call keyword arguments (line 438)
        kwargs_53905 = {}
        
        # Call to super(...): (line 438)
        # Processing the call arguments (line 438)
        # Getting the type of 'BdfDenseOutput' (line 438)
        BdfDenseOutput_53898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 14), 'BdfDenseOutput', False)
        # Getting the type of 'self' (line 438)
        self_53899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 30), 'self', False)
        # Processing the call keyword arguments (line 438)
        kwargs_53900 = {}
        # Getting the type of 'super' (line 438)
        super_53897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'super', False)
        # Calling super(args, kwargs) (line 438)
        super_call_result_53901 = invoke(stypy.reporting.localization.Localization(__file__, 438, 8), super_53897, *[BdfDenseOutput_53898, self_53899], **kwargs_53900)
        
        # Obtaining the member '__init__' of a type (line 438)
        init___53902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 8), super_call_result_53901, '__init__')
        # Calling __init__(args, kwargs) (line 438)
        init___call_result_53906 = invoke(stypy.reporting.localization.Localization(__file__, 438, 8), init___53902, *[t_old_53903, t_53904], **kwargs_53905)
        
        
        # Assigning a Name to a Attribute (line 439):
        
        # Assigning a Name to a Attribute (line 439):
        # Getting the type of 'order' (line 439)
        order_53907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 21), 'order')
        # Getting the type of 'self' (line 439)
        self_53908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'self')
        # Setting the type of the member 'order' of a type (line 439)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 8), self_53908, 'order', order_53907)
        
        # Assigning a BinOp to a Attribute (line 440):
        
        # Assigning a BinOp to a Attribute (line 440):
        # Getting the type of 'self' (line 440)
        self_53909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 23), 'self')
        # Obtaining the member 't' of a type (line 440)
        t_53910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 23), self_53909, 't')
        # Getting the type of 'h' (line 440)
        h_53911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 32), 'h')
        
        # Call to arange(...): (line 440)
        # Processing the call arguments (line 440)
        # Getting the type of 'self' (line 440)
        self_53914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 46), 'self', False)
        # Obtaining the member 'order' of a type (line 440)
        order_53915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 46), self_53914, 'order')
        # Processing the call keyword arguments (line 440)
        kwargs_53916 = {}
        # Getting the type of 'np' (line 440)
        np_53912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 36), 'np', False)
        # Obtaining the member 'arange' of a type (line 440)
        arange_53913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 36), np_53912, 'arange')
        # Calling arange(args, kwargs) (line 440)
        arange_call_result_53917 = invoke(stypy.reporting.localization.Localization(__file__, 440, 36), arange_53913, *[order_53915], **kwargs_53916)
        
        # Applying the binary operator '*' (line 440)
        result_mul_53918 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 32), '*', h_53911, arange_call_result_53917)
        
        # Applying the binary operator '-' (line 440)
        result_sub_53919 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 23), '-', t_53910, result_mul_53918)
        
        # Getting the type of 'self' (line 440)
        self_53920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'self')
        # Setting the type of the member 't_shift' of a type (line 440)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 8), self_53920, 't_shift', result_sub_53919)
        
        # Assigning a BinOp to a Attribute (line 441):
        
        # Assigning a BinOp to a Attribute (line 441):
        # Getting the type of 'h' (line 441)
        h_53921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 21), 'h')
        int_53922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 26), 'int')
        
        # Call to arange(...): (line 441)
        # Processing the call arguments (line 441)
        # Getting the type of 'self' (line 441)
        self_53925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 40), 'self', False)
        # Obtaining the member 'order' of a type (line 441)
        order_53926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 40), self_53925, 'order')
        # Processing the call keyword arguments (line 441)
        kwargs_53927 = {}
        # Getting the type of 'np' (line 441)
        np_53923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 30), 'np', False)
        # Obtaining the member 'arange' of a type (line 441)
        arange_53924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 30), np_53923, 'arange')
        # Calling arange(args, kwargs) (line 441)
        arange_call_result_53928 = invoke(stypy.reporting.localization.Localization(__file__, 441, 30), arange_53924, *[order_53926], **kwargs_53927)
        
        # Applying the binary operator '+' (line 441)
        result_add_53929 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 26), '+', int_53922, arange_call_result_53928)
        
        # Applying the binary operator '*' (line 441)
        result_mul_53930 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 21), '*', h_53921, result_add_53929)
        
        # Getting the type of 'self' (line 441)
        self_53931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'self')
        # Setting the type of the member 'denom' of a type (line 441)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 8), self_53931, 'denom', result_mul_53930)
        
        # Assigning a Name to a Attribute (line 442):
        
        # Assigning a Name to a Attribute (line 442):
        # Getting the type of 'D' (line 442)
        D_53932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 17), 'D')
        # Getting the type of 'self' (line 442)
        self_53933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'self')
        # Setting the type of the member 'D' of a type (line 442)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 8), self_53933, 'D', D_53932)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _call_impl(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_call_impl'
        module_type_store = module_type_store.open_function_context('_call_impl', 444, 4, False)
        # Assigning a type to the variable 'self' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BdfDenseOutput._call_impl.__dict__.__setitem__('stypy_localization', localization)
        BdfDenseOutput._call_impl.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BdfDenseOutput._call_impl.__dict__.__setitem__('stypy_type_store', module_type_store)
        BdfDenseOutput._call_impl.__dict__.__setitem__('stypy_function_name', 'BdfDenseOutput._call_impl')
        BdfDenseOutput._call_impl.__dict__.__setitem__('stypy_param_names_list', ['t'])
        BdfDenseOutput._call_impl.__dict__.__setitem__('stypy_varargs_param_name', None)
        BdfDenseOutput._call_impl.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BdfDenseOutput._call_impl.__dict__.__setitem__('stypy_call_defaults', defaults)
        BdfDenseOutput._call_impl.__dict__.__setitem__('stypy_call_varargs', varargs)
        BdfDenseOutput._call_impl.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BdfDenseOutput._call_impl.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BdfDenseOutput._call_impl', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_call_impl', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_call_impl(...)' code ##################

        
        
        # Getting the type of 't' (line 445)
        t_53934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 11), 't')
        # Obtaining the member 'ndim' of a type (line 445)
        ndim_53935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 11), t_53934, 'ndim')
        int_53936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 21), 'int')
        # Applying the binary operator '==' (line 445)
        result_eq_53937 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 11), '==', ndim_53935, int_53936)
        
        # Testing the type of an if condition (line 445)
        if_condition_53938 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 445, 8), result_eq_53937)
        # Assigning a type to the variable 'if_condition_53938' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'if_condition_53938', if_condition_53938)
        # SSA begins for if statement (line 445)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 446):
        
        # Assigning a BinOp to a Name (line 446):
        # Getting the type of 't' (line 446)
        t_53939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 17), 't')
        # Getting the type of 'self' (line 446)
        self_53940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 21), 'self')
        # Obtaining the member 't_shift' of a type (line 446)
        t_shift_53941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 21), self_53940, 't_shift')
        # Applying the binary operator '-' (line 446)
        result_sub_53942 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 17), '-', t_53939, t_shift_53941)
        
        # Getting the type of 'self' (line 446)
        self_53943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 37), 'self')
        # Obtaining the member 'denom' of a type (line 446)
        denom_53944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 37), self_53943, 'denom')
        # Applying the binary operator 'div' (line 446)
        result_div_53945 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 16), 'div', result_sub_53942, denom_53944)
        
        # Assigning a type to the variable 'x' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 12), 'x', result_div_53945)
        
        # Assigning a Call to a Name (line 447):
        
        # Assigning a Call to a Name (line 447):
        
        # Call to cumprod(...): (line 447)
        # Processing the call arguments (line 447)
        # Getting the type of 'x' (line 447)
        x_53948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 27), 'x', False)
        # Processing the call keyword arguments (line 447)
        kwargs_53949 = {}
        # Getting the type of 'np' (line 447)
        np_53946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 16), 'np', False)
        # Obtaining the member 'cumprod' of a type (line 447)
        cumprod_53947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 16), np_53946, 'cumprod')
        # Calling cumprod(args, kwargs) (line 447)
        cumprod_call_result_53950 = invoke(stypy.reporting.localization.Localization(__file__, 447, 16), cumprod_53947, *[x_53948], **kwargs_53949)
        
        # Assigning a type to the variable 'p' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'p', cumprod_call_result_53950)
        # SSA branch for the else part of an if statement (line 445)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 449):
        
        # Assigning a BinOp to a Name (line 449):
        # Getting the type of 't' (line 449)
        t_53951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 17), 't')
        
        # Obtaining the type of the subscript
        slice_53952 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 449, 21), None, None, None)
        # Getting the type of 'None' (line 449)
        None_53953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 37), 'None')
        # Getting the type of 'self' (line 449)
        self_53954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 21), 'self')
        # Obtaining the member 't_shift' of a type (line 449)
        t_shift_53955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 21), self_53954, 't_shift')
        # Obtaining the member '__getitem__' of a type (line 449)
        getitem___53956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 21), t_shift_53955, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 449)
        subscript_call_result_53957 = invoke(stypy.reporting.localization.Localization(__file__, 449, 21), getitem___53956, (slice_53952, None_53953))
        
        # Applying the binary operator '-' (line 449)
        result_sub_53958 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 17), '-', t_53951, subscript_call_result_53957)
        
        
        # Obtaining the type of the subscript
        slice_53959 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 449, 46), None, None, None)
        # Getting the type of 'None' (line 449)
        None_53960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 60), 'None')
        # Getting the type of 'self' (line 449)
        self_53961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 46), 'self')
        # Obtaining the member 'denom' of a type (line 449)
        denom_53962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 46), self_53961, 'denom')
        # Obtaining the member '__getitem__' of a type (line 449)
        getitem___53963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 46), denom_53962, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 449)
        subscript_call_result_53964 = invoke(stypy.reporting.localization.Localization(__file__, 449, 46), getitem___53963, (slice_53959, None_53960))
        
        # Applying the binary operator 'div' (line 449)
        result_div_53965 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 16), 'div', result_sub_53958, subscript_call_result_53964)
        
        # Assigning a type to the variable 'x' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 'x', result_div_53965)
        
        # Assigning a Call to a Name (line 450):
        
        # Assigning a Call to a Name (line 450):
        
        # Call to cumprod(...): (line 450)
        # Processing the call arguments (line 450)
        # Getting the type of 'x' (line 450)
        x_53968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 27), 'x', False)
        # Processing the call keyword arguments (line 450)
        int_53969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 35), 'int')
        keyword_53970 = int_53969
        kwargs_53971 = {'axis': keyword_53970}
        # Getting the type of 'np' (line 450)
        np_53966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 16), 'np', False)
        # Obtaining the member 'cumprod' of a type (line 450)
        cumprod_53967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 16), np_53966, 'cumprod')
        # Calling cumprod(args, kwargs) (line 450)
        cumprod_call_result_53972 = invoke(stypy.reporting.localization.Localization(__file__, 450, 16), cumprod_53967, *[x_53968], **kwargs_53971)
        
        # Assigning a type to the variable 'p' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 12), 'p', cumprod_call_result_53972)
        # SSA join for if statement (line 445)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 452):
        
        # Assigning a Call to a Name (line 452):
        
        # Call to dot(...): (line 452)
        # Processing the call arguments (line 452)
        
        # Obtaining the type of the subscript
        int_53975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 26), 'int')
        slice_53976 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 452, 19), int_53975, None, None)
        # Getting the type of 'self' (line 452)
        self_53977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 19), 'self', False)
        # Obtaining the member 'D' of a type (line 452)
        D_53978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 19), self_53977, 'D')
        # Obtaining the member '__getitem__' of a type (line 452)
        getitem___53979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 19), D_53978, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 452)
        subscript_call_result_53980 = invoke(stypy.reporting.localization.Localization(__file__, 452, 19), getitem___53979, slice_53976)
        
        # Obtaining the member 'T' of a type (line 452)
        T_53981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 19), subscript_call_result_53980, 'T')
        # Getting the type of 'p' (line 452)
        p_53982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 33), 'p', False)
        # Processing the call keyword arguments (line 452)
        kwargs_53983 = {}
        # Getting the type of 'np' (line 452)
        np_53973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 12), 'np', False)
        # Obtaining the member 'dot' of a type (line 452)
        dot_53974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 12), np_53973, 'dot')
        # Calling dot(args, kwargs) (line 452)
        dot_call_result_53984 = invoke(stypy.reporting.localization.Localization(__file__, 452, 12), dot_53974, *[T_53981, p_53982], **kwargs_53983)
        
        # Assigning a type to the variable 'y' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'y', dot_call_result_53984)
        
        
        # Getting the type of 'y' (line 453)
        y_53985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 11), 'y')
        # Obtaining the member 'ndim' of a type (line 453)
        ndim_53986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 11), y_53985, 'ndim')
        int_53987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 21), 'int')
        # Applying the binary operator '==' (line 453)
        result_eq_53988 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 11), '==', ndim_53986, int_53987)
        
        # Testing the type of an if condition (line 453)
        if_condition_53989 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 453, 8), result_eq_53988)
        # Assigning a type to the variable 'if_condition_53989' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'if_condition_53989', if_condition_53989)
        # SSA begins for if statement (line 453)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'y' (line 454)
        y_53990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 12), 'y')
        
        # Obtaining the type of the subscript
        int_53991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 24), 'int')
        # Getting the type of 'self' (line 454)
        self_53992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 17), 'self')
        # Obtaining the member 'D' of a type (line 454)
        D_53993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 17), self_53992, 'D')
        # Obtaining the member '__getitem__' of a type (line 454)
        getitem___53994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 17), D_53993, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 454)
        subscript_call_result_53995 = invoke(stypy.reporting.localization.Localization(__file__, 454, 17), getitem___53994, int_53991)
        
        # Applying the binary operator '+=' (line 454)
        result_iadd_53996 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 12), '+=', y_53990, subscript_call_result_53995)
        # Assigning a type to the variable 'y' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 12), 'y', result_iadd_53996)
        
        # SSA branch for the else part of an if statement (line 453)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'y' (line 456)
        y_53997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'y')
        
        # Obtaining the type of the subscript
        int_53998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 24), 'int')
        slice_53999 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 456, 17), None, None, None)
        # Getting the type of 'None' (line 456)
        None_54000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 30), 'None')
        # Getting the type of 'self' (line 456)
        self_54001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 17), 'self')
        # Obtaining the member 'D' of a type (line 456)
        D_54002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 17), self_54001, 'D')
        # Obtaining the member '__getitem__' of a type (line 456)
        getitem___54003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 17), D_54002, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 456)
        subscript_call_result_54004 = invoke(stypy.reporting.localization.Localization(__file__, 456, 17), getitem___54003, (int_53998, slice_53999, None_54000))
        
        # Applying the binary operator '+=' (line 456)
        result_iadd_54005 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 12), '+=', y_53997, subscript_call_result_54004)
        # Assigning a type to the variable 'y' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'y', result_iadd_54005)
        
        # SSA join for if statement (line 453)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'y' (line 458)
        y_54006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 15), 'y')
        # Assigning a type to the variable 'stypy_return_type' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'stypy_return_type', y_54006)
        
        # ################# End of '_call_impl(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_call_impl' in the type store
        # Getting the type of 'stypy_return_type' (line 444)
        stypy_return_type_54007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_54007)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_call_impl'
        return stypy_return_type_54007


# Assigning a type to the variable 'BdfDenseOutput' (line 436)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 0), 'BdfDenseOutput', BdfDenseOutput)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
