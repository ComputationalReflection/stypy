
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: import numpy as np
3: from scipy.linalg import lu_factor, lu_solve
4: from scipy.sparse import csc_matrix, issparse, eye
5: from scipy.sparse.linalg import splu
6: from scipy.optimize._numdiff import group_columns
7: from .common import (validate_max_step, validate_tol, select_initial_step,
8:                      norm, num_jac, EPS, warn_extraneous)
9: from .base import OdeSolver, DenseOutput
10: 
11: S6 = 6 ** 0.5
12: 
13: # Butcher tableau. A is not used directly, see below.
14: C = np.array([(4 - S6) / 10, (4 + S6) / 10, 1])
15: E = np.array([-13 - 7 * S6, -13 + 7 * S6, -1]) / 3
16: 
17: # Eigendecomposition of A is done: A = T L T**-1. There is 1 real eigenvalue
18: # and a complex conjugate pair. They are written below.
19: MU_REAL = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)
20: MU_COMPLEX = (3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
21:               - 0.5j * (3 ** (5 / 6) + 3 ** (7 / 6)))
22: 
23: # These are transformation matrices.
24: T = np.array([
25:     [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
26:     [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
27:     [1, 1, 0]])
28: TI = np.array([
29:     [4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
30:     [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
31:     [0.50287263494578682, -2.57192694985560522, 0.59603920482822492]])
32: # These linear combinations are used in the algorithm.
33: TI_REAL = TI[0]
34: TI_COMPLEX = TI[1] + 1j * TI[2]
35: 
36: # Interpolator coefficients.
37: P = np.array([
38:     [13/3 + 7*S6/3, -23/3 - 22*S6/3, 10/3 + 5 * S6],
39:     [13/3 - 7*S6/3, -23/3 + 22*S6/3, 10/3 - 5 * S6],
40:     [1/3, -8/3, 10/3]])
41: 
42: 
43: NEWTON_MAXITER = 6  # Maximum number of Newton iterations.
44: MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
45: MAX_FACTOR = 10  # Maximum allowed increase in a step size.
46: 
47: 
48: def solve_collocation_system(fun, t, y, h, Z0, scale, tol,
49:                              LU_real, LU_complex, solve_lu):
50:     '''Solve the collocation system.
51: 
52:     Parameters
53:     ----------
54:     fun : callable
55:         Right-hand side of the system.
56:     t : float
57:         Current time.
58:     y : ndarray, shape (n,)
59:         Current state.
60:     h : float
61:         Step to try.
62:     Z0 : ndarray, shape (3, n)
63:         Initial guess for the solution. It determines new values of `y` at
64:         ``t + h * C`` as ``y + Z0``, where ``C`` is the Radau method constants.
65:     scale : float
66:         Problem tolerance scale, i.e. ``rtol * abs(y) + atol``.
67:     tol : float
68:         Tolerance to which solve the system. This value is compared with
69:         the normalized by `scale` error.
70:     LU_real, LU_complex
71:         LU decompositions of the system Jacobians.
72:     solve_lu : callable
73:         Callable which solves a linear system given a LU decomposition. The
74:         signature is ``solve_lu(LU, b)``.
75: 
76:     Returns
77:     -------
78:     converged : bool
79:         Whether iterations converged.
80:     n_iter : int
81:         Number of completed iterations.
82:     Z : ndarray, shape (3, n)
83:         Found solution.
84:     rate : float
85:         The rate of convergence.
86:     '''
87:     n = y.shape[0]
88:     M_real = MU_REAL / h
89:     M_complex = MU_COMPLEX / h
90: 
91:     W = TI.dot(Z0)
92:     Z = Z0
93: 
94:     F = np.empty((3, n))
95:     ch = h * C
96: 
97:     dW_norm_old = None
98:     dW = np.empty_like(W)
99:     converged = False
100:     for k in range(NEWTON_MAXITER):
101:         for i in range(3):
102:             F[i] = fun(t + ch[i], y + Z[i])
103: 
104:         if not np.all(np.isfinite(F)):
105:             break
106: 
107:         f_real = F.T.dot(TI_REAL) - M_real * W[0]
108:         f_complex = F.T.dot(TI_COMPLEX) - M_complex * (W[1] + 1j * W[2])
109: 
110:         dW_real = solve_lu(LU_real, f_real)
111:         dW_complex = solve_lu(LU_complex, f_complex)
112: 
113:         dW[0] = dW_real
114:         dW[1] = dW_complex.real
115:         dW[2] = dW_complex.imag
116: 
117:         dW_norm = norm(dW / scale)
118:         if dW_norm_old is not None:
119:             rate = dW_norm / dW_norm_old
120:         else:
121:             rate = None
122: 
123:         if (rate is not None and (rate >= 1 or
124:                 rate ** (NEWTON_MAXITER - k) / (1 - rate) * dW_norm > tol)):
125:             break
126: 
127:         W += dW
128:         Z = T.dot(W)
129: 
130:         if (dW_norm == 0 or
131:                 rate is not None and rate / (1 - rate) * dW_norm < tol):
132:             converged = True
133:             break
134: 
135:         dW_norm_old = dW_norm
136: 
137:     return converged, k + 1, Z, rate
138: 
139: 
140: def predict_factor(h_abs, h_abs_old, error_norm, error_norm_old):
141:     '''Predict by which factor to increase/decrease the step size.
142: 
143:     The algorithm is described in [1]_.
144: 
145:     Parameters
146:     ----------
147:     h_abs, h_abs_old : float
148:         Current and previous values of the step size, `h_abs_old` can be None
149:         (see Notes).
150:     error_norm, error_norm_old : float
151:         Current and previous values of the error norm, `error_norm_old` can
152:         be None (see Notes).
153: 
154:     Returns
155:     -------
156:     factor : float
157:         Predicted factor.
158: 
159:     Notes
160:     -----
161:     If `h_abs_old` and `error_norm_old` are both not None then a two-step
162:     algorithm is used, otherwise a one-step algorithm is used.
163: 
164:     References
165:     ----------
166:     .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
167:            Equations II: Stiff and Differential-Algebraic Problems", Sec. IV.8.
168:     '''
169:     if error_norm_old is None or h_abs_old is None or error_norm == 0:
170:         multiplier = 1
171:     else:
172:         multiplier = h_abs / h_abs_old * (error_norm_old / error_norm) ** 0.25
173: 
174:     with np.errstate(divide='ignore'):
175:         factor = min(1, multiplier) * error_norm ** -0.25
176: 
177:     return factor
178: 
179: 
180: class Radau(OdeSolver):
181:     '''Implicit Runge-Kutta method of Radau IIA family of order 5.
182: 
183:     Implementation follows [1]_. The error is controlled for a 3rd order
184:     accurate embedded formula. A cubic polynomial which satisfies the
185:     collocation conditions is used for the dense output.
186: 
187:     Parameters
188:     ----------
189:     fun : callable
190:         Right-hand side of the system. The calling signature is ``fun(t, y)``.
191:         Here ``t`` is a scalar and there are two options for ndarray ``y``.
192:         It can either have shape (n,), then ``fun`` must return array_like with
193:         shape (n,). Or alternatively it can have shape (n, k), then ``fun``
194:         must return array_like with shape (n, k), i.e. each column
195:         corresponds to a single column in ``y``. The choice between the two
196:         options is determined by `vectorized` argument (see below). The
197:         vectorized implementation allows faster approximation of the Jacobian
198:         by finite differences.
199:     t0 : float
200:         Initial time.
201:     y0 : array_like, shape (n,)
202:         Initial state.
203:     t_bound : float
204:         Boundary time --- the integration won't continue beyond it. It also
205:         determines the direction of the integration.
206:     max_step : float, optional
207:         Maximum allowed step size. Default is np.inf, i.e. the step is not
208:         bounded and determined solely by the solver.
209:     rtol, atol : float and array_like, optional
210:         Relative and absolute tolerances. The solver keeps the local error
211:         estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
212:         relative accuracy (number of correct digits). But if a component of `y`
213:         is approximately below `atol` then the error only needs to fall within
214:         the same `atol` threshold, and the number of correct digits is not
215:         guaranteed. If components of y have different scales, it might be
216:         beneficial to set different `atol` values for different components by
217:         passing array_like with shape (n,) for `atol`. Default values are
218:         1e-3 for `rtol` and 1e-6 for `atol`.
219:     jac : {None, array_like, sparse_matrix, callable}, optional
220:         Jacobian matrix of the right-hand side of the system with respect to
221:         y, required only by 'Radau' and 'BDF' methods. The Jacobian matrix
222:         has shape (n, n) and its element (i, j) is equal to ``d f_i / d y_j``.
223:         There are 3 ways to define the Jacobian:
224: 
225:             * If array_like or sparse_matrix, then the Jacobian is assumed to
226:               be constant.
227:             * If callable, then the Jacobian is assumed to depend on both
228:               t and y, and will be called as ``jac(t, y)`` as necessary. The
229:               return value might be a sparse matrix.
230:             * If None (default), then the Jacobian will be approximated by
231:               finite differences.
232: 
233:         It is generally recommended to provide the Jacobian rather than
234:         relying on a finite difference approximation.
235:     jac_sparsity : {None, array_like, sparse matrix}, optional
236:         Defines a sparsity structure of the Jacobian matrix for a finite
237:         difference approximation, its shape must be (n, n). If the Jacobian has
238:         only few non-zero elements in *each* row, providing the sparsity
239:         structure will greatly speed up the computations [2]_. A zero
240:         entry means that a corresponding element in the Jacobian is identically
241:         zero. If None (default), the Jacobian is assumed to be dense.
242:     vectorized : bool, optional
243:         Whether `fun` is implemented in a vectorized fashion. Default is False.
244: 
245:     Attributes
246:     ----------
247:     n : int
248:         Number of equations.
249:     status : string
250:         Current status of the solver: 'running', 'finished' or 'failed'.
251:     t_bound : float
252:         Boundary time.
253:     direction : float
254:         Integration direction: +1 or -1.
255:     t : float
256:         Current time.
257:     y : ndarray
258:         Current state.
259:     t_old : float
260:         Previous time. None if no steps were made yet.
261:     step_size : float
262:         Size of the last successful step. None if no steps were made yet.
263:     nfev : int
264:         Number of the system's rhs evaluations.
265:     njev : int
266:         Number of the Jacobian evaluations.
267:     nlu : int
268:         Number of LU decompositions.
269: 
270:     References
271:     ----------
272:     .. [1] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
273:            Stiff and Differential-Algebraic Problems", Sec. IV.8.
274:     .. [2] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
275:            sparse Jacobian matrices", Journal of the Institute of Mathematics
276:            and its Applications, 13, pp. 117-120, 1974.
277:     '''
278:     def __init__(self, fun, t0, y0, t_bound, max_step=np.inf,
279:                  rtol=1e-3, atol=1e-6, jac=None, jac_sparsity=None,
280:                  vectorized=False, **extraneous):
281:         warn_extraneous(extraneous)
282:         super(Radau, self).__init__(fun, t0, y0, t_bound, vectorized)
283:         self.y_old = None
284:         self.max_step = validate_max_step(max_step)
285:         self.rtol, self.atol = validate_tol(rtol, atol, self.n)
286:         self.f = self.fun(self.t, self.y)
287:         # Select initial step assuming the same order which is used to control
288:         # the error.
289:         self.h_abs = select_initial_step(
290:             self.fun, self.t, self.y, self.f, self.direction,
291:             3, self.rtol, self.atol)
292:         self.h_abs_old = None
293:         self.error_norm_old = None
294: 
295:         self.newton_tol = max(10 * EPS / rtol, min(0.03, rtol ** 0.5))
296:         self.sol = None
297: 
298:         self.jac_factor = None
299:         self.jac, self.J = self._validate_jac(jac, jac_sparsity)
300:         if issparse(self.J):
301:             def lu(A):
302:                 self.nlu += 1
303:                 return splu(A)
304: 
305:             def solve_lu(LU, b):
306:                 return LU.solve(b)
307: 
308:             I = eye(self.n, format='csc')
309:         else:
310:             def lu(A):
311:                 self.nlu += 1
312:                 return lu_factor(A, overwrite_a=True)
313: 
314:             def solve_lu(LU, b):
315:                 return lu_solve(LU, b, overwrite_b=True)
316: 
317:             I = np.identity(self.n)
318: 
319:         self.lu = lu
320:         self.solve_lu = solve_lu
321:         self.I = I
322: 
323:         self.current_jac = True
324:         self.LU_real = None
325:         self.LU_complex = None
326:         self.Z = None
327: 
328:     def _validate_jac(self, jac, sparsity):
329:         t0 = self.t
330:         y0 = self.y
331: 
332:         if jac is None:
333:             if sparsity is not None:
334:                 if issparse(sparsity):
335:                     sparsity = csc_matrix(sparsity)
336:                 groups = group_columns(sparsity)
337:                 sparsity = (sparsity, groups)
338: 
339:             def jac_wrapped(t, y, f):
340:                 self.njev += 1
341:                 J, self.jac_factor = num_jac(self.fun_vectorized, t, y, f,
342:                                              self.atol, self.jac_factor,
343:                                              sparsity)
344:                 return J
345:             J = jac_wrapped(t0, y0, self.f)
346:         elif callable(jac):
347:             J = jac(t0, y0)
348:             self.njev = 1
349:             if issparse(J):
350:                 J = csc_matrix(J)
351: 
352:                 def jac_wrapped(t, y, _=None):
353:                     self.njev += 1
354:                     return csc_matrix(jac(t, y), dtype=float)
355: 
356:             else:
357:                 J = np.asarray(J, dtype=float)
358: 
359:                 def jac_wrapped(t, y, _=None):
360:                     self.njev += 1
361:                     return np.asarray(jac(t, y), dtype=float)
362: 
363:             if J.shape != (self.n, self.n):
364:                 raise ValueError("`jac` is expected to have shape {}, but "
365:                                  "actually has {}."
366:                                  .format((self.n, self.n), J.shape))
367:         else:
368:             if issparse(jac):
369:                 J = csc_matrix(jac)
370:             else:
371:                 J = np.asarray(jac, dtype=float)
372: 
373:             if J.shape != (self.n, self.n):
374:                 raise ValueError("`jac` is expected to have shape {}, but "
375:                                  "actually has {}."
376:                                  .format((self.n, self.n), J.shape))
377:             jac_wrapped = None
378: 
379:         return jac_wrapped, J
380: 
381:     def _step_impl(self):
382:         t = self.t
383:         y = self.y
384:         f = self.f
385: 
386:         max_step = self.max_step
387:         atol = self.atol
388:         rtol = self.rtol
389: 
390:         min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
391:         if self.h_abs > max_step:
392:             h_abs = max_step
393:             h_abs_old = None
394:             error_norm_old = None
395:         elif self.h_abs < min_step:
396:             h_abs = min_step
397:             h_abs_old = None
398:             error_norm_old = None
399:         else:
400:             h_abs = self.h_abs
401:             h_abs_old = self.h_abs_old
402:             error_norm_old = self.error_norm_old
403: 
404:         J = self.J
405:         LU_real = self.LU_real
406:         LU_complex = self.LU_complex
407: 
408:         current_jac = self.current_jac
409:         jac = self.jac
410: 
411:         rejected = False
412:         step_accepted = False
413:         message = None
414:         while not step_accepted:
415:             if h_abs < min_step:
416:                 return False, self.TOO_SMALL_STEP
417: 
418:             h = h_abs * self.direction
419:             t_new = t + h
420: 
421:             if self.direction * (t_new - self.t_bound) > 0:
422:                 t_new = self.t_bound
423: 
424:             h = t_new - t
425:             h_abs = np.abs(h)
426: 
427:             if self.sol is None:
428:                 Z0 = np.zeros((3, y.shape[0]))
429:             else:
430:                 Z0 = self.sol(t + h * C).T - y
431: 
432:             scale = atol + np.abs(y) * rtol
433: 
434:             converged = False
435:             while not converged:
436:                 if LU_real is None or LU_complex is None:
437:                     LU_real = self.lu(MU_REAL / h * self.I - J)
438:                     LU_complex = self.lu(MU_COMPLEX / h * self.I - J)
439: 
440:                 converged, n_iter, Z, rate = solve_collocation_system(
441:                     self.fun, t, y, h, Z0, scale, self.newton_tol,
442:                     LU_real, LU_complex, self.solve_lu)
443: 
444:                 if not converged:
445:                     if current_jac:
446:                         break
447: 
448:                     J = self.jac(t, y, f)
449:                     current_jac = True
450:                     LU_real = None
451:                     LU_complex = None
452: 
453:             if not converged:
454:                 h_abs *= 0.5
455:                 LU_real = None
456:                 LU_complex = None
457:                 continue
458: 
459:             y_new = y + Z[-1]
460:             ZE = Z.T.dot(E) / h
461:             error = self.solve_lu(LU_real, f + ZE)
462:             scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
463:             error_norm = norm(error / scale)
464:             safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER
465:                                                        + n_iter)
466: 
467:             if rejected and error_norm > 1:
468:                 error = self.solve_lu(LU_real, self.fun(t, y + error) + ZE)
469:                 error_norm = norm(error / scale)
470: 
471:             if error_norm > 1:
472:                 factor = predict_factor(h_abs, h_abs_old,
473:                                         error_norm, error_norm_old)
474:                 h_abs *= max(MIN_FACTOR, safety * factor)
475: 
476:                 LU_real = None
477:                 LU_complex = None
478:                 rejected = True
479:             else:
480:                 step_accepted = True
481: 
482:         recompute_jac = jac is not None and n_iter > 2 and rate > 1e-3
483: 
484:         factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old)
485:         factor = min(MAX_FACTOR, safety * factor)
486: 
487:         if not recompute_jac and factor < 1.2:
488:             factor = 1
489:         else:
490:             LU_real = None
491:             LU_complex = None
492: 
493:         f_new = self.fun(t_new, y_new)
494:         if recompute_jac:
495:             J = jac(t_new, y_new, f_new)
496:             current_jac = True
497:         elif jac is not None:
498:             current_jac = False
499: 
500:         self.h_abs_old = self.h_abs
501:         self.error_norm_old = error_norm
502: 
503:         self.h_abs = h_abs * factor
504: 
505:         self.y_old = y
506: 
507:         self.t = t_new
508:         self.y = y_new
509:         self.f = f_new
510: 
511:         self.Z = Z
512: 
513:         self.LU_real = LU_real
514:         self.LU_complex = LU_complex
515:         self.current_jac = current_jac
516:         self.J = J
517: 
518:         self.t_old = t
519:         self.sol = self._compute_dense_output()
520: 
521:         return step_accepted, message
522: 
523:     def _compute_dense_output(self):
524:         Q = np.dot(self.Z.T, P)
525:         return RadauDenseOutput(self.t_old, self.t, self.y_old, Q)
526: 
527:     def _dense_output_impl(self):
528:         return self.sol
529: 
530: 
531: class RadauDenseOutput(DenseOutput):
532:     def __init__(self, t_old, t, y_old, Q):
533:         super(RadauDenseOutput, self).__init__(t_old, t)
534:         self.h = t - t_old
535:         self.Q = Q
536:         self.order = Q.shape[1] - 1
537:         self.y_old = y_old
538: 
539:     def _call_impl(self, t):
540:         x = (t - self.t_old) / self.h
541:         if t.ndim == 0:
542:             p = np.tile(x, self.order + 1)
543:             p = np.cumprod(p)
544:         else:
545:             p = np.tile(x, (self.order + 1, 1))
546:             p = np.cumprod(p, axis=0)
547:         # Here we don't multiply by h, not a mistake.
548:         y = np.dot(self.Q, p)
549:         if y.ndim == 2:
550:             y += self.y_old[:, None]
551:         else:
552:             y += self.y_old
553: 
554:         return y
555: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import numpy' statement (line 2)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_56714 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy')

if (type(import_56714) is not StypyTypeError):

    if (import_56714 != 'pyd_module'):
        __import__(import_56714)
        sys_modules_56715 = sys.modules[import_56714]
        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'np', sys_modules_56715.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy', import_56714)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from scipy.linalg import lu_factor, lu_solve' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_56716 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.linalg')

if (type(import_56716) is not StypyTypeError):

    if (import_56716 != 'pyd_module'):
        __import__(import_56716)
        sys_modules_56717 = sys.modules[import_56716]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.linalg', sys_modules_56717.module_type_store, module_type_store, ['lu_factor', 'lu_solve'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_56717, sys_modules_56717.module_type_store, module_type_store)
    else:
        from scipy.linalg import lu_factor, lu_solve

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.linalg', None, module_type_store, ['lu_factor', 'lu_solve'], [lu_factor, lu_solve])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.linalg', import_56716)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from scipy.sparse import csc_matrix, issparse, eye' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_56718 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.sparse')

if (type(import_56718) is not StypyTypeError):

    if (import_56718 != 'pyd_module'):
        __import__(import_56718)
        sys_modules_56719 = sys.modules[import_56718]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.sparse', sys_modules_56719.module_type_store, module_type_store, ['csc_matrix', 'issparse', 'eye'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_56719, sys_modules_56719.module_type_store, module_type_store)
    else:
        from scipy.sparse import csc_matrix, issparse, eye

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.sparse', None, module_type_store, ['csc_matrix', 'issparse', 'eye'], [csc_matrix, issparse, eye])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.sparse', import_56718)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.sparse.linalg import splu' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_56720 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse.linalg')

if (type(import_56720) is not StypyTypeError):

    if (import_56720 != 'pyd_module'):
        __import__(import_56720)
        sys_modules_56721 = sys.modules[import_56720]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse.linalg', sys_modules_56721.module_type_store, module_type_store, ['splu'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_56721, sys_modules_56721.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import splu

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse.linalg', None, module_type_store, ['splu'], [splu])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse.linalg', import_56720)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.optimize._numdiff import group_columns' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_56722 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize._numdiff')

if (type(import_56722) is not StypyTypeError):

    if (import_56722 != 'pyd_module'):
        __import__(import_56722)
        sys_modules_56723 = sys.modules[import_56722]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize._numdiff', sys_modules_56723.module_type_store, module_type_store, ['group_columns'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_56723, sys_modules_56723.module_type_store, module_type_store)
    else:
        from scipy.optimize._numdiff import group_columns

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize._numdiff', None, module_type_store, ['group_columns'], [group_columns])

else:
    # Assigning a type to the variable 'scipy.optimize._numdiff' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize._numdiff', import_56722)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.integrate._ivp.common import validate_max_step, validate_tol, select_initial_step, norm, num_jac, EPS, warn_extraneous' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_56724 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.integrate._ivp.common')

if (type(import_56724) is not StypyTypeError):

    if (import_56724 != 'pyd_module'):
        __import__(import_56724)
        sys_modules_56725 = sys.modules[import_56724]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.integrate._ivp.common', sys_modules_56725.module_type_store, module_type_store, ['validate_max_step', 'validate_tol', 'select_initial_step', 'norm', 'num_jac', 'EPS', 'warn_extraneous'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_56725, sys_modules_56725.module_type_store, module_type_store)
    else:
        from scipy.integrate._ivp.common import validate_max_step, validate_tol, select_initial_step, norm, num_jac, EPS, warn_extraneous

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.integrate._ivp.common', None, module_type_store, ['validate_max_step', 'validate_tol', 'select_initial_step', 'norm', 'num_jac', 'EPS', 'warn_extraneous'], [validate_max_step, validate_tol, select_initial_step, norm, num_jac, EPS, warn_extraneous])

else:
    # Assigning a type to the variable 'scipy.integrate._ivp.common' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.integrate._ivp.common', import_56724)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.integrate._ivp.base import OdeSolver, DenseOutput' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_56726 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.integrate._ivp.base')

if (type(import_56726) is not StypyTypeError):

    if (import_56726 != 'pyd_module'):
        __import__(import_56726)
        sys_modules_56727 = sys.modules[import_56726]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.integrate._ivp.base', sys_modules_56727.module_type_store, module_type_store, ['OdeSolver', 'DenseOutput'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_56727, sys_modules_56727.module_type_store, module_type_store)
    else:
        from scipy.integrate._ivp.base import OdeSolver, DenseOutput

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.integrate._ivp.base', None, module_type_store, ['OdeSolver', 'DenseOutput'], [OdeSolver, DenseOutput])

else:
    # Assigning a type to the variable 'scipy.integrate._ivp.base' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.integrate._ivp.base', import_56726)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')


# Assigning a BinOp to a Name (line 11):

# Assigning a BinOp to a Name (line 11):
int_56728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 5), 'int')
float_56729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 10), 'float')
# Applying the binary operator '**' (line 11)
result_pow_56730 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 5), '**', int_56728, float_56729)

# Assigning a type to the variable 'S6' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'S6', result_pow_56730)

# Assigning a Call to a Name (line 14):

# Assigning a Call to a Name (line 14):

# Call to array(...): (line 14)
# Processing the call arguments (line 14)

# Obtaining an instance of the builtin type 'list' (line 14)
list_56733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
int_56734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 15), 'int')
# Getting the type of 'S6' (line 14)
S6_56735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 19), 'S6', False)
# Applying the binary operator '-' (line 14)
result_sub_56736 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 15), '-', int_56734, S6_56735)

int_56737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 25), 'int')
# Applying the binary operator 'div' (line 14)
result_div_56738 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 14), 'div', result_sub_56736, int_56737)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 13), list_56733, result_div_56738)
# Adding element type (line 14)
int_56739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 30), 'int')
# Getting the type of 'S6' (line 14)
S6_56740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 34), 'S6', False)
# Applying the binary operator '+' (line 14)
result_add_56741 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 30), '+', int_56739, S6_56740)

int_56742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 40), 'int')
# Applying the binary operator 'div' (line 14)
result_div_56743 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 29), 'div', result_add_56741, int_56742)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 13), list_56733, result_div_56743)
# Adding element type (line 14)
int_56744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 44), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 13), list_56733, int_56744)

# Processing the call keyword arguments (line 14)
kwargs_56745 = {}
# Getting the type of 'np' (line 14)
np_56731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'np', False)
# Obtaining the member 'array' of a type (line 14)
array_56732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 4), np_56731, 'array')
# Calling array(args, kwargs) (line 14)
array_call_result_56746 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), array_56732, *[list_56733], **kwargs_56745)

# Assigning a type to the variable 'C' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'C', array_call_result_56746)

# Assigning a BinOp to a Name (line 15):

# Assigning a BinOp to a Name (line 15):

# Call to array(...): (line 15)
# Processing the call arguments (line 15)

# Obtaining an instance of the builtin type 'list' (line 15)
list_56749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)
int_56750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 14), 'int')
int_56751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 20), 'int')
# Getting the type of 'S6' (line 15)
S6_56752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 24), 'S6', False)
# Applying the binary operator '*' (line 15)
result_mul_56753 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 20), '*', int_56751, S6_56752)

# Applying the binary operator '-' (line 15)
result_sub_56754 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 14), '-', int_56750, result_mul_56753)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 13), list_56749, result_sub_56754)
# Adding element type (line 15)
int_56755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 28), 'int')
int_56756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 34), 'int')
# Getting the type of 'S6' (line 15)
S6_56757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 38), 'S6', False)
# Applying the binary operator '*' (line 15)
result_mul_56758 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 34), '*', int_56756, S6_56757)

# Applying the binary operator '+' (line 15)
result_add_56759 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 28), '+', int_56755, result_mul_56758)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 13), list_56749, result_add_56759)
# Adding element type (line 15)
int_56760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 42), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 13), list_56749, int_56760)

# Processing the call keyword arguments (line 15)
kwargs_56761 = {}
# Getting the type of 'np' (line 15)
np_56747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'np', False)
# Obtaining the member 'array' of a type (line 15)
array_56748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 4), np_56747, 'array')
# Calling array(args, kwargs) (line 15)
array_call_result_56762 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), array_56748, *[list_56749], **kwargs_56761)

int_56763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 49), 'int')
# Applying the binary operator 'div' (line 15)
result_div_56764 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 4), 'div', array_call_result_56762, int_56763)

# Assigning a type to the variable 'E' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'E', result_div_56764)

# Assigning a BinOp to a Name (line 19):

# Assigning a BinOp to a Name (line 19):
int_56765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 10), 'int')
int_56766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 14), 'int')
int_56767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 20), 'int')
int_56768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 24), 'int')
# Applying the binary operator 'div' (line 19)
result_div_56769 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 20), 'div', int_56767, int_56768)

# Applying the binary operator '**' (line 19)
result_pow_56770 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 14), '**', int_56766, result_div_56769)

# Applying the binary operator '+' (line 19)
result_add_56771 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 10), '+', int_56765, result_pow_56770)

int_56772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 29), 'int')
int_56773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 35), 'int')
int_56774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 39), 'int')
# Applying the binary operator 'div' (line 19)
result_div_56775 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 35), 'div', int_56773, int_56774)

# Applying the binary operator '**' (line 19)
result_pow_56776 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 29), '**', int_56772, result_div_56775)

# Applying the binary operator '-' (line 19)
result_sub_56777 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 27), '-', result_add_56771, result_pow_56776)

# Assigning a type to the variable 'MU_REAL' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'MU_REAL', result_sub_56777)

# Assigning a BinOp to a Name (line 20):

# Assigning a BinOp to a Name (line 20):
int_56778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 14), 'int')
float_56779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 18), 'float')
int_56780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 25), 'int')
int_56781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 31), 'int')
int_56782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 35), 'int')
# Applying the binary operator 'div' (line 20)
result_div_56783 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 31), 'div', int_56781, int_56782)

# Applying the binary operator '**' (line 20)
result_pow_56784 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 25), '**', int_56780, result_div_56783)

int_56785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 40), 'int')
int_56786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 46), 'int')
int_56787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 50), 'int')
# Applying the binary operator 'div' (line 20)
result_div_56788 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 46), 'div', int_56786, int_56787)

# Applying the binary operator '**' (line 20)
result_pow_56789 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 40), '**', int_56785, result_div_56788)

# Applying the binary operator '-' (line 20)
result_sub_56790 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 25), '-', result_pow_56784, result_pow_56789)

# Applying the binary operator '*' (line 20)
result_mul_56791 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 18), '*', float_56779, result_sub_56790)

# Applying the binary operator '+' (line 20)
result_add_56792 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 14), '+', int_56778, result_mul_56791)

complex_56793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 16), 'complex')
int_56794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 24), 'int')
int_56795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 30), 'int')
int_56796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 34), 'int')
# Applying the binary operator 'div' (line 21)
result_div_56797 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 30), 'div', int_56795, int_56796)

# Applying the binary operator '**' (line 21)
result_pow_56798 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 24), '**', int_56794, result_div_56797)

int_56799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 39), 'int')
int_56800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 45), 'int')
int_56801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 49), 'int')
# Applying the binary operator 'div' (line 21)
result_div_56802 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 45), 'div', int_56800, int_56801)

# Applying the binary operator '**' (line 21)
result_pow_56803 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 39), '**', int_56799, result_div_56802)

# Applying the binary operator '+' (line 21)
result_add_56804 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 24), '+', result_pow_56798, result_pow_56803)

# Applying the binary operator '*' (line 21)
result_mul_56805 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 16), '*', complex_56793, result_add_56804)

# Applying the binary operator '-' (line 21)
result_sub_56806 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 14), '-', result_add_56792, result_mul_56805)

# Assigning a type to the variable 'MU_COMPLEX' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'MU_COMPLEX', result_sub_56806)

# Assigning a Call to a Name (line 24):

# Assigning a Call to a Name (line 24):

# Call to array(...): (line 24)
# Processing the call arguments (line 24)

# Obtaining an instance of the builtin type 'list' (line 24)
list_56809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 24)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'list' (line 25)
list_56810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 25)
# Adding element type (line 25)
float_56811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 5), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 4), list_56810, float_56811)
# Adding element type (line 25)
float_56812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 26), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 4), list_56810, float_56812)
# Adding element type (line 25)
float_56813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 48), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 4), list_56810, float_56813)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 13), list_56809, list_56810)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'list' (line 26)
list_56814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 26)
# Adding element type (line 26)
float_56815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 5), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 4), list_56814, float_56815)
# Adding element type (line 26)
float_56816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 26), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 4), list_56814, float_56816)
# Adding element type (line 26)
float_56817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 47), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 4), list_56814, float_56817)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 13), list_56809, list_56814)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'list' (line 27)
list_56818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 27)
# Adding element type (line 27)
int_56819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 4), list_56818, int_56819)
# Adding element type (line 27)
int_56820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 4), list_56818, int_56820)
# Adding element type (line 27)
int_56821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 4), list_56818, int_56821)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 13), list_56809, list_56818)

# Processing the call keyword arguments (line 24)
kwargs_56822 = {}
# Getting the type of 'np' (line 24)
np_56807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'np', False)
# Obtaining the member 'array' of a type (line 24)
array_56808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 4), np_56807, 'array')
# Calling array(args, kwargs) (line 24)
array_call_result_56823 = invoke(stypy.reporting.localization.Localization(__file__, 24, 4), array_56808, *[list_56809], **kwargs_56822)

# Assigning a type to the variable 'T' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'T', array_call_result_56823)

# Assigning a Call to a Name (line 28):

# Assigning a Call to a Name (line 28):

# Call to array(...): (line 28)
# Processing the call arguments (line 28)

# Obtaining an instance of the builtin type 'list' (line 28)
list_56826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 28)
# Adding element type (line 28)

# Obtaining an instance of the builtin type 'list' (line 29)
list_56827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 29)
# Adding element type (line 29)
float_56828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 5), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 4), list_56827, float_56828)
# Adding element type (line 29)
float_56829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 26), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 4), list_56827, float_56829)
# Adding element type (line 29)
float_56830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 47), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 4), list_56827, float_56830)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 14), list_56826, list_56827)
# Adding element type (line 28)

# Obtaining an instance of the builtin type 'list' (line 30)
list_56831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 30)
# Adding element type (line 30)
float_56832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 5), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 4), list_56831, float_56832)
# Adding element type (line 30)
float_56833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 27), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 4), list_56831, float_56833)
# Adding element type (line 30)
float_56834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 49), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 4), list_56831, float_56834)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 14), list_56826, list_56831)
# Adding element type (line 28)

# Obtaining an instance of the builtin type 'list' (line 31)
list_56835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 31)
# Adding element type (line 31)
float_56836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 5), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 4), list_56835, float_56836)
# Adding element type (line 31)
float_56837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 26), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 4), list_56835, float_56837)
# Adding element type (line 31)
float_56838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 48), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 4), list_56835, float_56838)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 14), list_56826, list_56835)

# Processing the call keyword arguments (line 28)
kwargs_56839 = {}
# Getting the type of 'np' (line 28)
np_56824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 5), 'np', False)
# Obtaining the member 'array' of a type (line 28)
array_56825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 5), np_56824, 'array')
# Calling array(args, kwargs) (line 28)
array_call_result_56840 = invoke(stypy.reporting.localization.Localization(__file__, 28, 5), array_56825, *[list_56826], **kwargs_56839)

# Assigning a type to the variable 'TI' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'TI', array_call_result_56840)

# Assigning a Subscript to a Name (line 33):

# Assigning a Subscript to a Name (line 33):

# Obtaining the type of the subscript
int_56841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 13), 'int')
# Getting the type of 'TI' (line 33)
TI_56842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 10), 'TI')
# Obtaining the member '__getitem__' of a type (line 33)
getitem___56843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 10), TI_56842, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 33)
subscript_call_result_56844 = invoke(stypy.reporting.localization.Localization(__file__, 33, 10), getitem___56843, int_56841)

# Assigning a type to the variable 'TI_REAL' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'TI_REAL', subscript_call_result_56844)

# Assigning a BinOp to a Name (line 34):

# Assigning a BinOp to a Name (line 34):

# Obtaining the type of the subscript
int_56845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 16), 'int')
# Getting the type of 'TI' (line 34)
TI_56846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 13), 'TI')
# Obtaining the member '__getitem__' of a type (line 34)
getitem___56847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 13), TI_56846, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 34)
subscript_call_result_56848 = invoke(stypy.reporting.localization.Localization(__file__, 34, 13), getitem___56847, int_56845)

complex_56849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 21), 'complex')

# Obtaining the type of the subscript
int_56850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 29), 'int')
# Getting the type of 'TI' (line 34)
TI_56851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 26), 'TI')
# Obtaining the member '__getitem__' of a type (line 34)
getitem___56852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 26), TI_56851, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 34)
subscript_call_result_56853 = invoke(stypy.reporting.localization.Localization(__file__, 34, 26), getitem___56852, int_56850)

# Applying the binary operator '*' (line 34)
result_mul_56854 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 21), '*', complex_56849, subscript_call_result_56853)

# Applying the binary operator '+' (line 34)
result_add_56855 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 13), '+', subscript_call_result_56848, result_mul_56854)

# Assigning a type to the variable 'TI_COMPLEX' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'TI_COMPLEX', result_add_56855)

# Assigning a Call to a Name (line 37):

# Assigning a Call to a Name (line 37):

# Call to array(...): (line 37)
# Processing the call arguments (line 37)

# Obtaining an instance of the builtin type 'list' (line 37)
list_56858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 37)
# Adding element type (line 37)

# Obtaining an instance of the builtin type 'list' (line 38)
list_56859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 38)
# Adding element type (line 38)
int_56860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 5), 'int')
int_56861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 8), 'int')
# Applying the binary operator 'div' (line 38)
result_div_56862 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 5), 'div', int_56860, int_56861)

int_56863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 12), 'int')
# Getting the type of 'S6' (line 38)
S6_56864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 14), 'S6', False)
# Applying the binary operator '*' (line 38)
result_mul_56865 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 12), '*', int_56863, S6_56864)

int_56866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 17), 'int')
# Applying the binary operator 'div' (line 38)
result_div_56867 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 16), 'div', result_mul_56865, int_56866)

# Applying the binary operator '+' (line 38)
result_add_56868 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 5), '+', result_div_56862, result_div_56867)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 4), list_56859, result_add_56868)
# Adding element type (line 38)
int_56869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 20), 'int')
int_56870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 24), 'int')
# Applying the binary operator 'div' (line 38)
result_div_56871 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 20), 'div', int_56869, int_56870)

int_56872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 28), 'int')
# Getting the type of 'S6' (line 38)
S6_56873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 31), 'S6', False)
# Applying the binary operator '*' (line 38)
result_mul_56874 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 28), '*', int_56872, S6_56873)

int_56875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 34), 'int')
# Applying the binary operator 'div' (line 38)
result_div_56876 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 33), 'div', result_mul_56874, int_56875)

# Applying the binary operator '-' (line 38)
result_sub_56877 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 20), '-', result_div_56871, result_div_56876)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 4), list_56859, result_sub_56877)
# Adding element type (line 38)
int_56878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 37), 'int')
int_56879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 40), 'int')
# Applying the binary operator 'div' (line 38)
result_div_56880 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 37), 'div', int_56878, int_56879)

int_56881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 44), 'int')
# Getting the type of 'S6' (line 38)
S6_56882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 48), 'S6', False)
# Applying the binary operator '*' (line 38)
result_mul_56883 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 44), '*', int_56881, S6_56882)

# Applying the binary operator '+' (line 38)
result_add_56884 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 37), '+', result_div_56880, result_mul_56883)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 4), list_56859, result_add_56884)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 13), list_56858, list_56859)
# Adding element type (line 37)

# Obtaining an instance of the builtin type 'list' (line 39)
list_56885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 39)
# Adding element type (line 39)
int_56886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 5), 'int')
int_56887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 8), 'int')
# Applying the binary operator 'div' (line 39)
result_div_56888 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 5), 'div', int_56886, int_56887)

int_56889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 12), 'int')
# Getting the type of 'S6' (line 39)
S6_56890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 14), 'S6', False)
# Applying the binary operator '*' (line 39)
result_mul_56891 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 12), '*', int_56889, S6_56890)

int_56892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 17), 'int')
# Applying the binary operator 'div' (line 39)
result_div_56893 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 16), 'div', result_mul_56891, int_56892)

# Applying the binary operator '-' (line 39)
result_sub_56894 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 5), '-', result_div_56888, result_div_56893)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 4), list_56885, result_sub_56894)
# Adding element type (line 39)
int_56895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 20), 'int')
int_56896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 24), 'int')
# Applying the binary operator 'div' (line 39)
result_div_56897 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 20), 'div', int_56895, int_56896)

int_56898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 28), 'int')
# Getting the type of 'S6' (line 39)
S6_56899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 31), 'S6', False)
# Applying the binary operator '*' (line 39)
result_mul_56900 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 28), '*', int_56898, S6_56899)

int_56901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 34), 'int')
# Applying the binary operator 'div' (line 39)
result_div_56902 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 33), 'div', result_mul_56900, int_56901)

# Applying the binary operator '+' (line 39)
result_add_56903 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 20), '+', result_div_56897, result_div_56902)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 4), list_56885, result_add_56903)
# Adding element type (line 39)
int_56904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 37), 'int')
int_56905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 40), 'int')
# Applying the binary operator 'div' (line 39)
result_div_56906 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 37), 'div', int_56904, int_56905)

int_56907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 44), 'int')
# Getting the type of 'S6' (line 39)
S6_56908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 48), 'S6', False)
# Applying the binary operator '*' (line 39)
result_mul_56909 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 44), '*', int_56907, S6_56908)

# Applying the binary operator '-' (line 39)
result_sub_56910 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 37), '-', result_div_56906, result_mul_56909)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 4), list_56885, result_sub_56910)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 13), list_56858, list_56885)
# Adding element type (line 37)

# Obtaining an instance of the builtin type 'list' (line 40)
list_56911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 40)
# Adding element type (line 40)
int_56912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 5), 'int')
int_56913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 7), 'int')
# Applying the binary operator 'div' (line 40)
result_div_56914 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 5), 'div', int_56912, int_56913)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 4), list_56911, result_div_56914)
# Adding element type (line 40)
int_56915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 10), 'int')
int_56916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 13), 'int')
# Applying the binary operator 'div' (line 40)
result_div_56917 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 10), 'div', int_56915, int_56916)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 4), list_56911, result_div_56917)
# Adding element type (line 40)
int_56918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 16), 'int')
int_56919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 19), 'int')
# Applying the binary operator 'div' (line 40)
result_div_56920 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 16), 'div', int_56918, int_56919)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 4), list_56911, result_div_56920)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 13), list_56858, list_56911)

# Processing the call keyword arguments (line 37)
kwargs_56921 = {}
# Getting the type of 'np' (line 37)
np_56856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'np', False)
# Obtaining the member 'array' of a type (line 37)
array_56857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 4), np_56856, 'array')
# Calling array(args, kwargs) (line 37)
array_call_result_56922 = invoke(stypy.reporting.localization.Localization(__file__, 37, 4), array_56857, *[list_56858], **kwargs_56921)

# Assigning a type to the variable 'P' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'P', array_call_result_56922)

# Assigning a Num to a Name (line 43):

# Assigning a Num to a Name (line 43):
int_56923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 17), 'int')
# Assigning a type to the variable 'NEWTON_MAXITER' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'NEWTON_MAXITER', int_56923)

# Assigning a Num to a Name (line 44):

# Assigning a Num to a Name (line 44):
float_56924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 13), 'float')
# Assigning a type to the variable 'MIN_FACTOR' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'MIN_FACTOR', float_56924)

# Assigning a Num to a Name (line 45):

# Assigning a Num to a Name (line 45):
int_56925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 13), 'int')
# Assigning a type to the variable 'MAX_FACTOR' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'MAX_FACTOR', int_56925)

@norecursion
def solve_collocation_system(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'solve_collocation_system'
    module_type_store = module_type_store.open_function_context('solve_collocation_system', 48, 0, False)
    
    # Passed parameters checking function
    solve_collocation_system.stypy_localization = localization
    solve_collocation_system.stypy_type_of_self = None
    solve_collocation_system.stypy_type_store = module_type_store
    solve_collocation_system.stypy_function_name = 'solve_collocation_system'
    solve_collocation_system.stypy_param_names_list = ['fun', 't', 'y', 'h', 'Z0', 'scale', 'tol', 'LU_real', 'LU_complex', 'solve_lu']
    solve_collocation_system.stypy_varargs_param_name = None
    solve_collocation_system.stypy_kwargs_param_name = None
    solve_collocation_system.stypy_call_defaults = defaults
    solve_collocation_system.stypy_call_varargs = varargs
    solve_collocation_system.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'solve_collocation_system', ['fun', 't', 'y', 'h', 'Z0', 'scale', 'tol', 'LU_real', 'LU_complex', 'solve_lu'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'solve_collocation_system', localization, ['fun', 't', 'y', 'h', 'Z0', 'scale', 'tol', 'LU_real', 'LU_complex', 'solve_lu'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'solve_collocation_system(...)' code ##################

    str_56926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, (-1)), 'str', 'Solve the collocation system.\n\n    Parameters\n    ----------\n    fun : callable\n        Right-hand side of the system.\n    t : float\n        Current time.\n    y : ndarray, shape (n,)\n        Current state.\n    h : float\n        Step to try.\n    Z0 : ndarray, shape (3, n)\n        Initial guess for the solution. It determines new values of `y` at\n        ``t + h * C`` as ``y + Z0``, where ``C`` is the Radau method constants.\n    scale : float\n        Problem tolerance scale, i.e. ``rtol * abs(y) + atol``.\n    tol : float\n        Tolerance to which solve the system. This value is compared with\n        the normalized by `scale` error.\n    LU_real, LU_complex\n        LU decompositions of the system Jacobians.\n    solve_lu : callable\n        Callable which solves a linear system given a LU decomposition. The\n        signature is ``solve_lu(LU, b)``.\n\n    Returns\n    -------\n    converged : bool\n        Whether iterations converged.\n    n_iter : int\n        Number of completed iterations.\n    Z : ndarray, shape (3, n)\n        Found solution.\n    rate : float\n        The rate of convergence.\n    ')
    
    # Assigning a Subscript to a Name (line 87):
    
    # Assigning a Subscript to a Name (line 87):
    
    # Obtaining the type of the subscript
    int_56927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 16), 'int')
    # Getting the type of 'y' (line 87)
    y_56928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'y')
    # Obtaining the member 'shape' of a type (line 87)
    shape_56929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), y_56928, 'shape')
    # Obtaining the member '__getitem__' of a type (line 87)
    getitem___56930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), shape_56929, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 87)
    subscript_call_result_56931 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), getitem___56930, int_56927)
    
    # Assigning a type to the variable 'n' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'n', subscript_call_result_56931)
    
    # Assigning a BinOp to a Name (line 88):
    
    # Assigning a BinOp to a Name (line 88):
    # Getting the type of 'MU_REAL' (line 88)
    MU_REAL_56932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 13), 'MU_REAL')
    # Getting the type of 'h' (line 88)
    h_56933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 23), 'h')
    # Applying the binary operator 'div' (line 88)
    result_div_56934 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 13), 'div', MU_REAL_56932, h_56933)
    
    # Assigning a type to the variable 'M_real' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'M_real', result_div_56934)
    
    # Assigning a BinOp to a Name (line 89):
    
    # Assigning a BinOp to a Name (line 89):
    # Getting the type of 'MU_COMPLEX' (line 89)
    MU_COMPLEX_56935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'MU_COMPLEX')
    # Getting the type of 'h' (line 89)
    h_56936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 29), 'h')
    # Applying the binary operator 'div' (line 89)
    result_div_56937 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 16), 'div', MU_COMPLEX_56935, h_56936)
    
    # Assigning a type to the variable 'M_complex' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'M_complex', result_div_56937)
    
    # Assigning a Call to a Name (line 91):
    
    # Assigning a Call to a Name (line 91):
    
    # Call to dot(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'Z0' (line 91)
    Z0_56940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 15), 'Z0', False)
    # Processing the call keyword arguments (line 91)
    kwargs_56941 = {}
    # Getting the type of 'TI' (line 91)
    TI_56938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'TI', False)
    # Obtaining the member 'dot' of a type (line 91)
    dot_56939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), TI_56938, 'dot')
    # Calling dot(args, kwargs) (line 91)
    dot_call_result_56942 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), dot_56939, *[Z0_56940], **kwargs_56941)
    
    # Assigning a type to the variable 'W' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'W', dot_call_result_56942)
    
    # Assigning a Name to a Name (line 92):
    
    # Assigning a Name to a Name (line 92):
    # Getting the type of 'Z0' (line 92)
    Z0_56943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'Z0')
    # Assigning a type to the variable 'Z' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'Z', Z0_56943)
    
    # Assigning a Call to a Name (line 94):
    
    # Assigning a Call to a Name (line 94):
    
    # Call to empty(...): (line 94)
    # Processing the call arguments (line 94)
    
    # Obtaining an instance of the builtin type 'tuple' (line 94)
    tuple_56946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 94)
    # Adding element type (line 94)
    int_56947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 18), tuple_56946, int_56947)
    # Adding element type (line 94)
    # Getting the type of 'n' (line 94)
    n_56948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 21), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 18), tuple_56946, n_56948)
    
    # Processing the call keyword arguments (line 94)
    kwargs_56949 = {}
    # Getting the type of 'np' (line 94)
    np_56944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'np', False)
    # Obtaining the member 'empty' of a type (line 94)
    empty_56945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), np_56944, 'empty')
    # Calling empty(args, kwargs) (line 94)
    empty_call_result_56950 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), empty_56945, *[tuple_56946], **kwargs_56949)
    
    # Assigning a type to the variable 'F' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'F', empty_call_result_56950)
    
    # Assigning a BinOp to a Name (line 95):
    
    # Assigning a BinOp to a Name (line 95):
    # Getting the type of 'h' (line 95)
    h_56951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 9), 'h')
    # Getting the type of 'C' (line 95)
    C_56952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 13), 'C')
    # Applying the binary operator '*' (line 95)
    result_mul_56953 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 9), '*', h_56951, C_56952)
    
    # Assigning a type to the variable 'ch' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'ch', result_mul_56953)
    
    # Assigning a Name to a Name (line 97):
    
    # Assigning a Name to a Name (line 97):
    # Getting the type of 'None' (line 97)
    None_56954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 18), 'None')
    # Assigning a type to the variable 'dW_norm_old' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'dW_norm_old', None_56954)
    
    # Assigning a Call to a Name (line 98):
    
    # Assigning a Call to a Name (line 98):
    
    # Call to empty_like(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'W' (line 98)
    W_56957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 23), 'W', False)
    # Processing the call keyword arguments (line 98)
    kwargs_56958 = {}
    # Getting the type of 'np' (line 98)
    np_56955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 9), 'np', False)
    # Obtaining the member 'empty_like' of a type (line 98)
    empty_like_56956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 9), np_56955, 'empty_like')
    # Calling empty_like(args, kwargs) (line 98)
    empty_like_call_result_56959 = invoke(stypy.reporting.localization.Localization(__file__, 98, 9), empty_like_56956, *[W_56957], **kwargs_56958)
    
    # Assigning a type to the variable 'dW' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'dW', empty_like_call_result_56959)
    
    # Assigning a Name to a Name (line 99):
    
    # Assigning a Name to a Name (line 99):
    # Getting the type of 'False' (line 99)
    False_56960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'False')
    # Assigning a type to the variable 'converged' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'converged', False_56960)
    
    
    # Call to range(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'NEWTON_MAXITER' (line 100)
    NEWTON_MAXITER_56962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 19), 'NEWTON_MAXITER', False)
    # Processing the call keyword arguments (line 100)
    kwargs_56963 = {}
    # Getting the type of 'range' (line 100)
    range_56961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 13), 'range', False)
    # Calling range(args, kwargs) (line 100)
    range_call_result_56964 = invoke(stypy.reporting.localization.Localization(__file__, 100, 13), range_56961, *[NEWTON_MAXITER_56962], **kwargs_56963)
    
    # Testing the type of a for loop iterable (line 100)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 100, 4), range_call_result_56964)
    # Getting the type of the for loop variable (line 100)
    for_loop_var_56965 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 100, 4), range_call_result_56964)
    # Assigning a type to the variable 'k' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'k', for_loop_var_56965)
    # SSA begins for a for statement (line 100)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 101)
    # Processing the call arguments (line 101)
    int_56967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 23), 'int')
    # Processing the call keyword arguments (line 101)
    kwargs_56968 = {}
    # Getting the type of 'range' (line 101)
    range_56966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 17), 'range', False)
    # Calling range(args, kwargs) (line 101)
    range_call_result_56969 = invoke(stypy.reporting.localization.Localization(__file__, 101, 17), range_56966, *[int_56967], **kwargs_56968)
    
    # Testing the type of a for loop iterable (line 101)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 101, 8), range_call_result_56969)
    # Getting the type of the for loop variable (line 101)
    for_loop_var_56970 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 101, 8), range_call_result_56969)
    # Assigning a type to the variable 'i' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'i', for_loop_var_56970)
    # SSA begins for a for statement (line 101)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 102):
    
    # Assigning a Call to a Subscript (line 102):
    
    # Call to fun(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 't' (line 102)
    t_56972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 23), 't', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 102)
    i_56973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 30), 'i', False)
    # Getting the type of 'ch' (line 102)
    ch_56974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 27), 'ch', False)
    # Obtaining the member '__getitem__' of a type (line 102)
    getitem___56975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 27), ch_56974, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 102)
    subscript_call_result_56976 = invoke(stypy.reporting.localization.Localization(__file__, 102, 27), getitem___56975, i_56973)
    
    # Applying the binary operator '+' (line 102)
    result_add_56977 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 23), '+', t_56972, subscript_call_result_56976)
    
    # Getting the type of 'y' (line 102)
    y_56978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 34), 'y', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 102)
    i_56979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 40), 'i', False)
    # Getting the type of 'Z' (line 102)
    Z_56980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 38), 'Z', False)
    # Obtaining the member '__getitem__' of a type (line 102)
    getitem___56981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 38), Z_56980, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 102)
    subscript_call_result_56982 = invoke(stypy.reporting.localization.Localization(__file__, 102, 38), getitem___56981, i_56979)
    
    # Applying the binary operator '+' (line 102)
    result_add_56983 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 34), '+', y_56978, subscript_call_result_56982)
    
    # Processing the call keyword arguments (line 102)
    kwargs_56984 = {}
    # Getting the type of 'fun' (line 102)
    fun_56971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 19), 'fun', False)
    # Calling fun(args, kwargs) (line 102)
    fun_call_result_56985 = invoke(stypy.reporting.localization.Localization(__file__, 102, 19), fun_56971, *[result_add_56977, result_add_56983], **kwargs_56984)
    
    # Getting the type of 'F' (line 102)
    F_56986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'F')
    # Getting the type of 'i' (line 102)
    i_56987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 14), 'i')
    # Storing an element on a container (line 102)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 12), F_56986, (i_56987, fun_call_result_56985))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to all(...): (line 104)
    # Processing the call arguments (line 104)
    
    # Call to isfinite(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'F' (line 104)
    F_56992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 34), 'F', False)
    # Processing the call keyword arguments (line 104)
    kwargs_56993 = {}
    # Getting the type of 'np' (line 104)
    np_56990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 22), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 104)
    isfinite_56991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 22), np_56990, 'isfinite')
    # Calling isfinite(args, kwargs) (line 104)
    isfinite_call_result_56994 = invoke(stypy.reporting.localization.Localization(__file__, 104, 22), isfinite_56991, *[F_56992], **kwargs_56993)
    
    # Processing the call keyword arguments (line 104)
    kwargs_56995 = {}
    # Getting the type of 'np' (line 104)
    np_56988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'np', False)
    # Obtaining the member 'all' of a type (line 104)
    all_56989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 15), np_56988, 'all')
    # Calling all(args, kwargs) (line 104)
    all_call_result_56996 = invoke(stypy.reporting.localization.Localization(__file__, 104, 15), all_56989, *[isfinite_call_result_56994], **kwargs_56995)
    
    # Applying the 'not' unary operator (line 104)
    result_not__56997 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 11), 'not', all_call_result_56996)
    
    # Testing the type of an if condition (line 104)
    if_condition_56998 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 8), result_not__56997)
    # Assigning a type to the variable 'if_condition_56998' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'if_condition_56998', if_condition_56998)
    # SSA begins for if statement (line 104)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 104)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 107):
    
    # Assigning a BinOp to a Name (line 107):
    
    # Call to dot(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'TI_REAL' (line 107)
    TI_REAL_57002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 25), 'TI_REAL', False)
    # Processing the call keyword arguments (line 107)
    kwargs_57003 = {}
    # Getting the type of 'F' (line 107)
    F_56999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 17), 'F', False)
    # Obtaining the member 'T' of a type (line 107)
    T_57000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 17), F_56999, 'T')
    # Obtaining the member 'dot' of a type (line 107)
    dot_57001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 17), T_57000, 'dot')
    # Calling dot(args, kwargs) (line 107)
    dot_call_result_57004 = invoke(stypy.reporting.localization.Localization(__file__, 107, 17), dot_57001, *[TI_REAL_57002], **kwargs_57003)
    
    # Getting the type of 'M_real' (line 107)
    M_real_57005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 36), 'M_real')
    
    # Obtaining the type of the subscript
    int_57006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 47), 'int')
    # Getting the type of 'W' (line 107)
    W_57007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 45), 'W')
    # Obtaining the member '__getitem__' of a type (line 107)
    getitem___57008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 45), W_57007, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 107)
    subscript_call_result_57009 = invoke(stypy.reporting.localization.Localization(__file__, 107, 45), getitem___57008, int_57006)
    
    # Applying the binary operator '*' (line 107)
    result_mul_57010 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 36), '*', M_real_57005, subscript_call_result_57009)
    
    # Applying the binary operator '-' (line 107)
    result_sub_57011 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 17), '-', dot_call_result_57004, result_mul_57010)
    
    # Assigning a type to the variable 'f_real' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'f_real', result_sub_57011)
    
    # Assigning a BinOp to a Name (line 108):
    
    # Assigning a BinOp to a Name (line 108):
    
    # Call to dot(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'TI_COMPLEX' (line 108)
    TI_COMPLEX_57015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 28), 'TI_COMPLEX', False)
    # Processing the call keyword arguments (line 108)
    kwargs_57016 = {}
    # Getting the type of 'F' (line 108)
    F_57012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 20), 'F', False)
    # Obtaining the member 'T' of a type (line 108)
    T_57013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 20), F_57012, 'T')
    # Obtaining the member 'dot' of a type (line 108)
    dot_57014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 20), T_57013, 'dot')
    # Calling dot(args, kwargs) (line 108)
    dot_call_result_57017 = invoke(stypy.reporting.localization.Localization(__file__, 108, 20), dot_57014, *[TI_COMPLEX_57015], **kwargs_57016)
    
    # Getting the type of 'M_complex' (line 108)
    M_complex_57018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 42), 'M_complex')
    
    # Obtaining the type of the subscript
    int_57019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 57), 'int')
    # Getting the type of 'W' (line 108)
    W_57020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 55), 'W')
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___57021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 55), W_57020, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_57022 = invoke(stypy.reporting.localization.Localization(__file__, 108, 55), getitem___57021, int_57019)
    
    complex_57023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 62), 'complex')
    
    # Obtaining the type of the subscript
    int_57024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 69), 'int')
    # Getting the type of 'W' (line 108)
    W_57025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 67), 'W')
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___57026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 67), W_57025, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_57027 = invoke(stypy.reporting.localization.Localization(__file__, 108, 67), getitem___57026, int_57024)
    
    # Applying the binary operator '*' (line 108)
    result_mul_57028 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 62), '*', complex_57023, subscript_call_result_57027)
    
    # Applying the binary operator '+' (line 108)
    result_add_57029 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 55), '+', subscript_call_result_57022, result_mul_57028)
    
    # Applying the binary operator '*' (line 108)
    result_mul_57030 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 42), '*', M_complex_57018, result_add_57029)
    
    # Applying the binary operator '-' (line 108)
    result_sub_57031 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 20), '-', dot_call_result_57017, result_mul_57030)
    
    # Assigning a type to the variable 'f_complex' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'f_complex', result_sub_57031)
    
    # Assigning a Call to a Name (line 110):
    
    # Assigning a Call to a Name (line 110):
    
    # Call to solve_lu(...): (line 110)
    # Processing the call arguments (line 110)
    # Getting the type of 'LU_real' (line 110)
    LU_real_57033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 27), 'LU_real', False)
    # Getting the type of 'f_real' (line 110)
    f_real_57034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 36), 'f_real', False)
    # Processing the call keyword arguments (line 110)
    kwargs_57035 = {}
    # Getting the type of 'solve_lu' (line 110)
    solve_lu_57032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 18), 'solve_lu', False)
    # Calling solve_lu(args, kwargs) (line 110)
    solve_lu_call_result_57036 = invoke(stypy.reporting.localization.Localization(__file__, 110, 18), solve_lu_57032, *[LU_real_57033, f_real_57034], **kwargs_57035)
    
    # Assigning a type to the variable 'dW_real' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'dW_real', solve_lu_call_result_57036)
    
    # Assigning a Call to a Name (line 111):
    
    # Assigning a Call to a Name (line 111):
    
    # Call to solve_lu(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'LU_complex' (line 111)
    LU_complex_57038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 30), 'LU_complex', False)
    # Getting the type of 'f_complex' (line 111)
    f_complex_57039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 42), 'f_complex', False)
    # Processing the call keyword arguments (line 111)
    kwargs_57040 = {}
    # Getting the type of 'solve_lu' (line 111)
    solve_lu_57037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 21), 'solve_lu', False)
    # Calling solve_lu(args, kwargs) (line 111)
    solve_lu_call_result_57041 = invoke(stypy.reporting.localization.Localization(__file__, 111, 21), solve_lu_57037, *[LU_complex_57038, f_complex_57039], **kwargs_57040)
    
    # Assigning a type to the variable 'dW_complex' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'dW_complex', solve_lu_call_result_57041)
    
    # Assigning a Name to a Subscript (line 113):
    
    # Assigning a Name to a Subscript (line 113):
    # Getting the type of 'dW_real' (line 113)
    dW_real_57042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'dW_real')
    # Getting the type of 'dW' (line 113)
    dW_57043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'dW')
    int_57044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 11), 'int')
    # Storing an element on a container (line 113)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 8), dW_57043, (int_57044, dW_real_57042))
    
    # Assigning a Attribute to a Subscript (line 114):
    
    # Assigning a Attribute to a Subscript (line 114):
    # Getting the type of 'dW_complex' (line 114)
    dW_complex_57045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'dW_complex')
    # Obtaining the member 'real' of a type (line 114)
    real_57046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 16), dW_complex_57045, 'real')
    # Getting the type of 'dW' (line 114)
    dW_57047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'dW')
    int_57048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 11), 'int')
    # Storing an element on a container (line 114)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 8), dW_57047, (int_57048, real_57046))
    
    # Assigning a Attribute to a Subscript (line 115):
    
    # Assigning a Attribute to a Subscript (line 115):
    # Getting the type of 'dW_complex' (line 115)
    dW_complex_57049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'dW_complex')
    # Obtaining the member 'imag' of a type (line 115)
    imag_57050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 16), dW_complex_57049, 'imag')
    # Getting the type of 'dW' (line 115)
    dW_57051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'dW')
    int_57052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 11), 'int')
    # Storing an element on a container (line 115)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 8), dW_57051, (int_57052, imag_57050))
    
    # Assigning a Call to a Name (line 117):
    
    # Assigning a Call to a Name (line 117):
    
    # Call to norm(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'dW' (line 117)
    dW_57054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 23), 'dW', False)
    # Getting the type of 'scale' (line 117)
    scale_57055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 28), 'scale', False)
    # Applying the binary operator 'div' (line 117)
    result_div_57056 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 23), 'div', dW_57054, scale_57055)
    
    # Processing the call keyword arguments (line 117)
    kwargs_57057 = {}
    # Getting the type of 'norm' (line 117)
    norm_57053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 18), 'norm', False)
    # Calling norm(args, kwargs) (line 117)
    norm_call_result_57058 = invoke(stypy.reporting.localization.Localization(__file__, 117, 18), norm_57053, *[result_div_57056], **kwargs_57057)
    
    # Assigning a type to the variable 'dW_norm' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'dW_norm', norm_call_result_57058)
    
    # Type idiom detected: calculating its left and rigth part (line 118)
    # Getting the type of 'dW_norm_old' (line 118)
    dW_norm_old_57059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'dW_norm_old')
    # Getting the type of 'None' (line 118)
    None_57060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 30), 'None')
    
    (may_be_57061, more_types_in_union_57062) = may_not_be_none(dW_norm_old_57059, None_57060)

    if may_be_57061:

        if more_types_in_union_57062:
            # Runtime conditional SSA (line 118)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 119):
        
        # Assigning a BinOp to a Name (line 119):
        # Getting the type of 'dW_norm' (line 119)
        dW_norm_57063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 19), 'dW_norm')
        # Getting the type of 'dW_norm_old' (line 119)
        dW_norm_old_57064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 29), 'dW_norm_old')
        # Applying the binary operator 'div' (line 119)
        result_div_57065 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 19), 'div', dW_norm_57063, dW_norm_old_57064)
        
        # Assigning a type to the variable 'rate' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'rate', result_div_57065)

        if more_types_in_union_57062:
            # Runtime conditional SSA for else branch (line 118)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_57061) or more_types_in_union_57062):
        
        # Assigning a Name to a Name (line 121):
        
        # Assigning a Name to a Name (line 121):
        # Getting the type of 'None' (line 121)
        None_57066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'None')
        # Assigning a type to the variable 'rate' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'rate', None_57066)

        if (may_be_57061 and more_types_in_union_57062):
            # SSA join for if statement (line 118)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'rate' (line 123)
    rate_57067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'rate')
    # Getting the type of 'None' (line 123)
    None_57068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 24), 'None')
    # Applying the binary operator 'isnot' (line 123)
    result_is_not_57069 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 12), 'isnot', rate_57067, None_57068)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'rate' (line 123)
    rate_57070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 34), 'rate')
    int_57071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 42), 'int')
    # Applying the binary operator '>=' (line 123)
    result_ge_57072 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 34), '>=', rate_57070, int_57071)
    
    
    # Getting the type of 'rate' (line 124)
    rate_57073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'rate')
    # Getting the type of 'NEWTON_MAXITER' (line 124)
    NEWTON_MAXITER_57074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 25), 'NEWTON_MAXITER')
    # Getting the type of 'k' (line 124)
    k_57075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 42), 'k')
    # Applying the binary operator '-' (line 124)
    result_sub_57076 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 25), '-', NEWTON_MAXITER_57074, k_57075)
    
    # Applying the binary operator '**' (line 124)
    result_pow_57077 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 16), '**', rate_57073, result_sub_57076)
    
    int_57078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 48), 'int')
    # Getting the type of 'rate' (line 124)
    rate_57079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 52), 'rate')
    # Applying the binary operator '-' (line 124)
    result_sub_57080 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 48), '-', int_57078, rate_57079)
    
    # Applying the binary operator 'div' (line 124)
    result_div_57081 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 16), 'div', result_pow_57077, result_sub_57080)
    
    # Getting the type of 'dW_norm' (line 124)
    dW_norm_57082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 60), 'dW_norm')
    # Applying the binary operator '*' (line 124)
    result_mul_57083 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 58), '*', result_div_57081, dW_norm_57082)
    
    # Getting the type of 'tol' (line 124)
    tol_57084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 70), 'tol')
    # Applying the binary operator '>' (line 124)
    result_gt_57085 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 16), '>', result_mul_57083, tol_57084)
    
    # Applying the binary operator 'or' (line 123)
    result_or_keyword_57086 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 34), 'or', result_ge_57072, result_gt_57085)
    
    # Applying the binary operator 'and' (line 123)
    result_and_keyword_57087 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 12), 'and', result_is_not_57069, result_or_keyword_57086)
    
    # Testing the type of an if condition (line 123)
    if_condition_57088 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 8), result_and_keyword_57087)
    # Assigning a type to the variable 'if_condition_57088' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'if_condition_57088', if_condition_57088)
    # SSA begins for if statement (line 123)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 123)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'W' (line 127)
    W_57089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'W')
    # Getting the type of 'dW' (line 127)
    dW_57090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 13), 'dW')
    # Applying the binary operator '+=' (line 127)
    result_iadd_57091 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 8), '+=', W_57089, dW_57090)
    # Assigning a type to the variable 'W' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'W', result_iadd_57091)
    
    
    # Assigning a Call to a Name (line 128):
    
    # Assigning a Call to a Name (line 128):
    
    # Call to dot(...): (line 128)
    # Processing the call arguments (line 128)
    # Getting the type of 'W' (line 128)
    W_57094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 18), 'W', False)
    # Processing the call keyword arguments (line 128)
    kwargs_57095 = {}
    # Getting the type of 'T' (line 128)
    T_57092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'T', False)
    # Obtaining the member 'dot' of a type (line 128)
    dot_57093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 12), T_57092, 'dot')
    # Calling dot(args, kwargs) (line 128)
    dot_call_result_57096 = invoke(stypy.reporting.localization.Localization(__file__, 128, 12), dot_57093, *[W_57094], **kwargs_57095)
    
    # Assigning a type to the variable 'Z' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'Z', dot_call_result_57096)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'dW_norm' (line 130)
    dW_norm_57097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'dW_norm')
    int_57098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 23), 'int')
    # Applying the binary operator '==' (line 130)
    result_eq_57099 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 12), '==', dW_norm_57097, int_57098)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'rate' (line 131)
    rate_57100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'rate')
    # Getting the type of 'None' (line 131)
    None_57101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 28), 'None')
    # Applying the binary operator 'isnot' (line 131)
    result_is_not_57102 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 16), 'isnot', rate_57100, None_57101)
    
    
    # Getting the type of 'rate' (line 131)
    rate_57103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 37), 'rate')
    int_57104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 45), 'int')
    # Getting the type of 'rate' (line 131)
    rate_57105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 49), 'rate')
    # Applying the binary operator '-' (line 131)
    result_sub_57106 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 45), '-', int_57104, rate_57105)
    
    # Applying the binary operator 'div' (line 131)
    result_div_57107 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 37), 'div', rate_57103, result_sub_57106)
    
    # Getting the type of 'dW_norm' (line 131)
    dW_norm_57108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 57), 'dW_norm')
    # Applying the binary operator '*' (line 131)
    result_mul_57109 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 55), '*', result_div_57107, dW_norm_57108)
    
    # Getting the type of 'tol' (line 131)
    tol_57110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 67), 'tol')
    # Applying the binary operator '<' (line 131)
    result_lt_57111 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 37), '<', result_mul_57109, tol_57110)
    
    # Applying the binary operator 'and' (line 131)
    result_and_keyword_57112 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 16), 'and', result_is_not_57102, result_lt_57111)
    
    # Applying the binary operator 'or' (line 130)
    result_or_keyword_57113 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 12), 'or', result_eq_57099, result_and_keyword_57112)
    
    # Testing the type of an if condition (line 130)
    if_condition_57114 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 8), result_or_keyword_57113)
    # Assigning a type to the variable 'if_condition_57114' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'if_condition_57114', if_condition_57114)
    # SSA begins for if statement (line 130)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 132):
    
    # Assigning a Name to a Name (line 132):
    # Getting the type of 'True' (line 132)
    True_57115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 24), 'True')
    # Assigning a type to the variable 'converged' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'converged', True_57115)
    # SSA join for if statement (line 130)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 135):
    
    # Assigning a Name to a Name (line 135):
    # Getting the type of 'dW_norm' (line 135)
    dW_norm_57116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 22), 'dW_norm')
    # Assigning a type to the variable 'dW_norm_old' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'dW_norm_old', dW_norm_57116)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 137)
    tuple_57117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 137)
    # Adding element type (line 137)
    # Getting the type of 'converged' (line 137)
    converged_57118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'converged')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 11), tuple_57117, converged_57118)
    # Adding element type (line 137)
    # Getting the type of 'k' (line 137)
    k_57119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 22), 'k')
    int_57120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 26), 'int')
    # Applying the binary operator '+' (line 137)
    result_add_57121 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 22), '+', k_57119, int_57120)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 11), tuple_57117, result_add_57121)
    # Adding element type (line 137)
    # Getting the type of 'Z' (line 137)
    Z_57122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 29), 'Z')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 11), tuple_57117, Z_57122)
    # Adding element type (line 137)
    # Getting the type of 'rate' (line 137)
    rate_57123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 32), 'rate')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 11), tuple_57117, rate_57123)
    
    # Assigning a type to the variable 'stypy_return_type' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'stypy_return_type', tuple_57117)
    
    # ################# End of 'solve_collocation_system(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'solve_collocation_system' in the type store
    # Getting the type of 'stypy_return_type' (line 48)
    stypy_return_type_57124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_57124)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'solve_collocation_system'
    return stypy_return_type_57124

# Assigning a type to the variable 'solve_collocation_system' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'solve_collocation_system', solve_collocation_system)

@norecursion
def predict_factor(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'predict_factor'
    module_type_store = module_type_store.open_function_context('predict_factor', 140, 0, False)
    
    # Passed parameters checking function
    predict_factor.stypy_localization = localization
    predict_factor.stypy_type_of_self = None
    predict_factor.stypy_type_store = module_type_store
    predict_factor.stypy_function_name = 'predict_factor'
    predict_factor.stypy_param_names_list = ['h_abs', 'h_abs_old', 'error_norm', 'error_norm_old']
    predict_factor.stypy_varargs_param_name = None
    predict_factor.stypy_kwargs_param_name = None
    predict_factor.stypy_call_defaults = defaults
    predict_factor.stypy_call_varargs = varargs
    predict_factor.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'predict_factor', ['h_abs', 'h_abs_old', 'error_norm', 'error_norm_old'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'predict_factor', localization, ['h_abs', 'h_abs_old', 'error_norm', 'error_norm_old'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'predict_factor(...)' code ##################

    str_57125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, (-1)), 'str', 'Predict by which factor to increase/decrease the step size.\n\n    The algorithm is described in [1]_.\n\n    Parameters\n    ----------\n    h_abs, h_abs_old : float\n        Current and previous values of the step size, `h_abs_old` can be None\n        (see Notes).\n    error_norm, error_norm_old : float\n        Current and previous values of the error norm, `error_norm_old` can\n        be None (see Notes).\n\n    Returns\n    -------\n    factor : float\n        Predicted factor.\n\n    Notes\n    -----\n    If `h_abs_old` and `error_norm_old` are both not None then a two-step\n    algorithm is used, otherwise a one-step algorithm is used.\n\n    References\n    ----------\n    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential\n           Equations II: Stiff and Differential-Algebraic Problems", Sec. IV.8.\n    ')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'error_norm_old' (line 169)
    error_norm_old_57126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 7), 'error_norm_old')
    # Getting the type of 'None' (line 169)
    None_57127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 25), 'None')
    # Applying the binary operator 'is' (line 169)
    result_is__57128 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 7), 'is', error_norm_old_57126, None_57127)
    
    
    # Getting the type of 'h_abs_old' (line 169)
    h_abs_old_57129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 33), 'h_abs_old')
    # Getting the type of 'None' (line 169)
    None_57130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 46), 'None')
    # Applying the binary operator 'is' (line 169)
    result_is__57131 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 33), 'is', h_abs_old_57129, None_57130)
    
    # Applying the binary operator 'or' (line 169)
    result_or_keyword_57132 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 7), 'or', result_is__57128, result_is__57131)
    
    # Getting the type of 'error_norm' (line 169)
    error_norm_57133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 54), 'error_norm')
    int_57134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 68), 'int')
    # Applying the binary operator '==' (line 169)
    result_eq_57135 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 54), '==', error_norm_57133, int_57134)
    
    # Applying the binary operator 'or' (line 169)
    result_or_keyword_57136 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 7), 'or', result_or_keyword_57132, result_eq_57135)
    
    # Testing the type of an if condition (line 169)
    if_condition_57137 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 4), result_or_keyword_57136)
    # Assigning a type to the variable 'if_condition_57137' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'if_condition_57137', if_condition_57137)
    # SSA begins for if statement (line 169)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 170):
    
    # Assigning a Num to a Name (line 170):
    int_57138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 21), 'int')
    # Assigning a type to the variable 'multiplier' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'multiplier', int_57138)
    # SSA branch for the else part of an if statement (line 169)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 172):
    
    # Assigning a BinOp to a Name (line 172):
    # Getting the type of 'h_abs' (line 172)
    h_abs_57139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 21), 'h_abs')
    # Getting the type of 'h_abs_old' (line 172)
    h_abs_old_57140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 29), 'h_abs_old')
    # Applying the binary operator 'div' (line 172)
    result_div_57141 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 21), 'div', h_abs_57139, h_abs_old_57140)
    
    # Getting the type of 'error_norm_old' (line 172)
    error_norm_old_57142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 42), 'error_norm_old')
    # Getting the type of 'error_norm' (line 172)
    error_norm_57143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 59), 'error_norm')
    # Applying the binary operator 'div' (line 172)
    result_div_57144 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 42), 'div', error_norm_old_57142, error_norm_57143)
    
    float_57145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 74), 'float')
    # Applying the binary operator '**' (line 172)
    result_pow_57146 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 41), '**', result_div_57144, float_57145)
    
    # Applying the binary operator '*' (line 172)
    result_mul_57147 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 39), '*', result_div_57141, result_pow_57146)
    
    # Assigning a type to the variable 'multiplier' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'multiplier', result_mul_57147)
    # SSA join for if statement (line 169)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to errstate(...): (line 174)
    # Processing the call keyword arguments (line 174)
    str_57150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 28), 'str', 'ignore')
    keyword_57151 = str_57150
    kwargs_57152 = {'divide': keyword_57151}
    # Getting the type of 'np' (line 174)
    np_57148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 9), 'np', False)
    # Obtaining the member 'errstate' of a type (line 174)
    errstate_57149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 9), np_57148, 'errstate')
    # Calling errstate(args, kwargs) (line 174)
    errstate_call_result_57153 = invoke(stypy.reporting.localization.Localization(__file__, 174, 9), errstate_57149, *[], **kwargs_57152)
    
    with_57154 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 174, 9), errstate_call_result_57153, 'with parameter', '__enter__', '__exit__')

    if with_57154:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 174)
        enter___57155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 9), errstate_call_result_57153, '__enter__')
        with_enter_57156 = invoke(stypy.reporting.localization.Localization(__file__, 174, 9), enter___57155)
        
        # Assigning a BinOp to a Name (line 175):
        
        # Assigning a BinOp to a Name (line 175):
        
        # Call to min(...): (line 175)
        # Processing the call arguments (line 175)
        int_57158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 21), 'int')
        # Getting the type of 'multiplier' (line 175)
        multiplier_57159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 24), 'multiplier', False)
        # Processing the call keyword arguments (line 175)
        kwargs_57160 = {}
        # Getting the type of 'min' (line 175)
        min_57157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 17), 'min', False)
        # Calling min(args, kwargs) (line 175)
        min_call_result_57161 = invoke(stypy.reporting.localization.Localization(__file__, 175, 17), min_57157, *[int_57158, multiplier_57159], **kwargs_57160)
        
        # Getting the type of 'error_norm' (line 175)
        error_norm_57162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 38), 'error_norm')
        float_57163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 52), 'float')
        # Applying the binary operator '**' (line 175)
        result_pow_57164 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 38), '**', error_norm_57162, float_57163)
        
        # Applying the binary operator '*' (line 175)
        result_mul_57165 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 17), '*', min_call_result_57161, result_pow_57164)
        
        # Assigning a type to the variable 'factor' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'factor', result_mul_57165)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 174)
        exit___57166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 9), errstate_call_result_57153, '__exit__')
        with_exit_57167 = invoke(stypy.reporting.localization.Localization(__file__, 174, 9), exit___57166, None, None, None)

    # Getting the type of 'factor' (line 177)
    factor_57168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 11), 'factor')
    # Assigning a type to the variable 'stypy_return_type' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'stypy_return_type', factor_57168)
    
    # ################# End of 'predict_factor(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'predict_factor' in the type store
    # Getting the type of 'stypy_return_type' (line 140)
    stypy_return_type_57169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_57169)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'predict_factor'
    return stypy_return_type_57169

# Assigning a type to the variable 'predict_factor' (line 140)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 0), 'predict_factor', predict_factor)
# Declaration of the 'Radau' class
# Getting the type of 'OdeSolver' (line 180)
OdeSolver_57170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'OdeSolver')

class Radau(OdeSolver_57170, ):
    str_57171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, (-1)), 'str', 'Implicit Runge-Kutta method of Radau IIA family of order 5.\n\n    Implementation follows [1]_. The error is controlled for a 3rd order\n    accurate embedded formula. A cubic polynomial which satisfies the\n    collocation conditions is used for the dense output.\n\n    Parameters\n    ----------\n    fun : callable\n        Right-hand side of the system. The calling signature is ``fun(t, y)``.\n        Here ``t`` is a scalar and there are two options for ndarray ``y``.\n        It can either have shape (n,), then ``fun`` must return array_like with\n        shape (n,). Or alternatively it can have shape (n, k), then ``fun``\n        must return array_like with shape (n, k), i.e. each column\n        corresponds to a single column in ``y``. The choice between the two\n        options is determined by `vectorized` argument (see below). The\n        vectorized implementation allows faster approximation of the Jacobian\n        by finite differences.\n    t0 : float\n        Initial time.\n    y0 : array_like, shape (n,)\n        Initial state.\n    t_bound : float\n        Boundary time --- the integration won\'t continue beyond it. It also\n        determines the direction of the integration.\n    max_step : float, optional\n        Maximum allowed step size. Default is np.inf, i.e. the step is not\n        bounded and determined solely by the solver.\n    rtol, atol : float and array_like, optional\n        Relative and absolute tolerances. The solver keeps the local error\n        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a\n        relative accuracy (number of correct digits). But if a component of `y`\n        is approximately below `atol` then the error only needs to fall within\n        the same `atol` threshold, and the number of correct digits is not\n        guaranteed. If components of y have different scales, it might be\n        beneficial to set different `atol` values for different components by\n        passing array_like with shape (n,) for `atol`. Default values are\n        1e-3 for `rtol` and 1e-6 for `atol`.\n    jac : {None, array_like, sparse_matrix, callable}, optional\n        Jacobian matrix of the right-hand side of the system with respect to\n        y, required only by \'Radau\' and \'BDF\' methods. The Jacobian matrix\n        has shape (n, n) and its element (i, j) is equal to ``d f_i / d y_j``.\n        There are 3 ways to define the Jacobian:\n\n            * If array_like or sparse_matrix, then the Jacobian is assumed to\n              be constant.\n            * If callable, then the Jacobian is assumed to depend on both\n              t and y, and will be called as ``jac(t, y)`` as necessary. The\n              return value might be a sparse matrix.\n            * If None (default), then the Jacobian will be approximated by\n              finite differences.\n\n        It is generally recommended to provide the Jacobian rather than\n        relying on a finite difference approximation.\n    jac_sparsity : {None, array_like, sparse matrix}, optional\n        Defines a sparsity structure of the Jacobian matrix for a finite\n        difference approximation, its shape must be (n, n). If the Jacobian has\n        only few non-zero elements in *each* row, providing the sparsity\n        structure will greatly speed up the computations [2]_. A zero\n        entry means that a corresponding element in the Jacobian is identically\n        zero. If None (default), the Jacobian is assumed to be dense.\n    vectorized : bool, optional\n        Whether `fun` is implemented in a vectorized fashion. Default is False.\n\n    Attributes\n    ----------\n    n : int\n        Number of equations.\n    status : string\n        Current status of the solver: \'running\', \'finished\' or \'failed\'.\n    t_bound : float\n        Boundary time.\n    direction : float\n        Integration direction: +1 or -1.\n    t : float\n        Current time.\n    y : ndarray\n        Current state.\n    t_old : float\n        Previous time. None if no steps were made yet.\n    step_size : float\n        Size of the last successful step. None if no steps were made yet.\n    nfev : int\n        Number of the system\'s rhs evaluations.\n    njev : int\n        Number of the Jacobian evaluations.\n    nlu : int\n        Number of LU decompositions.\n\n    References\n    ----------\n    .. [1] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:\n           Stiff and Differential-Algebraic Problems", Sec. IV.8.\n    .. [2] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of\n           sparse Jacobian matrices", Journal of the Institute of Mathematics\n           and its Applications, 13, pp. 117-120, 1974.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'np' (line 278)
        np_57172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 54), 'np')
        # Obtaining the member 'inf' of a type (line 278)
        inf_57173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 54), np_57172, 'inf')
        float_57174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 22), 'float')
        float_57175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 33), 'float')
        # Getting the type of 'None' (line 279)
        None_57176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 43), 'None')
        # Getting the type of 'None' (line 279)
        None_57177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 62), 'None')
        # Getting the type of 'False' (line 280)
        False_57178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 28), 'False')
        defaults = [inf_57173, float_57174, float_57175, None_57176, None_57177, False_57178]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 278, 4, False)
        # Assigning a type to the variable 'self' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Radau.__init__', ['fun', 't0', 'y0', 't_bound', 'max_step', 'rtol', 'atol', 'jac', 'jac_sparsity', 'vectorized'], None, 'extraneous', defaults, varargs, kwargs)

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

        
        # Call to warn_extraneous(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 'extraneous' (line 281)
        extraneous_57180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 24), 'extraneous', False)
        # Processing the call keyword arguments (line 281)
        kwargs_57181 = {}
        # Getting the type of 'warn_extraneous' (line 281)
        warn_extraneous_57179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'warn_extraneous', False)
        # Calling warn_extraneous(args, kwargs) (line 281)
        warn_extraneous_call_result_57182 = invoke(stypy.reporting.localization.Localization(__file__, 281, 8), warn_extraneous_57179, *[extraneous_57180], **kwargs_57181)
        
        
        # Call to __init__(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'fun' (line 282)
        fun_57189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 36), 'fun', False)
        # Getting the type of 't0' (line 282)
        t0_57190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 41), 't0', False)
        # Getting the type of 'y0' (line 282)
        y0_57191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 45), 'y0', False)
        # Getting the type of 't_bound' (line 282)
        t_bound_57192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 49), 't_bound', False)
        # Getting the type of 'vectorized' (line 282)
        vectorized_57193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 58), 'vectorized', False)
        # Processing the call keyword arguments (line 282)
        kwargs_57194 = {}
        
        # Call to super(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'Radau' (line 282)
        Radau_57184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 14), 'Radau', False)
        # Getting the type of 'self' (line 282)
        self_57185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 21), 'self', False)
        # Processing the call keyword arguments (line 282)
        kwargs_57186 = {}
        # Getting the type of 'super' (line 282)
        super_57183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'super', False)
        # Calling super(args, kwargs) (line 282)
        super_call_result_57187 = invoke(stypy.reporting.localization.Localization(__file__, 282, 8), super_57183, *[Radau_57184, self_57185], **kwargs_57186)
        
        # Obtaining the member '__init__' of a type (line 282)
        init___57188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 8), super_call_result_57187, '__init__')
        # Calling __init__(args, kwargs) (line 282)
        init___call_result_57195 = invoke(stypy.reporting.localization.Localization(__file__, 282, 8), init___57188, *[fun_57189, t0_57190, y0_57191, t_bound_57192, vectorized_57193], **kwargs_57194)
        
        
        # Assigning a Name to a Attribute (line 283):
        
        # Assigning a Name to a Attribute (line 283):
        # Getting the type of 'None' (line 283)
        None_57196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 21), 'None')
        # Getting the type of 'self' (line 283)
        self_57197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'self')
        # Setting the type of the member 'y_old' of a type (line 283)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), self_57197, 'y_old', None_57196)
        
        # Assigning a Call to a Attribute (line 284):
        
        # Assigning a Call to a Attribute (line 284):
        
        # Call to validate_max_step(...): (line 284)
        # Processing the call arguments (line 284)
        # Getting the type of 'max_step' (line 284)
        max_step_57199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 42), 'max_step', False)
        # Processing the call keyword arguments (line 284)
        kwargs_57200 = {}
        # Getting the type of 'validate_max_step' (line 284)
        validate_max_step_57198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 24), 'validate_max_step', False)
        # Calling validate_max_step(args, kwargs) (line 284)
        validate_max_step_call_result_57201 = invoke(stypy.reporting.localization.Localization(__file__, 284, 24), validate_max_step_57198, *[max_step_57199], **kwargs_57200)
        
        # Getting the type of 'self' (line 284)
        self_57202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'self')
        # Setting the type of the member 'max_step' of a type (line 284)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 8), self_57202, 'max_step', validate_max_step_call_result_57201)
        
        # Assigning a Call to a Tuple (line 285):
        
        # Assigning a Subscript to a Name (line 285):
        
        # Obtaining the type of the subscript
        int_57203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 8), 'int')
        
        # Call to validate_tol(...): (line 285)
        # Processing the call arguments (line 285)
        # Getting the type of 'rtol' (line 285)
        rtol_57205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 44), 'rtol', False)
        # Getting the type of 'atol' (line 285)
        atol_57206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 50), 'atol', False)
        # Getting the type of 'self' (line 285)
        self_57207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 56), 'self', False)
        # Obtaining the member 'n' of a type (line 285)
        n_57208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 56), self_57207, 'n')
        # Processing the call keyword arguments (line 285)
        kwargs_57209 = {}
        # Getting the type of 'validate_tol' (line 285)
        validate_tol_57204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 31), 'validate_tol', False)
        # Calling validate_tol(args, kwargs) (line 285)
        validate_tol_call_result_57210 = invoke(stypy.reporting.localization.Localization(__file__, 285, 31), validate_tol_57204, *[rtol_57205, atol_57206, n_57208], **kwargs_57209)
        
        # Obtaining the member '__getitem__' of a type (line 285)
        getitem___57211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 8), validate_tol_call_result_57210, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 285)
        subscript_call_result_57212 = invoke(stypy.reporting.localization.Localization(__file__, 285, 8), getitem___57211, int_57203)
        
        # Assigning a type to the variable 'tuple_var_assignment_56704' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'tuple_var_assignment_56704', subscript_call_result_57212)
        
        # Assigning a Subscript to a Name (line 285):
        
        # Obtaining the type of the subscript
        int_57213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 8), 'int')
        
        # Call to validate_tol(...): (line 285)
        # Processing the call arguments (line 285)
        # Getting the type of 'rtol' (line 285)
        rtol_57215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 44), 'rtol', False)
        # Getting the type of 'atol' (line 285)
        atol_57216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 50), 'atol', False)
        # Getting the type of 'self' (line 285)
        self_57217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 56), 'self', False)
        # Obtaining the member 'n' of a type (line 285)
        n_57218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 56), self_57217, 'n')
        # Processing the call keyword arguments (line 285)
        kwargs_57219 = {}
        # Getting the type of 'validate_tol' (line 285)
        validate_tol_57214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 31), 'validate_tol', False)
        # Calling validate_tol(args, kwargs) (line 285)
        validate_tol_call_result_57220 = invoke(stypy.reporting.localization.Localization(__file__, 285, 31), validate_tol_57214, *[rtol_57215, atol_57216, n_57218], **kwargs_57219)
        
        # Obtaining the member '__getitem__' of a type (line 285)
        getitem___57221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 8), validate_tol_call_result_57220, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 285)
        subscript_call_result_57222 = invoke(stypy.reporting.localization.Localization(__file__, 285, 8), getitem___57221, int_57213)
        
        # Assigning a type to the variable 'tuple_var_assignment_56705' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'tuple_var_assignment_56705', subscript_call_result_57222)
        
        # Assigning a Name to a Attribute (line 285):
        # Getting the type of 'tuple_var_assignment_56704' (line 285)
        tuple_var_assignment_56704_57223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'tuple_var_assignment_56704')
        # Getting the type of 'self' (line 285)
        self_57224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'self')
        # Setting the type of the member 'rtol' of a type (line 285)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 8), self_57224, 'rtol', tuple_var_assignment_56704_57223)
        
        # Assigning a Name to a Attribute (line 285):
        # Getting the type of 'tuple_var_assignment_56705' (line 285)
        tuple_var_assignment_56705_57225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'tuple_var_assignment_56705')
        # Getting the type of 'self' (line 285)
        self_57226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 19), 'self')
        # Setting the type of the member 'atol' of a type (line 285)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 19), self_57226, 'atol', tuple_var_assignment_56705_57225)
        
        # Assigning a Call to a Attribute (line 286):
        
        # Assigning a Call to a Attribute (line 286):
        
        # Call to fun(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 'self' (line 286)
        self_57229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 26), 'self', False)
        # Obtaining the member 't' of a type (line 286)
        t_57230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 26), self_57229, 't')
        # Getting the type of 'self' (line 286)
        self_57231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 34), 'self', False)
        # Obtaining the member 'y' of a type (line 286)
        y_57232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 34), self_57231, 'y')
        # Processing the call keyword arguments (line 286)
        kwargs_57233 = {}
        # Getting the type of 'self' (line 286)
        self_57227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 17), 'self', False)
        # Obtaining the member 'fun' of a type (line 286)
        fun_57228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 17), self_57227, 'fun')
        # Calling fun(args, kwargs) (line 286)
        fun_call_result_57234 = invoke(stypy.reporting.localization.Localization(__file__, 286, 17), fun_57228, *[t_57230, y_57232], **kwargs_57233)
        
        # Getting the type of 'self' (line 286)
        self_57235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'self')
        # Setting the type of the member 'f' of a type (line 286)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 8), self_57235, 'f', fun_call_result_57234)
        
        # Assigning a Call to a Attribute (line 289):
        
        # Assigning a Call to a Attribute (line 289):
        
        # Call to select_initial_step(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'self' (line 290)
        self_57237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'self', False)
        # Obtaining the member 'fun' of a type (line 290)
        fun_57238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 12), self_57237, 'fun')
        # Getting the type of 'self' (line 290)
        self_57239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 22), 'self', False)
        # Obtaining the member 't' of a type (line 290)
        t_57240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 22), self_57239, 't')
        # Getting the type of 'self' (line 290)
        self_57241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 30), 'self', False)
        # Obtaining the member 'y' of a type (line 290)
        y_57242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 30), self_57241, 'y')
        # Getting the type of 'self' (line 290)
        self_57243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 38), 'self', False)
        # Obtaining the member 'f' of a type (line 290)
        f_57244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 38), self_57243, 'f')
        # Getting the type of 'self' (line 290)
        self_57245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 46), 'self', False)
        # Obtaining the member 'direction' of a type (line 290)
        direction_57246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 46), self_57245, 'direction')
        int_57247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 12), 'int')
        # Getting the type of 'self' (line 291)
        self_57248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 15), 'self', False)
        # Obtaining the member 'rtol' of a type (line 291)
        rtol_57249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 15), self_57248, 'rtol')
        # Getting the type of 'self' (line 291)
        self_57250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 26), 'self', False)
        # Obtaining the member 'atol' of a type (line 291)
        atol_57251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 26), self_57250, 'atol')
        # Processing the call keyword arguments (line 289)
        kwargs_57252 = {}
        # Getting the type of 'select_initial_step' (line 289)
        select_initial_step_57236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 21), 'select_initial_step', False)
        # Calling select_initial_step(args, kwargs) (line 289)
        select_initial_step_call_result_57253 = invoke(stypy.reporting.localization.Localization(__file__, 289, 21), select_initial_step_57236, *[fun_57238, t_57240, y_57242, f_57244, direction_57246, int_57247, rtol_57249, atol_57251], **kwargs_57252)
        
        # Getting the type of 'self' (line 289)
        self_57254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'self')
        # Setting the type of the member 'h_abs' of a type (line 289)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 8), self_57254, 'h_abs', select_initial_step_call_result_57253)
        
        # Assigning a Name to a Attribute (line 292):
        
        # Assigning a Name to a Attribute (line 292):
        # Getting the type of 'None' (line 292)
        None_57255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 25), 'None')
        # Getting the type of 'self' (line 292)
        self_57256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'self')
        # Setting the type of the member 'h_abs_old' of a type (line 292)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), self_57256, 'h_abs_old', None_57255)
        
        # Assigning a Name to a Attribute (line 293):
        
        # Assigning a Name to a Attribute (line 293):
        # Getting the type of 'None' (line 293)
        None_57257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 30), 'None')
        # Getting the type of 'self' (line 293)
        self_57258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'self')
        # Setting the type of the member 'error_norm_old' of a type (line 293)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 8), self_57258, 'error_norm_old', None_57257)
        
        # Assigning a Call to a Attribute (line 295):
        
        # Assigning a Call to a Attribute (line 295):
        
        # Call to max(...): (line 295)
        # Processing the call arguments (line 295)
        int_57260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 30), 'int')
        # Getting the type of 'EPS' (line 295)
        EPS_57261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 35), 'EPS', False)
        # Applying the binary operator '*' (line 295)
        result_mul_57262 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 30), '*', int_57260, EPS_57261)
        
        # Getting the type of 'rtol' (line 295)
        rtol_57263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 41), 'rtol', False)
        # Applying the binary operator 'div' (line 295)
        result_div_57264 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 39), 'div', result_mul_57262, rtol_57263)
        
        
        # Call to min(...): (line 295)
        # Processing the call arguments (line 295)
        float_57266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 51), 'float')
        # Getting the type of 'rtol' (line 295)
        rtol_57267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 57), 'rtol', False)
        float_57268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 65), 'float')
        # Applying the binary operator '**' (line 295)
        result_pow_57269 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 57), '**', rtol_57267, float_57268)
        
        # Processing the call keyword arguments (line 295)
        kwargs_57270 = {}
        # Getting the type of 'min' (line 295)
        min_57265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 47), 'min', False)
        # Calling min(args, kwargs) (line 295)
        min_call_result_57271 = invoke(stypy.reporting.localization.Localization(__file__, 295, 47), min_57265, *[float_57266, result_pow_57269], **kwargs_57270)
        
        # Processing the call keyword arguments (line 295)
        kwargs_57272 = {}
        # Getting the type of 'max' (line 295)
        max_57259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 26), 'max', False)
        # Calling max(args, kwargs) (line 295)
        max_call_result_57273 = invoke(stypy.reporting.localization.Localization(__file__, 295, 26), max_57259, *[result_div_57264, min_call_result_57271], **kwargs_57272)
        
        # Getting the type of 'self' (line 295)
        self_57274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'self')
        # Setting the type of the member 'newton_tol' of a type (line 295)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 8), self_57274, 'newton_tol', max_call_result_57273)
        
        # Assigning a Name to a Attribute (line 296):
        
        # Assigning a Name to a Attribute (line 296):
        # Getting the type of 'None' (line 296)
        None_57275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 19), 'None')
        # Getting the type of 'self' (line 296)
        self_57276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'self')
        # Setting the type of the member 'sol' of a type (line 296)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 8), self_57276, 'sol', None_57275)
        
        # Assigning a Name to a Attribute (line 298):
        
        # Assigning a Name to a Attribute (line 298):
        # Getting the type of 'None' (line 298)
        None_57277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 26), 'None')
        # Getting the type of 'self' (line 298)
        self_57278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'self')
        # Setting the type of the member 'jac_factor' of a type (line 298)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 8), self_57278, 'jac_factor', None_57277)
        
        # Assigning a Call to a Tuple (line 299):
        
        # Assigning a Subscript to a Name (line 299):
        
        # Obtaining the type of the subscript
        int_57279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 8), 'int')
        
        # Call to _validate_jac(...): (line 299)
        # Processing the call arguments (line 299)
        # Getting the type of 'jac' (line 299)
        jac_57282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 46), 'jac', False)
        # Getting the type of 'jac_sparsity' (line 299)
        jac_sparsity_57283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 51), 'jac_sparsity', False)
        # Processing the call keyword arguments (line 299)
        kwargs_57284 = {}
        # Getting the type of 'self' (line 299)
        self_57280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 27), 'self', False)
        # Obtaining the member '_validate_jac' of a type (line 299)
        _validate_jac_57281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 27), self_57280, '_validate_jac')
        # Calling _validate_jac(args, kwargs) (line 299)
        _validate_jac_call_result_57285 = invoke(stypy.reporting.localization.Localization(__file__, 299, 27), _validate_jac_57281, *[jac_57282, jac_sparsity_57283], **kwargs_57284)
        
        # Obtaining the member '__getitem__' of a type (line 299)
        getitem___57286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 8), _validate_jac_call_result_57285, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 299)
        subscript_call_result_57287 = invoke(stypy.reporting.localization.Localization(__file__, 299, 8), getitem___57286, int_57279)
        
        # Assigning a type to the variable 'tuple_var_assignment_56706' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'tuple_var_assignment_56706', subscript_call_result_57287)
        
        # Assigning a Subscript to a Name (line 299):
        
        # Obtaining the type of the subscript
        int_57288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 8), 'int')
        
        # Call to _validate_jac(...): (line 299)
        # Processing the call arguments (line 299)
        # Getting the type of 'jac' (line 299)
        jac_57291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 46), 'jac', False)
        # Getting the type of 'jac_sparsity' (line 299)
        jac_sparsity_57292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 51), 'jac_sparsity', False)
        # Processing the call keyword arguments (line 299)
        kwargs_57293 = {}
        # Getting the type of 'self' (line 299)
        self_57289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 27), 'self', False)
        # Obtaining the member '_validate_jac' of a type (line 299)
        _validate_jac_57290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 27), self_57289, '_validate_jac')
        # Calling _validate_jac(args, kwargs) (line 299)
        _validate_jac_call_result_57294 = invoke(stypy.reporting.localization.Localization(__file__, 299, 27), _validate_jac_57290, *[jac_57291, jac_sparsity_57292], **kwargs_57293)
        
        # Obtaining the member '__getitem__' of a type (line 299)
        getitem___57295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 8), _validate_jac_call_result_57294, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 299)
        subscript_call_result_57296 = invoke(stypy.reporting.localization.Localization(__file__, 299, 8), getitem___57295, int_57288)
        
        # Assigning a type to the variable 'tuple_var_assignment_56707' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'tuple_var_assignment_56707', subscript_call_result_57296)
        
        # Assigning a Name to a Attribute (line 299):
        # Getting the type of 'tuple_var_assignment_56706' (line 299)
        tuple_var_assignment_56706_57297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'tuple_var_assignment_56706')
        # Getting the type of 'self' (line 299)
        self_57298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'self')
        # Setting the type of the member 'jac' of a type (line 299)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 8), self_57298, 'jac', tuple_var_assignment_56706_57297)
        
        # Assigning a Name to a Attribute (line 299):
        # Getting the type of 'tuple_var_assignment_56707' (line 299)
        tuple_var_assignment_56707_57299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'tuple_var_assignment_56707')
        # Getting the type of 'self' (line 299)
        self_57300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 18), 'self')
        # Setting the type of the member 'J' of a type (line 299)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 18), self_57300, 'J', tuple_var_assignment_56707_57299)
        
        
        # Call to issparse(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 'self' (line 300)
        self_57302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 20), 'self', False)
        # Obtaining the member 'J' of a type (line 300)
        J_57303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 20), self_57302, 'J')
        # Processing the call keyword arguments (line 300)
        kwargs_57304 = {}
        # Getting the type of 'issparse' (line 300)
        issparse_57301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 11), 'issparse', False)
        # Calling issparse(args, kwargs) (line 300)
        issparse_call_result_57305 = invoke(stypy.reporting.localization.Localization(__file__, 300, 11), issparse_57301, *[J_57303], **kwargs_57304)
        
        # Testing the type of an if condition (line 300)
        if_condition_57306 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 300, 8), issparse_call_result_57305)
        # Assigning a type to the variable 'if_condition_57306' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'if_condition_57306', if_condition_57306)
        # SSA begins for if statement (line 300)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

        @norecursion
        def lu(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'lu'
            module_type_store = module_type_store.open_function_context('lu', 301, 12, False)
            
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

            
            # Getting the type of 'self' (line 302)
            self_57307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 16), 'self')
            # Obtaining the member 'nlu' of a type (line 302)
            nlu_57308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 16), self_57307, 'nlu')
            int_57309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 28), 'int')
            # Applying the binary operator '+=' (line 302)
            result_iadd_57310 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 16), '+=', nlu_57308, int_57309)
            # Getting the type of 'self' (line 302)
            self_57311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 16), 'self')
            # Setting the type of the member 'nlu' of a type (line 302)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 16), self_57311, 'nlu', result_iadd_57310)
            
            
            # Call to splu(...): (line 303)
            # Processing the call arguments (line 303)
            # Getting the type of 'A' (line 303)
            A_57313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 28), 'A', False)
            # Processing the call keyword arguments (line 303)
            kwargs_57314 = {}
            # Getting the type of 'splu' (line 303)
            splu_57312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 23), 'splu', False)
            # Calling splu(args, kwargs) (line 303)
            splu_call_result_57315 = invoke(stypy.reporting.localization.Localization(__file__, 303, 23), splu_57312, *[A_57313], **kwargs_57314)
            
            # Assigning a type to the variable 'stypy_return_type' (line 303)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'stypy_return_type', splu_call_result_57315)
            
            # ################# End of 'lu(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'lu' in the type store
            # Getting the type of 'stypy_return_type' (line 301)
            stypy_return_type_57316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_57316)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'lu'
            return stypy_return_type_57316

        # Assigning a type to the variable 'lu' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'lu', lu)

        @norecursion
        def solve_lu(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'solve_lu'
            module_type_store = module_type_store.open_function_context('solve_lu', 305, 12, False)
            
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

            
            # Call to solve(...): (line 306)
            # Processing the call arguments (line 306)
            # Getting the type of 'b' (line 306)
            b_57319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 32), 'b', False)
            # Processing the call keyword arguments (line 306)
            kwargs_57320 = {}
            # Getting the type of 'LU' (line 306)
            LU_57317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 23), 'LU', False)
            # Obtaining the member 'solve' of a type (line 306)
            solve_57318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 23), LU_57317, 'solve')
            # Calling solve(args, kwargs) (line 306)
            solve_call_result_57321 = invoke(stypy.reporting.localization.Localization(__file__, 306, 23), solve_57318, *[b_57319], **kwargs_57320)
            
            # Assigning a type to the variable 'stypy_return_type' (line 306)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'stypy_return_type', solve_call_result_57321)
            
            # ################# End of 'solve_lu(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'solve_lu' in the type store
            # Getting the type of 'stypy_return_type' (line 305)
            stypy_return_type_57322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_57322)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'solve_lu'
            return stypy_return_type_57322

        # Assigning a type to the variable 'solve_lu' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'solve_lu', solve_lu)
        
        # Assigning a Call to a Name (line 308):
        
        # Assigning a Call to a Name (line 308):
        
        # Call to eye(...): (line 308)
        # Processing the call arguments (line 308)
        # Getting the type of 'self' (line 308)
        self_57324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 20), 'self', False)
        # Obtaining the member 'n' of a type (line 308)
        n_57325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 20), self_57324, 'n')
        # Processing the call keyword arguments (line 308)
        str_57326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 35), 'str', 'csc')
        keyword_57327 = str_57326
        kwargs_57328 = {'format': keyword_57327}
        # Getting the type of 'eye' (line 308)
        eye_57323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 16), 'eye', False)
        # Calling eye(args, kwargs) (line 308)
        eye_call_result_57329 = invoke(stypy.reporting.localization.Localization(__file__, 308, 16), eye_57323, *[n_57325], **kwargs_57328)
        
        # Assigning a type to the variable 'I' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'I', eye_call_result_57329)
        # SSA branch for the else part of an if statement (line 300)
        module_type_store.open_ssa_branch('else')

        @norecursion
        def lu(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'lu'
            module_type_store = module_type_store.open_function_context('lu', 310, 12, False)
            
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

            
            # Getting the type of 'self' (line 311)
            self_57330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 16), 'self')
            # Obtaining the member 'nlu' of a type (line 311)
            nlu_57331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 16), self_57330, 'nlu')
            int_57332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 28), 'int')
            # Applying the binary operator '+=' (line 311)
            result_iadd_57333 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 16), '+=', nlu_57331, int_57332)
            # Getting the type of 'self' (line 311)
            self_57334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 16), 'self')
            # Setting the type of the member 'nlu' of a type (line 311)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 16), self_57334, 'nlu', result_iadd_57333)
            
            
            # Call to lu_factor(...): (line 312)
            # Processing the call arguments (line 312)
            # Getting the type of 'A' (line 312)
            A_57336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 33), 'A', False)
            # Processing the call keyword arguments (line 312)
            # Getting the type of 'True' (line 312)
            True_57337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 48), 'True', False)
            keyword_57338 = True_57337
            kwargs_57339 = {'overwrite_a': keyword_57338}
            # Getting the type of 'lu_factor' (line 312)
            lu_factor_57335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 23), 'lu_factor', False)
            # Calling lu_factor(args, kwargs) (line 312)
            lu_factor_call_result_57340 = invoke(stypy.reporting.localization.Localization(__file__, 312, 23), lu_factor_57335, *[A_57336], **kwargs_57339)
            
            # Assigning a type to the variable 'stypy_return_type' (line 312)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 16), 'stypy_return_type', lu_factor_call_result_57340)
            
            # ################# End of 'lu(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'lu' in the type store
            # Getting the type of 'stypy_return_type' (line 310)
            stypy_return_type_57341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_57341)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'lu'
            return stypy_return_type_57341

        # Assigning a type to the variable 'lu' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'lu', lu)

        @norecursion
        def solve_lu(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'solve_lu'
            module_type_store = module_type_store.open_function_context('solve_lu', 314, 12, False)
            
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

            
            # Call to lu_solve(...): (line 315)
            # Processing the call arguments (line 315)
            # Getting the type of 'LU' (line 315)
            LU_57343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 32), 'LU', False)
            # Getting the type of 'b' (line 315)
            b_57344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 36), 'b', False)
            # Processing the call keyword arguments (line 315)
            # Getting the type of 'True' (line 315)
            True_57345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 51), 'True', False)
            keyword_57346 = True_57345
            kwargs_57347 = {'overwrite_b': keyword_57346}
            # Getting the type of 'lu_solve' (line 315)
            lu_solve_57342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 23), 'lu_solve', False)
            # Calling lu_solve(args, kwargs) (line 315)
            lu_solve_call_result_57348 = invoke(stypy.reporting.localization.Localization(__file__, 315, 23), lu_solve_57342, *[LU_57343, b_57344], **kwargs_57347)
            
            # Assigning a type to the variable 'stypy_return_type' (line 315)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 16), 'stypy_return_type', lu_solve_call_result_57348)
            
            # ################# End of 'solve_lu(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'solve_lu' in the type store
            # Getting the type of 'stypy_return_type' (line 314)
            stypy_return_type_57349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_57349)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'solve_lu'
            return stypy_return_type_57349

        # Assigning a type to the variable 'solve_lu' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'solve_lu', solve_lu)
        
        # Assigning a Call to a Name (line 317):
        
        # Assigning a Call to a Name (line 317):
        
        # Call to identity(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'self' (line 317)
        self_57352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 28), 'self', False)
        # Obtaining the member 'n' of a type (line 317)
        n_57353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 28), self_57352, 'n')
        # Processing the call keyword arguments (line 317)
        kwargs_57354 = {}
        # Getting the type of 'np' (line 317)
        np_57350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 16), 'np', False)
        # Obtaining the member 'identity' of a type (line 317)
        identity_57351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 16), np_57350, 'identity')
        # Calling identity(args, kwargs) (line 317)
        identity_call_result_57355 = invoke(stypy.reporting.localization.Localization(__file__, 317, 16), identity_57351, *[n_57353], **kwargs_57354)
        
        # Assigning a type to the variable 'I' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'I', identity_call_result_57355)
        # SSA join for if statement (line 300)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 319):
        
        # Assigning a Name to a Attribute (line 319):
        # Getting the type of 'lu' (line 319)
        lu_57356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 18), 'lu')
        # Getting the type of 'self' (line 319)
        self_57357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'self')
        # Setting the type of the member 'lu' of a type (line 319)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 8), self_57357, 'lu', lu_57356)
        
        # Assigning a Name to a Attribute (line 320):
        
        # Assigning a Name to a Attribute (line 320):
        # Getting the type of 'solve_lu' (line 320)
        solve_lu_57358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 24), 'solve_lu')
        # Getting the type of 'self' (line 320)
        self_57359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'self')
        # Setting the type of the member 'solve_lu' of a type (line 320)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 8), self_57359, 'solve_lu', solve_lu_57358)
        
        # Assigning a Name to a Attribute (line 321):
        
        # Assigning a Name to a Attribute (line 321):
        # Getting the type of 'I' (line 321)
        I_57360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 17), 'I')
        # Getting the type of 'self' (line 321)
        self_57361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'self')
        # Setting the type of the member 'I' of a type (line 321)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 8), self_57361, 'I', I_57360)
        
        # Assigning a Name to a Attribute (line 323):
        
        # Assigning a Name to a Attribute (line 323):
        # Getting the type of 'True' (line 323)
        True_57362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 27), 'True')
        # Getting the type of 'self' (line 323)
        self_57363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'self')
        # Setting the type of the member 'current_jac' of a type (line 323)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 8), self_57363, 'current_jac', True_57362)
        
        # Assigning a Name to a Attribute (line 324):
        
        # Assigning a Name to a Attribute (line 324):
        # Getting the type of 'None' (line 324)
        None_57364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 23), 'None')
        # Getting the type of 'self' (line 324)
        self_57365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'self')
        # Setting the type of the member 'LU_real' of a type (line 324)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 8), self_57365, 'LU_real', None_57364)
        
        # Assigning a Name to a Attribute (line 325):
        
        # Assigning a Name to a Attribute (line 325):
        # Getting the type of 'None' (line 325)
        None_57366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 26), 'None')
        # Getting the type of 'self' (line 325)
        self_57367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'self')
        # Setting the type of the member 'LU_complex' of a type (line 325)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 8), self_57367, 'LU_complex', None_57366)
        
        # Assigning a Name to a Attribute (line 326):
        
        # Assigning a Name to a Attribute (line 326):
        # Getting the type of 'None' (line 326)
        None_57368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 17), 'None')
        # Getting the type of 'self' (line 326)
        self_57369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'self')
        # Setting the type of the member 'Z' of a type (line 326)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 8), self_57369, 'Z', None_57368)
        
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
        module_type_store = module_type_store.open_function_context('_validate_jac', 328, 4, False)
        # Assigning a type to the variable 'self' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Radau._validate_jac.__dict__.__setitem__('stypy_localization', localization)
        Radau._validate_jac.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Radau._validate_jac.__dict__.__setitem__('stypy_type_store', module_type_store)
        Radau._validate_jac.__dict__.__setitem__('stypy_function_name', 'Radau._validate_jac')
        Radau._validate_jac.__dict__.__setitem__('stypy_param_names_list', ['jac', 'sparsity'])
        Radau._validate_jac.__dict__.__setitem__('stypy_varargs_param_name', None)
        Radau._validate_jac.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Radau._validate_jac.__dict__.__setitem__('stypy_call_defaults', defaults)
        Radau._validate_jac.__dict__.__setitem__('stypy_call_varargs', varargs)
        Radau._validate_jac.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Radau._validate_jac.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Radau._validate_jac', ['jac', 'sparsity'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Name (line 329):
        
        # Assigning a Attribute to a Name (line 329):
        # Getting the type of 'self' (line 329)
        self_57370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 13), 'self')
        # Obtaining the member 't' of a type (line 329)
        t_57371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 13), self_57370, 't')
        # Assigning a type to the variable 't0' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 't0', t_57371)
        
        # Assigning a Attribute to a Name (line 330):
        
        # Assigning a Attribute to a Name (line 330):
        # Getting the type of 'self' (line 330)
        self_57372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 13), 'self')
        # Obtaining the member 'y' of a type (line 330)
        y_57373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 13), self_57372, 'y')
        # Assigning a type to the variable 'y0' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'y0', y_57373)
        
        # Type idiom detected: calculating its left and rigth part (line 332)
        # Getting the type of 'jac' (line 332)
        jac_57374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 11), 'jac')
        # Getting the type of 'None' (line 332)
        None_57375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 18), 'None')
        
        (may_be_57376, more_types_in_union_57377) = may_be_none(jac_57374, None_57375)

        if may_be_57376:

            if more_types_in_union_57377:
                # Runtime conditional SSA (line 332)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Type idiom detected: calculating its left and rigth part (line 333)
            # Getting the type of 'sparsity' (line 333)
            sparsity_57378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'sparsity')
            # Getting the type of 'None' (line 333)
            None_57379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 31), 'None')
            
            (may_be_57380, more_types_in_union_57381) = may_not_be_none(sparsity_57378, None_57379)

            if may_be_57380:

                if more_types_in_union_57381:
                    # Runtime conditional SSA (line 333)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                
                # Call to issparse(...): (line 334)
                # Processing the call arguments (line 334)
                # Getting the type of 'sparsity' (line 334)
                sparsity_57383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 28), 'sparsity', False)
                # Processing the call keyword arguments (line 334)
                kwargs_57384 = {}
                # Getting the type of 'issparse' (line 334)
                issparse_57382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 19), 'issparse', False)
                # Calling issparse(args, kwargs) (line 334)
                issparse_call_result_57385 = invoke(stypy.reporting.localization.Localization(__file__, 334, 19), issparse_57382, *[sparsity_57383], **kwargs_57384)
                
                # Testing the type of an if condition (line 334)
                if_condition_57386 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 16), issparse_call_result_57385)
                # Assigning a type to the variable 'if_condition_57386' (line 334)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'if_condition_57386', if_condition_57386)
                # SSA begins for if statement (line 334)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 335):
                
                # Assigning a Call to a Name (line 335):
                
                # Call to csc_matrix(...): (line 335)
                # Processing the call arguments (line 335)
                # Getting the type of 'sparsity' (line 335)
                sparsity_57388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 42), 'sparsity', False)
                # Processing the call keyword arguments (line 335)
                kwargs_57389 = {}
                # Getting the type of 'csc_matrix' (line 335)
                csc_matrix_57387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 31), 'csc_matrix', False)
                # Calling csc_matrix(args, kwargs) (line 335)
                csc_matrix_call_result_57390 = invoke(stypy.reporting.localization.Localization(__file__, 335, 31), csc_matrix_57387, *[sparsity_57388], **kwargs_57389)
                
                # Assigning a type to the variable 'sparsity' (line 335)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 20), 'sparsity', csc_matrix_call_result_57390)
                # SSA join for if statement (line 334)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # Assigning a Call to a Name (line 336):
                
                # Assigning a Call to a Name (line 336):
                
                # Call to group_columns(...): (line 336)
                # Processing the call arguments (line 336)
                # Getting the type of 'sparsity' (line 336)
                sparsity_57392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 39), 'sparsity', False)
                # Processing the call keyword arguments (line 336)
                kwargs_57393 = {}
                # Getting the type of 'group_columns' (line 336)
                group_columns_57391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 25), 'group_columns', False)
                # Calling group_columns(args, kwargs) (line 336)
                group_columns_call_result_57394 = invoke(stypy.reporting.localization.Localization(__file__, 336, 25), group_columns_57391, *[sparsity_57392], **kwargs_57393)
                
                # Assigning a type to the variable 'groups' (line 336)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 16), 'groups', group_columns_call_result_57394)
                
                # Assigning a Tuple to a Name (line 337):
                
                # Assigning a Tuple to a Name (line 337):
                
                # Obtaining an instance of the builtin type 'tuple' (line 337)
                tuple_57395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 28), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 337)
                # Adding element type (line 337)
                # Getting the type of 'sparsity' (line 337)
                sparsity_57396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 28), 'sparsity')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 28), tuple_57395, sparsity_57396)
                # Adding element type (line 337)
                # Getting the type of 'groups' (line 337)
                groups_57397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 38), 'groups')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 28), tuple_57395, groups_57397)
                
                # Assigning a type to the variable 'sparsity' (line 337)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 16), 'sparsity', tuple_57395)

                if more_types_in_union_57381:
                    # SSA join for if statement (line 333)
                    module_type_store = module_type_store.join_ssa_context()


            

            @norecursion
            def jac_wrapped(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'jac_wrapped'
                module_type_store = module_type_store.open_function_context('jac_wrapped', 339, 12, False)
                
                # Passed parameters checking function
                jac_wrapped.stypy_localization = localization
                jac_wrapped.stypy_type_of_self = None
                jac_wrapped.stypy_type_store = module_type_store
                jac_wrapped.stypy_function_name = 'jac_wrapped'
                jac_wrapped.stypy_param_names_list = ['t', 'y', 'f']
                jac_wrapped.stypy_varargs_param_name = None
                jac_wrapped.stypy_kwargs_param_name = None
                jac_wrapped.stypy_call_defaults = defaults
                jac_wrapped.stypy_call_varargs = varargs
                jac_wrapped.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, 'jac_wrapped', ['t', 'y', 'f'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'jac_wrapped', localization, ['t', 'y', 'f'], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'jac_wrapped(...)' code ##################

                
                # Getting the type of 'self' (line 340)
                self_57398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 16), 'self')
                # Obtaining the member 'njev' of a type (line 340)
                njev_57399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 16), self_57398, 'njev')
                int_57400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 29), 'int')
                # Applying the binary operator '+=' (line 340)
                result_iadd_57401 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 16), '+=', njev_57399, int_57400)
                # Getting the type of 'self' (line 340)
                self_57402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 16), 'self')
                # Setting the type of the member 'njev' of a type (line 340)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 16), self_57402, 'njev', result_iadd_57401)
                
                
                # Assigning a Call to a Tuple (line 341):
                
                # Assigning a Subscript to a Name (line 341):
                
                # Obtaining the type of the subscript
                int_57403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 16), 'int')
                
                # Call to num_jac(...): (line 341)
                # Processing the call arguments (line 341)
                # Getting the type of 'self' (line 341)
                self_57405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 45), 'self', False)
                # Obtaining the member 'fun_vectorized' of a type (line 341)
                fun_vectorized_57406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 45), self_57405, 'fun_vectorized')
                # Getting the type of 't' (line 341)
                t_57407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 66), 't', False)
                # Getting the type of 'y' (line 341)
                y_57408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 69), 'y', False)
                # Getting the type of 'f' (line 341)
                f_57409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 72), 'f', False)
                # Getting the type of 'self' (line 342)
                self_57410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 45), 'self', False)
                # Obtaining the member 'atol' of a type (line 342)
                atol_57411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 45), self_57410, 'atol')
                # Getting the type of 'self' (line 342)
                self_57412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 56), 'self', False)
                # Obtaining the member 'jac_factor' of a type (line 342)
                jac_factor_57413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 56), self_57412, 'jac_factor')
                # Getting the type of 'sparsity' (line 343)
                sparsity_57414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 45), 'sparsity', False)
                # Processing the call keyword arguments (line 341)
                kwargs_57415 = {}
                # Getting the type of 'num_jac' (line 341)
                num_jac_57404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 37), 'num_jac', False)
                # Calling num_jac(args, kwargs) (line 341)
                num_jac_call_result_57416 = invoke(stypy.reporting.localization.Localization(__file__, 341, 37), num_jac_57404, *[fun_vectorized_57406, t_57407, y_57408, f_57409, atol_57411, jac_factor_57413, sparsity_57414], **kwargs_57415)
                
                # Obtaining the member '__getitem__' of a type (line 341)
                getitem___57417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 16), num_jac_call_result_57416, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 341)
                subscript_call_result_57418 = invoke(stypy.reporting.localization.Localization(__file__, 341, 16), getitem___57417, int_57403)
                
                # Assigning a type to the variable 'tuple_var_assignment_56708' (line 341)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 16), 'tuple_var_assignment_56708', subscript_call_result_57418)
                
                # Assigning a Subscript to a Name (line 341):
                
                # Obtaining the type of the subscript
                int_57419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 16), 'int')
                
                # Call to num_jac(...): (line 341)
                # Processing the call arguments (line 341)
                # Getting the type of 'self' (line 341)
                self_57421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 45), 'self', False)
                # Obtaining the member 'fun_vectorized' of a type (line 341)
                fun_vectorized_57422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 45), self_57421, 'fun_vectorized')
                # Getting the type of 't' (line 341)
                t_57423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 66), 't', False)
                # Getting the type of 'y' (line 341)
                y_57424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 69), 'y', False)
                # Getting the type of 'f' (line 341)
                f_57425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 72), 'f', False)
                # Getting the type of 'self' (line 342)
                self_57426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 45), 'self', False)
                # Obtaining the member 'atol' of a type (line 342)
                atol_57427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 45), self_57426, 'atol')
                # Getting the type of 'self' (line 342)
                self_57428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 56), 'self', False)
                # Obtaining the member 'jac_factor' of a type (line 342)
                jac_factor_57429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 56), self_57428, 'jac_factor')
                # Getting the type of 'sparsity' (line 343)
                sparsity_57430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 45), 'sparsity', False)
                # Processing the call keyword arguments (line 341)
                kwargs_57431 = {}
                # Getting the type of 'num_jac' (line 341)
                num_jac_57420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 37), 'num_jac', False)
                # Calling num_jac(args, kwargs) (line 341)
                num_jac_call_result_57432 = invoke(stypy.reporting.localization.Localization(__file__, 341, 37), num_jac_57420, *[fun_vectorized_57422, t_57423, y_57424, f_57425, atol_57427, jac_factor_57429, sparsity_57430], **kwargs_57431)
                
                # Obtaining the member '__getitem__' of a type (line 341)
                getitem___57433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 16), num_jac_call_result_57432, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 341)
                subscript_call_result_57434 = invoke(stypy.reporting.localization.Localization(__file__, 341, 16), getitem___57433, int_57419)
                
                # Assigning a type to the variable 'tuple_var_assignment_56709' (line 341)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 16), 'tuple_var_assignment_56709', subscript_call_result_57434)
                
                # Assigning a Name to a Name (line 341):
                # Getting the type of 'tuple_var_assignment_56708' (line 341)
                tuple_var_assignment_56708_57435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 16), 'tuple_var_assignment_56708')
                # Assigning a type to the variable 'J' (line 341)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 16), 'J', tuple_var_assignment_56708_57435)
                
                # Assigning a Name to a Attribute (line 341):
                # Getting the type of 'tuple_var_assignment_56709' (line 341)
                tuple_var_assignment_56709_57436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 16), 'tuple_var_assignment_56709')
                # Getting the type of 'self' (line 341)
                self_57437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 19), 'self')
                # Setting the type of the member 'jac_factor' of a type (line 341)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 19), self_57437, 'jac_factor', tuple_var_assignment_56709_57436)
                # Getting the type of 'J' (line 344)
                J_57438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 23), 'J')
                # Assigning a type to the variable 'stypy_return_type' (line 344)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 16), 'stypy_return_type', J_57438)
                
                # ################# End of 'jac_wrapped(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'jac_wrapped' in the type store
                # Getting the type of 'stypy_return_type' (line 339)
                stypy_return_type_57439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_57439)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'jac_wrapped'
                return stypy_return_type_57439

            # Assigning a type to the variable 'jac_wrapped' (line 339)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'jac_wrapped', jac_wrapped)
            
            # Assigning a Call to a Name (line 345):
            
            # Assigning a Call to a Name (line 345):
            
            # Call to jac_wrapped(...): (line 345)
            # Processing the call arguments (line 345)
            # Getting the type of 't0' (line 345)
            t0_57441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 28), 't0', False)
            # Getting the type of 'y0' (line 345)
            y0_57442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 32), 'y0', False)
            # Getting the type of 'self' (line 345)
            self_57443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 36), 'self', False)
            # Obtaining the member 'f' of a type (line 345)
            f_57444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 36), self_57443, 'f')
            # Processing the call keyword arguments (line 345)
            kwargs_57445 = {}
            # Getting the type of 'jac_wrapped' (line 345)
            jac_wrapped_57440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 16), 'jac_wrapped', False)
            # Calling jac_wrapped(args, kwargs) (line 345)
            jac_wrapped_call_result_57446 = invoke(stypy.reporting.localization.Localization(__file__, 345, 16), jac_wrapped_57440, *[t0_57441, y0_57442, f_57444], **kwargs_57445)
            
            # Assigning a type to the variable 'J' (line 345)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'J', jac_wrapped_call_result_57446)

            if more_types_in_union_57377:
                # Runtime conditional SSA for else branch (line 332)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_57376) or more_types_in_union_57377):
            
            
            # Call to callable(...): (line 346)
            # Processing the call arguments (line 346)
            # Getting the type of 'jac' (line 346)
            jac_57448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 22), 'jac', False)
            # Processing the call keyword arguments (line 346)
            kwargs_57449 = {}
            # Getting the type of 'callable' (line 346)
            callable_57447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 13), 'callable', False)
            # Calling callable(args, kwargs) (line 346)
            callable_call_result_57450 = invoke(stypy.reporting.localization.Localization(__file__, 346, 13), callable_57447, *[jac_57448], **kwargs_57449)
            
            # Testing the type of an if condition (line 346)
            if_condition_57451 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 346, 13), callable_call_result_57450)
            # Assigning a type to the variable 'if_condition_57451' (line 346)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 13), 'if_condition_57451', if_condition_57451)
            # SSA begins for if statement (line 346)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 347):
            
            # Assigning a Call to a Name (line 347):
            
            # Call to jac(...): (line 347)
            # Processing the call arguments (line 347)
            # Getting the type of 't0' (line 347)
            t0_57453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 20), 't0', False)
            # Getting the type of 'y0' (line 347)
            y0_57454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 24), 'y0', False)
            # Processing the call keyword arguments (line 347)
            kwargs_57455 = {}
            # Getting the type of 'jac' (line 347)
            jac_57452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 16), 'jac', False)
            # Calling jac(args, kwargs) (line 347)
            jac_call_result_57456 = invoke(stypy.reporting.localization.Localization(__file__, 347, 16), jac_57452, *[t0_57453, y0_57454], **kwargs_57455)
            
            # Assigning a type to the variable 'J' (line 347)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'J', jac_call_result_57456)
            
            # Assigning a Num to a Attribute (line 348):
            
            # Assigning a Num to a Attribute (line 348):
            int_57457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 24), 'int')
            # Getting the type of 'self' (line 348)
            self_57458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'self')
            # Setting the type of the member 'njev' of a type (line 348)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 12), self_57458, 'njev', int_57457)
            
            
            # Call to issparse(...): (line 349)
            # Processing the call arguments (line 349)
            # Getting the type of 'J' (line 349)
            J_57460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 24), 'J', False)
            # Processing the call keyword arguments (line 349)
            kwargs_57461 = {}
            # Getting the type of 'issparse' (line 349)
            issparse_57459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 15), 'issparse', False)
            # Calling issparse(args, kwargs) (line 349)
            issparse_call_result_57462 = invoke(stypy.reporting.localization.Localization(__file__, 349, 15), issparse_57459, *[J_57460], **kwargs_57461)
            
            # Testing the type of an if condition (line 349)
            if_condition_57463 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 349, 12), issparse_call_result_57462)
            # Assigning a type to the variable 'if_condition_57463' (line 349)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'if_condition_57463', if_condition_57463)
            # SSA begins for if statement (line 349)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 350):
            
            # Assigning a Call to a Name (line 350):
            
            # Call to csc_matrix(...): (line 350)
            # Processing the call arguments (line 350)
            # Getting the type of 'J' (line 350)
            J_57465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 31), 'J', False)
            # Processing the call keyword arguments (line 350)
            kwargs_57466 = {}
            # Getting the type of 'csc_matrix' (line 350)
            csc_matrix_57464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 20), 'csc_matrix', False)
            # Calling csc_matrix(args, kwargs) (line 350)
            csc_matrix_call_result_57467 = invoke(stypy.reporting.localization.Localization(__file__, 350, 20), csc_matrix_57464, *[J_57465], **kwargs_57466)
            
            # Assigning a type to the variable 'J' (line 350)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 16), 'J', csc_matrix_call_result_57467)

            @norecursion
            def jac_wrapped(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                # Getting the type of 'None' (line 352)
                None_57468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 40), 'None')
                defaults = [None_57468]
                # Create a new context for function 'jac_wrapped'
                module_type_store = module_type_store.open_function_context('jac_wrapped', 352, 16, False)
                
                # Passed parameters checking function
                jac_wrapped.stypy_localization = localization
                jac_wrapped.stypy_type_of_self = None
                jac_wrapped.stypy_type_store = module_type_store
                jac_wrapped.stypy_function_name = 'jac_wrapped'
                jac_wrapped.stypy_param_names_list = ['t', 'y', '_']
                jac_wrapped.stypy_varargs_param_name = None
                jac_wrapped.stypy_kwargs_param_name = None
                jac_wrapped.stypy_call_defaults = defaults
                jac_wrapped.stypy_call_varargs = varargs
                jac_wrapped.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, 'jac_wrapped', ['t', 'y', '_'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'jac_wrapped', localization, ['t', 'y', '_'], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'jac_wrapped(...)' code ##################

                
                # Getting the type of 'self' (line 353)
                self_57469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 20), 'self')
                # Obtaining the member 'njev' of a type (line 353)
                njev_57470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 20), self_57469, 'njev')
                int_57471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 33), 'int')
                # Applying the binary operator '+=' (line 353)
                result_iadd_57472 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 20), '+=', njev_57470, int_57471)
                # Getting the type of 'self' (line 353)
                self_57473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 20), 'self')
                # Setting the type of the member 'njev' of a type (line 353)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 20), self_57473, 'njev', result_iadd_57472)
                
                
                # Call to csc_matrix(...): (line 354)
                # Processing the call arguments (line 354)
                
                # Call to jac(...): (line 354)
                # Processing the call arguments (line 354)
                # Getting the type of 't' (line 354)
                t_57476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 42), 't', False)
                # Getting the type of 'y' (line 354)
                y_57477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 45), 'y', False)
                # Processing the call keyword arguments (line 354)
                kwargs_57478 = {}
                # Getting the type of 'jac' (line 354)
                jac_57475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 38), 'jac', False)
                # Calling jac(args, kwargs) (line 354)
                jac_call_result_57479 = invoke(stypy.reporting.localization.Localization(__file__, 354, 38), jac_57475, *[t_57476, y_57477], **kwargs_57478)
                
                # Processing the call keyword arguments (line 354)
                # Getting the type of 'float' (line 354)
                float_57480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 55), 'float', False)
                keyword_57481 = float_57480
                kwargs_57482 = {'dtype': keyword_57481}
                # Getting the type of 'csc_matrix' (line 354)
                csc_matrix_57474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 27), 'csc_matrix', False)
                # Calling csc_matrix(args, kwargs) (line 354)
                csc_matrix_call_result_57483 = invoke(stypy.reporting.localization.Localization(__file__, 354, 27), csc_matrix_57474, *[jac_call_result_57479], **kwargs_57482)
                
                # Assigning a type to the variable 'stypy_return_type' (line 354)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 20), 'stypy_return_type', csc_matrix_call_result_57483)
                
                # ################# End of 'jac_wrapped(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'jac_wrapped' in the type store
                # Getting the type of 'stypy_return_type' (line 352)
                stypy_return_type_57484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 16), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_57484)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'jac_wrapped'
                return stypy_return_type_57484

            # Assigning a type to the variable 'jac_wrapped' (line 352)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 16), 'jac_wrapped', jac_wrapped)
            # SSA branch for the else part of an if statement (line 349)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 357):
            
            # Assigning a Call to a Name (line 357):
            
            # Call to asarray(...): (line 357)
            # Processing the call arguments (line 357)
            # Getting the type of 'J' (line 357)
            J_57487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 31), 'J', False)
            # Processing the call keyword arguments (line 357)
            # Getting the type of 'float' (line 357)
            float_57488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 40), 'float', False)
            keyword_57489 = float_57488
            kwargs_57490 = {'dtype': keyword_57489}
            # Getting the type of 'np' (line 357)
            np_57485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 20), 'np', False)
            # Obtaining the member 'asarray' of a type (line 357)
            asarray_57486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 20), np_57485, 'asarray')
            # Calling asarray(args, kwargs) (line 357)
            asarray_call_result_57491 = invoke(stypy.reporting.localization.Localization(__file__, 357, 20), asarray_57486, *[J_57487], **kwargs_57490)
            
            # Assigning a type to the variable 'J' (line 357)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 16), 'J', asarray_call_result_57491)

            @norecursion
            def jac_wrapped(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                # Getting the type of 'None' (line 359)
                None_57492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 40), 'None')
                defaults = [None_57492]
                # Create a new context for function 'jac_wrapped'
                module_type_store = module_type_store.open_function_context('jac_wrapped', 359, 16, False)
                
                # Passed parameters checking function
                jac_wrapped.stypy_localization = localization
                jac_wrapped.stypy_type_of_self = None
                jac_wrapped.stypy_type_store = module_type_store
                jac_wrapped.stypy_function_name = 'jac_wrapped'
                jac_wrapped.stypy_param_names_list = ['t', 'y', '_']
                jac_wrapped.stypy_varargs_param_name = None
                jac_wrapped.stypy_kwargs_param_name = None
                jac_wrapped.stypy_call_defaults = defaults
                jac_wrapped.stypy_call_varargs = varargs
                jac_wrapped.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, 'jac_wrapped', ['t', 'y', '_'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'jac_wrapped', localization, ['t', 'y', '_'], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'jac_wrapped(...)' code ##################

                
                # Getting the type of 'self' (line 360)
                self_57493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 20), 'self')
                # Obtaining the member 'njev' of a type (line 360)
                njev_57494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 20), self_57493, 'njev')
                int_57495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 33), 'int')
                # Applying the binary operator '+=' (line 360)
                result_iadd_57496 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 20), '+=', njev_57494, int_57495)
                # Getting the type of 'self' (line 360)
                self_57497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 20), 'self')
                # Setting the type of the member 'njev' of a type (line 360)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 20), self_57497, 'njev', result_iadd_57496)
                
                
                # Call to asarray(...): (line 361)
                # Processing the call arguments (line 361)
                
                # Call to jac(...): (line 361)
                # Processing the call arguments (line 361)
                # Getting the type of 't' (line 361)
                t_57501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 42), 't', False)
                # Getting the type of 'y' (line 361)
                y_57502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 45), 'y', False)
                # Processing the call keyword arguments (line 361)
                kwargs_57503 = {}
                # Getting the type of 'jac' (line 361)
                jac_57500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 38), 'jac', False)
                # Calling jac(args, kwargs) (line 361)
                jac_call_result_57504 = invoke(stypy.reporting.localization.Localization(__file__, 361, 38), jac_57500, *[t_57501, y_57502], **kwargs_57503)
                
                # Processing the call keyword arguments (line 361)
                # Getting the type of 'float' (line 361)
                float_57505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 55), 'float', False)
                keyword_57506 = float_57505
                kwargs_57507 = {'dtype': keyword_57506}
                # Getting the type of 'np' (line 361)
                np_57498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 27), 'np', False)
                # Obtaining the member 'asarray' of a type (line 361)
                asarray_57499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 27), np_57498, 'asarray')
                # Calling asarray(args, kwargs) (line 361)
                asarray_call_result_57508 = invoke(stypy.reporting.localization.Localization(__file__, 361, 27), asarray_57499, *[jac_call_result_57504], **kwargs_57507)
                
                # Assigning a type to the variable 'stypy_return_type' (line 361)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 20), 'stypy_return_type', asarray_call_result_57508)
                
                # ################# End of 'jac_wrapped(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'jac_wrapped' in the type store
                # Getting the type of 'stypy_return_type' (line 359)
                stypy_return_type_57509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 16), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_57509)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'jac_wrapped'
                return stypy_return_type_57509

            # Assigning a type to the variable 'jac_wrapped' (line 359)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 16), 'jac_wrapped', jac_wrapped)
            # SSA join for if statement (line 349)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Getting the type of 'J' (line 363)
            J_57510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 15), 'J')
            # Obtaining the member 'shape' of a type (line 363)
            shape_57511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 15), J_57510, 'shape')
            
            # Obtaining an instance of the builtin type 'tuple' (line 363)
            tuple_57512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 27), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 363)
            # Adding element type (line 363)
            # Getting the type of 'self' (line 363)
            self_57513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 27), 'self')
            # Obtaining the member 'n' of a type (line 363)
            n_57514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 27), self_57513, 'n')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 27), tuple_57512, n_57514)
            # Adding element type (line 363)
            # Getting the type of 'self' (line 363)
            self_57515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 35), 'self')
            # Obtaining the member 'n' of a type (line 363)
            n_57516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 35), self_57515, 'n')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 27), tuple_57512, n_57516)
            
            # Applying the binary operator '!=' (line 363)
            result_ne_57517 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 15), '!=', shape_57511, tuple_57512)
            
            # Testing the type of an if condition (line 363)
            if_condition_57518 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 363, 12), result_ne_57517)
            # Assigning a type to the variable 'if_condition_57518' (line 363)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'if_condition_57518', if_condition_57518)
            # SSA begins for if statement (line 363)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 364)
            # Processing the call arguments (line 364)
            
            # Call to format(...): (line 364)
            # Processing the call arguments (line 364)
            
            # Obtaining an instance of the builtin type 'tuple' (line 366)
            tuple_57522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 42), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 366)
            # Adding element type (line 366)
            # Getting the type of 'self' (line 366)
            self_57523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 42), 'self', False)
            # Obtaining the member 'n' of a type (line 366)
            n_57524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 42), self_57523, 'n')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 42), tuple_57522, n_57524)
            # Adding element type (line 366)
            # Getting the type of 'self' (line 366)
            self_57525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 50), 'self', False)
            # Obtaining the member 'n' of a type (line 366)
            n_57526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 50), self_57525, 'n')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 42), tuple_57522, n_57526)
            
            # Getting the type of 'J' (line 366)
            J_57527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 59), 'J', False)
            # Obtaining the member 'shape' of a type (line 366)
            shape_57528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 59), J_57527, 'shape')
            # Processing the call keyword arguments (line 364)
            kwargs_57529 = {}
            str_57520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 33), 'str', '`jac` is expected to have shape {}, but actually has {}.')
            # Obtaining the member 'format' of a type (line 364)
            format_57521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 33), str_57520, 'format')
            # Calling format(args, kwargs) (line 364)
            format_call_result_57530 = invoke(stypy.reporting.localization.Localization(__file__, 364, 33), format_57521, *[tuple_57522, shape_57528], **kwargs_57529)
            
            # Processing the call keyword arguments (line 364)
            kwargs_57531 = {}
            # Getting the type of 'ValueError' (line 364)
            ValueError_57519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 364)
            ValueError_call_result_57532 = invoke(stypy.reporting.localization.Localization(__file__, 364, 22), ValueError_57519, *[format_call_result_57530], **kwargs_57531)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 364, 16), ValueError_call_result_57532, 'raise parameter', BaseException)
            # SSA join for if statement (line 363)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 346)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to issparse(...): (line 368)
            # Processing the call arguments (line 368)
            # Getting the type of 'jac' (line 368)
            jac_57534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 24), 'jac', False)
            # Processing the call keyword arguments (line 368)
            kwargs_57535 = {}
            # Getting the type of 'issparse' (line 368)
            issparse_57533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 15), 'issparse', False)
            # Calling issparse(args, kwargs) (line 368)
            issparse_call_result_57536 = invoke(stypy.reporting.localization.Localization(__file__, 368, 15), issparse_57533, *[jac_57534], **kwargs_57535)
            
            # Testing the type of an if condition (line 368)
            if_condition_57537 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 368, 12), issparse_call_result_57536)
            # Assigning a type to the variable 'if_condition_57537' (line 368)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'if_condition_57537', if_condition_57537)
            # SSA begins for if statement (line 368)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 369):
            
            # Assigning a Call to a Name (line 369):
            
            # Call to csc_matrix(...): (line 369)
            # Processing the call arguments (line 369)
            # Getting the type of 'jac' (line 369)
            jac_57539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 31), 'jac', False)
            # Processing the call keyword arguments (line 369)
            kwargs_57540 = {}
            # Getting the type of 'csc_matrix' (line 369)
            csc_matrix_57538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 20), 'csc_matrix', False)
            # Calling csc_matrix(args, kwargs) (line 369)
            csc_matrix_call_result_57541 = invoke(stypy.reporting.localization.Localization(__file__, 369, 20), csc_matrix_57538, *[jac_57539], **kwargs_57540)
            
            # Assigning a type to the variable 'J' (line 369)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 16), 'J', csc_matrix_call_result_57541)
            # SSA branch for the else part of an if statement (line 368)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 371):
            
            # Assigning a Call to a Name (line 371):
            
            # Call to asarray(...): (line 371)
            # Processing the call arguments (line 371)
            # Getting the type of 'jac' (line 371)
            jac_57544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 31), 'jac', False)
            # Processing the call keyword arguments (line 371)
            # Getting the type of 'float' (line 371)
            float_57545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 42), 'float', False)
            keyword_57546 = float_57545
            kwargs_57547 = {'dtype': keyword_57546}
            # Getting the type of 'np' (line 371)
            np_57542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 20), 'np', False)
            # Obtaining the member 'asarray' of a type (line 371)
            asarray_57543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 20), np_57542, 'asarray')
            # Calling asarray(args, kwargs) (line 371)
            asarray_call_result_57548 = invoke(stypy.reporting.localization.Localization(__file__, 371, 20), asarray_57543, *[jac_57544], **kwargs_57547)
            
            # Assigning a type to the variable 'J' (line 371)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 16), 'J', asarray_call_result_57548)
            # SSA join for if statement (line 368)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Getting the type of 'J' (line 373)
            J_57549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 15), 'J')
            # Obtaining the member 'shape' of a type (line 373)
            shape_57550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 15), J_57549, 'shape')
            
            # Obtaining an instance of the builtin type 'tuple' (line 373)
            tuple_57551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 27), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 373)
            # Adding element type (line 373)
            # Getting the type of 'self' (line 373)
            self_57552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 27), 'self')
            # Obtaining the member 'n' of a type (line 373)
            n_57553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 27), self_57552, 'n')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 27), tuple_57551, n_57553)
            # Adding element type (line 373)
            # Getting the type of 'self' (line 373)
            self_57554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 35), 'self')
            # Obtaining the member 'n' of a type (line 373)
            n_57555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 35), self_57554, 'n')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 27), tuple_57551, n_57555)
            
            # Applying the binary operator '!=' (line 373)
            result_ne_57556 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 15), '!=', shape_57550, tuple_57551)
            
            # Testing the type of an if condition (line 373)
            if_condition_57557 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 373, 12), result_ne_57556)
            # Assigning a type to the variable 'if_condition_57557' (line 373)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'if_condition_57557', if_condition_57557)
            # SSA begins for if statement (line 373)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 374)
            # Processing the call arguments (line 374)
            
            # Call to format(...): (line 374)
            # Processing the call arguments (line 374)
            
            # Obtaining an instance of the builtin type 'tuple' (line 376)
            tuple_57561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 42), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 376)
            # Adding element type (line 376)
            # Getting the type of 'self' (line 376)
            self_57562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 42), 'self', False)
            # Obtaining the member 'n' of a type (line 376)
            n_57563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 42), self_57562, 'n')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 42), tuple_57561, n_57563)
            # Adding element type (line 376)
            # Getting the type of 'self' (line 376)
            self_57564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 50), 'self', False)
            # Obtaining the member 'n' of a type (line 376)
            n_57565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 50), self_57564, 'n')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 42), tuple_57561, n_57565)
            
            # Getting the type of 'J' (line 376)
            J_57566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 59), 'J', False)
            # Obtaining the member 'shape' of a type (line 376)
            shape_57567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 59), J_57566, 'shape')
            # Processing the call keyword arguments (line 374)
            kwargs_57568 = {}
            str_57559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 33), 'str', '`jac` is expected to have shape {}, but actually has {}.')
            # Obtaining the member 'format' of a type (line 374)
            format_57560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 33), str_57559, 'format')
            # Calling format(args, kwargs) (line 374)
            format_call_result_57569 = invoke(stypy.reporting.localization.Localization(__file__, 374, 33), format_57560, *[tuple_57561, shape_57567], **kwargs_57568)
            
            # Processing the call keyword arguments (line 374)
            kwargs_57570 = {}
            # Getting the type of 'ValueError' (line 374)
            ValueError_57558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 374)
            ValueError_call_result_57571 = invoke(stypy.reporting.localization.Localization(__file__, 374, 22), ValueError_57558, *[format_call_result_57569], **kwargs_57570)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 374, 16), ValueError_call_result_57571, 'raise parameter', BaseException)
            # SSA join for if statement (line 373)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Name to a Name (line 377):
            
            # Assigning a Name to a Name (line 377):
            # Getting the type of 'None' (line 377)
            None_57572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 26), 'None')
            # Assigning a type to the variable 'jac_wrapped' (line 377)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 12), 'jac_wrapped', None_57572)
            # SSA join for if statement (line 346)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_57376 and more_types_in_union_57377):
                # SSA join for if statement (line 332)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Obtaining an instance of the builtin type 'tuple' (line 379)
        tuple_57573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 379)
        # Adding element type (line 379)
        # Getting the type of 'jac_wrapped' (line 379)
        jac_wrapped_57574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 15), 'jac_wrapped')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 15), tuple_57573, jac_wrapped_57574)
        # Adding element type (line 379)
        # Getting the type of 'J' (line 379)
        J_57575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 28), 'J')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 15), tuple_57573, J_57575)
        
        # Assigning a type to the variable 'stypy_return_type' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'stypy_return_type', tuple_57573)
        
        # ################# End of '_validate_jac(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_validate_jac' in the type store
        # Getting the type of 'stypy_return_type' (line 328)
        stypy_return_type_57576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_57576)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_validate_jac'
        return stypy_return_type_57576


    @norecursion
    def _step_impl(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_step_impl'
        module_type_store = module_type_store.open_function_context('_step_impl', 381, 4, False)
        # Assigning a type to the variable 'self' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Radau._step_impl.__dict__.__setitem__('stypy_localization', localization)
        Radau._step_impl.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Radau._step_impl.__dict__.__setitem__('stypy_type_store', module_type_store)
        Radau._step_impl.__dict__.__setitem__('stypy_function_name', 'Radau._step_impl')
        Radau._step_impl.__dict__.__setitem__('stypy_param_names_list', [])
        Radau._step_impl.__dict__.__setitem__('stypy_varargs_param_name', None)
        Radau._step_impl.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Radau._step_impl.__dict__.__setitem__('stypy_call_defaults', defaults)
        Radau._step_impl.__dict__.__setitem__('stypy_call_varargs', varargs)
        Radau._step_impl.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Radau._step_impl.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Radau._step_impl', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Name (line 382):
        
        # Assigning a Attribute to a Name (line 382):
        # Getting the type of 'self' (line 382)
        self_57577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'self')
        # Obtaining the member 't' of a type (line 382)
        t_57578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 12), self_57577, 't')
        # Assigning a type to the variable 't' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 't', t_57578)
        
        # Assigning a Attribute to a Name (line 383):
        
        # Assigning a Attribute to a Name (line 383):
        # Getting the type of 'self' (line 383)
        self_57579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'self')
        # Obtaining the member 'y' of a type (line 383)
        y_57580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 12), self_57579, 'y')
        # Assigning a type to the variable 'y' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'y', y_57580)
        
        # Assigning a Attribute to a Name (line 384):
        
        # Assigning a Attribute to a Name (line 384):
        # Getting the type of 'self' (line 384)
        self_57581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'self')
        # Obtaining the member 'f' of a type (line 384)
        f_57582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 12), self_57581, 'f')
        # Assigning a type to the variable 'f' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'f', f_57582)
        
        # Assigning a Attribute to a Name (line 386):
        
        # Assigning a Attribute to a Name (line 386):
        # Getting the type of 'self' (line 386)
        self_57583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 19), 'self')
        # Obtaining the member 'max_step' of a type (line 386)
        max_step_57584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 19), self_57583, 'max_step')
        # Assigning a type to the variable 'max_step' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'max_step', max_step_57584)
        
        # Assigning a Attribute to a Name (line 387):
        
        # Assigning a Attribute to a Name (line 387):
        # Getting the type of 'self' (line 387)
        self_57585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 15), 'self')
        # Obtaining the member 'atol' of a type (line 387)
        atol_57586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 15), self_57585, 'atol')
        # Assigning a type to the variable 'atol' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'atol', atol_57586)
        
        # Assigning a Attribute to a Name (line 388):
        
        # Assigning a Attribute to a Name (line 388):
        # Getting the type of 'self' (line 388)
        self_57587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 15), 'self')
        # Obtaining the member 'rtol' of a type (line 388)
        rtol_57588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 15), self_57587, 'rtol')
        # Assigning a type to the variable 'rtol' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'rtol', rtol_57588)
        
        # Assigning a BinOp to a Name (line 390):
        
        # Assigning a BinOp to a Name (line 390):
        int_57589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 19), 'int')
        
        # Call to abs(...): (line 390)
        # Processing the call arguments (line 390)
        
        # Call to nextafter(...): (line 390)
        # Processing the call arguments (line 390)
        # Getting the type of 't' (line 390)
        t_57594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 44), 't', False)
        # Getting the type of 'self' (line 390)
        self_57595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 47), 'self', False)
        # Obtaining the member 'direction' of a type (line 390)
        direction_57596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 47), self_57595, 'direction')
        # Getting the type of 'np' (line 390)
        np_57597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 64), 'np', False)
        # Obtaining the member 'inf' of a type (line 390)
        inf_57598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 64), np_57597, 'inf')
        # Applying the binary operator '*' (line 390)
        result_mul_57599 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 47), '*', direction_57596, inf_57598)
        
        # Processing the call keyword arguments (line 390)
        kwargs_57600 = {}
        # Getting the type of 'np' (line 390)
        np_57592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 31), 'np', False)
        # Obtaining the member 'nextafter' of a type (line 390)
        nextafter_57593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 31), np_57592, 'nextafter')
        # Calling nextafter(args, kwargs) (line 390)
        nextafter_call_result_57601 = invoke(stypy.reporting.localization.Localization(__file__, 390, 31), nextafter_57593, *[t_57594, result_mul_57599], **kwargs_57600)
        
        # Getting the type of 't' (line 390)
        t_57602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 74), 't', False)
        # Applying the binary operator '-' (line 390)
        result_sub_57603 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 31), '-', nextafter_call_result_57601, t_57602)
        
        # Processing the call keyword arguments (line 390)
        kwargs_57604 = {}
        # Getting the type of 'np' (line 390)
        np_57590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 24), 'np', False)
        # Obtaining the member 'abs' of a type (line 390)
        abs_57591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 24), np_57590, 'abs')
        # Calling abs(args, kwargs) (line 390)
        abs_call_result_57605 = invoke(stypy.reporting.localization.Localization(__file__, 390, 24), abs_57591, *[result_sub_57603], **kwargs_57604)
        
        # Applying the binary operator '*' (line 390)
        result_mul_57606 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 19), '*', int_57589, abs_call_result_57605)
        
        # Assigning a type to the variable 'min_step' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'min_step', result_mul_57606)
        
        
        # Getting the type of 'self' (line 391)
        self_57607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 11), 'self')
        # Obtaining the member 'h_abs' of a type (line 391)
        h_abs_57608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 11), self_57607, 'h_abs')
        # Getting the type of 'max_step' (line 391)
        max_step_57609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 24), 'max_step')
        # Applying the binary operator '>' (line 391)
        result_gt_57610 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 11), '>', h_abs_57608, max_step_57609)
        
        # Testing the type of an if condition (line 391)
        if_condition_57611 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 391, 8), result_gt_57610)
        # Assigning a type to the variable 'if_condition_57611' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'if_condition_57611', if_condition_57611)
        # SSA begins for if statement (line 391)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 392):
        
        # Assigning a Name to a Name (line 392):
        # Getting the type of 'max_step' (line 392)
        max_step_57612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 20), 'max_step')
        # Assigning a type to the variable 'h_abs' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'h_abs', max_step_57612)
        
        # Assigning a Name to a Name (line 393):
        
        # Assigning a Name to a Name (line 393):
        # Getting the type of 'None' (line 393)
        None_57613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 24), 'None')
        # Assigning a type to the variable 'h_abs_old' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'h_abs_old', None_57613)
        
        # Assigning a Name to a Name (line 394):
        
        # Assigning a Name to a Name (line 394):
        # Getting the type of 'None' (line 394)
        None_57614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 29), 'None')
        # Assigning a type to the variable 'error_norm_old' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'error_norm_old', None_57614)
        # SSA branch for the else part of an if statement (line 391)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 395)
        self_57615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 13), 'self')
        # Obtaining the member 'h_abs' of a type (line 395)
        h_abs_57616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 13), self_57615, 'h_abs')
        # Getting the type of 'min_step' (line 395)
        min_step_57617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 26), 'min_step')
        # Applying the binary operator '<' (line 395)
        result_lt_57618 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 13), '<', h_abs_57616, min_step_57617)
        
        # Testing the type of an if condition (line 395)
        if_condition_57619 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 395, 13), result_lt_57618)
        # Assigning a type to the variable 'if_condition_57619' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 13), 'if_condition_57619', if_condition_57619)
        # SSA begins for if statement (line 395)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 396):
        
        # Assigning a Name to a Name (line 396):
        # Getting the type of 'min_step' (line 396)
        min_step_57620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 20), 'min_step')
        # Assigning a type to the variable 'h_abs' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'h_abs', min_step_57620)
        
        # Assigning a Name to a Name (line 397):
        
        # Assigning a Name to a Name (line 397):
        # Getting the type of 'None' (line 397)
        None_57621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 24), 'None')
        # Assigning a type to the variable 'h_abs_old' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'h_abs_old', None_57621)
        
        # Assigning a Name to a Name (line 398):
        
        # Assigning a Name to a Name (line 398):
        # Getting the type of 'None' (line 398)
        None_57622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 29), 'None')
        # Assigning a type to the variable 'error_norm_old' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'error_norm_old', None_57622)
        # SSA branch for the else part of an if statement (line 395)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 400):
        
        # Assigning a Attribute to a Name (line 400):
        # Getting the type of 'self' (line 400)
        self_57623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 20), 'self')
        # Obtaining the member 'h_abs' of a type (line 400)
        h_abs_57624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 20), self_57623, 'h_abs')
        # Assigning a type to the variable 'h_abs' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'h_abs', h_abs_57624)
        
        # Assigning a Attribute to a Name (line 401):
        
        # Assigning a Attribute to a Name (line 401):
        # Getting the type of 'self' (line 401)
        self_57625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 24), 'self')
        # Obtaining the member 'h_abs_old' of a type (line 401)
        h_abs_old_57626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 24), self_57625, 'h_abs_old')
        # Assigning a type to the variable 'h_abs_old' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'h_abs_old', h_abs_old_57626)
        
        # Assigning a Attribute to a Name (line 402):
        
        # Assigning a Attribute to a Name (line 402):
        # Getting the type of 'self' (line 402)
        self_57627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 29), 'self')
        # Obtaining the member 'error_norm_old' of a type (line 402)
        error_norm_old_57628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 29), self_57627, 'error_norm_old')
        # Assigning a type to the variable 'error_norm_old' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'error_norm_old', error_norm_old_57628)
        # SSA join for if statement (line 395)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 391)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 404):
        
        # Assigning a Attribute to a Name (line 404):
        # Getting the type of 'self' (line 404)
        self_57629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 12), 'self')
        # Obtaining the member 'J' of a type (line 404)
        J_57630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 12), self_57629, 'J')
        # Assigning a type to the variable 'J' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'J', J_57630)
        
        # Assigning a Attribute to a Name (line 405):
        
        # Assigning a Attribute to a Name (line 405):
        # Getting the type of 'self' (line 405)
        self_57631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 18), 'self')
        # Obtaining the member 'LU_real' of a type (line 405)
        LU_real_57632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 18), self_57631, 'LU_real')
        # Assigning a type to the variable 'LU_real' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'LU_real', LU_real_57632)
        
        # Assigning a Attribute to a Name (line 406):
        
        # Assigning a Attribute to a Name (line 406):
        # Getting the type of 'self' (line 406)
        self_57633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 21), 'self')
        # Obtaining the member 'LU_complex' of a type (line 406)
        LU_complex_57634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 21), self_57633, 'LU_complex')
        # Assigning a type to the variable 'LU_complex' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'LU_complex', LU_complex_57634)
        
        # Assigning a Attribute to a Name (line 408):
        
        # Assigning a Attribute to a Name (line 408):
        # Getting the type of 'self' (line 408)
        self_57635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 22), 'self')
        # Obtaining the member 'current_jac' of a type (line 408)
        current_jac_57636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 22), self_57635, 'current_jac')
        # Assigning a type to the variable 'current_jac' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'current_jac', current_jac_57636)
        
        # Assigning a Attribute to a Name (line 409):
        
        # Assigning a Attribute to a Name (line 409):
        # Getting the type of 'self' (line 409)
        self_57637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 14), 'self')
        # Obtaining the member 'jac' of a type (line 409)
        jac_57638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 14), self_57637, 'jac')
        # Assigning a type to the variable 'jac' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'jac', jac_57638)
        
        # Assigning a Name to a Name (line 411):
        
        # Assigning a Name to a Name (line 411):
        # Getting the type of 'False' (line 411)
        False_57639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 19), 'False')
        # Assigning a type to the variable 'rejected' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'rejected', False_57639)
        
        # Assigning a Name to a Name (line 412):
        
        # Assigning a Name to a Name (line 412):
        # Getting the type of 'False' (line 412)
        False_57640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 24), 'False')
        # Assigning a type to the variable 'step_accepted' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'step_accepted', False_57640)
        
        # Assigning a Name to a Name (line 413):
        
        # Assigning a Name to a Name (line 413):
        # Getting the type of 'None' (line 413)
        None_57641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 18), 'None')
        # Assigning a type to the variable 'message' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'message', None_57641)
        
        
        # Getting the type of 'step_accepted' (line 414)
        step_accepted_57642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 18), 'step_accepted')
        # Applying the 'not' unary operator (line 414)
        result_not__57643 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 14), 'not', step_accepted_57642)
        
        # Testing the type of an if condition (line 414)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 414, 8), result_not__57643)
        # SSA begins for while statement (line 414)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        
        # Getting the type of 'h_abs' (line 415)
        h_abs_57644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 15), 'h_abs')
        # Getting the type of 'min_step' (line 415)
        min_step_57645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 23), 'min_step')
        # Applying the binary operator '<' (line 415)
        result_lt_57646 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 15), '<', h_abs_57644, min_step_57645)
        
        # Testing the type of an if condition (line 415)
        if_condition_57647 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 415, 12), result_lt_57646)
        # Assigning a type to the variable 'if_condition_57647' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 12), 'if_condition_57647', if_condition_57647)
        # SSA begins for if statement (line 415)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 416)
        tuple_57648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 416)
        # Adding element type (line 416)
        # Getting the type of 'False' (line 416)
        False_57649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 23), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 23), tuple_57648, False_57649)
        # Adding element type (line 416)
        # Getting the type of 'self' (line 416)
        self_57650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 30), 'self')
        # Obtaining the member 'TOO_SMALL_STEP' of a type (line 416)
        TOO_SMALL_STEP_57651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 30), self_57650, 'TOO_SMALL_STEP')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 23), tuple_57648, TOO_SMALL_STEP_57651)
        
        # Assigning a type to the variable 'stypy_return_type' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 16), 'stypy_return_type', tuple_57648)
        # SSA join for if statement (line 415)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 418):
        
        # Assigning a BinOp to a Name (line 418):
        # Getting the type of 'h_abs' (line 418)
        h_abs_57652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 16), 'h_abs')
        # Getting the type of 'self' (line 418)
        self_57653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 24), 'self')
        # Obtaining the member 'direction' of a type (line 418)
        direction_57654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 24), self_57653, 'direction')
        # Applying the binary operator '*' (line 418)
        result_mul_57655 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 16), '*', h_abs_57652, direction_57654)
        
        # Assigning a type to the variable 'h' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), 'h', result_mul_57655)
        
        # Assigning a BinOp to a Name (line 419):
        
        # Assigning a BinOp to a Name (line 419):
        # Getting the type of 't' (line 419)
        t_57656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 20), 't')
        # Getting the type of 'h' (line 419)
        h_57657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 24), 'h')
        # Applying the binary operator '+' (line 419)
        result_add_57658 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 20), '+', t_57656, h_57657)
        
        # Assigning a type to the variable 't_new' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 12), 't_new', result_add_57658)
        
        
        # Getting the type of 'self' (line 421)
        self_57659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 15), 'self')
        # Obtaining the member 'direction' of a type (line 421)
        direction_57660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 15), self_57659, 'direction')
        # Getting the type of 't_new' (line 421)
        t_new_57661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 33), 't_new')
        # Getting the type of 'self' (line 421)
        self_57662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 41), 'self')
        # Obtaining the member 't_bound' of a type (line 421)
        t_bound_57663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 41), self_57662, 't_bound')
        # Applying the binary operator '-' (line 421)
        result_sub_57664 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 33), '-', t_new_57661, t_bound_57663)
        
        # Applying the binary operator '*' (line 421)
        result_mul_57665 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 15), '*', direction_57660, result_sub_57664)
        
        int_57666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 57), 'int')
        # Applying the binary operator '>' (line 421)
        result_gt_57667 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 15), '>', result_mul_57665, int_57666)
        
        # Testing the type of an if condition (line 421)
        if_condition_57668 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 421, 12), result_gt_57667)
        # Assigning a type to the variable 'if_condition_57668' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 12), 'if_condition_57668', if_condition_57668)
        # SSA begins for if statement (line 421)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 422):
        
        # Assigning a Attribute to a Name (line 422):
        # Getting the type of 'self' (line 422)
        self_57669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 24), 'self')
        # Obtaining the member 't_bound' of a type (line 422)
        t_bound_57670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 24), self_57669, 't_bound')
        # Assigning a type to the variable 't_new' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 16), 't_new', t_bound_57670)
        # SSA join for if statement (line 421)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 424):
        
        # Assigning a BinOp to a Name (line 424):
        # Getting the type of 't_new' (line 424)
        t_new_57671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 16), 't_new')
        # Getting the type of 't' (line 424)
        t_57672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 24), 't')
        # Applying the binary operator '-' (line 424)
        result_sub_57673 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 16), '-', t_new_57671, t_57672)
        
        # Assigning a type to the variable 'h' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'h', result_sub_57673)
        
        # Assigning a Call to a Name (line 425):
        
        # Assigning a Call to a Name (line 425):
        
        # Call to abs(...): (line 425)
        # Processing the call arguments (line 425)
        # Getting the type of 'h' (line 425)
        h_57676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 27), 'h', False)
        # Processing the call keyword arguments (line 425)
        kwargs_57677 = {}
        # Getting the type of 'np' (line 425)
        np_57674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 20), 'np', False)
        # Obtaining the member 'abs' of a type (line 425)
        abs_57675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 20), np_57674, 'abs')
        # Calling abs(args, kwargs) (line 425)
        abs_call_result_57678 = invoke(stypy.reporting.localization.Localization(__file__, 425, 20), abs_57675, *[h_57676], **kwargs_57677)
        
        # Assigning a type to the variable 'h_abs' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'h_abs', abs_call_result_57678)
        
        # Type idiom detected: calculating its left and rigth part (line 427)
        # Getting the type of 'self' (line 427)
        self_57679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 15), 'self')
        # Obtaining the member 'sol' of a type (line 427)
        sol_57680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 15), self_57679, 'sol')
        # Getting the type of 'None' (line 427)
        None_57681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 27), 'None')
        
        (may_be_57682, more_types_in_union_57683) = may_be_none(sol_57680, None_57681)

        if may_be_57682:

            if more_types_in_union_57683:
                # Runtime conditional SSA (line 427)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 428):
            
            # Assigning a Call to a Name (line 428):
            
            # Call to zeros(...): (line 428)
            # Processing the call arguments (line 428)
            
            # Obtaining an instance of the builtin type 'tuple' (line 428)
            tuple_57686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 31), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 428)
            # Adding element type (line 428)
            int_57687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 31), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 31), tuple_57686, int_57687)
            # Adding element type (line 428)
            
            # Obtaining the type of the subscript
            int_57688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 42), 'int')
            # Getting the type of 'y' (line 428)
            y_57689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 34), 'y', False)
            # Obtaining the member 'shape' of a type (line 428)
            shape_57690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 34), y_57689, 'shape')
            # Obtaining the member '__getitem__' of a type (line 428)
            getitem___57691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 34), shape_57690, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 428)
            subscript_call_result_57692 = invoke(stypy.reporting.localization.Localization(__file__, 428, 34), getitem___57691, int_57688)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 31), tuple_57686, subscript_call_result_57692)
            
            # Processing the call keyword arguments (line 428)
            kwargs_57693 = {}
            # Getting the type of 'np' (line 428)
            np_57684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 21), 'np', False)
            # Obtaining the member 'zeros' of a type (line 428)
            zeros_57685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 21), np_57684, 'zeros')
            # Calling zeros(args, kwargs) (line 428)
            zeros_call_result_57694 = invoke(stypy.reporting.localization.Localization(__file__, 428, 21), zeros_57685, *[tuple_57686], **kwargs_57693)
            
            # Assigning a type to the variable 'Z0' (line 428)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 16), 'Z0', zeros_call_result_57694)

            if more_types_in_union_57683:
                # Runtime conditional SSA for else branch (line 427)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_57682) or more_types_in_union_57683):
            
            # Assigning a BinOp to a Name (line 430):
            
            # Assigning a BinOp to a Name (line 430):
            
            # Call to sol(...): (line 430)
            # Processing the call arguments (line 430)
            # Getting the type of 't' (line 430)
            t_57697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 30), 't', False)
            # Getting the type of 'h' (line 430)
            h_57698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 34), 'h', False)
            # Getting the type of 'C' (line 430)
            C_57699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 38), 'C', False)
            # Applying the binary operator '*' (line 430)
            result_mul_57700 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 34), '*', h_57698, C_57699)
            
            # Applying the binary operator '+' (line 430)
            result_add_57701 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 30), '+', t_57697, result_mul_57700)
            
            # Processing the call keyword arguments (line 430)
            kwargs_57702 = {}
            # Getting the type of 'self' (line 430)
            self_57695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 21), 'self', False)
            # Obtaining the member 'sol' of a type (line 430)
            sol_57696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 21), self_57695, 'sol')
            # Calling sol(args, kwargs) (line 430)
            sol_call_result_57703 = invoke(stypy.reporting.localization.Localization(__file__, 430, 21), sol_57696, *[result_add_57701], **kwargs_57702)
            
            # Obtaining the member 'T' of a type (line 430)
            T_57704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 21), sol_call_result_57703, 'T')
            # Getting the type of 'y' (line 430)
            y_57705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 45), 'y')
            # Applying the binary operator '-' (line 430)
            result_sub_57706 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 21), '-', T_57704, y_57705)
            
            # Assigning a type to the variable 'Z0' (line 430)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 16), 'Z0', result_sub_57706)

            if (may_be_57682 and more_types_in_union_57683):
                # SSA join for if statement (line 427)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BinOp to a Name (line 432):
        
        # Assigning a BinOp to a Name (line 432):
        # Getting the type of 'atol' (line 432)
        atol_57707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 20), 'atol')
        
        # Call to abs(...): (line 432)
        # Processing the call arguments (line 432)
        # Getting the type of 'y' (line 432)
        y_57710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 34), 'y', False)
        # Processing the call keyword arguments (line 432)
        kwargs_57711 = {}
        # Getting the type of 'np' (line 432)
        np_57708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 27), 'np', False)
        # Obtaining the member 'abs' of a type (line 432)
        abs_57709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 27), np_57708, 'abs')
        # Calling abs(args, kwargs) (line 432)
        abs_call_result_57712 = invoke(stypy.reporting.localization.Localization(__file__, 432, 27), abs_57709, *[y_57710], **kwargs_57711)
        
        # Getting the type of 'rtol' (line 432)
        rtol_57713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 39), 'rtol')
        # Applying the binary operator '*' (line 432)
        result_mul_57714 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 27), '*', abs_call_result_57712, rtol_57713)
        
        # Applying the binary operator '+' (line 432)
        result_add_57715 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 20), '+', atol_57707, result_mul_57714)
        
        # Assigning a type to the variable 'scale' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'scale', result_add_57715)
        
        # Assigning a Name to a Name (line 434):
        
        # Assigning a Name to a Name (line 434):
        # Getting the type of 'False' (line 434)
        False_57716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 24), 'False')
        # Assigning a type to the variable 'converged' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'converged', False_57716)
        
        
        # Getting the type of 'converged' (line 435)
        converged_57717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 22), 'converged')
        # Applying the 'not' unary operator (line 435)
        result_not__57718 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 18), 'not', converged_57717)
        
        # Testing the type of an if condition (line 435)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 435, 12), result_not__57718)
        # SSA begins for while statement (line 435)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'LU_real' (line 436)
        LU_real_57719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 19), 'LU_real')
        # Getting the type of 'None' (line 436)
        None_57720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 30), 'None')
        # Applying the binary operator 'is' (line 436)
        result_is__57721 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 19), 'is', LU_real_57719, None_57720)
        
        
        # Getting the type of 'LU_complex' (line 436)
        LU_complex_57722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 38), 'LU_complex')
        # Getting the type of 'None' (line 436)
        None_57723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 52), 'None')
        # Applying the binary operator 'is' (line 436)
        result_is__57724 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 38), 'is', LU_complex_57722, None_57723)
        
        # Applying the binary operator 'or' (line 436)
        result_or_keyword_57725 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 19), 'or', result_is__57721, result_is__57724)
        
        # Testing the type of an if condition (line 436)
        if_condition_57726 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 436, 16), result_or_keyword_57725)
        # Assigning a type to the variable 'if_condition_57726' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 16), 'if_condition_57726', if_condition_57726)
        # SSA begins for if statement (line 436)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 437):
        
        # Assigning a Call to a Name (line 437):
        
        # Call to lu(...): (line 437)
        # Processing the call arguments (line 437)
        # Getting the type of 'MU_REAL' (line 437)
        MU_REAL_57729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 38), 'MU_REAL', False)
        # Getting the type of 'h' (line 437)
        h_57730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 48), 'h', False)
        # Applying the binary operator 'div' (line 437)
        result_div_57731 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 38), 'div', MU_REAL_57729, h_57730)
        
        # Getting the type of 'self' (line 437)
        self_57732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 52), 'self', False)
        # Obtaining the member 'I' of a type (line 437)
        I_57733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 52), self_57732, 'I')
        # Applying the binary operator '*' (line 437)
        result_mul_57734 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 50), '*', result_div_57731, I_57733)
        
        # Getting the type of 'J' (line 437)
        J_57735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 61), 'J', False)
        # Applying the binary operator '-' (line 437)
        result_sub_57736 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 38), '-', result_mul_57734, J_57735)
        
        # Processing the call keyword arguments (line 437)
        kwargs_57737 = {}
        # Getting the type of 'self' (line 437)
        self_57727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 30), 'self', False)
        # Obtaining the member 'lu' of a type (line 437)
        lu_57728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 30), self_57727, 'lu')
        # Calling lu(args, kwargs) (line 437)
        lu_call_result_57738 = invoke(stypy.reporting.localization.Localization(__file__, 437, 30), lu_57728, *[result_sub_57736], **kwargs_57737)
        
        # Assigning a type to the variable 'LU_real' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 20), 'LU_real', lu_call_result_57738)
        
        # Assigning a Call to a Name (line 438):
        
        # Assigning a Call to a Name (line 438):
        
        # Call to lu(...): (line 438)
        # Processing the call arguments (line 438)
        # Getting the type of 'MU_COMPLEX' (line 438)
        MU_COMPLEX_57741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 41), 'MU_COMPLEX', False)
        # Getting the type of 'h' (line 438)
        h_57742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 54), 'h', False)
        # Applying the binary operator 'div' (line 438)
        result_div_57743 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 41), 'div', MU_COMPLEX_57741, h_57742)
        
        # Getting the type of 'self' (line 438)
        self_57744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 58), 'self', False)
        # Obtaining the member 'I' of a type (line 438)
        I_57745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 58), self_57744, 'I')
        # Applying the binary operator '*' (line 438)
        result_mul_57746 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 56), '*', result_div_57743, I_57745)
        
        # Getting the type of 'J' (line 438)
        J_57747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 67), 'J', False)
        # Applying the binary operator '-' (line 438)
        result_sub_57748 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 41), '-', result_mul_57746, J_57747)
        
        # Processing the call keyword arguments (line 438)
        kwargs_57749 = {}
        # Getting the type of 'self' (line 438)
        self_57739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 33), 'self', False)
        # Obtaining the member 'lu' of a type (line 438)
        lu_57740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 33), self_57739, 'lu')
        # Calling lu(args, kwargs) (line 438)
        lu_call_result_57750 = invoke(stypy.reporting.localization.Localization(__file__, 438, 33), lu_57740, *[result_sub_57748], **kwargs_57749)
        
        # Assigning a type to the variable 'LU_complex' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 20), 'LU_complex', lu_call_result_57750)
        # SSA join for if statement (line 436)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 440):
        
        # Assigning a Subscript to a Name (line 440):
        
        # Obtaining the type of the subscript
        int_57751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 16), 'int')
        
        # Call to solve_collocation_system(...): (line 440)
        # Processing the call arguments (line 440)
        # Getting the type of 'self' (line 441)
        self_57753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 20), 'self', False)
        # Obtaining the member 'fun' of a type (line 441)
        fun_57754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 20), self_57753, 'fun')
        # Getting the type of 't' (line 441)
        t_57755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 30), 't', False)
        # Getting the type of 'y' (line 441)
        y_57756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 33), 'y', False)
        # Getting the type of 'h' (line 441)
        h_57757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 36), 'h', False)
        # Getting the type of 'Z0' (line 441)
        Z0_57758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 39), 'Z0', False)
        # Getting the type of 'scale' (line 441)
        scale_57759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 43), 'scale', False)
        # Getting the type of 'self' (line 441)
        self_57760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 50), 'self', False)
        # Obtaining the member 'newton_tol' of a type (line 441)
        newton_tol_57761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 50), self_57760, 'newton_tol')
        # Getting the type of 'LU_real' (line 442)
        LU_real_57762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 20), 'LU_real', False)
        # Getting the type of 'LU_complex' (line 442)
        LU_complex_57763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 29), 'LU_complex', False)
        # Getting the type of 'self' (line 442)
        self_57764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 41), 'self', False)
        # Obtaining the member 'solve_lu' of a type (line 442)
        solve_lu_57765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 41), self_57764, 'solve_lu')
        # Processing the call keyword arguments (line 440)
        kwargs_57766 = {}
        # Getting the type of 'solve_collocation_system' (line 440)
        solve_collocation_system_57752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 45), 'solve_collocation_system', False)
        # Calling solve_collocation_system(args, kwargs) (line 440)
        solve_collocation_system_call_result_57767 = invoke(stypy.reporting.localization.Localization(__file__, 440, 45), solve_collocation_system_57752, *[fun_57754, t_57755, y_57756, h_57757, Z0_57758, scale_57759, newton_tol_57761, LU_real_57762, LU_complex_57763, solve_lu_57765], **kwargs_57766)
        
        # Obtaining the member '__getitem__' of a type (line 440)
        getitem___57768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 16), solve_collocation_system_call_result_57767, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 440)
        subscript_call_result_57769 = invoke(stypy.reporting.localization.Localization(__file__, 440, 16), getitem___57768, int_57751)
        
        # Assigning a type to the variable 'tuple_var_assignment_56710' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 16), 'tuple_var_assignment_56710', subscript_call_result_57769)
        
        # Assigning a Subscript to a Name (line 440):
        
        # Obtaining the type of the subscript
        int_57770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 16), 'int')
        
        # Call to solve_collocation_system(...): (line 440)
        # Processing the call arguments (line 440)
        # Getting the type of 'self' (line 441)
        self_57772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 20), 'self', False)
        # Obtaining the member 'fun' of a type (line 441)
        fun_57773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 20), self_57772, 'fun')
        # Getting the type of 't' (line 441)
        t_57774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 30), 't', False)
        # Getting the type of 'y' (line 441)
        y_57775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 33), 'y', False)
        # Getting the type of 'h' (line 441)
        h_57776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 36), 'h', False)
        # Getting the type of 'Z0' (line 441)
        Z0_57777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 39), 'Z0', False)
        # Getting the type of 'scale' (line 441)
        scale_57778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 43), 'scale', False)
        # Getting the type of 'self' (line 441)
        self_57779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 50), 'self', False)
        # Obtaining the member 'newton_tol' of a type (line 441)
        newton_tol_57780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 50), self_57779, 'newton_tol')
        # Getting the type of 'LU_real' (line 442)
        LU_real_57781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 20), 'LU_real', False)
        # Getting the type of 'LU_complex' (line 442)
        LU_complex_57782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 29), 'LU_complex', False)
        # Getting the type of 'self' (line 442)
        self_57783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 41), 'self', False)
        # Obtaining the member 'solve_lu' of a type (line 442)
        solve_lu_57784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 41), self_57783, 'solve_lu')
        # Processing the call keyword arguments (line 440)
        kwargs_57785 = {}
        # Getting the type of 'solve_collocation_system' (line 440)
        solve_collocation_system_57771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 45), 'solve_collocation_system', False)
        # Calling solve_collocation_system(args, kwargs) (line 440)
        solve_collocation_system_call_result_57786 = invoke(stypy.reporting.localization.Localization(__file__, 440, 45), solve_collocation_system_57771, *[fun_57773, t_57774, y_57775, h_57776, Z0_57777, scale_57778, newton_tol_57780, LU_real_57781, LU_complex_57782, solve_lu_57784], **kwargs_57785)
        
        # Obtaining the member '__getitem__' of a type (line 440)
        getitem___57787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 16), solve_collocation_system_call_result_57786, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 440)
        subscript_call_result_57788 = invoke(stypy.reporting.localization.Localization(__file__, 440, 16), getitem___57787, int_57770)
        
        # Assigning a type to the variable 'tuple_var_assignment_56711' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 16), 'tuple_var_assignment_56711', subscript_call_result_57788)
        
        # Assigning a Subscript to a Name (line 440):
        
        # Obtaining the type of the subscript
        int_57789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 16), 'int')
        
        # Call to solve_collocation_system(...): (line 440)
        # Processing the call arguments (line 440)
        # Getting the type of 'self' (line 441)
        self_57791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 20), 'self', False)
        # Obtaining the member 'fun' of a type (line 441)
        fun_57792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 20), self_57791, 'fun')
        # Getting the type of 't' (line 441)
        t_57793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 30), 't', False)
        # Getting the type of 'y' (line 441)
        y_57794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 33), 'y', False)
        # Getting the type of 'h' (line 441)
        h_57795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 36), 'h', False)
        # Getting the type of 'Z0' (line 441)
        Z0_57796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 39), 'Z0', False)
        # Getting the type of 'scale' (line 441)
        scale_57797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 43), 'scale', False)
        # Getting the type of 'self' (line 441)
        self_57798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 50), 'self', False)
        # Obtaining the member 'newton_tol' of a type (line 441)
        newton_tol_57799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 50), self_57798, 'newton_tol')
        # Getting the type of 'LU_real' (line 442)
        LU_real_57800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 20), 'LU_real', False)
        # Getting the type of 'LU_complex' (line 442)
        LU_complex_57801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 29), 'LU_complex', False)
        # Getting the type of 'self' (line 442)
        self_57802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 41), 'self', False)
        # Obtaining the member 'solve_lu' of a type (line 442)
        solve_lu_57803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 41), self_57802, 'solve_lu')
        # Processing the call keyword arguments (line 440)
        kwargs_57804 = {}
        # Getting the type of 'solve_collocation_system' (line 440)
        solve_collocation_system_57790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 45), 'solve_collocation_system', False)
        # Calling solve_collocation_system(args, kwargs) (line 440)
        solve_collocation_system_call_result_57805 = invoke(stypy.reporting.localization.Localization(__file__, 440, 45), solve_collocation_system_57790, *[fun_57792, t_57793, y_57794, h_57795, Z0_57796, scale_57797, newton_tol_57799, LU_real_57800, LU_complex_57801, solve_lu_57803], **kwargs_57804)
        
        # Obtaining the member '__getitem__' of a type (line 440)
        getitem___57806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 16), solve_collocation_system_call_result_57805, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 440)
        subscript_call_result_57807 = invoke(stypy.reporting.localization.Localization(__file__, 440, 16), getitem___57806, int_57789)
        
        # Assigning a type to the variable 'tuple_var_assignment_56712' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 16), 'tuple_var_assignment_56712', subscript_call_result_57807)
        
        # Assigning a Subscript to a Name (line 440):
        
        # Obtaining the type of the subscript
        int_57808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 16), 'int')
        
        # Call to solve_collocation_system(...): (line 440)
        # Processing the call arguments (line 440)
        # Getting the type of 'self' (line 441)
        self_57810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 20), 'self', False)
        # Obtaining the member 'fun' of a type (line 441)
        fun_57811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 20), self_57810, 'fun')
        # Getting the type of 't' (line 441)
        t_57812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 30), 't', False)
        # Getting the type of 'y' (line 441)
        y_57813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 33), 'y', False)
        # Getting the type of 'h' (line 441)
        h_57814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 36), 'h', False)
        # Getting the type of 'Z0' (line 441)
        Z0_57815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 39), 'Z0', False)
        # Getting the type of 'scale' (line 441)
        scale_57816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 43), 'scale', False)
        # Getting the type of 'self' (line 441)
        self_57817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 50), 'self', False)
        # Obtaining the member 'newton_tol' of a type (line 441)
        newton_tol_57818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 50), self_57817, 'newton_tol')
        # Getting the type of 'LU_real' (line 442)
        LU_real_57819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 20), 'LU_real', False)
        # Getting the type of 'LU_complex' (line 442)
        LU_complex_57820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 29), 'LU_complex', False)
        # Getting the type of 'self' (line 442)
        self_57821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 41), 'self', False)
        # Obtaining the member 'solve_lu' of a type (line 442)
        solve_lu_57822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 41), self_57821, 'solve_lu')
        # Processing the call keyword arguments (line 440)
        kwargs_57823 = {}
        # Getting the type of 'solve_collocation_system' (line 440)
        solve_collocation_system_57809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 45), 'solve_collocation_system', False)
        # Calling solve_collocation_system(args, kwargs) (line 440)
        solve_collocation_system_call_result_57824 = invoke(stypy.reporting.localization.Localization(__file__, 440, 45), solve_collocation_system_57809, *[fun_57811, t_57812, y_57813, h_57814, Z0_57815, scale_57816, newton_tol_57818, LU_real_57819, LU_complex_57820, solve_lu_57822], **kwargs_57823)
        
        # Obtaining the member '__getitem__' of a type (line 440)
        getitem___57825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 16), solve_collocation_system_call_result_57824, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 440)
        subscript_call_result_57826 = invoke(stypy.reporting.localization.Localization(__file__, 440, 16), getitem___57825, int_57808)
        
        # Assigning a type to the variable 'tuple_var_assignment_56713' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 16), 'tuple_var_assignment_56713', subscript_call_result_57826)
        
        # Assigning a Name to a Name (line 440):
        # Getting the type of 'tuple_var_assignment_56710' (line 440)
        tuple_var_assignment_56710_57827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 16), 'tuple_var_assignment_56710')
        # Assigning a type to the variable 'converged' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 16), 'converged', tuple_var_assignment_56710_57827)
        
        # Assigning a Name to a Name (line 440):
        # Getting the type of 'tuple_var_assignment_56711' (line 440)
        tuple_var_assignment_56711_57828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 16), 'tuple_var_assignment_56711')
        # Assigning a type to the variable 'n_iter' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 27), 'n_iter', tuple_var_assignment_56711_57828)
        
        # Assigning a Name to a Name (line 440):
        # Getting the type of 'tuple_var_assignment_56712' (line 440)
        tuple_var_assignment_56712_57829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 16), 'tuple_var_assignment_56712')
        # Assigning a type to the variable 'Z' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 35), 'Z', tuple_var_assignment_56712_57829)
        
        # Assigning a Name to a Name (line 440):
        # Getting the type of 'tuple_var_assignment_56713' (line 440)
        tuple_var_assignment_56713_57830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 16), 'tuple_var_assignment_56713')
        # Assigning a type to the variable 'rate' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 38), 'rate', tuple_var_assignment_56713_57830)
        
        
        # Getting the type of 'converged' (line 444)
        converged_57831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 23), 'converged')
        # Applying the 'not' unary operator (line 444)
        result_not__57832 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 19), 'not', converged_57831)
        
        # Testing the type of an if condition (line 444)
        if_condition_57833 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 444, 16), result_not__57832)
        # Assigning a type to the variable 'if_condition_57833' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 16), 'if_condition_57833', if_condition_57833)
        # SSA begins for if statement (line 444)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'current_jac' (line 445)
        current_jac_57834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 23), 'current_jac')
        # Testing the type of an if condition (line 445)
        if_condition_57835 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 445, 20), current_jac_57834)
        # Assigning a type to the variable 'if_condition_57835' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 20), 'if_condition_57835', if_condition_57835)
        # SSA begins for if statement (line 445)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 445)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 448):
        
        # Assigning a Call to a Name (line 448):
        
        # Call to jac(...): (line 448)
        # Processing the call arguments (line 448)
        # Getting the type of 't' (line 448)
        t_57838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 33), 't', False)
        # Getting the type of 'y' (line 448)
        y_57839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 36), 'y', False)
        # Getting the type of 'f' (line 448)
        f_57840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 39), 'f', False)
        # Processing the call keyword arguments (line 448)
        kwargs_57841 = {}
        # Getting the type of 'self' (line 448)
        self_57836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 24), 'self', False)
        # Obtaining the member 'jac' of a type (line 448)
        jac_57837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 24), self_57836, 'jac')
        # Calling jac(args, kwargs) (line 448)
        jac_call_result_57842 = invoke(stypy.reporting.localization.Localization(__file__, 448, 24), jac_57837, *[t_57838, y_57839, f_57840], **kwargs_57841)
        
        # Assigning a type to the variable 'J' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 20), 'J', jac_call_result_57842)
        
        # Assigning a Name to a Name (line 449):
        
        # Assigning a Name to a Name (line 449):
        # Getting the type of 'True' (line 449)
        True_57843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 34), 'True')
        # Assigning a type to the variable 'current_jac' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 20), 'current_jac', True_57843)
        
        # Assigning a Name to a Name (line 450):
        
        # Assigning a Name to a Name (line 450):
        # Getting the type of 'None' (line 450)
        None_57844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 30), 'None')
        # Assigning a type to the variable 'LU_real' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 20), 'LU_real', None_57844)
        
        # Assigning a Name to a Name (line 451):
        
        # Assigning a Name to a Name (line 451):
        # Getting the type of 'None' (line 451)
        None_57845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 33), 'None')
        # Assigning a type to the variable 'LU_complex' (line 451)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 20), 'LU_complex', None_57845)
        # SSA join for if statement (line 444)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 435)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'converged' (line 453)
        converged_57846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 19), 'converged')
        # Applying the 'not' unary operator (line 453)
        result_not__57847 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 15), 'not', converged_57846)
        
        # Testing the type of an if condition (line 453)
        if_condition_57848 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 453, 12), result_not__57847)
        # Assigning a type to the variable 'if_condition_57848' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 12), 'if_condition_57848', if_condition_57848)
        # SSA begins for if statement (line 453)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'h_abs' (line 454)
        h_abs_57849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 16), 'h_abs')
        float_57850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 25), 'float')
        # Applying the binary operator '*=' (line 454)
        result_imul_57851 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 16), '*=', h_abs_57849, float_57850)
        # Assigning a type to the variable 'h_abs' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 16), 'h_abs', result_imul_57851)
        
        
        # Assigning a Name to a Name (line 455):
        
        # Assigning a Name to a Name (line 455):
        # Getting the type of 'None' (line 455)
        None_57852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 26), 'None')
        # Assigning a type to the variable 'LU_real' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 16), 'LU_real', None_57852)
        
        # Assigning a Name to a Name (line 456):
        
        # Assigning a Name to a Name (line 456):
        # Getting the type of 'None' (line 456)
        None_57853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 29), 'None')
        # Assigning a type to the variable 'LU_complex' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'LU_complex', None_57853)
        # SSA join for if statement (line 453)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 459):
        
        # Assigning a BinOp to a Name (line 459):
        # Getting the type of 'y' (line 459)
        y_57854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 20), 'y')
        
        # Obtaining the type of the subscript
        int_57855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 26), 'int')
        # Getting the type of 'Z' (line 459)
        Z_57856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 24), 'Z')
        # Obtaining the member '__getitem__' of a type (line 459)
        getitem___57857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 24), Z_57856, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 459)
        subscript_call_result_57858 = invoke(stypy.reporting.localization.Localization(__file__, 459, 24), getitem___57857, int_57855)
        
        # Applying the binary operator '+' (line 459)
        result_add_57859 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 20), '+', y_57854, subscript_call_result_57858)
        
        # Assigning a type to the variable 'y_new' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'y_new', result_add_57859)
        
        # Assigning a BinOp to a Name (line 460):
        
        # Assigning a BinOp to a Name (line 460):
        
        # Call to dot(...): (line 460)
        # Processing the call arguments (line 460)
        # Getting the type of 'E' (line 460)
        E_57863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 25), 'E', False)
        # Processing the call keyword arguments (line 460)
        kwargs_57864 = {}
        # Getting the type of 'Z' (line 460)
        Z_57860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 17), 'Z', False)
        # Obtaining the member 'T' of a type (line 460)
        T_57861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 17), Z_57860, 'T')
        # Obtaining the member 'dot' of a type (line 460)
        dot_57862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 17), T_57861, 'dot')
        # Calling dot(args, kwargs) (line 460)
        dot_call_result_57865 = invoke(stypy.reporting.localization.Localization(__file__, 460, 17), dot_57862, *[E_57863], **kwargs_57864)
        
        # Getting the type of 'h' (line 460)
        h_57866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 30), 'h')
        # Applying the binary operator 'div' (line 460)
        result_div_57867 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 17), 'div', dot_call_result_57865, h_57866)
        
        # Assigning a type to the variable 'ZE' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'ZE', result_div_57867)
        
        # Assigning a Call to a Name (line 461):
        
        # Assigning a Call to a Name (line 461):
        
        # Call to solve_lu(...): (line 461)
        # Processing the call arguments (line 461)
        # Getting the type of 'LU_real' (line 461)
        LU_real_57870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 34), 'LU_real', False)
        # Getting the type of 'f' (line 461)
        f_57871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 43), 'f', False)
        # Getting the type of 'ZE' (line 461)
        ZE_57872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 47), 'ZE', False)
        # Applying the binary operator '+' (line 461)
        result_add_57873 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 43), '+', f_57871, ZE_57872)
        
        # Processing the call keyword arguments (line 461)
        kwargs_57874 = {}
        # Getting the type of 'self' (line 461)
        self_57868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 20), 'self', False)
        # Obtaining the member 'solve_lu' of a type (line 461)
        solve_lu_57869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 20), self_57868, 'solve_lu')
        # Calling solve_lu(args, kwargs) (line 461)
        solve_lu_call_result_57875 = invoke(stypy.reporting.localization.Localization(__file__, 461, 20), solve_lu_57869, *[LU_real_57870, result_add_57873], **kwargs_57874)
        
        # Assigning a type to the variable 'error' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'error', solve_lu_call_result_57875)
        
        # Assigning a BinOp to a Name (line 462):
        
        # Assigning a BinOp to a Name (line 462):
        # Getting the type of 'atol' (line 462)
        atol_57876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 20), 'atol')
        
        # Call to maximum(...): (line 462)
        # Processing the call arguments (line 462)
        
        # Call to abs(...): (line 462)
        # Processing the call arguments (line 462)
        # Getting the type of 'y' (line 462)
        y_57881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 45), 'y', False)
        # Processing the call keyword arguments (line 462)
        kwargs_57882 = {}
        # Getting the type of 'np' (line 462)
        np_57879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 38), 'np', False)
        # Obtaining the member 'abs' of a type (line 462)
        abs_57880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 38), np_57879, 'abs')
        # Calling abs(args, kwargs) (line 462)
        abs_call_result_57883 = invoke(stypy.reporting.localization.Localization(__file__, 462, 38), abs_57880, *[y_57881], **kwargs_57882)
        
        
        # Call to abs(...): (line 462)
        # Processing the call arguments (line 462)
        # Getting the type of 'y_new' (line 462)
        y_new_57886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 56), 'y_new', False)
        # Processing the call keyword arguments (line 462)
        kwargs_57887 = {}
        # Getting the type of 'np' (line 462)
        np_57884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 49), 'np', False)
        # Obtaining the member 'abs' of a type (line 462)
        abs_57885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 49), np_57884, 'abs')
        # Calling abs(args, kwargs) (line 462)
        abs_call_result_57888 = invoke(stypy.reporting.localization.Localization(__file__, 462, 49), abs_57885, *[y_new_57886], **kwargs_57887)
        
        # Processing the call keyword arguments (line 462)
        kwargs_57889 = {}
        # Getting the type of 'np' (line 462)
        np_57877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 27), 'np', False)
        # Obtaining the member 'maximum' of a type (line 462)
        maximum_57878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 27), np_57877, 'maximum')
        # Calling maximum(args, kwargs) (line 462)
        maximum_call_result_57890 = invoke(stypy.reporting.localization.Localization(__file__, 462, 27), maximum_57878, *[abs_call_result_57883, abs_call_result_57888], **kwargs_57889)
        
        # Getting the type of 'rtol' (line 462)
        rtol_57891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 66), 'rtol')
        # Applying the binary operator '*' (line 462)
        result_mul_57892 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 27), '*', maximum_call_result_57890, rtol_57891)
        
        # Applying the binary operator '+' (line 462)
        result_add_57893 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 20), '+', atol_57876, result_mul_57892)
        
        # Assigning a type to the variable 'scale' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 12), 'scale', result_add_57893)
        
        # Assigning a Call to a Name (line 463):
        
        # Assigning a Call to a Name (line 463):
        
        # Call to norm(...): (line 463)
        # Processing the call arguments (line 463)
        # Getting the type of 'error' (line 463)
        error_57895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 30), 'error', False)
        # Getting the type of 'scale' (line 463)
        scale_57896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 38), 'scale', False)
        # Applying the binary operator 'div' (line 463)
        result_div_57897 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 30), 'div', error_57895, scale_57896)
        
        # Processing the call keyword arguments (line 463)
        kwargs_57898 = {}
        # Getting the type of 'norm' (line 463)
        norm_57894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 25), 'norm', False)
        # Calling norm(args, kwargs) (line 463)
        norm_call_result_57899 = invoke(stypy.reporting.localization.Localization(__file__, 463, 25), norm_57894, *[result_div_57897], **kwargs_57898)
        
        # Assigning a type to the variable 'error_norm' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 12), 'error_norm', norm_call_result_57899)
        
        # Assigning a BinOp to a Name (line 464):
        
        # Assigning a BinOp to a Name (line 464):
        float_57900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 21), 'float')
        int_57901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 28), 'int')
        # Getting the type of 'NEWTON_MAXITER' (line 464)
        NEWTON_MAXITER_57902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 32), 'NEWTON_MAXITER')
        # Applying the binary operator '*' (line 464)
        result_mul_57903 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 28), '*', int_57901, NEWTON_MAXITER_57902)
        
        int_57904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 49), 'int')
        # Applying the binary operator '+' (line 464)
        result_add_57905 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 28), '+', result_mul_57903, int_57904)
        
        # Applying the binary operator '*' (line 464)
        result_mul_57906 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 21), '*', float_57900, result_add_57905)
        
        int_57907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 55), 'int')
        # Getting the type of 'NEWTON_MAXITER' (line 464)
        NEWTON_MAXITER_57908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 59), 'NEWTON_MAXITER')
        # Applying the binary operator '*' (line 464)
        result_mul_57909 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 55), '*', int_57907, NEWTON_MAXITER_57908)
        
        # Getting the type of 'n_iter' (line 465)
        n_iter_57910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 57), 'n_iter')
        # Applying the binary operator '+' (line 464)
        result_add_57911 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 55), '+', result_mul_57909, n_iter_57910)
        
        # Applying the binary operator 'div' (line 464)
        result_div_57912 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 52), 'div', result_mul_57906, result_add_57911)
        
        # Assigning a type to the variable 'safety' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 12), 'safety', result_div_57912)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'rejected' (line 467)
        rejected_57913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 15), 'rejected')
        
        # Getting the type of 'error_norm' (line 467)
        error_norm_57914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 28), 'error_norm')
        int_57915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 41), 'int')
        # Applying the binary operator '>' (line 467)
        result_gt_57916 = python_operator(stypy.reporting.localization.Localization(__file__, 467, 28), '>', error_norm_57914, int_57915)
        
        # Applying the binary operator 'and' (line 467)
        result_and_keyword_57917 = python_operator(stypy.reporting.localization.Localization(__file__, 467, 15), 'and', rejected_57913, result_gt_57916)
        
        # Testing the type of an if condition (line 467)
        if_condition_57918 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 467, 12), result_and_keyword_57917)
        # Assigning a type to the variable 'if_condition_57918' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), 'if_condition_57918', if_condition_57918)
        # SSA begins for if statement (line 467)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 468):
        
        # Assigning a Call to a Name (line 468):
        
        # Call to solve_lu(...): (line 468)
        # Processing the call arguments (line 468)
        # Getting the type of 'LU_real' (line 468)
        LU_real_57921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 38), 'LU_real', False)
        
        # Call to fun(...): (line 468)
        # Processing the call arguments (line 468)
        # Getting the type of 't' (line 468)
        t_57924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 56), 't', False)
        # Getting the type of 'y' (line 468)
        y_57925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 59), 'y', False)
        # Getting the type of 'error' (line 468)
        error_57926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 63), 'error', False)
        # Applying the binary operator '+' (line 468)
        result_add_57927 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 59), '+', y_57925, error_57926)
        
        # Processing the call keyword arguments (line 468)
        kwargs_57928 = {}
        # Getting the type of 'self' (line 468)
        self_57922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 47), 'self', False)
        # Obtaining the member 'fun' of a type (line 468)
        fun_57923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 47), self_57922, 'fun')
        # Calling fun(args, kwargs) (line 468)
        fun_call_result_57929 = invoke(stypy.reporting.localization.Localization(__file__, 468, 47), fun_57923, *[t_57924, result_add_57927], **kwargs_57928)
        
        # Getting the type of 'ZE' (line 468)
        ZE_57930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 72), 'ZE', False)
        # Applying the binary operator '+' (line 468)
        result_add_57931 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 47), '+', fun_call_result_57929, ZE_57930)
        
        # Processing the call keyword arguments (line 468)
        kwargs_57932 = {}
        # Getting the type of 'self' (line 468)
        self_57919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 24), 'self', False)
        # Obtaining the member 'solve_lu' of a type (line 468)
        solve_lu_57920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 24), self_57919, 'solve_lu')
        # Calling solve_lu(args, kwargs) (line 468)
        solve_lu_call_result_57933 = invoke(stypy.reporting.localization.Localization(__file__, 468, 24), solve_lu_57920, *[LU_real_57921, result_add_57931], **kwargs_57932)
        
        # Assigning a type to the variable 'error' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 16), 'error', solve_lu_call_result_57933)
        
        # Assigning a Call to a Name (line 469):
        
        # Assigning a Call to a Name (line 469):
        
        # Call to norm(...): (line 469)
        # Processing the call arguments (line 469)
        # Getting the type of 'error' (line 469)
        error_57935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 34), 'error', False)
        # Getting the type of 'scale' (line 469)
        scale_57936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 42), 'scale', False)
        # Applying the binary operator 'div' (line 469)
        result_div_57937 = python_operator(stypy.reporting.localization.Localization(__file__, 469, 34), 'div', error_57935, scale_57936)
        
        # Processing the call keyword arguments (line 469)
        kwargs_57938 = {}
        # Getting the type of 'norm' (line 469)
        norm_57934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 29), 'norm', False)
        # Calling norm(args, kwargs) (line 469)
        norm_call_result_57939 = invoke(stypy.reporting.localization.Localization(__file__, 469, 29), norm_57934, *[result_div_57937], **kwargs_57938)
        
        # Assigning a type to the variable 'error_norm' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 16), 'error_norm', norm_call_result_57939)
        # SSA join for if statement (line 467)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'error_norm' (line 471)
        error_norm_57940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 15), 'error_norm')
        int_57941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 28), 'int')
        # Applying the binary operator '>' (line 471)
        result_gt_57942 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 15), '>', error_norm_57940, int_57941)
        
        # Testing the type of an if condition (line 471)
        if_condition_57943 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 471, 12), result_gt_57942)
        # Assigning a type to the variable 'if_condition_57943' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 12), 'if_condition_57943', if_condition_57943)
        # SSA begins for if statement (line 471)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 472):
        
        # Assigning a Call to a Name (line 472):
        
        # Call to predict_factor(...): (line 472)
        # Processing the call arguments (line 472)
        # Getting the type of 'h_abs' (line 472)
        h_abs_57945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 40), 'h_abs', False)
        # Getting the type of 'h_abs_old' (line 472)
        h_abs_old_57946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 47), 'h_abs_old', False)
        # Getting the type of 'error_norm' (line 473)
        error_norm_57947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 40), 'error_norm', False)
        # Getting the type of 'error_norm_old' (line 473)
        error_norm_old_57948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 52), 'error_norm_old', False)
        # Processing the call keyword arguments (line 472)
        kwargs_57949 = {}
        # Getting the type of 'predict_factor' (line 472)
        predict_factor_57944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 25), 'predict_factor', False)
        # Calling predict_factor(args, kwargs) (line 472)
        predict_factor_call_result_57950 = invoke(stypy.reporting.localization.Localization(__file__, 472, 25), predict_factor_57944, *[h_abs_57945, h_abs_old_57946, error_norm_57947, error_norm_old_57948], **kwargs_57949)
        
        # Assigning a type to the variable 'factor' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 16), 'factor', predict_factor_call_result_57950)
        
        # Getting the type of 'h_abs' (line 474)
        h_abs_57951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 16), 'h_abs')
        
        # Call to max(...): (line 474)
        # Processing the call arguments (line 474)
        # Getting the type of 'MIN_FACTOR' (line 474)
        MIN_FACTOR_57953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 29), 'MIN_FACTOR', False)
        # Getting the type of 'safety' (line 474)
        safety_57954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 41), 'safety', False)
        # Getting the type of 'factor' (line 474)
        factor_57955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 50), 'factor', False)
        # Applying the binary operator '*' (line 474)
        result_mul_57956 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 41), '*', safety_57954, factor_57955)
        
        # Processing the call keyword arguments (line 474)
        kwargs_57957 = {}
        # Getting the type of 'max' (line 474)
        max_57952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 25), 'max', False)
        # Calling max(args, kwargs) (line 474)
        max_call_result_57958 = invoke(stypy.reporting.localization.Localization(__file__, 474, 25), max_57952, *[MIN_FACTOR_57953, result_mul_57956], **kwargs_57957)
        
        # Applying the binary operator '*=' (line 474)
        result_imul_57959 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 16), '*=', h_abs_57951, max_call_result_57958)
        # Assigning a type to the variable 'h_abs' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 16), 'h_abs', result_imul_57959)
        
        
        # Assigning a Name to a Name (line 476):
        
        # Assigning a Name to a Name (line 476):
        # Getting the type of 'None' (line 476)
        None_57960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 26), 'None')
        # Assigning a type to the variable 'LU_real' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 16), 'LU_real', None_57960)
        
        # Assigning a Name to a Name (line 477):
        
        # Assigning a Name to a Name (line 477):
        # Getting the type of 'None' (line 477)
        None_57961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 29), 'None')
        # Assigning a type to the variable 'LU_complex' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 16), 'LU_complex', None_57961)
        
        # Assigning a Name to a Name (line 478):
        
        # Assigning a Name to a Name (line 478):
        # Getting the type of 'True' (line 478)
        True_57962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 27), 'True')
        # Assigning a type to the variable 'rejected' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 16), 'rejected', True_57962)
        # SSA branch for the else part of an if statement (line 471)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 480):
        
        # Assigning a Name to a Name (line 480):
        # Getting the type of 'True' (line 480)
        True_57963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 32), 'True')
        # Assigning a type to the variable 'step_accepted' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 16), 'step_accepted', True_57963)
        # SSA join for if statement (line 471)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 414)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BoolOp to a Name (line 482):
        
        # Assigning a BoolOp to a Name (line 482):
        
        # Evaluating a boolean operation
        
        # Getting the type of 'jac' (line 482)
        jac_57964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 24), 'jac')
        # Getting the type of 'None' (line 482)
        None_57965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 35), 'None')
        # Applying the binary operator 'isnot' (line 482)
        result_is_not_57966 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 24), 'isnot', jac_57964, None_57965)
        
        
        # Getting the type of 'n_iter' (line 482)
        n_iter_57967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 44), 'n_iter')
        int_57968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 53), 'int')
        # Applying the binary operator '>' (line 482)
        result_gt_57969 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 44), '>', n_iter_57967, int_57968)
        
        # Applying the binary operator 'and' (line 482)
        result_and_keyword_57970 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 24), 'and', result_is_not_57966, result_gt_57969)
        
        # Getting the type of 'rate' (line 482)
        rate_57971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 59), 'rate')
        float_57972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 66), 'float')
        # Applying the binary operator '>' (line 482)
        result_gt_57973 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 59), '>', rate_57971, float_57972)
        
        # Applying the binary operator 'and' (line 482)
        result_and_keyword_57974 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 24), 'and', result_and_keyword_57970, result_gt_57973)
        
        # Assigning a type to the variable 'recompute_jac' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), 'recompute_jac', result_and_keyword_57974)
        
        # Assigning a Call to a Name (line 484):
        
        # Assigning a Call to a Name (line 484):
        
        # Call to predict_factor(...): (line 484)
        # Processing the call arguments (line 484)
        # Getting the type of 'h_abs' (line 484)
        h_abs_57976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 32), 'h_abs', False)
        # Getting the type of 'h_abs_old' (line 484)
        h_abs_old_57977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 39), 'h_abs_old', False)
        # Getting the type of 'error_norm' (line 484)
        error_norm_57978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 50), 'error_norm', False)
        # Getting the type of 'error_norm_old' (line 484)
        error_norm_old_57979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 62), 'error_norm_old', False)
        # Processing the call keyword arguments (line 484)
        kwargs_57980 = {}
        # Getting the type of 'predict_factor' (line 484)
        predict_factor_57975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 17), 'predict_factor', False)
        # Calling predict_factor(args, kwargs) (line 484)
        predict_factor_call_result_57981 = invoke(stypy.reporting.localization.Localization(__file__, 484, 17), predict_factor_57975, *[h_abs_57976, h_abs_old_57977, error_norm_57978, error_norm_old_57979], **kwargs_57980)
        
        # Assigning a type to the variable 'factor' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'factor', predict_factor_call_result_57981)
        
        # Assigning a Call to a Name (line 485):
        
        # Assigning a Call to a Name (line 485):
        
        # Call to min(...): (line 485)
        # Processing the call arguments (line 485)
        # Getting the type of 'MAX_FACTOR' (line 485)
        MAX_FACTOR_57983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 21), 'MAX_FACTOR', False)
        # Getting the type of 'safety' (line 485)
        safety_57984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 33), 'safety', False)
        # Getting the type of 'factor' (line 485)
        factor_57985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 42), 'factor', False)
        # Applying the binary operator '*' (line 485)
        result_mul_57986 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 33), '*', safety_57984, factor_57985)
        
        # Processing the call keyword arguments (line 485)
        kwargs_57987 = {}
        # Getting the type of 'min' (line 485)
        min_57982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 17), 'min', False)
        # Calling min(args, kwargs) (line 485)
        min_call_result_57988 = invoke(stypy.reporting.localization.Localization(__file__, 485, 17), min_57982, *[MAX_FACTOR_57983, result_mul_57986], **kwargs_57987)
        
        # Assigning a type to the variable 'factor' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'factor', min_call_result_57988)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'recompute_jac' (line 487)
        recompute_jac_57989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 15), 'recompute_jac')
        # Applying the 'not' unary operator (line 487)
        result_not__57990 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 11), 'not', recompute_jac_57989)
        
        
        # Getting the type of 'factor' (line 487)
        factor_57991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 33), 'factor')
        float_57992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 42), 'float')
        # Applying the binary operator '<' (line 487)
        result_lt_57993 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 33), '<', factor_57991, float_57992)
        
        # Applying the binary operator 'and' (line 487)
        result_and_keyword_57994 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 11), 'and', result_not__57990, result_lt_57993)
        
        # Testing the type of an if condition (line 487)
        if_condition_57995 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 487, 8), result_and_keyword_57994)
        # Assigning a type to the variable 'if_condition_57995' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'if_condition_57995', if_condition_57995)
        # SSA begins for if statement (line 487)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 488):
        
        # Assigning a Num to a Name (line 488):
        int_57996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 21), 'int')
        # Assigning a type to the variable 'factor' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), 'factor', int_57996)
        # SSA branch for the else part of an if statement (line 487)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 490):
        
        # Assigning a Name to a Name (line 490):
        # Getting the type of 'None' (line 490)
        None_57997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 22), 'None')
        # Assigning a type to the variable 'LU_real' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'LU_real', None_57997)
        
        # Assigning a Name to a Name (line 491):
        
        # Assigning a Name to a Name (line 491):
        # Getting the type of 'None' (line 491)
        None_57998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 25), 'None')
        # Assigning a type to the variable 'LU_complex' (line 491)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'LU_complex', None_57998)
        # SSA join for if statement (line 487)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 493):
        
        # Assigning a Call to a Name (line 493):
        
        # Call to fun(...): (line 493)
        # Processing the call arguments (line 493)
        # Getting the type of 't_new' (line 493)
        t_new_58001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 25), 't_new', False)
        # Getting the type of 'y_new' (line 493)
        y_new_58002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 32), 'y_new', False)
        # Processing the call keyword arguments (line 493)
        kwargs_58003 = {}
        # Getting the type of 'self' (line 493)
        self_57999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 16), 'self', False)
        # Obtaining the member 'fun' of a type (line 493)
        fun_58000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 16), self_57999, 'fun')
        # Calling fun(args, kwargs) (line 493)
        fun_call_result_58004 = invoke(stypy.reporting.localization.Localization(__file__, 493, 16), fun_58000, *[t_new_58001, y_new_58002], **kwargs_58003)
        
        # Assigning a type to the variable 'f_new' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'f_new', fun_call_result_58004)
        
        # Getting the type of 'recompute_jac' (line 494)
        recompute_jac_58005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 11), 'recompute_jac')
        # Testing the type of an if condition (line 494)
        if_condition_58006 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 494, 8), recompute_jac_58005)
        # Assigning a type to the variable 'if_condition_58006' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'if_condition_58006', if_condition_58006)
        # SSA begins for if statement (line 494)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 495):
        
        # Assigning a Call to a Name (line 495):
        
        # Call to jac(...): (line 495)
        # Processing the call arguments (line 495)
        # Getting the type of 't_new' (line 495)
        t_new_58008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 20), 't_new', False)
        # Getting the type of 'y_new' (line 495)
        y_new_58009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 27), 'y_new', False)
        # Getting the type of 'f_new' (line 495)
        f_new_58010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 34), 'f_new', False)
        # Processing the call keyword arguments (line 495)
        kwargs_58011 = {}
        # Getting the type of 'jac' (line 495)
        jac_58007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 16), 'jac', False)
        # Calling jac(args, kwargs) (line 495)
        jac_call_result_58012 = invoke(stypy.reporting.localization.Localization(__file__, 495, 16), jac_58007, *[t_new_58008, y_new_58009, f_new_58010], **kwargs_58011)
        
        # Assigning a type to the variable 'J' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), 'J', jac_call_result_58012)
        
        # Assigning a Name to a Name (line 496):
        
        # Assigning a Name to a Name (line 496):
        # Getting the type of 'True' (line 496)
        True_58013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 26), 'True')
        # Assigning a type to the variable 'current_jac' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'current_jac', True_58013)
        # SSA branch for the else part of an if statement (line 494)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 497)
        # Getting the type of 'jac' (line 497)
        jac_58014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 13), 'jac')
        # Getting the type of 'None' (line 497)
        None_58015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 24), 'None')
        
        (may_be_58016, more_types_in_union_58017) = may_not_be_none(jac_58014, None_58015)

        if may_be_58016:

            if more_types_in_union_58017:
                # Runtime conditional SSA (line 497)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 498):
            
            # Assigning a Name to a Name (line 498):
            # Getting the type of 'False' (line 498)
            False_58018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 26), 'False')
            # Assigning a type to the variable 'current_jac' (line 498)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 12), 'current_jac', False_58018)

            if more_types_in_union_58017:
                # SSA join for if statement (line 497)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 494)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Attribute (line 500):
        
        # Assigning a Attribute to a Attribute (line 500):
        # Getting the type of 'self' (line 500)
        self_58019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 25), 'self')
        # Obtaining the member 'h_abs' of a type (line 500)
        h_abs_58020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 25), self_58019, 'h_abs')
        # Getting the type of 'self' (line 500)
        self_58021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'self')
        # Setting the type of the member 'h_abs_old' of a type (line 500)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 8), self_58021, 'h_abs_old', h_abs_58020)
        
        # Assigning a Name to a Attribute (line 501):
        
        # Assigning a Name to a Attribute (line 501):
        # Getting the type of 'error_norm' (line 501)
        error_norm_58022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 30), 'error_norm')
        # Getting the type of 'self' (line 501)
        self_58023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 'self')
        # Setting the type of the member 'error_norm_old' of a type (line 501)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 8), self_58023, 'error_norm_old', error_norm_58022)
        
        # Assigning a BinOp to a Attribute (line 503):
        
        # Assigning a BinOp to a Attribute (line 503):
        # Getting the type of 'h_abs' (line 503)
        h_abs_58024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 21), 'h_abs')
        # Getting the type of 'factor' (line 503)
        factor_58025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 29), 'factor')
        # Applying the binary operator '*' (line 503)
        result_mul_58026 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 21), '*', h_abs_58024, factor_58025)
        
        # Getting the type of 'self' (line 503)
        self_58027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'self')
        # Setting the type of the member 'h_abs' of a type (line 503)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 8), self_58027, 'h_abs', result_mul_58026)
        
        # Assigning a Name to a Attribute (line 505):
        
        # Assigning a Name to a Attribute (line 505):
        # Getting the type of 'y' (line 505)
        y_58028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 21), 'y')
        # Getting the type of 'self' (line 505)
        self_58029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), 'self')
        # Setting the type of the member 'y_old' of a type (line 505)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 8), self_58029, 'y_old', y_58028)
        
        # Assigning a Name to a Attribute (line 507):
        
        # Assigning a Name to a Attribute (line 507):
        # Getting the type of 't_new' (line 507)
        t_new_58030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 17), 't_new')
        # Getting the type of 'self' (line 507)
        self_58031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'self')
        # Setting the type of the member 't' of a type (line 507)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 8), self_58031, 't', t_new_58030)
        
        # Assigning a Name to a Attribute (line 508):
        
        # Assigning a Name to a Attribute (line 508):
        # Getting the type of 'y_new' (line 508)
        y_new_58032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 17), 'y_new')
        # Getting the type of 'self' (line 508)
        self_58033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'self')
        # Setting the type of the member 'y' of a type (line 508)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 8), self_58033, 'y', y_new_58032)
        
        # Assigning a Name to a Attribute (line 509):
        
        # Assigning a Name to a Attribute (line 509):
        # Getting the type of 'f_new' (line 509)
        f_new_58034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 17), 'f_new')
        # Getting the type of 'self' (line 509)
        self_58035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'self')
        # Setting the type of the member 'f' of a type (line 509)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 8), self_58035, 'f', f_new_58034)
        
        # Assigning a Name to a Attribute (line 511):
        
        # Assigning a Name to a Attribute (line 511):
        # Getting the type of 'Z' (line 511)
        Z_58036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 17), 'Z')
        # Getting the type of 'self' (line 511)
        self_58037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 8), 'self')
        # Setting the type of the member 'Z' of a type (line 511)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 8), self_58037, 'Z', Z_58036)
        
        # Assigning a Name to a Attribute (line 513):
        
        # Assigning a Name to a Attribute (line 513):
        # Getting the type of 'LU_real' (line 513)
        LU_real_58038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 23), 'LU_real')
        # Getting the type of 'self' (line 513)
        self_58039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'self')
        # Setting the type of the member 'LU_real' of a type (line 513)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 8), self_58039, 'LU_real', LU_real_58038)
        
        # Assigning a Name to a Attribute (line 514):
        
        # Assigning a Name to a Attribute (line 514):
        # Getting the type of 'LU_complex' (line 514)
        LU_complex_58040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 26), 'LU_complex')
        # Getting the type of 'self' (line 514)
        self_58041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 8), 'self')
        # Setting the type of the member 'LU_complex' of a type (line 514)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 8), self_58041, 'LU_complex', LU_complex_58040)
        
        # Assigning a Name to a Attribute (line 515):
        
        # Assigning a Name to a Attribute (line 515):
        # Getting the type of 'current_jac' (line 515)
        current_jac_58042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 27), 'current_jac')
        # Getting the type of 'self' (line 515)
        self_58043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 'self')
        # Setting the type of the member 'current_jac' of a type (line 515)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 8), self_58043, 'current_jac', current_jac_58042)
        
        # Assigning a Name to a Attribute (line 516):
        
        # Assigning a Name to a Attribute (line 516):
        # Getting the type of 'J' (line 516)
        J_58044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 17), 'J')
        # Getting the type of 'self' (line 516)
        self_58045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'self')
        # Setting the type of the member 'J' of a type (line 516)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 8), self_58045, 'J', J_58044)
        
        # Assigning a Name to a Attribute (line 518):
        
        # Assigning a Name to a Attribute (line 518):
        # Getting the type of 't' (line 518)
        t_58046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 21), 't')
        # Getting the type of 'self' (line 518)
        self_58047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'self')
        # Setting the type of the member 't_old' of a type (line 518)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 8), self_58047, 't_old', t_58046)
        
        # Assigning a Call to a Attribute (line 519):
        
        # Assigning a Call to a Attribute (line 519):
        
        # Call to _compute_dense_output(...): (line 519)
        # Processing the call keyword arguments (line 519)
        kwargs_58050 = {}
        # Getting the type of 'self' (line 519)
        self_58048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 19), 'self', False)
        # Obtaining the member '_compute_dense_output' of a type (line 519)
        _compute_dense_output_58049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 19), self_58048, '_compute_dense_output')
        # Calling _compute_dense_output(args, kwargs) (line 519)
        _compute_dense_output_call_result_58051 = invoke(stypy.reporting.localization.Localization(__file__, 519, 19), _compute_dense_output_58049, *[], **kwargs_58050)
        
        # Getting the type of 'self' (line 519)
        self_58052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'self')
        # Setting the type of the member 'sol' of a type (line 519)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 8), self_58052, 'sol', _compute_dense_output_call_result_58051)
        
        # Obtaining an instance of the builtin type 'tuple' (line 521)
        tuple_58053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 521)
        # Adding element type (line 521)
        # Getting the type of 'step_accepted' (line 521)
        step_accepted_58054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 15), 'step_accepted')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 15), tuple_58053, step_accepted_58054)
        # Adding element type (line 521)
        # Getting the type of 'message' (line 521)
        message_58055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 30), 'message')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 15), tuple_58053, message_58055)
        
        # Assigning a type to the variable 'stypy_return_type' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'stypy_return_type', tuple_58053)
        
        # ################# End of '_step_impl(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_step_impl' in the type store
        # Getting the type of 'stypy_return_type' (line 381)
        stypy_return_type_58056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58056)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_step_impl'
        return stypy_return_type_58056


    @norecursion
    def _compute_dense_output(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_compute_dense_output'
        module_type_store = module_type_store.open_function_context('_compute_dense_output', 523, 4, False)
        # Assigning a type to the variable 'self' (line 524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Radau._compute_dense_output.__dict__.__setitem__('stypy_localization', localization)
        Radau._compute_dense_output.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Radau._compute_dense_output.__dict__.__setitem__('stypy_type_store', module_type_store)
        Radau._compute_dense_output.__dict__.__setitem__('stypy_function_name', 'Radau._compute_dense_output')
        Radau._compute_dense_output.__dict__.__setitem__('stypy_param_names_list', [])
        Radau._compute_dense_output.__dict__.__setitem__('stypy_varargs_param_name', None)
        Radau._compute_dense_output.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Radau._compute_dense_output.__dict__.__setitem__('stypy_call_defaults', defaults)
        Radau._compute_dense_output.__dict__.__setitem__('stypy_call_varargs', varargs)
        Radau._compute_dense_output.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Radau._compute_dense_output.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Radau._compute_dense_output', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_compute_dense_output', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_compute_dense_output(...)' code ##################

        
        # Assigning a Call to a Name (line 524):
        
        # Assigning a Call to a Name (line 524):
        
        # Call to dot(...): (line 524)
        # Processing the call arguments (line 524)
        # Getting the type of 'self' (line 524)
        self_58059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 19), 'self', False)
        # Obtaining the member 'Z' of a type (line 524)
        Z_58060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 19), self_58059, 'Z')
        # Obtaining the member 'T' of a type (line 524)
        T_58061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 19), Z_58060, 'T')
        # Getting the type of 'P' (line 524)
        P_58062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 29), 'P', False)
        # Processing the call keyword arguments (line 524)
        kwargs_58063 = {}
        # Getting the type of 'np' (line 524)
        np_58057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 12), 'np', False)
        # Obtaining the member 'dot' of a type (line 524)
        dot_58058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 12), np_58057, 'dot')
        # Calling dot(args, kwargs) (line 524)
        dot_call_result_58064 = invoke(stypy.reporting.localization.Localization(__file__, 524, 12), dot_58058, *[T_58061, P_58062], **kwargs_58063)
        
        # Assigning a type to the variable 'Q' (line 524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 8), 'Q', dot_call_result_58064)
        
        # Call to RadauDenseOutput(...): (line 525)
        # Processing the call arguments (line 525)
        # Getting the type of 'self' (line 525)
        self_58066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 32), 'self', False)
        # Obtaining the member 't_old' of a type (line 525)
        t_old_58067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 32), self_58066, 't_old')
        # Getting the type of 'self' (line 525)
        self_58068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 44), 'self', False)
        # Obtaining the member 't' of a type (line 525)
        t_58069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 44), self_58068, 't')
        # Getting the type of 'self' (line 525)
        self_58070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 52), 'self', False)
        # Obtaining the member 'y_old' of a type (line 525)
        y_old_58071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 52), self_58070, 'y_old')
        # Getting the type of 'Q' (line 525)
        Q_58072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 64), 'Q', False)
        # Processing the call keyword arguments (line 525)
        kwargs_58073 = {}
        # Getting the type of 'RadauDenseOutput' (line 525)
        RadauDenseOutput_58065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 15), 'RadauDenseOutput', False)
        # Calling RadauDenseOutput(args, kwargs) (line 525)
        RadauDenseOutput_call_result_58074 = invoke(stypy.reporting.localization.Localization(__file__, 525, 15), RadauDenseOutput_58065, *[t_old_58067, t_58069, y_old_58071, Q_58072], **kwargs_58073)
        
        # Assigning a type to the variable 'stypy_return_type' (line 525)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'stypy_return_type', RadauDenseOutput_call_result_58074)
        
        # ################# End of '_compute_dense_output(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_compute_dense_output' in the type store
        # Getting the type of 'stypy_return_type' (line 523)
        stypy_return_type_58075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58075)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_compute_dense_output'
        return stypy_return_type_58075


    @norecursion
    def _dense_output_impl(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_dense_output_impl'
        module_type_store = module_type_store.open_function_context('_dense_output_impl', 527, 4, False)
        # Assigning a type to the variable 'self' (line 528)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Radau._dense_output_impl.__dict__.__setitem__('stypy_localization', localization)
        Radau._dense_output_impl.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Radau._dense_output_impl.__dict__.__setitem__('stypy_type_store', module_type_store)
        Radau._dense_output_impl.__dict__.__setitem__('stypy_function_name', 'Radau._dense_output_impl')
        Radau._dense_output_impl.__dict__.__setitem__('stypy_param_names_list', [])
        Radau._dense_output_impl.__dict__.__setitem__('stypy_varargs_param_name', None)
        Radau._dense_output_impl.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Radau._dense_output_impl.__dict__.__setitem__('stypy_call_defaults', defaults)
        Radau._dense_output_impl.__dict__.__setitem__('stypy_call_varargs', varargs)
        Radau._dense_output_impl.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Radau._dense_output_impl.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Radau._dense_output_impl', [], None, None, defaults, varargs, kwargs)

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

        # Getting the type of 'self' (line 528)
        self_58076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 15), 'self')
        # Obtaining the member 'sol' of a type (line 528)
        sol_58077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 15), self_58076, 'sol')
        # Assigning a type to the variable 'stypy_return_type' (line 528)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 8), 'stypy_return_type', sol_58077)
        
        # ################# End of '_dense_output_impl(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_dense_output_impl' in the type store
        # Getting the type of 'stypy_return_type' (line 527)
        stypy_return_type_58078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58078)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_dense_output_impl'
        return stypy_return_type_58078


# Assigning a type to the variable 'Radau' (line 180)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 0), 'Radau', Radau)
# Declaration of the 'RadauDenseOutput' class
# Getting the type of 'DenseOutput' (line 531)
DenseOutput_58079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 23), 'DenseOutput')

class RadauDenseOutput(DenseOutput_58079, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 532, 4, False)
        # Assigning a type to the variable 'self' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RadauDenseOutput.__init__', ['t_old', 't', 'y_old', 'Q'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['t_old', 't', 'y_old', 'Q'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 533)
        # Processing the call arguments (line 533)
        # Getting the type of 't_old' (line 533)
        t_old_58086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 47), 't_old', False)
        # Getting the type of 't' (line 533)
        t_58087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 54), 't', False)
        # Processing the call keyword arguments (line 533)
        kwargs_58088 = {}
        
        # Call to super(...): (line 533)
        # Processing the call arguments (line 533)
        # Getting the type of 'RadauDenseOutput' (line 533)
        RadauDenseOutput_58081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 14), 'RadauDenseOutput', False)
        # Getting the type of 'self' (line 533)
        self_58082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 32), 'self', False)
        # Processing the call keyword arguments (line 533)
        kwargs_58083 = {}
        # Getting the type of 'super' (line 533)
        super_58080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'super', False)
        # Calling super(args, kwargs) (line 533)
        super_call_result_58084 = invoke(stypy.reporting.localization.Localization(__file__, 533, 8), super_58080, *[RadauDenseOutput_58081, self_58082], **kwargs_58083)
        
        # Obtaining the member '__init__' of a type (line 533)
        init___58085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 8), super_call_result_58084, '__init__')
        # Calling __init__(args, kwargs) (line 533)
        init___call_result_58089 = invoke(stypy.reporting.localization.Localization(__file__, 533, 8), init___58085, *[t_old_58086, t_58087], **kwargs_58088)
        
        
        # Assigning a BinOp to a Attribute (line 534):
        
        # Assigning a BinOp to a Attribute (line 534):
        # Getting the type of 't' (line 534)
        t_58090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 17), 't')
        # Getting the type of 't_old' (line 534)
        t_old_58091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 21), 't_old')
        # Applying the binary operator '-' (line 534)
        result_sub_58092 = python_operator(stypy.reporting.localization.Localization(__file__, 534, 17), '-', t_58090, t_old_58091)
        
        # Getting the type of 'self' (line 534)
        self_58093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 8), 'self')
        # Setting the type of the member 'h' of a type (line 534)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 8), self_58093, 'h', result_sub_58092)
        
        # Assigning a Name to a Attribute (line 535):
        
        # Assigning a Name to a Attribute (line 535):
        # Getting the type of 'Q' (line 535)
        Q_58094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 17), 'Q')
        # Getting the type of 'self' (line 535)
        self_58095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'self')
        # Setting the type of the member 'Q' of a type (line 535)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 8), self_58095, 'Q', Q_58094)
        
        # Assigning a BinOp to a Attribute (line 536):
        
        # Assigning a BinOp to a Attribute (line 536):
        
        # Obtaining the type of the subscript
        int_58096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 29), 'int')
        # Getting the type of 'Q' (line 536)
        Q_58097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 21), 'Q')
        # Obtaining the member 'shape' of a type (line 536)
        shape_58098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 21), Q_58097, 'shape')
        # Obtaining the member '__getitem__' of a type (line 536)
        getitem___58099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 21), shape_58098, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 536)
        subscript_call_result_58100 = invoke(stypy.reporting.localization.Localization(__file__, 536, 21), getitem___58099, int_58096)
        
        int_58101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 34), 'int')
        # Applying the binary operator '-' (line 536)
        result_sub_58102 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 21), '-', subscript_call_result_58100, int_58101)
        
        # Getting the type of 'self' (line 536)
        self_58103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'self')
        # Setting the type of the member 'order' of a type (line 536)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 8), self_58103, 'order', result_sub_58102)
        
        # Assigning a Name to a Attribute (line 537):
        
        # Assigning a Name to a Attribute (line 537):
        # Getting the type of 'y_old' (line 537)
        y_old_58104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 21), 'y_old')
        # Getting the type of 'self' (line 537)
        self_58105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'self')
        # Setting the type of the member 'y_old' of a type (line 537)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 8), self_58105, 'y_old', y_old_58104)
        
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
        module_type_store = module_type_store.open_function_context('_call_impl', 539, 4, False)
        # Assigning a type to the variable 'self' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RadauDenseOutput._call_impl.__dict__.__setitem__('stypy_localization', localization)
        RadauDenseOutput._call_impl.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RadauDenseOutput._call_impl.__dict__.__setitem__('stypy_type_store', module_type_store)
        RadauDenseOutput._call_impl.__dict__.__setitem__('stypy_function_name', 'RadauDenseOutput._call_impl')
        RadauDenseOutput._call_impl.__dict__.__setitem__('stypy_param_names_list', ['t'])
        RadauDenseOutput._call_impl.__dict__.__setitem__('stypy_varargs_param_name', None)
        RadauDenseOutput._call_impl.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RadauDenseOutput._call_impl.__dict__.__setitem__('stypy_call_defaults', defaults)
        RadauDenseOutput._call_impl.__dict__.__setitem__('stypy_call_varargs', varargs)
        RadauDenseOutput._call_impl.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RadauDenseOutput._call_impl.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RadauDenseOutput._call_impl', ['t'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a BinOp to a Name (line 540):
        
        # Assigning a BinOp to a Name (line 540):
        # Getting the type of 't' (line 540)
        t_58106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 13), 't')
        # Getting the type of 'self' (line 540)
        self_58107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 17), 'self')
        # Obtaining the member 't_old' of a type (line 540)
        t_old_58108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 17), self_58107, 't_old')
        # Applying the binary operator '-' (line 540)
        result_sub_58109 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 13), '-', t_58106, t_old_58108)
        
        # Getting the type of 'self' (line 540)
        self_58110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 31), 'self')
        # Obtaining the member 'h' of a type (line 540)
        h_58111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 31), self_58110, 'h')
        # Applying the binary operator 'div' (line 540)
        result_div_58112 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 12), 'div', result_sub_58109, h_58111)
        
        # Assigning a type to the variable 'x' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'x', result_div_58112)
        
        
        # Getting the type of 't' (line 541)
        t_58113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 11), 't')
        # Obtaining the member 'ndim' of a type (line 541)
        ndim_58114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 11), t_58113, 'ndim')
        int_58115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 21), 'int')
        # Applying the binary operator '==' (line 541)
        result_eq_58116 = python_operator(stypy.reporting.localization.Localization(__file__, 541, 11), '==', ndim_58114, int_58115)
        
        # Testing the type of an if condition (line 541)
        if_condition_58117 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 541, 8), result_eq_58116)
        # Assigning a type to the variable 'if_condition_58117' (line 541)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'if_condition_58117', if_condition_58117)
        # SSA begins for if statement (line 541)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 542):
        
        # Assigning a Call to a Name (line 542):
        
        # Call to tile(...): (line 542)
        # Processing the call arguments (line 542)
        # Getting the type of 'x' (line 542)
        x_58120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 24), 'x', False)
        # Getting the type of 'self' (line 542)
        self_58121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 27), 'self', False)
        # Obtaining the member 'order' of a type (line 542)
        order_58122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 27), self_58121, 'order')
        int_58123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 40), 'int')
        # Applying the binary operator '+' (line 542)
        result_add_58124 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 27), '+', order_58122, int_58123)
        
        # Processing the call keyword arguments (line 542)
        kwargs_58125 = {}
        # Getting the type of 'np' (line 542)
        np_58118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 16), 'np', False)
        # Obtaining the member 'tile' of a type (line 542)
        tile_58119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 16), np_58118, 'tile')
        # Calling tile(args, kwargs) (line 542)
        tile_call_result_58126 = invoke(stypy.reporting.localization.Localization(__file__, 542, 16), tile_58119, *[x_58120, result_add_58124], **kwargs_58125)
        
        # Assigning a type to the variable 'p' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 12), 'p', tile_call_result_58126)
        
        # Assigning a Call to a Name (line 543):
        
        # Assigning a Call to a Name (line 543):
        
        # Call to cumprod(...): (line 543)
        # Processing the call arguments (line 543)
        # Getting the type of 'p' (line 543)
        p_58129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 27), 'p', False)
        # Processing the call keyword arguments (line 543)
        kwargs_58130 = {}
        # Getting the type of 'np' (line 543)
        np_58127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 16), 'np', False)
        # Obtaining the member 'cumprod' of a type (line 543)
        cumprod_58128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 16), np_58127, 'cumprod')
        # Calling cumprod(args, kwargs) (line 543)
        cumprod_call_result_58131 = invoke(stypy.reporting.localization.Localization(__file__, 543, 16), cumprod_58128, *[p_58129], **kwargs_58130)
        
        # Assigning a type to the variable 'p' (line 543)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 12), 'p', cumprod_call_result_58131)
        # SSA branch for the else part of an if statement (line 541)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 545):
        
        # Assigning a Call to a Name (line 545):
        
        # Call to tile(...): (line 545)
        # Processing the call arguments (line 545)
        # Getting the type of 'x' (line 545)
        x_58134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 24), 'x', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 545)
        tuple_58135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 545)
        # Adding element type (line 545)
        # Getting the type of 'self' (line 545)
        self_58136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 28), 'self', False)
        # Obtaining the member 'order' of a type (line 545)
        order_58137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 28), self_58136, 'order')
        int_58138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 41), 'int')
        # Applying the binary operator '+' (line 545)
        result_add_58139 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 28), '+', order_58137, int_58138)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 545, 28), tuple_58135, result_add_58139)
        # Adding element type (line 545)
        int_58140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 545, 28), tuple_58135, int_58140)
        
        # Processing the call keyword arguments (line 545)
        kwargs_58141 = {}
        # Getting the type of 'np' (line 545)
        np_58132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 16), 'np', False)
        # Obtaining the member 'tile' of a type (line 545)
        tile_58133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 16), np_58132, 'tile')
        # Calling tile(args, kwargs) (line 545)
        tile_call_result_58142 = invoke(stypy.reporting.localization.Localization(__file__, 545, 16), tile_58133, *[x_58134, tuple_58135], **kwargs_58141)
        
        # Assigning a type to the variable 'p' (line 545)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 12), 'p', tile_call_result_58142)
        
        # Assigning a Call to a Name (line 546):
        
        # Assigning a Call to a Name (line 546):
        
        # Call to cumprod(...): (line 546)
        # Processing the call arguments (line 546)
        # Getting the type of 'p' (line 546)
        p_58145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 27), 'p', False)
        # Processing the call keyword arguments (line 546)
        int_58146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 35), 'int')
        keyword_58147 = int_58146
        kwargs_58148 = {'axis': keyword_58147}
        # Getting the type of 'np' (line 546)
        np_58143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 16), 'np', False)
        # Obtaining the member 'cumprod' of a type (line 546)
        cumprod_58144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 16), np_58143, 'cumprod')
        # Calling cumprod(args, kwargs) (line 546)
        cumprod_call_result_58149 = invoke(stypy.reporting.localization.Localization(__file__, 546, 16), cumprod_58144, *[p_58145], **kwargs_58148)
        
        # Assigning a type to the variable 'p' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 12), 'p', cumprod_call_result_58149)
        # SSA join for if statement (line 541)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 548):
        
        # Assigning a Call to a Name (line 548):
        
        # Call to dot(...): (line 548)
        # Processing the call arguments (line 548)
        # Getting the type of 'self' (line 548)
        self_58152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 19), 'self', False)
        # Obtaining the member 'Q' of a type (line 548)
        Q_58153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 19), self_58152, 'Q')
        # Getting the type of 'p' (line 548)
        p_58154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 27), 'p', False)
        # Processing the call keyword arguments (line 548)
        kwargs_58155 = {}
        # Getting the type of 'np' (line 548)
        np_58150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 12), 'np', False)
        # Obtaining the member 'dot' of a type (line 548)
        dot_58151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 12), np_58150, 'dot')
        # Calling dot(args, kwargs) (line 548)
        dot_call_result_58156 = invoke(stypy.reporting.localization.Localization(__file__, 548, 12), dot_58151, *[Q_58153, p_58154], **kwargs_58155)
        
        # Assigning a type to the variable 'y' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'y', dot_call_result_58156)
        
        
        # Getting the type of 'y' (line 549)
        y_58157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 11), 'y')
        # Obtaining the member 'ndim' of a type (line 549)
        ndim_58158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 11), y_58157, 'ndim')
        int_58159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 21), 'int')
        # Applying the binary operator '==' (line 549)
        result_eq_58160 = python_operator(stypy.reporting.localization.Localization(__file__, 549, 11), '==', ndim_58158, int_58159)
        
        # Testing the type of an if condition (line 549)
        if_condition_58161 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 549, 8), result_eq_58160)
        # Assigning a type to the variable 'if_condition_58161' (line 549)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 8), 'if_condition_58161', if_condition_58161)
        # SSA begins for if statement (line 549)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'y' (line 550)
        y_58162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 12), 'y')
        
        # Obtaining the type of the subscript
        slice_58163 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 550, 17), None, None, None)
        # Getting the type of 'None' (line 550)
        None_58164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 31), 'None')
        # Getting the type of 'self' (line 550)
        self_58165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 17), 'self')
        # Obtaining the member 'y_old' of a type (line 550)
        y_old_58166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 17), self_58165, 'y_old')
        # Obtaining the member '__getitem__' of a type (line 550)
        getitem___58167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 17), y_old_58166, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 550)
        subscript_call_result_58168 = invoke(stypy.reporting.localization.Localization(__file__, 550, 17), getitem___58167, (slice_58163, None_58164))
        
        # Applying the binary operator '+=' (line 550)
        result_iadd_58169 = python_operator(stypy.reporting.localization.Localization(__file__, 550, 12), '+=', y_58162, subscript_call_result_58168)
        # Assigning a type to the variable 'y' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 12), 'y', result_iadd_58169)
        
        # SSA branch for the else part of an if statement (line 549)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'y' (line 552)
        y_58170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 12), 'y')
        # Getting the type of 'self' (line 552)
        self_58171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 17), 'self')
        # Obtaining the member 'y_old' of a type (line 552)
        y_old_58172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 17), self_58171, 'y_old')
        # Applying the binary operator '+=' (line 552)
        result_iadd_58173 = python_operator(stypy.reporting.localization.Localization(__file__, 552, 12), '+=', y_58170, y_old_58172)
        # Assigning a type to the variable 'y' (line 552)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 12), 'y', result_iadd_58173)
        
        # SSA join for if statement (line 549)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'y' (line 554)
        y_58174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 15), 'y')
        # Assigning a type to the variable 'stypy_return_type' (line 554)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 8), 'stypy_return_type', y_58174)
        
        # ################# End of '_call_impl(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_call_impl' in the type store
        # Getting the type of 'stypy_return_type' (line 539)
        stypy_return_type_58175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58175)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_call_impl'
        return stypy_return_type_58175


# Assigning a type to the variable 'RadauDenseOutput' (line 531)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 0), 'RadauDenseOutput', RadauDenseOutput)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
