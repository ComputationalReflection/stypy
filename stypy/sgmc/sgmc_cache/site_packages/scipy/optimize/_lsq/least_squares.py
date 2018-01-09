
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Generic interface for least-square minimization.'''
2: from __future__ import division, print_function, absolute_import
3: 
4: from warnings import warn
5: 
6: import numpy as np
7: from numpy.linalg import norm
8: 
9: from scipy.sparse import issparse, csr_matrix
10: from scipy.sparse.linalg import LinearOperator
11: from scipy.optimize import _minpack, OptimizeResult
12: from scipy.optimize._numdiff import approx_derivative, group_columns
13: from scipy._lib.six import string_types
14: 
15: from .trf import trf
16: from .dogbox import dogbox
17: from .common import EPS, in_bounds, make_strictly_feasible
18: 
19: 
20: TERMINATION_MESSAGES = {
21:     -1: "Improper input parameters status returned from `leastsq`",
22:     0: "The maximum number of function evaluations is exceeded.",
23:     1: "`gtol` termination condition is satisfied.",
24:     2: "`ftol` termination condition is satisfied.",
25:     3: "`xtol` termination condition is satisfied.",
26:     4: "Both `ftol` and `xtol` termination conditions are satisfied."
27: }
28: 
29: 
30: FROM_MINPACK_TO_COMMON = {
31:     0: -1,  # Improper input parameters from MINPACK.
32:     1: 2,
33:     2: 3,
34:     3: 4,
35:     4: 1,
36:     5: 0
37:     # There are 6, 7, 8 for too small tolerance parameters,
38:     # but we guard against it by checking ftol, xtol, gtol beforehand.
39: }
40: 
41: 
42: def call_minpack(fun, x0, jac, ftol, xtol, gtol, max_nfev, x_scale, diff_step):
43:     n = x0.size
44: 
45:     if diff_step is None:
46:         epsfcn = EPS
47:     else:
48:         epsfcn = diff_step**2
49: 
50:     # Compute MINPACK's `diag`, which is inverse of our `x_scale` and
51:     # ``x_scale='jac'`` corresponds to ``diag=None``.
52:     if isinstance(x_scale, string_types) and x_scale == 'jac':
53:         diag = None
54:     else:
55:         diag = 1 / x_scale
56: 
57:     full_output = True
58:     col_deriv = False
59:     factor = 100.0
60: 
61:     if jac is None:
62:         if max_nfev is None:
63:             # n squared to account for Jacobian evaluations.
64:             max_nfev = 100 * n * (n + 1)
65:         x, info, status = _minpack._lmdif(
66:             fun, x0, (), full_output, ftol, xtol, gtol,
67:             max_nfev, epsfcn, factor, diag)
68:     else:
69:         if max_nfev is None:
70:             max_nfev = 100 * n
71:         x, info, status = _minpack._lmder(
72:             fun, jac, x0, (), full_output, col_deriv,
73:             ftol, xtol, gtol, max_nfev, factor, diag)
74: 
75:     f = info['fvec']
76: 
77:     if callable(jac):
78:         J = jac(x)
79:     else:
80:         J = np.atleast_2d(approx_derivative(fun, x))
81: 
82:     cost = 0.5 * np.dot(f, f)
83:     g = J.T.dot(f)
84:     g_norm = norm(g, ord=np.inf)
85: 
86:     nfev = info['nfev']
87:     njev = info.get('njev', None)
88: 
89:     status = FROM_MINPACK_TO_COMMON[status]
90:     active_mask = np.zeros_like(x0, dtype=int)
91: 
92:     return OptimizeResult(
93:         x=x, cost=cost, fun=f, jac=J, grad=g, optimality=g_norm,
94:         active_mask=active_mask, nfev=nfev, njev=njev, status=status)
95: 
96: 
97: def prepare_bounds(bounds, n):
98:     lb, ub = [np.asarray(b, dtype=float) for b in bounds]
99:     if lb.ndim == 0:
100:         lb = np.resize(lb, n)
101: 
102:     if ub.ndim == 0:
103:         ub = np.resize(ub, n)
104: 
105:     return lb, ub
106: 
107: 
108: def check_tolerance(ftol, xtol, gtol):
109:     message = "{} is too low, setting to machine epsilon {}."
110:     if ftol < EPS:
111:         warn(message.format("`ftol`", EPS))
112:         ftol = EPS
113:     if xtol < EPS:
114:         warn(message.format("`xtol`", EPS))
115:         xtol = EPS
116:     if gtol < EPS:
117:         warn(message.format("`gtol`", EPS))
118:         gtol = EPS
119: 
120:     return ftol, xtol, gtol
121: 
122: 
123: def check_x_scale(x_scale, x0):
124:     if isinstance(x_scale, string_types) and x_scale == 'jac':
125:         return x_scale
126: 
127:     try:
128:         x_scale = np.asarray(x_scale, dtype=float)
129:         valid = np.all(np.isfinite(x_scale)) and np.all(x_scale > 0)
130:     except (ValueError, TypeError):
131:         valid = False
132: 
133:     if not valid:
134:         raise ValueError("`x_scale` must be 'jac' or array_like with "
135:                          "positive numbers.")
136: 
137:     if x_scale.ndim == 0:
138:         x_scale = np.resize(x_scale, x0.shape)
139: 
140:     if x_scale.shape != x0.shape:
141:         raise ValueError("Inconsistent shapes between `x_scale` and `x0`.")
142: 
143:     return x_scale
144: 
145: 
146: def check_jac_sparsity(jac_sparsity, m, n):
147:     if jac_sparsity is None:
148:         return None
149: 
150:     if not issparse(jac_sparsity):
151:         jac_sparsity = np.atleast_2d(jac_sparsity)
152: 
153:     if jac_sparsity.shape != (m, n):
154:         raise ValueError("`jac_sparsity` has wrong shape.")
155: 
156:     return jac_sparsity, group_columns(jac_sparsity)
157: 
158: 
159: # Loss functions.
160: 
161: 
162: def huber(z, rho, cost_only):
163:     mask = z <= 1
164:     rho[0, mask] = z[mask]
165:     rho[0, ~mask] = 2 * z[~mask]**0.5 - 1
166:     if cost_only:
167:         return
168:     rho[1, mask] = 1
169:     rho[1, ~mask] = z[~mask]**-0.5
170:     rho[2, mask] = 0
171:     rho[2, ~mask] = -0.5 * z[~mask]**-1.5
172: 
173: 
174: def soft_l1(z, rho, cost_only):
175:     t = 1 + z
176:     rho[0] = 2 * (t**0.5 - 1)
177:     if cost_only:
178:         return
179:     rho[1] = t**-0.5
180:     rho[2] = -0.5 * t**-1.5
181: 
182: 
183: def cauchy(z, rho, cost_only):
184:     rho[0] = np.log1p(z)
185:     if cost_only:
186:         return
187:     t = 1 + z
188:     rho[1] = 1 / t
189:     rho[2] = -1 / t**2
190: 
191: 
192: def arctan(z, rho, cost_only):
193:     rho[0] = np.arctan(z)
194:     if cost_only:
195:         return
196:     t = 1 + z**2
197:     rho[1] = 1 / t
198:     rho[2] = -2 * z / t**2
199: 
200: 
201: IMPLEMENTED_LOSSES = dict(linear=None, huber=huber, soft_l1=soft_l1,
202:                           cauchy=cauchy, arctan=arctan)
203: 
204: 
205: def construct_loss_function(m, loss, f_scale):
206:     if loss == 'linear':
207:         return None
208: 
209:     if not callable(loss):
210:         loss = IMPLEMENTED_LOSSES[loss]
211:         rho = np.empty((3, m))
212: 
213:         def loss_function(f, cost_only=False):
214:             z = (f / f_scale) ** 2
215:             loss(z, rho, cost_only=cost_only)
216:             if cost_only:
217:                 return 0.5 * f_scale ** 2 * np.sum(rho[0])
218:             rho[0] *= f_scale ** 2
219:             rho[2] /= f_scale ** 2
220:             return rho
221:     else:
222:         def loss_function(f, cost_only=False):
223:             z = (f / f_scale) ** 2
224:             rho = loss(z)
225:             if cost_only:
226:                 return 0.5 * f_scale ** 2 * np.sum(rho[0])
227:             rho[0] *= f_scale ** 2
228:             rho[2] /= f_scale ** 2
229:             return rho
230: 
231:     return loss_function
232: 
233: 
234: def least_squares(
235:         fun, x0, jac='2-point', bounds=(-np.inf, np.inf), method='trf',
236:         ftol=1e-8, xtol=1e-8, gtol=1e-8, x_scale=1.0, loss='linear',
237:         f_scale=1.0, diff_step=None, tr_solver=None, tr_options={},
238:         jac_sparsity=None, max_nfev=None, verbose=0, args=(), kwargs={}):
239:     '''Solve a nonlinear least-squares problem with bounds on the variables.
240: 
241:     Given the residuals f(x) (an m-dimensional real function of n real
242:     variables) and the loss function rho(s) (a scalar function), `least_squares`
243:     finds a local minimum of the cost function F(x)::
244: 
245:         minimize F(x) = 0.5 * sum(rho(f_i(x)**2), i = 0, ..., m - 1)
246:         subject to lb <= x <= ub
247: 
248:     The purpose of the loss function rho(s) is to reduce the influence of
249:     outliers on the solution.
250: 
251:     Parameters
252:     ----------
253:     fun : callable
254:         Function which computes the vector of residuals, with the signature
255:         ``fun(x, *args, **kwargs)``, i.e., the minimization proceeds with
256:         respect to its first argument. The argument ``x`` passed to this
257:         function is an ndarray of shape (n,) (never a scalar, even for n=1).
258:         It must return a 1-d array_like of shape (m,) or a scalar. If the
259:         argument ``x`` is complex or the function ``fun`` returns complex
260:         residuals, it must be wrapped in a real function of real arguments,
261:         as shown at the end of the Examples section.
262:     x0 : array_like with shape (n,) or float
263:         Initial guess on independent variables. If float, it will be treated
264:         as a 1-d array with one element.
265:     jac : {'2-point', '3-point', 'cs', callable}, optional
266:         Method of computing the Jacobian matrix (an m-by-n matrix, where
267:         element (i, j) is the partial derivative of f[i] with respect to
268:         x[j]). The keywords select a finite difference scheme for numerical
269:         estimation. The scheme '3-point' is more accurate, but requires
270:         twice as much operations compared to '2-point' (default). The
271:         scheme 'cs' uses complex steps, and while potentially the most
272:         accurate, it is applicable only when `fun` correctly handles
273:         complex inputs and can be analytically continued to the complex
274:         plane. Method 'lm' always uses the '2-point' scheme. If callable,
275:         it is used as ``jac(x, *args, **kwargs)`` and should return a
276:         good approximation (or the exact value) for the Jacobian as an
277:         array_like (np.atleast_2d is applied), a sparse matrix or a
278:         `scipy.sparse.linalg.LinearOperator`.
279:     bounds : 2-tuple of array_like, optional
280:         Lower and upper bounds on independent variables. Defaults to no bounds.
281:         Each array must match the size of `x0` or be a scalar, in the latter
282:         case a bound will be the same for all variables. Use ``np.inf`` with
283:         an appropriate sign to disable bounds on all or some variables.
284:     method : {'trf', 'dogbox', 'lm'}, optional
285:         Algorithm to perform minimization.
286: 
287:             * 'trf' : Trust Region Reflective algorithm, particularly suitable
288:               for large sparse problems with bounds. Generally robust method.
289:             * 'dogbox' : dogleg algorithm with rectangular trust regions,
290:               typical use case is small problems with bounds. Not recommended
291:               for problems with rank-deficient Jacobian.
292:             * 'lm' : Levenberg-Marquardt algorithm as implemented in MINPACK.
293:               Doesn't handle bounds and sparse Jacobians. Usually the most
294:               efficient method for small unconstrained problems.
295: 
296:         Default is 'trf'. See Notes for more information.
297:     ftol : float, optional
298:         Tolerance for termination by the change of the cost function. Default
299:         is 1e-8. The optimization process is stopped when  ``dF < ftol * F``,
300:         and there was an adequate agreement between a local quadratic model and
301:         the true model in the last step.
302:     xtol : float, optional
303:         Tolerance for termination by the change of the independent variables.
304:         Default is 1e-8. The exact condition depends on the `method` used:
305: 
306:             * For 'trf' and 'dogbox' : ``norm(dx) < xtol * (xtol + norm(x))``
307:             * For 'lm' : ``Delta < xtol * norm(xs)``, where ``Delta`` is
308:               a trust-region radius and ``xs`` is the value of ``x``
309:               scaled according to `x_scale` parameter (see below).
310: 
311:     gtol : float, optional
312:         Tolerance for termination by the norm of the gradient. Default is 1e-8.
313:         The exact condition depends on a `method` used:
314: 
315:             * For 'trf' : ``norm(g_scaled, ord=np.inf) < gtol``, where
316:               ``g_scaled`` is the value of the gradient scaled to account for
317:               the presence of the bounds [STIR]_.
318:             * For 'dogbox' : ``norm(g_free, ord=np.inf) < gtol``, where
319:               ``g_free`` is the gradient with respect to the variables which
320:               are not in the optimal state on the boundary.
321:             * For 'lm' : the maximum absolute value of the cosine of angles
322:               between columns of the Jacobian and the residual vector is less
323:               than `gtol`, or the residual vector is zero.
324: 
325:     x_scale : array_like or 'jac', optional
326:         Characteristic scale of each variable. Setting `x_scale` is equivalent
327:         to reformulating the problem in scaled variables ``xs = x / x_scale``.
328:         An alternative view is that the size of a trust region along j-th
329:         dimension is proportional to ``x_scale[j]``. Improved convergence may
330:         be achieved by setting `x_scale` such that a step of a given size
331:         along any of the scaled variables has a similar effect on the cost
332:         function. If set to 'jac', the scale is iteratively updated using the
333:         inverse norms of the columns of the Jacobian matrix (as described in
334:         [JJMore]_).
335:     loss : str or callable, optional
336:         Determines the loss function. The following keyword values are allowed:
337: 
338:             * 'linear' (default) : ``rho(z) = z``. Gives a standard
339:               least-squares problem.
340:             * 'soft_l1' : ``rho(z) = 2 * ((1 + z)**0.5 - 1)``. The smooth
341:               approximation of l1 (absolute value) loss. Usually a good
342:               choice for robust least squares.
343:             * 'huber' : ``rho(z) = z if z <= 1 else 2*z**0.5 - 1``. Works
344:               similarly to 'soft_l1'.
345:             * 'cauchy' : ``rho(z) = ln(1 + z)``. Severely weakens outliers
346:               influence, but may cause difficulties in optimization process.
347:             * 'arctan' : ``rho(z) = arctan(z)``. Limits a maximum loss on
348:               a single residual, has properties similar to 'cauchy'.
349: 
350:         If callable, it must take a 1-d ndarray ``z=f**2`` and return an
351:         array_like with shape (3, m) where row 0 contains function values,
352:         row 1 contains first derivatives and row 2 contains second
353:         derivatives. Method 'lm' supports only 'linear' loss.
354:     f_scale : float, optional
355:         Value of soft margin between inlier and outlier residuals, default
356:         is 1.0. The loss function is evaluated as follows
357:         ``rho_(f**2) = C**2 * rho(f**2 / C**2)``, where ``C`` is `f_scale`,
358:         and ``rho`` is determined by `loss` parameter. This parameter has
359:         no effect with ``loss='linear'``, but for other `loss` values it is
360:         of crucial importance.
361:     max_nfev : None or int, optional
362:         Maximum number of function evaluations before the termination.
363:         If None (default), the value is chosen automatically:
364: 
365:             * For 'trf' and 'dogbox' : 100 * n.
366:             * For 'lm' :  100 * n if `jac` is callable and 100 * n * (n + 1)
367:               otherwise (because 'lm' counts function calls in Jacobian
368:               estimation).
369: 
370:     diff_step : None or array_like, optional
371:         Determines the relative step size for the finite difference
372:         approximation of the Jacobian. The actual step is computed as
373:         ``x * diff_step``. If None (default), then `diff_step` is taken to be
374:         a conventional "optimal" power of machine epsilon for the finite
375:         difference scheme used [NR]_.
376:     tr_solver : {None, 'exact', 'lsmr'}, optional
377:         Method for solving trust-region subproblems, relevant only for 'trf'
378:         and 'dogbox' methods.
379: 
380:             * 'exact' is suitable for not very large problems with dense
381:               Jacobian matrices. The computational complexity per iteration is
382:               comparable to a singular value decomposition of the Jacobian
383:               matrix.
384:             * 'lsmr' is suitable for problems with sparse and large Jacobian
385:               matrices. It uses the iterative procedure
386:               `scipy.sparse.linalg.lsmr` for finding a solution of a linear
387:               least-squares problem and only requires matrix-vector product
388:               evaluations.
389: 
390:         If None (default) the solver is chosen based on the type of Jacobian
391:         returned on the first iteration.
392:     tr_options : dict, optional
393:         Keyword options passed to trust-region solver.
394: 
395:             * ``tr_solver='exact'``: `tr_options` are ignored.
396:             * ``tr_solver='lsmr'``: options for `scipy.sparse.linalg.lsmr`.
397:               Additionally  ``method='trf'`` supports  'regularize' option
398:               (bool, default is True) which adds a regularization term to the
399:               normal equation, which improves convergence if the Jacobian is
400:               rank-deficient [Byrd]_ (eq. 3.4).
401: 
402:     jac_sparsity : {None, array_like, sparse matrix}, optional
403:         Defines the sparsity structure of the Jacobian matrix for finite
404:         difference estimation, its shape must be (m, n). If the Jacobian has
405:         only few non-zero elements in *each* row, providing the sparsity
406:         structure will greatly speed up the computations [Curtis]_. A zero
407:         entry means that a corresponding element in the Jacobian is identically
408:         zero. If provided, forces the use of 'lsmr' trust-region solver.
409:         If None (default) then dense differencing will be used. Has no effect
410:         for 'lm' method.
411:     verbose : {0, 1, 2}, optional
412:         Level of algorithm's verbosity:
413: 
414:             * 0 (default) : work silently.
415:             * 1 : display a termination report.
416:             * 2 : display progress during iterations (not supported by 'lm'
417:               method).
418: 
419:     args, kwargs : tuple and dict, optional
420:         Additional arguments passed to `fun` and `jac`. Both empty by default.
421:         The calling signature is ``fun(x, *args, **kwargs)`` and the same for
422:         `jac`.
423: 
424:     Returns
425:     -------
426:     `OptimizeResult` with the following fields defined:
427:     x : ndarray, shape (n,)
428:         Solution found.
429:     cost : float
430:         Value of the cost function at the solution.
431:     fun : ndarray, shape (m,)
432:         Vector of residuals at the solution.
433:     jac : ndarray, sparse matrix or LinearOperator, shape (m, n)
434:         Modified Jacobian matrix at the solution, in the sense that J^T J
435:         is a Gauss-Newton approximation of the Hessian of the cost function.
436:         The type is the same as the one used by the algorithm.
437:     grad : ndarray, shape (m,)
438:         Gradient of the cost function at the solution.
439:     optimality : float
440:         First-order optimality measure. In unconstrained problems, it is always
441:         the uniform norm of the gradient. In constrained problems, it is the
442:         quantity which was compared with `gtol` during iterations.
443:     active_mask : ndarray of int, shape (n,)
444:         Each component shows whether a corresponding constraint is active
445:         (that is, whether a variable is at the bound):
446: 
447:             *  0 : a constraint is not active.
448:             * -1 : a lower bound is active.
449:             *  1 : an upper bound is active.
450: 
451:         Might be somewhat arbitrary for 'trf' method as it generates a sequence
452:         of strictly feasible iterates and `active_mask` is determined within a
453:         tolerance threshold.
454:     nfev : int
455:         Number of function evaluations done. Methods 'trf' and 'dogbox' do not
456:         count function calls for numerical Jacobian approximation, as opposed
457:         to 'lm' method.
458:     njev : int or None
459:         Number of Jacobian evaluations done. If numerical Jacobian
460:         approximation is used in 'lm' method, it is set to None.
461:     status : int
462:         The reason for algorithm termination:
463: 
464:             * -1 : improper input parameters status returned from MINPACK.
465:             *  0 : the maximum number of function evaluations is exceeded.
466:             *  1 : `gtol` termination condition is satisfied.
467:             *  2 : `ftol` termination condition is satisfied.
468:             *  3 : `xtol` termination condition is satisfied.
469:             *  4 : Both `ftol` and `xtol` termination conditions are satisfied.
470: 
471:     message : str
472:         Verbal description of the termination reason.
473:     success : bool
474:         True if one of the convergence criteria is satisfied (`status` > 0).
475: 
476:     See Also
477:     --------
478:     leastsq : A legacy wrapper for the MINPACK implementation of the
479:               Levenberg-Marquadt algorithm.
480:     curve_fit : Least-squares minimization applied to a curve fitting problem.
481: 
482:     Notes
483:     -----
484:     Method 'lm' (Levenberg-Marquardt) calls a wrapper over least-squares
485:     algorithms implemented in MINPACK (lmder, lmdif). It runs the
486:     Levenberg-Marquardt algorithm formulated as a trust-region type algorithm.
487:     The implementation is based on paper [JJMore]_, it is very robust and
488:     efficient with a lot of smart tricks. It should be your first choice
489:     for unconstrained problems. Note that it doesn't support bounds. Also
490:     it doesn't work when m < n.
491: 
492:     Method 'trf' (Trust Region Reflective) is motivated by the process of
493:     solving a system of equations, which constitute the first-order optimality
494:     condition for a bound-constrained minimization problem as formulated in
495:     [STIR]_. The algorithm iteratively solves trust-region subproblems
496:     augmented by a special diagonal quadratic term and with trust-region shape
497:     determined by the distance from the bounds and the direction of the
498:     gradient. This enhancements help to avoid making steps directly into bounds
499:     and efficiently explore the whole space of variables. To further improve
500:     convergence, the algorithm considers search directions reflected from the
501:     bounds. To obey theoretical requirements, the algorithm keeps iterates
502:     strictly feasible. With dense Jacobians trust-region subproblems are
503:     solved by an exact method very similar to the one described in [JJMore]_
504:     (and implemented in MINPACK). The difference from the MINPACK
505:     implementation is that a singular value decomposition of a Jacobian
506:     matrix is done once per iteration, instead of a QR decomposition and series
507:     of Givens rotation eliminations. For large sparse Jacobians a 2-d subspace
508:     approach of solving trust-region subproblems is used [STIR]_, [Byrd]_.
509:     The subspace is spanned by a scaled gradient and an approximate
510:     Gauss-Newton solution delivered by `scipy.sparse.linalg.lsmr`. When no
511:     constraints are imposed the algorithm is very similar to MINPACK and has
512:     generally comparable performance. The algorithm works quite robust in
513:     unbounded and bounded problems, thus it is chosen as a default algorithm.
514: 
515:     Method 'dogbox' operates in a trust-region framework, but considers
516:     rectangular trust regions as opposed to conventional ellipsoids [Voglis]_.
517:     The intersection of a current trust region and initial bounds is again
518:     rectangular, so on each iteration a quadratic minimization problem subject
519:     to bound constraints is solved approximately by Powell's dogleg method
520:     [NumOpt]_. The required Gauss-Newton step can be computed exactly for
521:     dense Jacobians or approximately by `scipy.sparse.linalg.lsmr` for large
522:     sparse Jacobians. The algorithm is likely to exhibit slow convergence when
523:     the rank of Jacobian is less than the number of variables. The algorithm
524:     often outperforms 'trf' in bounded problems with a small number of
525:     variables.
526: 
527:     Robust loss functions are implemented as described in [BA]_. The idea
528:     is to modify a residual vector and a Jacobian matrix on each iteration
529:     such that computed gradient and Gauss-Newton Hessian approximation match
530:     the true gradient and Hessian approximation of the cost function. Then
531:     the algorithm proceeds in a normal way, i.e. robust loss functions are
532:     implemented as a simple wrapper over standard least-squares algorithms.
533: 
534:     .. versionadded:: 0.17.0
535: 
536:     References
537:     ----------
538:     .. [STIR] M. A. Branch, T. F. Coleman, and Y. Li, "A Subspace, Interior,
539:               and Conjugate Gradient Method for Large-Scale Bound-Constrained
540:               Minimization Problems," SIAM Journal on Scientific Computing,
541:               Vol. 21, Number 1, pp 1-23, 1999.
542:     .. [NR] William H. Press et. al., "Numerical Recipes. The Art of Scientific
543:             Computing. 3rd edition", Sec. 5.7.
544:     .. [Byrd] R. H. Byrd, R. B. Schnabel and G. A. Shultz, "Approximate
545:               solution of the trust region problem by minimization over
546:               two-dimensional subspaces", Math. Programming, 40, pp. 247-263,
547:               1988.
548:     .. [Curtis] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
549:                 sparse Jacobian matrices", Journal of the Institute of
550:                 Mathematics and its Applications, 13, pp. 117-120, 1974.
551:     .. [JJMore] J. J. More, "The Levenberg-Marquardt Algorithm: Implementation
552:                 and Theory," Numerical Analysis, ed. G. A. Watson, Lecture
553:                 Notes in Mathematics 630, Springer Verlag, pp. 105-116, 1977.
554:     .. [Voglis] C. Voglis and I. E. Lagaris, "A Rectangular Trust Region
555:                 Dogleg Approach for Unconstrained and Bound Constrained
556:                 Nonlinear Optimization", WSEAS International Conference on
557:                 Applied Mathematics, Corfu, Greece, 2004.
558:     .. [NumOpt] J. Nocedal and S. J. Wright, "Numerical optimization,
559:                 2nd edition", Chapter 4.
560:     .. [BA] B. Triggs et. al., "Bundle Adjustment - A Modern Synthesis",
561:             Proceedings of the International Workshop on Vision Algorithms:
562:             Theory and Practice, pp. 298-372, 1999.
563: 
564:     Examples
565:     --------
566:     In this example we find a minimum of the Rosenbrock function without bounds
567:     on independed variables.
568: 
569:     >>> def fun_rosenbrock(x):
570:     ...     return np.array([10 * (x[1] - x[0]**2), (1 - x[0])])
571: 
572:     Notice that we only provide the vector of the residuals. The algorithm
573:     constructs the cost function as a sum of squares of the residuals, which
574:     gives the Rosenbrock function. The exact minimum is at ``x = [1.0, 1.0]``.
575: 
576:     >>> from scipy.optimize import least_squares
577:     >>> x0_rosenbrock = np.array([2, 2])
578:     >>> res_1 = least_squares(fun_rosenbrock, x0_rosenbrock)
579:     >>> res_1.x
580:     array([ 1.,  1.])
581:     >>> res_1.cost
582:     9.8669242910846867e-30
583:     >>> res_1.optimality
584:     8.8928864934219529e-14
585: 
586:     We now constrain the variables, in such a way that the previous solution
587:     becomes infeasible. Specifically, we require that ``x[1] >= 1.5``, and
588:     ``x[0]`` left unconstrained. To this end, we specify the `bounds` parameter
589:     to `least_squares` in the form ``bounds=([-np.inf, 1.5], np.inf)``.
590: 
591:     We also provide the analytic Jacobian:
592: 
593:     >>> def jac_rosenbrock(x):
594:     ...     return np.array([
595:     ...         [-20 * x[0], 10],
596:     ...         [-1, 0]])
597: 
598:     Putting this all together, we see that the new solution lies on the bound:
599: 
600:     >>> res_2 = least_squares(fun_rosenbrock, x0_rosenbrock, jac_rosenbrock,
601:     ...                       bounds=([-np.inf, 1.5], np.inf))
602:     >>> res_2.x
603:     array([ 1.22437075,  1.5       ])
604:     >>> res_2.cost
605:     0.025213093946805685
606:     >>> res_2.optimality
607:     1.5885401433157753e-07
608: 
609:     Now we solve a system of equations (i.e., the cost function should be zero
610:     at a minimum) for a Broyden tridiagonal vector-valued function of 100000
611:     variables:
612: 
613:     >>> def fun_broyden(x):
614:     ...     f = (3 - x) * x + 1
615:     ...     f[1:] -= x[:-1]
616:     ...     f[:-1] -= 2 * x[1:]
617:     ...     return f
618: 
619:     The corresponding Jacobian matrix is sparse. We tell the algorithm to
620:     estimate it by finite differences and provide the sparsity structure of
621:     Jacobian to significantly speed up this process.
622: 
623:     >>> from scipy.sparse import lil_matrix
624:     >>> def sparsity_broyden(n):
625:     ...     sparsity = lil_matrix((n, n), dtype=int)
626:     ...     i = np.arange(n)
627:     ...     sparsity[i, i] = 1
628:     ...     i = np.arange(1, n)
629:     ...     sparsity[i, i - 1] = 1
630:     ...     i = np.arange(n - 1)
631:     ...     sparsity[i, i + 1] = 1
632:     ...     return sparsity
633:     ...
634:     >>> n = 100000
635:     >>> x0_broyden = -np.ones(n)
636:     ...
637:     >>> res_3 = least_squares(fun_broyden, x0_broyden,
638:     ...                       jac_sparsity=sparsity_broyden(n))
639:     >>> res_3.cost
640:     4.5687069299604613e-23
641:     >>> res_3.optimality
642:     1.1650454296851518e-11
643: 
644:     Let's also solve a curve fitting problem using robust loss function to
645:     take care of outliers in the data. Define the model function as
646:     ``y = a + b * exp(c * t)``, where t is a predictor variable, y is an
647:     observation and a, b, c are parameters to estimate.
648: 
649:     First, define the function which generates the data with noise and
650:     outliers, define the model parameters, and generate data:
651: 
652:     >>> def gen_data(t, a, b, c, noise=0, n_outliers=0, random_state=0):
653:     ...     y = a + b * np.exp(t * c)
654:     ...
655:     ...     rnd = np.random.RandomState(random_state)
656:     ...     error = noise * rnd.randn(t.size)
657:     ...     outliers = rnd.randint(0, t.size, n_outliers)
658:     ...     error[outliers] *= 10
659:     ...
660:     ...     return y + error
661:     ...
662:     >>> a = 0.5
663:     >>> b = 2.0
664:     >>> c = -1
665:     >>> t_min = 0
666:     >>> t_max = 10
667:     >>> n_points = 15
668:     ...
669:     >>> t_train = np.linspace(t_min, t_max, n_points)
670:     >>> y_train = gen_data(t_train, a, b, c, noise=0.1, n_outliers=3)
671: 
672:     Define function for computing residuals and initial estimate of
673:     parameters.
674: 
675:     >>> def fun(x, t, y):
676:     ...     return x[0] + x[1] * np.exp(x[2] * t) - y
677:     ...
678:     >>> x0 = np.array([1.0, 1.0, 0.0])
679: 
680:     Compute a standard least-squares solution:
681: 
682:     >>> res_lsq = least_squares(fun, x0, args=(t_train, y_train))
683: 
684:     Now compute two solutions with two different robust loss functions. The
685:     parameter `f_scale` is set to 0.1, meaning that inlier residuals should
686:     not significantly exceed 0.1 (the noise level used).
687: 
688:     >>> res_soft_l1 = least_squares(fun, x0, loss='soft_l1', f_scale=0.1,
689:     ...                             args=(t_train, y_train))
690:     >>> res_log = least_squares(fun, x0, loss='cauchy', f_scale=0.1,
691:     ...                         args=(t_train, y_train))
692: 
693:     And finally plot all the curves. We see that by selecting an appropriate
694:     `loss`  we can get estimates close to optimal even in the presence of
695:     strong outliers. But keep in mind that generally it is recommended to try
696:     'soft_l1' or 'huber' losses first (if at all necessary) as the other two
697:     options may cause difficulties in optimization process.
698: 
699:     >>> t_test = np.linspace(t_min, t_max, n_points * 10)
700:     >>> y_true = gen_data(t_test, a, b, c)
701:     >>> y_lsq = gen_data(t_test, *res_lsq.x)
702:     >>> y_soft_l1 = gen_data(t_test, *res_soft_l1.x)
703:     >>> y_log = gen_data(t_test, *res_log.x)
704:     ...
705:     >>> import matplotlib.pyplot as plt
706:     >>> plt.plot(t_train, y_train, 'o')
707:     >>> plt.plot(t_test, y_true, 'k', linewidth=2, label='true')
708:     >>> plt.plot(t_test, y_lsq, label='linear loss')
709:     >>> plt.plot(t_test, y_soft_l1, label='soft_l1 loss')
710:     >>> plt.plot(t_test, y_log, label='cauchy loss')
711:     >>> plt.xlabel("t")
712:     >>> plt.ylabel("y")
713:     >>> plt.legend()
714:     >>> plt.show()
715: 
716:     In the next example, we show how complex-valued residual functions of
717:     complex variables can be optimized with ``least_squares()``. Consider the
718:     following function:
719: 
720:     >>> def f(z):
721:     ...     return z - (0.5 + 0.5j)
722: 
723:     We wrap it into a function of real variables that returns real residuals
724:     by simply handling the real and imaginary parts as independent variables:
725: 
726:     >>> def f_wrap(x):
727:     ...     fx = f(x[0] + 1j*x[1])
728:     ...     return np.array([fx.real, fx.imag])
729: 
730:     Thus, instead of the original m-dimensional complex function of n complex
731:     variables we optimize a 2m-dimensional real function of 2n real variables:
732: 
733:     >>> from scipy.optimize import least_squares
734:     >>> res_wrapped = least_squares(f_wrap, (0.1, 0.1), bounds=([0, 0], [1, 1]))
735:     >>> z = res_wrapped.x[0] + res_wrapped.x[1]*1j
736:     >>> z
737:     (0.49999999999925893+0.49999999999925893j)
738: 
739:     '''
740:     if method not in ['trf', 'dogbox', 'lm']:
741:         raise ValueError("`method` must be 'trf', 'dogbox' or 'lm'.")
742: 
743:     if jac not in ['2-point', '3-point', 'cs'] and not callable(jac):
744:         raise ValueError("`jac` must be '2-point', '3-point', 'cs' or "
745:                          "callable.")
746: 
747:     if tr_solver not in [None, 'exact', 'lsmr']:
748:         raise ValueError("`tr_solver` must be None, 'exact' or 'lsmr'.")
749: 
750:     if loss not in IMPLEMENTED_LOSSES and not callable(loss):
751:         raise ValueError("`loss` must be one of {0} or a callable."
752:                          .format(IMPLEMENTED_LOSSES.keys()))
753: 
754:     if method == 'lm' and loss != 'linear':
755:         raise ValueError("method='lm' supports only 'linear' loss function.")
756: 
757:     if verbose not in [0, 1, 2]:
758:         raise ValueError("`verbose` must be in [0, 1, 2].")
759: 
760:     if len(bounds) != 2:
761:         raise ValueError("`bounds` must contain 2 elements.")
762: 
763:     if max_nfev is not None and max_nfev <= 0:
764:         raise ValueError("`max_nfev` must be None or positive integer.")
765: 
766:     if np.iscomplexobj(x0):
767:         raise ValueError("`x0` must be real.")
768: 
769:     x0 = np.atleast_1d(x0).astype(float)
770: 
771:     if x0.ndim > 1:
772:         raise ValueError("`x0` must have at most 1 dimension.")
773: 
774:     lb, ub = prepare_bounds(bounds, x0.shape[0])
775: 
776:     if method == 'lm' and not np.all((lb == -np.inf) & (ub == np.inf)):
777:         raise ValueError("Method 'lm' doesn't support bounds.")
778: 
779:     if lb.shape != x0.shape or ub.shape != x0.shape:
780:         raise ValueError("Inconsistent shapes between bounds and `x0`.")
781: 
782:     if np.any(lb >= ub):
783:         raise ValueError("Each lower bound must be strictly less than each "
784:                          "upper bound.")
785: 
786:     if not in_bounds(x0, lb, ub):
787:         raise ValueError("`x0` is infeasible.")
788: 
789:     x_scale = check_x_scale(x_scale, x0)
790: 
791:     ftol, xtol, gtol = check_tolerance(ftol, xtol, gtol)
792: 
793:     def fun_wrapped(x):
794:         return np.atleast_1d(fun(x, *args, **kwargs))
795: 
796:     if method == 'trf':
797:         x0 = make_strictly_feasible(x0, lb, ub)
798: 
799:     f0 = fun_wrapped(x0)
800: 
801:     if f0.ndim != 1:
802:         raise ValueError("`fun` must return at most 1-d array_like.")
803: 
804:     if not np.all(np.isfinite(f0)):
805:         raise ValueError("Residuals are not finite in the initial point.")
806: 
807:     n = x0.size
808:     m = f0.size
809: 
810:     if method == 'lm' and m < n:
811:         raise ValueError("Method 'lm' doesn't work when the number of "
812:                          "residuals is less than the number of variables.")
813: 
814:     loss_function = construct_loss_function(m, loss, f_scale)
815:     if callable(loss):
816:         rho = loss_function(f0)
817:         if rho.shape != (3, m):
818:             raise ValueError("The return value of `loss` callable has wrong "
819:                              "shape.")
820:         initial_cost = 0.5 * np.sum(rho[0])
821:     elif loss_function is not None:
822:         initial_cost = loss_function(f0, cost_only=True)
823:     else:
824:         initial_cost = 0.5 * np.dot(f0, f0)
825: 
826:     if callable(jac):
827:         J0 = jac(x0, *args, **kwargs)
828: 
829:         if issparse(J0):
830:             J0 = csr_matrix(J0)
831: 
832:             def jac_wrapped(x, _=None):
833:                 return csr_matrix(jac(x, *args, **kwargs))
834: 
835:         elif isinstance(J0, LinearOperator):
836:             def jac_wrapped(x, _=None):
837:                 return jac(x, *args, **kwargs)
838: 
839:         else:
840:             J0 = np.atleast_2d(J0)
841: 
842:             def jac_wrapped(x, _=None):
843:                 return np.atleast_2d(jac(x, *args, **kwargs))
844: 
845:     else:  # Estimate Jacobian by finite differences.
846:         if method == 'lm':
847:             if jac_sparsity is not None:
848:                 raise ValueError("method='lm' does not support "
849:                                  "`jac_sparsity`.")
850: 
851:             if jac != '2-point':
852:                 warn("jac='{0}' works equivalently to '2-point' "
853:                      "for method='lm'.".format(jac))
854: 
855:             J0 = jac_wrapped = None
856:         else:
857:             if jac_sparsity is not None and tr_solver == 'exact':
858:                 raise ValueError("tr_solver='exact' is incompatible "
859:                                  "with `jac_sparsity`.")
860: 
861:             jac_sparsity = check_jac_sparsity(jac_sparsity, m, n)
862: 
863:             def jac_wrapped(x, f):
864:                 J = approx_derivative(fun, x, rel_step=diff_step, method=jac,
865:                                       f0=f, bounds=bounds, args=args,
866:                                       kwargs=kwargs, sparsity=jac_sparsity)
867:                 if J.ndim != 2:  # J is guaranteed not sparse.
868:                     J = np.atleast_2d(J)
869: 
870:                 return J
871: 
872:             J0 = jac_wrapped(x0, f0)
873: 
874:     if J0 is not None:
875:         if J0.shape != (m, n):
876:             raise ValueError(
877:                 "The return value of `jac` has wrong shape: expected {0}, "
878:                 "actual {1}.".format((m, n), J0.shape))
879: 
880:         if not isinstance(J0, np.ndarray):
881:             if method == 'lm':
882:                 raise ValueError("method='lm' works only with dense "
883:                                  "Jacobian matrices.")
884: 
885:             if tr_solver == 'exact':
886:                 raise ValueError(
887:                     "tr_solver='exact' works only with dense "
888:                     "Jacobian matrices.")
889: 
890:         jac_scale = isinstance(x_scale, string_types) and x_scale == 'jac'
891:         if isinstance(J0, LinearOperator) and jac_scale:
892:             raise ValueError("x_scale='jac' can't be used when `jac` "
893:                              "returns LinearOperator.")
894: 
895:         if tr_solver is None:
896:             if isinstance(J0, np.ndarray):
897:                 tr_solver = 'exact'
898:             else:
899:                 tr_solver = 'lsmr'
900: 
901:     if method == 'lm':
902:         result = call_minpack(fun_wrapped, x0, jac_wrapped, ftol, xtol, gtol,
903:                               max_nfev, x_scale, diff_step)
904: 
905:     elif method == 'trf':
906:         result = trf(fun_wrapped, jac_wrapped, x0, f0, J0, lb, ub, ftol, xtol,
907:                      gtol, max_nfev, x_scale, loss_function, tr_solver,
908:                      tr_options.copy(), verbose)
909: 
910:     elif method == 'dogbox':
911:         if tr_solver == 'lsmr' and 'regularize' in tr_options:
912:             warn("The keyword 'regularize' in `tr_options` is not relevant "
913:                  "for 'dogbox' method.")
914:             tr_options = tr_options.copy()
915:             del tr_options['regularize']
916: 
917:         result = dogbox(fun_wrapped, jac_wrapped, x0, f0, J0, lb, ub, ftol,
918:                         xtol, gtol, max_nfev, x_scale, loss_function,
919:                         tr_solver, tr_options, verbose)
920: 
921:     result.message = TERMINATION_MESSAGES[result.status]
922:     result.success = result.status > 0
923: 
924:     if verbose >= 1:
925:         print(result.message)
926:         print("Function evaluations {0}, initial cost {1:.4e}, final cost "
927:               "{2:.4e}, first-order optimality {3:.2e}."
928:               .format(result.nfev, initial_cost, result.cost,
929:                       result.optimality))
930: 
931:     return result
932: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_250509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Generic interface for least-square minimization.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from warnings import warn' statement (line 4)
try:
    from warnings import warn

except:
    warn = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'warnings', None, module_type_store, ['warn'], [warn])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_250510 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_250510) is not StypyTypeError):

    if (import_250510 != 'pyd_module'):
        __import__(import_250510)
        sys_modules_250511 = sys.modules[import_250510]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_250511.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_250510)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.linalg import norm' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_250512 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.linalg')

if (type(import_250512) is not StypyTypeError):

    if (import_250512 != 'pyd_module'):
        __import__(import_250512)
        sys_modules_250513 = sys.modules[import_250512]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.linalg', sys_modules_250513.module_type_store, module_type_store, ['norm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_250513, sys_modules_250513.module_type_store, module_type_store)
    else:
        from numpy.linalg import norm

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.linalg', None, module_type_store, ['norm'], [norm])

else:
    # Assigning a type to the variable 'numpy.linalg' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.linalg', import_250512)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.sparse import issparse, csr_matrix' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_250514 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse')

if (type(import_250514) is not StypyTypeError):

    if (import_250514 != 'pyd_module'):
        __import__(import_250514)
        sys_modules_250515 = sys.modules[import_250514]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse', sys_modules_250515.module_type_store, module_type_store, ['issparse', 'csr_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_250515, sys_modules_250515.module_type_store, module_type_store)
    else:
        from scipy.sparse import issparse, csr_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse', None, module_type_store, ['issparse', 'csr_matrix'], [issparse, csr_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse', import_250514)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.sparse.linalg import LinearOperator' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_250516 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg')

if (type(import_250516) is not StypyTypeError):

    if (import_250516 != 'pyd_module'):
        __import__(import_250516)
        sys_modules_250517 = sys.modules[import_250516]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg', sys_modules_250517.module_type_store, module_type_store, ['LinearOperator'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_250517, sys_modules_250517.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import LinearOperator

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg', None, module_type_store, ['LinearOperator'], [LinearOperator])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg', import_250516)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.optimize import _minpack, OptimizeResult' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_250518 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize')

if (type(import_250518) is not StypyTypeError):

    if (import_250518 != 'pyd_module'):
        __import__(import_250518)
        sys_modules_250519 = sys.modules[import_250518]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize', sys_modules_250519.module_type_store, module_type_store, ['_minpack', 'OptimizeResult'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_250519, sys_modules_250519.module_type_store, module_type_store)
    else:
        from scipy.optimize import _minpack, OptimizeResult

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize', None, module_type_store, ['_minpack', 'OptimizeResult'], [_minpack, OptimizeResult])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize', import_250518)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.optimize._numdiff import approx_derivative, group_columns' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_250520 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.optimize._numdiff')

if (type(import_250520) is not StypyTypeError):

    if (import_250520 != 'pyd_module'):
        __import__(import_250520)
        sys_modules_250521 = sys.modules[import_250520]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.optimize._numdiff', sys_modules_250521.module_type_store, module_type_store, ['approx_derivative', 'group_columns'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_250521, sys_modules_250521.module_type_store, module_type_store)
    else:
        from scipy.optimize._numdiff import approx_derivative, group_columns

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.optimize._numdiff', None, module_type_store, ['approx_derivative', 'group_columns'], [approx_derivative, group_columns])

else:
    # Assigning a type to the variable 'scipy.optimize._numdiff' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.optimize._numdiff', import_250520)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy._lib.six import string_types' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_250522 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib.six')

if (type(import_250522) is not StypyTypeError):

    if (import_250522 != 'pyd_module'):
        __import__(import_250522)
        sys_modules_250523 = sys.modules[import_250522]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib.six', sys_modules_250523.module_type_store, module_type_store, ['string_types'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_250523, sys_modules_250523.module_type_store, module_type_store)
    else:
        from scipy._lib.six import string_types

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib.six', None, module_type_store, ['string_types'], [string_types])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib.six', import_250522)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from scipy.optimize._lsq.trf import trf' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_250524 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.optimize._lsq.trf')

if (type(import_250524) is not StypyTypeError):

    if (import_250524 != 'pyd_module'):
        __import__(import_250524)
        sys_modules_250525 = sys.modules[import_250524]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.optimize._lsq.trf', sys_modules_250525.module_type_store, module_type_store, ['trf'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_250525, sys_modules_250525.module_type_store, module_type_store)
    else:
        from scipy.optimize._lsq.trf import trf

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.optimize._lsq.trf', None, module_type_store, ['trf'], [trf])

else:
    # Assigning a type to the variable 'scipy.optimize._lsq.trf' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.optimize._lsq.trf', import_250524)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy.optimize._lsq.dogbox import dogbox' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_250526 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.optimize._lsq.dogbox')

if (type(import_250526) is not StypyTypeError):

    if (import_250526 != 'pyd_module'):
        __import__(import_250526)
        sys_modules_250527 = sys.modules[import_250526]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.optimize._lsq.dogbox', sys_modules_250527.module_type_store, module_type_store, ['dogbox'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_250527, sys_modules_250527.module_type_store, module_type_store)
    else:
        from scipy.optimize._lsq.dogbox import dogbox

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.optimize._lsq.dogbox', None, module_type_store, ['dogbox'], [dogbox])

else:
    # Assigning a type to the variable 'scipy.optimize._lsq.dogbox' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.optimize._lsq.dogbox', import_250526)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from scipy.optimize._lsq.common import EPS, in_bounds, make_strictly_feasible' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_250528 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.optimize._lsq.common')

if (type(import_250528) is not StypyTypeError):

    if (import_250528 != 'pyd_module'):
        __import__(import_250528)
        sys_modules_250529 = sys.modules[import_250528]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.optimize._lsq.common', sys_modules_250529.module_type_store, module_type_store, ['EPS', 'in_bounds', 'make_strictly_feasible'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_250529, sys_modules_250529.module_type_store, module_type_store)
    else:
        from scipy.optimize._lsq.common import EPS, in_bounds, make_strictly_feasible

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.optimize._lsq.common', None, module_type_store, ['EPS', 'in_bounds', 'make_strictly_feasible'], [EPS, in_bounds, make_strictly_feasible])

else:
    # Assigning a type to the variable 'scipy.optimize._lsq.common' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.optimize._lsq.common', import_250528)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')


# Assigning a Dict to a Name (line 20):

# Assigning a Dict to a Name (line 20):

# Obtaining an instance of the builtin type 'dict' (line 20)
dict_250530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 23), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 20)
# Adding element type (key, value) (line 20)
int_250531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 4), 'int')
str_250532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 8), 'str', 'Improper input parameters status returned from `leastsq`')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), dict_250530, (int_250531, str_250532))
# Adding element type (key, value) (line 20)
int_250533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 4), 'int')
str_250534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 7), 'str', 'The maximum number of function evaluations is exceeded.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), dict_250530, (int_250533, str_250534))
# Adding element type (key, value) (line 20)
int_250535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 4), 'int')
str_250536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 7), 'str', '`gtol` termination condition is satisfied.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), dict_250530, (int_250535, str_250536))
# Adding element type (key, value) (line 20)
int_250537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 4), 'int')
str_250538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 7), 'str', '`ftol` termination condition is satisfied.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), dict_250530, (int_250537, str_250538))
# Adding element type (key, value) (line 20)
int_250539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 4), 'int')
str_250540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 7), 'str', '`xtol` termination condition is satisfied.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), dict_250530, (int_250539, str_250540))
# Adding element type (key, value) (line 20)
int_250541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 4), 'int')
str_250542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 7), 'str', 'Both `ftol` and `xtol` termination conditions are satisfied.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), dict_250530, (int_250541, str_250542))

# Assigning a type to the variable 'TERMINATION_MESSAGES' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'TERMINATION_MESSAGES', dict_250530)

# Assigning a Dict to a Name (line 30):

# Assigning a Dict to a Name (line 30):

# Obtaining an instance of the builtin type 'dict' (line 30)
dict_250543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 25), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 30)
# Adding element type (key, value) (line 30)
int_250544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 4), 'int')
int_250545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 7), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 25), dict_250543, (int_250544, int_250545))
# Adding element type (key, value) (line 30)
int_250546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 4), 'int')
int_250547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 7), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 25), dict_250543, (int_250546, int_250547))
# Adding element type (key, value) (line 30)
int_250548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 4), 'int')
int_250549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 7), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 25), dict_250543, (int_250548, int_250549))
# Adding element type (key, value) (line 30)
int_250550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 4), 'int')
int_250551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 7), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 25), dict_250543, (int_250550, int_250551))
# Adding element type (key, value) (line 30)
int_250552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 4), 'int')
int_250553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 7), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 25), dict_250543, (int_250552, int_250553))
# Adding element type (key, value) (line 30)
int_250554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 4), 'int')
int_250555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 7), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 25), dict_250543, (int_250554, int_250555))

# Assigning a type to the variable 'FROM_MINPACK_TO_COMMON' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'FROM_MINPACK_TO_COMMON', dict_250543)

@norecursion
def call_minpack(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'call_minpack'
    module_type_store = module_type_store.open_function_context('call_minpack', 42, 0, False)
    
    # Passed parameters checking function
    call_minpack.stypy_localization = localization
    call_minpack.stypy_type_of_self = None
    call_minpack.stypy_type_store = module_type_store
    call_minpack.stypy_function_name = 'call_minpack'
    call_minpack.stypy_param_names_list = ['fun', 'x0', 'jac', 'ftol', 'xtol', 'gtol', 'max_nfev', 'x_scale', 'diff_step']
    call_minpack.stypy_varargs_param_name = None
    call_minpack.stypy_kwargs_param_name = None
    call_minpack.stypy_call_defaults = defaults
    call_minpack.stypy_call_varargs = varargs
    call_minpack.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'call_minpack', ['fun', 'x0', 'jac', 'ftol', 'xtol', 'gtol', 'max_nfev', 'x_scale', 'diff_step'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'call_minpack', localization, ['fun', 'x0', 'jac', 'ftol', 'xtol', 'gtol', 'max_nfev', 'x_scale', 'diff_step'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'call_minpack(...)' code ##################

    
    # Assigning a Attribute to a Name (line 43):
    
    # Assigning a Attribute to a Name (line 43):
    # Getting the type of 'x0' (line 43)
    x0_250556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'x0')
    # Obtaining the member 'size' of a type (line 43)
    size_250557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), x0_250556, 'size')
    # Assigning a type to the variable 'n' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'n', size_250557)
    
    # Type idiom detected: calculating its left and rigth part (line 45)
    # Getting the type of 'diff_step' (line 45)
    diff_step_250558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 7), 'diff_step')
    # Getting the type of 'None' (line 45)
    None_250559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 20), 'None')
    
    (may_be_250560, more_types_in_union_250561) = may_be_none(diff_step_250558, None_250559)

    if may_be_250560:

        if more_types_in_union_250561:
            # Runtime conditional SSA (line 45)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 46):
        
        # Assigning a Name to a Name (line 46):
        # Getting the type of 'EPS' (line 46)
        EPS_250562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 17), 'EPS')
        # Assigning a type to the variable 'epsfcn' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'epsfcn', EPS_250562)

        if more_types_in_union_250561:
            # Runtime conditional SSA for else branch (line 45)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_250560) or more_types_in_union_250561):
        
        # Assigning a BinOp to a Name (line 48):
        
        # Assigning a BinOp to a Name (line 48):
        # Getting the type of 'diff_step' (line 48)
        diff_step_250563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 17), 'diff_step')
        int_250564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 28), 'int')
        # Applying the binary operator '**' (line 48)
        result_pow_250565 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 17), '**', diff_step_250563, int_250564)
        
        # Assigning a type to the variable 'epsfcn' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'epsfcn', result_pow_250565)

        if (may_be_250560 and more_types_in_union_250561):
            # SSA join for if statement (line 45)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'x_scale' (line 52)
    x_scale_250567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 18), 'x_scale', False)
    # Getting the type of 'string_types' (line 52)
    string_types_250568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 27), 'string_types', False)
    # Processing the call keyword arguments (line 52)
    kwargs_250569 = {}
    # Getting the type of 'isinstance' (line 52)
    isinstance_250566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 52)
    isinstance_call_result_250570 = invoke(stypy.reporting.localization.Localization(__file__, 52, 7), isinstance_250566, *[x_scale_250567, string_types_250568], **kwargs_250569)
    
    
    # Getting the type of 'x_scale' (line 52)
    x_scale_250571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 45), 'x_scale')
    str_250572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 56), 'str', 'jac')
    # Applying the binary operator '==' (line 52)
    result_eq_250573 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 45), '==', x_scale_250571, str_250572)
    
    # Applying the binary operator 'and' (line 52)
    result_and_keyword_250574 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 7), 'and', isinstance_call_result_250570, result_eq_250573)
    
    # Testing the type of an if condition (line 52)
    if_condition_250575 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 4), result_and_keyword_250574)
    # Assigning a type to the variable 'if_condition_250575' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'if_condition_250575', if_condition_250575)
    # SSA begins for if statement (line 52)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 53):
    
    # Assigning a Name to a Name (line 53):
    # Getting the type of 'None' (line 53)
    None_250576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 15), 'None')
    # Assigning a type to the variable 'diag' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'diag', None_250576)
    # SSA branch for the else part of an if statement (line 52)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 55):
    
    # Assigning a BinOp to a Name (line 55):
    int_250577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 15), 'int')
    # Getting the type of 'x_scale' (line 55)
    x_scale_250578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 19), 'x_scale')
    # Applying the binary operator 'div' (line 55)
    result_div_250579 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 15), 'div', int_250577, x_scale_250578)
    
    # Assigning a type to the variable 'diag' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'diag', result_div_250579)
    # SSA join for if statement (line 52)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 57):
    
    # Assigning a Name to a Name (line 57):
    # Getting the type of 'True' (line 57)
    True_250580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 18), 'True')
    # Assigning a type to the variable 'full_output' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'full_output', True_250580)
    
    # Assigning a Name to a Name (line 58):
    
    # Assigning a Name to a Name (line 58):
    # Getting the type of 'False' (line 58)
    False_250581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'False')
    # Assigning a type to the variable 'col_deriv' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'col_deriv', False_250581)
    
    # Assigning a Num to a Name (line 59):
    
    # Assigning a Num to a Name (line 59):
    float_250582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 13), 'float')
    # Assigning a type to the variable 'factor' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'factor', float_250582)
    
    # Type idiom detected: calculating its left and rigth part (line 61)
    # Getting the type of 'jac' (line 61)
    jac_250583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 7), 'jac')
    # Getting the type of 'None' (line 61)
    None_250584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 14), 'None')
    
    (may_be_250585, more_types_in_union_250586) = may_be_none(jac_250583, None_250584)

    if may_be_250585:

        if more_types_in_union_250586:
            # Runtime conditional SSA (line 61)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Type idiom detected: calculating its left and rigth part (line 62)
        # Getting the type of 'max_nfev' (line 62)
        max_nfev_250587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 11), 'max_nfev')
        # Getting the type of 'None' (line 62)
        None_250588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 23), 'None')
        
        (may_be_250589, more_types_in_union_250590) = may_be_none(max_nfev_250587, None_250588)

        if may_be_250589:

            if more_types_in_union_250590:
                # Runtime conditional SSA (line 62)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 64):
            
            # Assigning a BinOp to a Name (line 64):
            int_250591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 23), 'int')
            # Getting the type of 'n' (line 64)
            n_250592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 29), 'n')
            # Applying the binary operator '*' (line 64)
            result_mul_250593 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 23), '*', int_250591, n_250592)
            
            # Getting the type of 'n' (line 64)
            n_250594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 34), 'n')
            int_250595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 38), 'int')
            # Applying the binary operator '+' (line 64)
            result_add_250596 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 34), '+', n_250594, int_250595)
            
            # Applying the binary operator '*' (line 64)
            result_mul_250597 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 31), '*', result_mul_250593, result_add_250596)
            
            # Assigning a type to the variable 'max_nfev' (line 64)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'max_nfev', result_mul_250597)

            if more_types_in_union_250590:
                # SSA join for if statement (line 62)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Tuple (line 65):
        
        # Assigning a Subscript to a Name (line 65):
        
        # Obtaining the type of the subscript
        int_250598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 8), 'int')
        
        # Call to _lmdif(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'fun' (line 66)
        fun_250601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'fun', False)
        # Getting the type of 'x0' (line 66)
        x0_250602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 17), 'x0', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 66)
        tuple_250603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 66)
        
        # Getting the type of 'full_output' (line 66)
        full_output_250604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'full_output', False)
        # Getting the type of 'ftol' (line 66)
        ftol_250605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 38), 'ftol', False)
        # Getting the type of 'xtol' (line 66)
        xtol_250606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 44), 'xtol', False)
        # Getting the type of 'gtol' (line 66)
        gtol_250607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 50), 'gtol', False)
        # Getting the type of 'max_nfev' (line 67)
        max_nfev_250608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'max_nfev', False)
        # Getting the type of 'epsfcn' (line 67)
        epsfcn_250609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 22), 'epsfcn', False)
        # Getting the type of 'factor' (line 67)
        factor_250610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 30), 'factor', False)
        # Getting the type of 'diag' (line 67)
        diag_250611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 38), 'diag', False)
        # Processing the call keyword arguments (line 65)
        kwargs_250612 = {}
        # Getting the type of '_minpack' (line 65)
        _minpack_250599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 26), '_minpack', False)
        # Obtaining the member '_lmdif' of a type (line 65)
        _lmdif_250600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 26), _minpack_250599, '_lmdif')
        # Calling _lmdif(args, kwargs) (line 65)
        _lmdif_call_result_250613 = invoke(stypy.reporting.localization.Localization(__file__, 65, 26), _lmdif_250600, *[fun_250601, x0_250602, tuple_250603, full_output_250604, ftol_250605, xtol_250606, gtol_250607, max_nfev_250608, epsfcn_250609, factor_250610, diag_250611], **kwargs_250612)
        
        # Obtaining the member '__getitem__' of a type (line 65)
        getitem___250614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), _lmdif_call_result_250613, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 65)
        subscript_call_result_250615 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), getitem___250614, int_250598)
        
        # Assigning a type to the variable 'tuple_var_assignment_250496' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'tuple_var_assignment_250496', subscript_call_result_250615)
        
        # Assigning a Subscript to a Name (line 65):
        
        # Obtaining the type of the subscript
        int_250616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 8), 'int')
        
        # Call to _lmdif(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'fun' (line 66)
        fun_250619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'fun', False)
        # Getting the type of 'x0' (line 66)
        x0_250620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 17), 'x0', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 66)
        tuple_250621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 66)
        
        # Getting the type of 'full_output' (line 66)
        full_output_250622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'full_output', False)
        # Getting the type of 'ftol' (line 66)
        ftol_250623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 38), 'ftol', False)
        # Getting the type of 'xtol' (line 66)
        xtol_250624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 44), 'xtol', False)
        # Getting the type of 'gtol' (line 66)
        gtol_250625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 50), 'gtol', False)
        # Getting the type of 'max_nfev' (line 67)
        max_nfev_250626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'max_nfev', False)
        # Getting the type of 'epsfcn' (line 67)
        epsfcn_250627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 22), 'epsfcn', False)
        # Getting the type of 'factor' (line 67)
        factor_250628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 30), 'factor', False)
        # Getting the type of 'diag' (line 67)
        diag_250629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 38), 'diag', False)
        # Processing the call keyword arguments (line 65)
        kwargs_250630 = {}
        # Getting the type of '_minpack' (line 65)
        _minpack_250617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 26), '_minpack', False)
        # Obtaining the member '_lmdif' of a type (line 65)
        _lmdif_250618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 26), _minpack_250617, '_lmdif')
        # Calling _lmdif(args, kwargs) (line 65)
        _lmdif_call_result_250631 = invoke(stypy.reporting.localization.Localization(__file__, 65, 26), _lmdif_250618, *[fun_250619, x0_250620, tuple_250621, full_output_250622, ftol_250623, xtol_250624, gtol_250625, max_nfev_250626, epsfcn_250627, factor_250628, diag_250629], **kwargs_250630)
        
        # Obtaining the member '__getitem__' of a type (line 65)
        getitem___250632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), _lmdif_call_result_250631, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 65)
        subscript_call_result_250633 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), getitem___250632, int_250616)
        
        # Assigning a type to the variable 'tuple_var_assignment_250497' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'tuple_var_assignment_250497', subscript_call_result_250633)
        
        # Assigning a Subscript to a Name (line 65):
        
        # Obtaining the type of the subscript
        int_250634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 8), 'int')
        
        # Call to _lmdif(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'fun' (line 66)
        fun_250637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'fun', False)
        # Getting the type of 'x0' (line 66)
        x0_250638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 17), 'x0', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 66)
        tuple_250639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 66)
        
        # Getting the type of 'full_output' (line 66)
        full_output_250640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'full_output', False)
        # Getting the type of 'ftol' (line 66)
        ftol_250641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 38), 'ftol', False)
        # Getting the type of 'xtol' (line 66)
        xtol_250642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 44), 'xtol', False)
        # Getting the type of 'gtol' (line 66)
        gtol_250643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 50), 'gtol', False)
        # Getting the type of 'max_nfev' (line 67)
        max_nfev_250644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'max_nfev', False)
        # Getting the type of 'epsfcn' (line 67)
        epsfcn_250645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 22), 'epsfcn', False)
        # Getting the type of 'factor' (line 67)
        factor_250646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 30), 'factor', False)
        # Getting the type of 'diag' (line 67)
        diag_250647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 38), 'diag', False)
        # Processing the call keyword arguments (line 65)
        kwargs_250648 = {}
        # Getting the type of '_minpack' (line 65)
        _minpack_250635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 26), '_minpack', False)
        # Obtaining the member '_lmdif' of a type (line 65)
        _lmdif_250636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 26), _minpack_250635, '_lmdif')
        # Calling _lmdif(args, kwargs) (line 65)
        _lmdif_call_result_250649 = invoke(stypy.reporting.localization.Localization(__file__, 65, 26), _lmdif_250636, *[fun_250637, x0_250638, tuple_250639, full_output_250640, ftol_250641, xtol_250642, gtol_250643, max_nfev_250644, epsfcn_250645, factor_250646, diag_250647], **kwargs_250648)
        
        # Obtaining the member '__getitem__' of a type (line 65)
        getitem___250650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), _lmdif_call_result_250649, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 65)
        subscript_call_result_250651 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), getitem___250650, int_250634)
        
        # Assigning a type to the variable 'tuple_var_assignment_250498' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'tuple_var_assignment_250498', subscript_call_result_250651)
        
        # Assigning a Name to a Name (line 65):
        # Getting the type of 'tuple_var_assignment_250496' (line 65)
        tuple_var_assignment_250496_250652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'tuple_var_assignment_250496')
        # Assigning a type to the variable 'x' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'x', tuple_var_assignment_250496_250652)
        
        # Assigning a Name to a Name (line 65):
        # Getting the type of 'tuple_var_assignment_250497' (line 65)
        tuple_var_assignment_250497_250653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'tuple_var_assignment_250497')
        # Assigning a type to the variable 'info' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 11), 'info', tuple_var_assignment_250497_250653)
        
        # Assigning a Name to a Name (line 65):
        # Getting the type of 'tuple_var_assignment_250498' (line 65)
        tuple_var_assignment_250498_250654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'tuple_var_assignment_250498')
        # Assigning a type to the variable 'status' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 17), 'status', tuple_var_assignment_250498_250654)

        if more_types_in_union_250586:
            # Runtime conditional SSA for else branch (line 61)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_250585) or more_types_in_union_250586):
        
        # Type idiom detected: calculating its left and rigth part (line 69)
        # Getting the type of 'max_nfev' (line 69)
        max_nfev_250655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), 'max_nfev')
        # Getting the type of 'None' (line 69)
        None_250656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 23), 'None')
        
        (may_be_250657, more_types_in_union_250658) = may_be_none(max_nfev_250655, None_250656)

        if may_be_250657:

            if more_types_in_union_250658:
                # Runtime conditional SSA (line 69)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 70):
            
            # Assigning a BinOp to a Name (line 70):
            int_250659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 23), 'int')
            # Getting the type of 'n' (line 70)
            n_250660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 29), 'n')
            # Applying the binary operator '*' (line 70)
            result_mul_250661 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 23), '*', int_250659, n_250660)
            
            # Assigning a type to the variable 'max_nfev' (line 70)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'max_nfev', result_mul_250661)

            if more_types_in_union_250658:
                # SSA join for if statement (line 69)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Tuple (line 71):
        
        # Assigning a Subscript to a Name (line 71):
        
        # Obtaining the type of the subscript
        int_250662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 8), 'int')
        
        # Call to _lmder(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'fun' (line 72)
        fun_250665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'fun', False)
        # Getting the type of 'jac' (line 72)
        jac_250666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 17), 'jac', False)
        # Getting the type of 'x0' (line 72)
        x0_250667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 22), 'x0', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 72)
        tuple_250668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 72)
        
        # Getting the type of 'full_output' (line 72)
        full_output_250669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 30), 'full_output', False)
        # Getting the type of 'col_deriv' (line 72)
        col_deriv_250670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 43), 'col_deriv', False)
        # Getting the type of 'ftol' (line 73)
        ftol_250671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'ftol', False)
        # Getting the type of 'xtol' (line 73)
        xtol_250672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 18), 'xtol', False)
        # Getting the type of 'gtol' (line 73)
        gtol_250673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 24), 'gtol', False)
        # Getting the type of 'max_nfev' (line 73)
        max_nfev_250674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 30), 'max_nfev', False)
        # Getting the type of 'factor' (line 73)
        factor_250675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 40), 'factor', False)
        # Getting the type of 'diag' (line 73)
        diag_250676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 48), 'diag', False)
        # Processing the call keyword arguments (line 71)
        kwargs_250677 = {}
        # Getting the type of '_minpack' (line 71)
        _minpack_250663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 26), '_minpack', False)
        # Obtaining the member '_lmder' of a type (line 71)
        _lmder_250664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 26), _minpack_250663, '_lmder')
        # Calling _lmder(args, kwargs) (line 71)
        _lmder_call_result_250678 = invoke(stypy.reporting.localization.Localization(__file__, 71, 26), _lmder_250664, *[fun_250665, jac_250666, x0_250667, tuple_250668, full_output_250669, col_deriv_250670, ftol_250671, xtol_250672, gtol_250673, max_nfev_250674, factor_250675, diag_250676], **kwargs_250677)
        
        # Obtaining the member '__getitem__' of a type (line 71)
        getitem___250679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), _lmder_call_result_250678, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 71)
        subscript_call_result_250680 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), getitem___250679, int_250662)
        
        # Assigning a type to the variable 'tuple_var_assignment_250499' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'tuple_var_assignment_250499', subscript_call_result_250680)
        
        # Assigning a Subscript to a Name (line 71):
        
        # Obtaining the type of the subscript
        int_250681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 8), 'int')
        
        # Call to _lmder(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'fun' (line 72)
        fun_250684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'fun', False)
        # Getting the type of 'jac' (line 72)
        jac_250685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 17), 'jac', False)
        # Getting the type of 'x0' (line 72)
        x0_250686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 22), 'x0', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 72)
        tuple_250687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 72)
        
        # Getting the type of 'full_output' (line 72)
        full_output_250688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 30), 'full_output', False)
        # Getting the type of 'col_deriv' (line 72)
        col_deriv_250689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 43), 'col_deriv', False)
        # Getting the type of 'ftol' (line 73)
        ftol_250690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'ftol', False)
        # Getting the type of 'xtol' (line 73)
        xtol_250691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 18), 'xtol', False)
        # Getting the type of 'gtol' (line 73)
        gtol_250692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 24), 'gtol', False)
        # Getting the type of 'max_nfev' (line 73)
        max_nfev_250693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 30), 'max_nfev', False)
        # Getting the type of 'factor' (line 73)
        factor_250694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 40), 'factor', False)
        # Getting the type of 'diag' (line 73)
        diag_250695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 48), 'diag', False)
        # Processing the call keyword arguments (line 71)
        kwargs_250696 = {}
        # Getting the type of '_minpack' (line 71)
        _minpack_250682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 26), '_minpack', False)
        # Obtaining the member '_lmder' of a type (line 71)
        _lmder_250683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 26), _minpack_250682, '_lmder')
        # Calling _lmder(args, kwargs) (line 71)
        _lmder_call_result_250697 = invoke(stypy.reporting.localization.Localization(__file__, 71, 26), _lmder_250683, *[fun_250684, jac_250685, x0_250686, tuple_250687, full_output_250688, col_deriv_250689, ftol_250690, xtol_250691, gtol_250692, max_nfev_250693, factor_250694, diag_250695], **kwargs_250696)
        
        # Obtaining the member '__getitem__' of a type (line 71)
        getitem___250698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), _lmder_call_result_250697, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 71)
        subscript_call_result_250699 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), getitem___250698, int_250681)
        
        # Assigning a type to the variable 'tuple_var_assignment_250500' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'tuple_var_assignment_250500', subscript_call_result_250699)
        
        # Assigning a Subscript to a Name (line 71):
        
        # Obtaining the type of the subscript
        int_250700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 8), 'int')
        
        # Call to _lmder(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'fun' (line 72)
        fun_250703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'fun', False)
        # Getting the type of 'jac' (line 72)
        jac_250704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 17), 'jac', False)
        # Getting the type of 'x0' (line 72)
        x0_250705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 22), 'x0', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 72)
        tuple_250706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 72)
        
        # Getting the type of 'full_output' (line 72)
        full_output_250707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 30), 'full_output', False)
        # Getting the type of 'col_deriv' (line 72)
        col_deriv_250708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 43), 'col_deriv', False)
        # Getting the type of 'ftol' (line 73)
        ftol_250709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'ftol', False)
        # Getting the type of 'xtol' (line 73)
        xtol_250710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 18), 'xtol', False)
        # Getting the type of 'gtol' (line 73)
        gtol_250711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 24), 'gtol', False)
        # Getting the type of 'max_nfev' (line 73)
        max_nfev_250712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 30), 'max_nfev', False)
        # Getting the type of 'factor' (line 73)
        factor_250713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 40), 'factor', False)
        # Getting the type of 'diag' (line 73)
        diag_250714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 48), 'diag', False)
        # Processing the call keyword arguments (line 71)
        kwargs_250715 = {}
        # Getting the type of '_minpack' (line 71)
        _minpack_250701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 26), '_minpack', False)
        # Obtaining the member '_lmder' of a type (line 71)
        _lmder_250702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 26), _minpack_250701, '_lmder')
        # Calling _lmder(args, kwargs) (line 71)
        _lmder_call_result_250716 = invoke(stypy.reporting.localization.Localization(__file__, 71, 26), _lmder_250702, *[fun_250703, jac_250704, x0_250705, tuple_250706, full_output_250707, col_deriv_250708, ftol_250709, xtol_250710, gtol_250711, max_nfev_250712, factor_250713, diag_250714], **kwargs_250715)
        
        # Obtaining the member '__getitem__' of a type (line 71)
        getitem___250717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), _lmder_call_result_250716, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 71)
        subscript_call_result_250718 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), getitem___250717, int_250700)
        
        # Assigning a type to the variable 'tuple_var_assignment_250501' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'tuple_var_assignment_250501', subscript_call_result_250718)
        
        # Assigning a Name to a Name (line 71):
        # Getting the type of 'tuple_var_assignment_250499' (line 71)
        tuple_var_assignment_250499_250719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'tuple_var_assignment_250499')
        # Assigning a type to the variable 'x' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'x', tuple_var_assignment_250499_250719)
        
        # Assigning a Name to a Name (line 71):
        # Getting the type of 'tuple_var_assignment_250500' (line 71)
        tuple_var_assignment_250500_250720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'tuple_var_assignment_250500')
        # Assigning a type to the variable 'info' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 11), 'info', tuple_var_assignment_250500_250720)
        
        # Assigning a Name to a Name (line 71):
        # Getting the type of 'tuple_var_assignment_250501' (line 71)
        tuple_var_assignment_250501_250721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'tuple_var_assignment_250501')
        # Assigning a type to the variable 'status' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 17), 'status', tuple_var_assignment_250501_250721)

        if (may_be_250585 and more_types_in_union_250586):
            # SSA join for if statement (line 61)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Subscript to a Name (line 75):
    
    # Assigning a Subscript to a Name (line 75):
    
    # Obtaining the type of the subscript
    str_250722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 13), 'str', 'fvec')
    # Getting the type of 'info' (line 75)
    info_250723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'info')
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___250724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), info_250723, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_250725 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), getitem___250724, str_250722)
    
    # Assigning a type to the variable 'f' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'f', subscript_call_result_250725)
    
    
    # Call to callable(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'jac' (line 77)
    jac_250727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 'jac', False)
    # Processing the call keyword arguments (line 77)
    kwargs_250728 = {}
    # Getting the type of 'callable' (line 77)
    callable_250726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 7), 'callable', False)
    # Calling callable(args, kwargs) (line 77)
    callable_call_result_250729 = invoke(stypy.reporting.localization.Localization(__file__, 77, 7), callable_250726, *[jac_250727], **kwargs_250728)
    
    # Testing the type of an if condition (line 77)
    if_condition_250730 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 4), callable_call_result_250729)
    # Assigning a type to the variable 'if_condition_250730' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'if_condition_250730', if_condition_250730)
    # SSA begins for if statement (line 77)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 78):
    
    # Assigning a Call to a Name (line 78):
    
    # Call to jac(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'x' (line 78)
    x_250732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'x', False)
    # Processing the call keyword arguments (line 78)
    kwargs_250733 = {}
    # Getting the type of 'jac' (line 78)
    jac_250731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'jac', False)
    # Calling jac(args, kwargs) (line 78)
    jac_call_result_250734 = invoke(stypy.reporting.localization.Localization(__file__, 78, 12), jac_250731, *[x_250732], **kwargs_250733)
    
    # Assigning a type to the variable 'J' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'J', jac_call_result_250734)
    # SSA branch for the else part of an if statement (line 77)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 80):
    
    # Assigning a Call to a Name (line 80):
    
    # Call to atleast_2d(...): (line 80)
    # Processing the call arguments (line 80)
    
    # Call to approx_derivative(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'fun' (line 80)
    fun_250738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 44), 'fun', False)
    # Getting the type of 'x' (line 80)
    x_250739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 49), 'x', False)
    # Processing the call keyword arguments (line 80)
    kwargs_250740 = {}
    # Getting the type of 'approx_derivative' (line 80)
    approx_derivative_250737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 26), 'approx_derivative', False)
    # Calling approx_derivative(args, kwargs) (line 80)
    approx_derivative_call_result_250741 = invoke(stypy.reporting.localization.Localization(__file__, 80, 26), approx_derivative_250737, *[fun_250738, x_250739], **kwargs_250740)
    
    # Processing the call keyword arguments (line 80)
    kwargs_250742 = {}
    # Getting the type of 'np' (line 80)
    np_250735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'np', False)
    # Obtaining the member 'atleast_2d' of a type (line 80)
    atleast_2d_250736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), np_250735, 'atleast_2d')
    # Calling atleast_2d(args, kwargs) (line 80)
    atleast_2d_call_result_250743 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), atleast_2d_250736, *[approx_derivative_call_result_250741], **kwargs_250742)
    
    # Assigning a type to the variable 'J' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'J', atleast_2d_call_result_250743)
    # SSA join for if statement (line 77)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 82):
    
    # Assigning a BinOp to a Name (line 82):
    float_250744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 11), 'float')
    
    # Call to dot(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'f' (line 82)
    f_250747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 24), 'f', False)
    # Getting the type of 'f' (line 82)
    f_250748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 27), 'f', False)
    # Processing the call keyword arguments (line 82)
    kwargs_250749 = {}
    # Getting the type of 'np' (line 82)
    np_250745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 17), 'np', False)
    # Obtaining the member 'dot' of a type (line 82)
    dot_250746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 17), np_250745, 'dot')
    # Calling dot(args, kwargs) (line 82)
    dot_call_result_250750 = invoke(stypy.reporting.localization.Localization(__file__, 82, 17), dot_250746, *[f_250747, f_250748], **kwargs_250749)
    
    # Applying the binary operator '*' (line 82)
    result_mul_250751 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 11), '*', float_250744, dot_call_result_250750)
    
    # Assigning a type to the variable 'cost' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'cost', result_mul_250751)
    
    # Assigning a Call to a Name (line 83):
    
    # Assigning a Call to a Name (line 83):
    
    # Call to dot(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'f' (line 83)
    f_250755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'f', False)
    # Processing the call keyword arguments (line 83)
    kwargs_250756 = {}
    # Getting the type of 'J' (line 83)
    J_250752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'J', False)
    # Obtaining the member 'T' of a type (line 83)
    T_250753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), J_250752, 'T')
    # Obtaining the member 'dot' of a type (line 83)
    dot_250754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), T_250753, 'dot')
    # Calling dot(args, kwargs) (line 83)
    dot_call_result_250757 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), dot_250754, *[f_250755], **kwargs_250756)
    
    # Assigning a type to the variable 'g' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'g', dot_call_result_250757)
    
    # Assigning a Call to a Name (line 84):
    
    # Assigning a Call to a Name (line 84):
    
    # Call to norm(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'g' (line 84)
    g_250759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 18), 'g', False)
    # Processing the call keyword arguments (line 84)
    # Getting the type of 'np' (line 84)
    np_250760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 25), 'np', False)
    # Obtaining the member 'inf' of a type (line 84)
    inf_250761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 25), np_250760, 'inf')
    keyword_250762 = inf_250761
    kwargs_250763 = {'ord': keyword_250762}
    # Getting the type of 'norm' (line 84)
    norm_250758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 13), 'norm', False)
    # Calling norm(args, kwargs) (line 84)
    norm_call_result_250764 = invoke(stypy.reporting.localization.Localization(__file__, 84, 13), norm_250758, *[g_250759], **kwargs_250763)
    
    # Assigning a type to the variable 'g_norm' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'g_norm', norm_call_result_250764)
    
    # Assigning a Subscript to a Name (line 86):
    
    # Assigning a Subscript to a Name (line 86):
    
    # Obtaining the type of the subscript
    str_250765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 16), 'str', 'nfev')
    # Getting the type of 'info' (line 86)
    info_250766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 11), 'info')
    # Obtaining the member '__getitem__' of a type (line 86)
    getitem___250767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 11), info_250766, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 86)
    subscript_call_result_250768 = invoke(stypy.reporting.localization.Localization(__file__, 86, 11), getitem___250767, str_250765)
    
    # Assigning a type to the variable 'nfev' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'nfev', subscript_call_result_250768)
    
    # Assigning a Call to a Name (line 87):
    
    # Assigning a Call to a Name (line 87):
    
    # Call to get(...): (line 87)
    # Processing the call arguments (line 87)
    str_250771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 20), 'str', 'njev')
    # Getting the type of 'None' (line 87)
    None_250772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 28), 'None', False)
    # Processing the call keyword arguments (line 87)
    kwargs_250773 = {}
    # Getting the type of 'info' (line 87)
    info_250769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 11), 'info', False)
    # Obtaining the member 'get' of a type (line 87)
    get_250770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 11), info_250769, 'get')
    # Calling get(args, kwargs) (line 87)
    get_call_result_250774 = invoke(stypy.reporting.localization.Localization(__file__, 87, 11), get_250770, *[str_250771, None_250772], **kwargs_250773)
    
    # Assigning a type to the variable 'njev' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'njev', get_call_result_250774)
    
    # Assigning a Subscript to a Name (line 89):
    
    # Assigning a Subscript to a Name (line 89):
    
    # Obtaining the type of the subscript
    # Getting the type of 'status' (line 89)
    status_250775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 36), 'status')
    # Getting the type of 'FROM_MINPACK_TO_COMMON' (line 89)
    FROM_MINPACK_TO_COMMON_250776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'FROM_MINPACK_TO_COMMON')
    # Obtaining the member '__getitem__' of a type (line 89)
    getitem___250777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 13), FROM_MINPACK_TO_COMMON_250776, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
    subscript_call_result_250778 = invoke(stypy.reporting.localization.Localization(__file__, 89, 13), getitem___250777, status_250775)
    
    # Assigning a type to the variable 'status' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'status', subscript_call_result_250778)
    
    # Assigning a Call to a Name (line 90):
    
    # Assigning a Call to a Name (line 90):
    
    # Call to zeros_like(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'x0' (line 90)
    x0_250781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 32), 'x0', False)
    # Processing the call keyword arguments (line 90)
    # Getting the type of 'int' (line 90)
    int_250782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 42), 'int', False)
    keyword_250783 = int_250782
    kwargs_250784 = {'dtype': keyword_250783}
    # Getting the type of 'np' (line 90)
    np_250779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 18), 'np', False)
    # Obtaining the member 'zeros_like' of a type (line 90)
    zeros_like_250780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 18), np_250779, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 90)
    zeros_like_call_result_250785 = invoke(stypy.reporting.localization.Localization(__file__, 90, 18), zeros_like_250780, *[x0_250781], **kwargs_250784)
    
    # Assigning a type to the variable 'active_mask' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'active_mask', zeros_like_call_result_250785)
    
    # Call to OptimizeResult(...): (line 92)
    # Processing the call keyword arguments (line 92)
    # Getting the type of 'x' (line 93)
    x_250787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 10), 'x', False)
    keyword_250788 = x_250787
    # Getting the type of 'cost' (line 93)
    cost_250789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 18), 'cost', False)
    keyword_250790 = cost_250789
    # Getting the type of 'f' (line 93)
    f_250791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 28), 'f', False)
    keyword_250792 = f_250791
    # Getting the type of 'J' (line 93)
    J_250793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 35), 'J', False)
    keyword_250794 = J_250793
    # Getting the type of 'g' (line 93)
    g_250795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 43), 'g', False)
    keyword_250796 = g_250795
    # Getting the type of 'g_norm' (line 93)
    g_norm_250797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 57), 'g_norm', False)
    keyword_250798 = g_norm_250797
    # Getting the type of 'active_mask' (line 94)
    active_mask_250799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'active_mask', False)
    keyword_250800 = active_mask_250799
    # Getting the type of 'nfev' (line 94)
    nfev_250801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 38), 'nfev', False)
    keyword_250802 = nfev_250801
    # Getting the type of 'njev' (line 94)
    njev_250803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 49), 'njev', False)
    keyword_250804 = njev_250803
    # Getting the type of 'status' (line 94)
    status_250805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 62), 'status', False)
    keyword_250806 = status_250805
    kwargs_250807 = {'status': keyword_250806, 'njev': keyword_250804, 'nfev': keyword_250802, 'active_mask': keyword_250800, 'cost': keyword_250790, 'optimality': keyword_250798, 'fun': keyword_250792, 'x': keyword_250788, 'grad': keyword_250796, 'jac': keyword_250794}
    # Getting the type of 'OptimizeResult' (line 92)
    OptimizeResult_250786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'OptimizeResult', False)
    # Calling OptimizeResult(args, kwargs) (line 92)
    OptimizeResult_call_result_250808 = invoke(stypy.reporting.localization.Localization(__file__, 92, 11), OptimizeResult_250786, *[], **kwargs_250807)
    
    # Assigning a type to the variable 'stypy_return_type' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type', OptimizeResult_call_result_250808)
    
    # ################# End of 'call_minpack(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'call_minpack' in the type store
    # Getting the type of 'stypy_return_type' (line 42)
    stypy_return_type_250809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_250809)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'call_minpack'
    return stypy_return_type_250809

# Assigning a type to the variable 'call_minpack' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'call_minpack', call_minpack)

@norecursion
def prepare_bounds(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'prepare_bounds'
    module_type_store = module_type_store.open_function_context('prepare_bounds', 97, 0, False)
    
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

    
    # Assigning a ListComp to a Tuple (line 98):
    
    # Assigning a Subscript to a Name (line 98):
    
    # Obtaining the type of the subscript
    int_250810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'bounds' (line 98)
    bounds_250818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 50), 'bounds')
    comprehension_250819 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 14), bounds_250818)
    # Assigning a type to the variable 'b' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 14), 'b', comprehension_250819)
    
    # Call to asarray(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'b' (line 98)
    b_250813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 25), 'b', False)
    # Processing the call keyword arguments (line 98)
    # Getting the type of 'float' (line 98)
    float_250814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 34), 'float', False)
    keyword_250815 = float_250814
    kwargs_250816 = {'dtype': keyword_250815}
    # Getting the type of 'np' (line 98)
    np_250811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 14), 'np', False)
    # Obtaining the member 'asarray' of a type (line 98)
    asarray_250812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 14), np_250811, 'asarray')
    # Calling asarray(args, kwargs) (line 98)
    asarray_call_result_250817 = invoke(stypy.reporting.localization.Localization(__file__, 98, 14), asarray_250812, *[b_250813], **kwargs_250816)
    
    list_250820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 14), list_250820, asarray_call_result_250817)
    # Obtaining the member '__getitem__' of a type (line 98)
    getitem___250821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 4), list_250820, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 98)
    subscript_call_result_250822 = invoke(stypy.reporting.localization.Localization(__file__, 98, 4), getitem___250821, int_250810)
    
    # Assigning a type to the variable 'tuple_var_assignment_250502' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_250502', subscript_call_result_250822)
    
    # Assigning a Subscript to a Name (line 98):
    
    # Obtaining the type of the subscript
    int_250823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'bounds' (line 98)
    bounds_250831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 50), 'bounds')
    comprehension_250832 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 14), bounds_250831)
    # Assigning a type to the variable 'b' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 14), 'b', comprehension_250832)
    
    # Call to asarray(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'b' (line 98)
    b_250826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 25), 'b', False)
    # Processing the call keyword arguments (line 98)
    # Getting the type of 'float' (line 98)
    float_250827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 34), 'float', False)
    keyword_250828 = float_250827
    kwargs_250829 = {'dtype': keyword_250828}
    # Getting the type of 'np' (line 98)
    np_250824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 14), 'np', False)
    # Obtaining the member 'asarray' of a type (line 98)
    asarray_250825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 14), np_250824, 'asarray')
    # Calling asarray(args, kwargs) (line 98)
    asarray_call_result_250830 = invoke(stypy.reporting.localization.Localization(__file__, 98, 14), asarray_250825, *[b_250826], **kwargs_250829)
    
    list_250833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 14), list_250833, asarray_call_result_250830)
    # Obtaining the member '__getitem__' of a type (line 98)
    getitem___250834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 4), list_250833, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 98)
    subscript_call_result_250835 = invoke(stypy.reporting.localization.Localization(__file__, 98, 4), getitem___250834, int_250823)
    
    # Assigning a type to the variable 'tuple_var_assignment_250503' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_250503', subscript_call_result_250835)
    
    # Assigning a Name to a Name (line 98):
    # Getting the type of 'tuple_var_assignment_250502' (line 98)
    tuple_var_assignment_250502_250836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_250502')
    # Assigning a type to the variable 'lb' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'lb', tuple_var_assignment_250502_250836)
    
    # Assigning a Name to a Name (line 98):
    # Getting the type of 'tuple_var_assignment_250503' (line 98)
    tuple_var_assignment_250503_250837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_250503')
    # Assigning a type to the variable 'ub' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'ub', tuple_var_assignment_250503_250837)
    
    
    # Getting the type of 'lb' (line 99)
    lb_250838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 7), 'lb')
    # Obtaining the member 'ndim' of a type (line 99)
    ndim_250839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 7), lb_250838, 'ndim')
    int_250840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 18), 'int')
    # Applying the binary operator '==' (line 99)
    result_eq_250841 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 7), '==', ndim_250839, int_250840)
    
    # Testing the type of an if condition (line 99)
    if_condition_250842 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 4), result_eq_250841)
    # Assigning a type to the variable 'if_condition_250842' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'if_condition_250842', if_condition_250842)
    # SSA begins for if statement (line 99)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 100):
    
    # Assigning a Call to a Name (line 100):
    
    # Call to resize(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'lb' (line 100)
    lb_250845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 23), 'lb', False)
    # Getting the type of 'n' (line 100)
    n_250846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'n', False)
    # Processing the call keyword arguments (line 100)
    kwargs_250847 = {}
    # Getting the type of 'np' (line 100)
    np_250843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 13), 'np', False)
    # Obtaining the member 'resize' of a type (line 100)
    resize_250844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 13), np_250843, 'resize')
    # Calling resize(args, kwargs) (line 100)
    resize_call_result_250848 = invoke(stypy.reporting.localization.Localization(__file__, 100, 13), resize_250844, *[lb_250845, n_250846], **kwargs_250847)
    
    # Assigning a type to the variable 'lb' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'lb', resize_call_result_250848)
    # SSA join for if statement (line 99)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'ub' (line 102)
    ub_250849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 7), 'ub')
    # Obtaining the member 'ndim' of a type (line 102)
    ndim_250850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 7), ub_250849, 'ndim')
    int_250851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 18), 'int')
    # Applying the binary operator '==' (line 102)
    result_eq_250852 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 7), '==', ndim_250850, int_250851)
    
    # Testing the type of an if condition (line 102)
    if_condition_250853 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 4), result_eq_250852)
    # Assigning a type to the variable 'if_condition_250853' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'if_condition_250853', if_condition_250853)
    # SSA begins for if statement (line 102)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 103):
    
    # Assigning a Call to a Name (line 103):
    
    # Call to resize(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'ub' (line 103)
    ub_250856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 23), 'ub', False)
    # Getting the type of 'n' (line 103)
    n_250857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'n', False)
    # Processing the call keyword arguments (line 103)
    kwargs_250858 = {}
    # Getting the type of 'np' (line 103)
    np_250854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 13), 'np', False)
    # Obtaining the member 'resize' of a type (line 103)
    resize_250855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 13), np_250854, 'resize')
    # Calling resize(args, kwargs) (line 103)
    resize_call_result_250859 = invoke(stypy.reporting.localization.Localization(__file__, 103, 13), resize_250855, *[ub_250856, n_250857], **kwargs_250858)
    
    # Assigning a type to the variable 'ub' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'ub', resize_call_result_250859)
    # SSA join for if statement (line 102)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 105)
    tuple_250860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 105)
    # Adding element type (line 105)
    # Getting the type of 'lb' (line 105)
    lb_250861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 'lb')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 11), tuple_250860, lb_250861)
    # Adding element type (line 105)
    # Getting the type of 'ub' (line 105)
    ub_250862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 15), 'ub')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 11), tuple_250860, ub_250862)
    
    # Assigning a type to the variable 'stypy_return_type' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type', tuple_250860)
    
    # ################# End of 'prepare_bounds(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'prepare_bounds' in the type store
    # Getting the type of 'stypy_return_type' (line 97)
    stypy_return_type_250863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_250863)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'prepare_bounds'
    return stypy_return_type_250863

# Assigning a type to the variable 'prepare_bounds' (line 97)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'prepare_bounds', prepare_bounds)

@norecursion
def check_tolerance(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_tolerance'
    module_type_store = module_type_store.open_function_context('check_tolerance', 108, 0, False)
    
    # Passed parameters checking function
    check_tolerance.stypy_localization = localization
    check_tolerance.stypy_type_of_self = None
    check_tolerance.stypy_type_store = module_type_store
    check_tolerance.stypy_function_name = 'check_tolerance'
    check_tolerance.stypy_param_names_list = ['ftol', 'xtol', 'gtol']
    check_tolerance.stypy_varargs_param_name = None
    check_tolerance.stypy_kwargs_param_name = None
    check_tolerance.stypy_call_defaults = defaults
    check_tolerance.stypy_call_varargs = varargs
    check_tolerance.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_tolerance', ['ftol', 'xtol', 'gtol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_tolerance', localization, ['ftol', 'xtol', 'gtol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_tolerance(...)' code ##################

    
    # Assigning a Str to a Name (line 109):
    
    # Assigning a Str to a Name (line 109):
    str_250864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 14), 'str', '{} is too low, setting to machine epsilon {}.')
    # Assigning a type to the variable 'message' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'message', str_250864)
    
    
    # Getting the type of 'ftol' (line 110)
    ftol_250865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 7), 'ftol')
    # Getting the type of 'EPS' (line 110)
    EPS_250866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 14), 'EPS')
    # Applying the binary operator '<' (line 110)
    result_lt_250867 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 7), '<', ftol_250865, EPS_250866)
    
    # Testing the type of an if condition (line 110)
    if_condition_250868 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 4), result_lt_250867)
    # Assigning a type to the variable 'if_condition_250868' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'if_condition_250868', if_condition_250868)
    # SSA begins for if statement (line 110)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 111)
    # Processing the call arguments (line 111)
    
    # Call to format(...): (line 111)
    # Processing the call arguments (line 111)
    str_250872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 28), 'str', '`ftol`')
    # Getting the type of 'EPS' (line 111)
    EPS_250873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 38), 'EPS', False)
    # Processing the call keyword arguments (line 111)
    kwargs_250874 = {}
    # Getting the type of 'message' (line 111)
    message_250870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 13), 'message', False)
    # Obtaining the member 'format' of a type (line 111)
    format_250871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 13), message_250870, 'format')
    # Calling format(args, kwargs) (line 111)
    format_call_result_250875 = invoke(stypy.reporting.localization.Localization(__file__, 111, 13), format_250871, *[str_250872, EPS_250873], **kwargs_250874)
    
    # Processing the call keyword arguments (line 111)
    kwargs_250876 = {}
    # Getting the type of 'warn' (line 111)
    warn_250869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'warn', False)
    # Calling warn(args, kwargs) (line 111)
    warn_call_result_250877 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), warn_250869, *[format_call_result_250875], **kwargs_250876)
    
    
    # Assigning a Name to a Name (line 112):
    
    # Assigning a Name to a Name (line 112):
    # Getting the type of 'EPS' (line 112)
    EPS_250878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 15), 'EPS')
    # Assigning a type to the variable 'ftol' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'ftol', EPS_250878)
    # SSA join for if statement (line 110)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'xtol' (line 113)
    xtol_250879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 7), 'xtol')
    # Getting the type of 'EPS' (line 113)
    EPS_250880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 14), 'EPS')
    # Applying the binary operator '<' (line 113)
    result_lt_250881 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 7), '<', xtol_250879, EPS_250880)
    
    # Testing the type of an if condition (line 113)
    if_condition_250882 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 4), result_lt_250881)
    # Assigning a type to the variable 'if_condition_250882' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'if_condition_250882', if_condition_250882)
    # SSA begins for if statement (line 113)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 114)
    # Processing the call arguments (line 114)
    
    # Call to format(...): (line 114)
    # Processing the call arguments (line 114)
    str_250886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 28), 'str', '`xtol`')
    # Getting the type of 'EPS' (line 114)
    EPS_250887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 38), 'EPS', False)
    # Processing the call keyword arguments (line 114)
    kwargs_250888 = {}
    # Getting the type of 'message' (line 114)
    message_250884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 13), 'message', False)
    # Obtaining the member 'format' of a type (line 114)
    format_250885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 13), message_250884, 'format')
    # Calling format(args, kwargs) (line 114)
    format_call_result_250889 = invoke(stypy.reporting.localization.Localization(__file__, 114, 13), format_250885, *[str_250886, EPS_250887], **kwargs_250888)
    
    # Processing the call keyword arguments (line 114)
    kwargs_250890 = {}
    # Getting the type of 'warn' (line 114)
    warn_250883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'warn', False)
    # Calling warn(args, kwargs) (line 114)
    warn_call_result_250891 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), warn_250883, *[format_call_result_250889], **kwargs_250890)
    
    
    # Assigning a Name to a Name (line 115):
    
    # Assigning a Name to a Name (line 115):
    # Getting the type of 'EPS' (line 115)
    EPS_250892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 15), 'EPS')
    # Assigning a type to the variable 'xtol' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'xtol', EPS_250892)
    # SSA join for if statement (line 113)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'gtol' (line 116)
    gtol_250893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 7), 'gtol')
    # Getting the type of 'EPS' (line 116)
    EPS_250894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 14), 'EPS')
    # Applying the binary operator '<' (line 116)
    result_lt_250895 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 7), '<', gtol_250893, EPS_250894)
    
    # Testing the type of an if condition (line 116)
    if_condition_250896 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 4), result_lt_250895)
    # Assigning a type to the variable 'if_condition_250896' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'if_condition_250896', if_condition_250896)
    # SSA begins for if statement (line 116)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 117)
    # Processing the call arguments (line 117)
    
    # Call to format(...): (line 117)
    # Processing the call arguments (line 117)
    str_250900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 28), 'str', '`gtol`')
    # Getting the type of 'EPS' (line 117)
    EPS_250901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 38), 'EPS', False)
    # Processing the call keyword arguments (line 117)
    kwargs_250902 = {}
    # Getting the type of 'message' (line 117)
    message_250898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 13), 'message', False)
    # Obtaining the member 'format' of a type (line 117)
    format_250899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 13), message_250898, 'format')
    # Calling format(args, kwargs) (line 117)
    format_call_result_250903 = invoke(stypy.reporting.localization.Localization(__file__, 117, 13), format_250899, *[str_250900, EPS_250901], **kwargs_250902)
    
    # Processing the call keyword arguments (line 117)
    kwargs_250904 = {}
    # Getting the type of 'warn' (line 117)
    warn_250897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'warn', False)
    # Calling warn(args, kwargs) (line 117)
    warn_call_result_250905 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), warn_250897, *[format_call_result_250903], **kwargs_250904)
    
    
    # Assigning a Name to a Name (line 118):
    
    # Assigning a Name to a Name (line 118):
    # Getting the type of 'EPS' (line 118)
    EPS_250906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'EPS')
    # Assigning a type to the variable 'gtol' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'gtol', EPS_250906)
    # SSA join for if statement (line 116)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 120)
    tuple_250907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 120)
    # Adding element type (line 120)
    # Getting the type of 'ftol' (line 120)
    ftol_250908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 11), 'ftol')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 11), tuple_250907, ftol_250908)
    # Adding element type (line 120)
    # Getting the type of 'xtol' (line 120)
    xtol_250909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 17), 'xtol')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 11), tuple_250907, xtol_250909)
    # Adding element type (line 120)
    # Getting the type of 'gtol' (line 120)
    gtol_250910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 23), 'gtol')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 11), tuple_250907, gtol_250910)
    
    # Assigning a type to the variable 'stypy_return_type' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'stypy_return_type', tuple_250907)
    
    # ################# End of 'check_tolerance(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_tolerance' in the type store
    # Getting the type of 'stypy_return_type' (line 108)
    stypy_return_type_250911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_250911)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_tolerance'
    return stypy_return_type_250911

# Assigning a type to the variable 'check_tolerance' (line 108)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'check_tolerance', check_tolerance)

@norecursion
def check_x_scale(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_x_scale'
    module_type_store = module_type_store.open_function_context('check_x_scale', 123, 0, False)
    
    # Passed parameters checking function
    check_x_scale.stypy_localization = localization
    check_x_scale.stypy_type_of_self = None
    check_x_scale.stypy_type_store = module_type_store
    check_x_scale.stypy_function_name = 'check_x_scale'
    check_x_scale.stypy_param_names_list = ['x_scale', 'x0']
    check_x_scale.stypy_varargs_param_name = None
    check_x_scale.stypy_kwargs_param_name = None
    check_x_scale.stypy_call_defaults = defaults
    check_x_scale.stypy_call_varargs = varargs
    check_x_scale.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_x_scale', ['x_scale', 'x0'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_x_scale', localization, ['x_scale', 'x0'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_x_scale(...)' code ##################

    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'x_scale' (line 124)
    x_scale_250913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 18), 'x_scale', False)
    # Getting the type of 'string_types' (line 124)
    string_types_250914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 27), 'string_types', False)
    # Processing the call keyword arguments (line 124)
    kwargs_250915 = {}
    # Getting the type of 'isinstance' (line 124)
    isinstance_250912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 124)
    isinstance_call_result_250916 = invoke(stypy.reporting.localization.Localization(__file__, 124, 7), isinstance_250912, *[x_scale_250913, string_types_250914], **kwargs_250915)
    
    
    # Getting the type of 'x_scale' (line 124)
    x_scale_250917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 45), 'x_scale')
    str_250918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 56), 'str', 'jac')
    # Applying the binary operator '==' (line 124)
    result_eq_250919 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 45), '==', x_scale_250917, str_250918)
    
    # Applying the binary operator 'and' (line 124)
    result_and_keyword_250920 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 7), 'and', isinstance_call_result_250916, result_eq_250919)
    
    # Testing the type of an if condition (line 124)
    if_condition_250921 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 4), result_and_keyword_250920)
    # Assigning a type to the variable 'if_condition_250921' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'if_condition_250921', if_condition_250921)
    # SSA begins for if statement (line 124)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'x_scale' (line 125)
    x_scale_250922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 15), 'x_scale')
    # Assigning a type to the variable 'stypy_return_type' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'stypy_return_type', x_scale_250922)
    # SSA join for if statement (line 124)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 127)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 128):
    
    # Assigning a Call to a Name (line 128):
    
    # Call to asarray(...): (line 128)
    # Processing the call arguments (line 128)
    # Getting the type of 'x_scale' (line 128)
    x_scale_250925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 29), 'x_scale', False)
    # Processing the call keyword arguments (line 128)
    # Getting the type of 'float' (line 128)
    float_250926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 44), 'float', False)
    keyword_250927 = float_250926
    kwargs_250928 = {'dtype': keyword_250927}
    # Getting the type of 'np' (line 128)
    np_250923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 18), 'np', False)
    # Obtaining the member 'asarray' of a type (line 128)
    asarray_250924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 18), np_250923, 'asarray')
    # Calling asarray(args, kwargs) (line 128)
    asarray_call_result_250929 = invoke(stypy.reporting.localization.Localization(__file__, 128, 18), asarray_250924, *[x_scale_250925], **kwargs_250928)
    
    # Assigning a type to the variable 'x_scale' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'x_scale', asarray_call_result_250929)
    
    # Assigning a BoolOp to a Name (line 129):
    
    # Assigning a BoolOp to a Name (line 129):
    
    # Evaluating a boolean operation
    
    # Call to all(...): (line 129)
    # Processing the call arguments (line 129)
    
    # Call to isfinite(...): (line 129)
    # Processing the call arguments (line 129)
    # Getting the type of 'x_scale' (line 129)
    x_scale_250934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 35), 'x_scale', False)
    # Processing the call keyword arguments (line 129)
    kwargs_250935 = {}
    # Getting the type of 'np' (line 129)
    np_250932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 23), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 129)
    isfinite_250933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 23), np_250932, 'isfinite')
    # Calling isfinite(args, kwargs) (line 129)
    isfinite_call_result_250936 = invoke(stypy.reporting.localization.Localization(__file__, 129, 23), isfinite_250933, *[x_scale_250934], **kwargs_250935)
    
    # Processing the call keyword arguments (line 129)
    kwargs_250937 = {}
    # Getting the type of 'np' (line 129)
    np_250930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'np', False)
    # Obtaining the member 'all' of a type (line 129)
    all_250931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 16), np_250930, 'all')
    # Calling all(args, kwargs) (line 129)
    all_call_result_250938 = invoke(stypy.reporting.localization.Localization(__file__, 129, 16), all_250931, *[isfinite_call_result_250936], **kwargs_250937)
    
    
    # Call to all(...): (line 129)
    # Processing the call arguments (line 129)
    
    # Getting the type of 'x_scale' (line 129)
    x_scale_250941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 56), 'x_scale', False)
    int_250942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 66), 'int')
    # Applying the binary operator '>' (line 129)
    result_gt_250943 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 56), '>', x_scale_250941, int_250942)
    
    # Processing the call keyword arguments (line 129)
    kwargs_250944 = {}
    # Getting the type of 'np' (line 129)
    np_250939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 49), 'np', False)
    # Obtaining the member 'all' of a type (line 129)
    all_250940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 49), np_250939, 'all')
    # Calling all(args, kwargs) (line 129)
    all_call_result_250945 = invoke(stypy.reporting.localization.Localization(__file__, 129, 49), all_250940, *[result_gt_250943], **kwargs_250944)
    
    # Applying the binary operator 'and' (line 129)
    result_and_keyword_250946 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 16), 'and', all_call_result_250938, all_call_result_250945)
    
    # Assigning a type to the variable 'valid' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'valid', result_and_keyword_250946)
    # SSA branch for the except part of a try statement (line 127)
    # SSA branch for the except 'Tuple' branch of a try statement (line 127)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 131):
    
    # Assigning a Name to a Name (line 131):
    # Getting the type of 'False' (line 131)
    False_250947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'False')
    # Assigning a type to the variable 'valid' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'valid', False_250947)
    # SSA join for try-except statement (line 127)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'valid' (line 133)
    valid_250948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 11), 'valid')
    # Applying the 'not' unary operator (line 133)
    result_not__250949 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 7), 'not', valid_250948)
    
    # Testing the type of an if condition (line 133)
    if_condition_250950 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 4), result_not__250949)
    # Assigning a type to the variable 'if_condition_250950' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'if_condition_250950', if_condition_250950)
    # SSA begins for if statement (line 133)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 134)
    # Processing the call arguments (line 134)
    str_250952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 25), 'str', "`x_scale` must be 'jac' or array_like with positive numbers.")
    # Processing the call keyword arguments (line 134)
    kwargs_250953 = {}
    # Getting the type of 'ValueError' (line 134)
    ValueError_250951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 134)
    ValueError_call_result_250954 = invoke(stypy.reporting.localization.Localization(__file__, 134, 14), ValueError_250951, *[str_250952], **kwargs_250953)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 134, 8), ValueError_call_result_250954, 'raise parameter', BaseException)
    # SSA join for if statement (line 133)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'x_scale' (line 137)
    x_scale_250955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 7), 'x_scale')
    # Obtaining the member 'ndim' of a type (line 137)
    ndim_250956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 7), x_scale_250955, 'ndim')
    int_250957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 23), 'int')
    # Applying the binary operator '==' (line 137)
    result_eq_250958 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 7), '==', ndim_250956, int_250957)
    
    # Testing the type of an if condition (line 137)
    if_condition_250959 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 4), result_eq_250958)
    # Assigning a type to the variable 'if_condition_250959' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'if_condition_250959', if_condition_250959)
    # SSA begins for if statement (line 137)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 138):
    
    # Assigning a Call to a Name (line 138):
    
    # Call to resize(...): (line 138)
    # Processing the call arguments (line 138)
    # Getting the type of 'x_scale' (line 138)
    x_scale_250962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 28), 'x_scale', False)
    # Getting the type of 'x0' (line 138)
    x0_250963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 37), 'x0', False)
    # Obtaining the member 'shape' of a type (line 138)
    shape_250964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 37), x0_250963, 'shape')
    # Processing the call keyword arguments (line 138)
    kwargs_250965 = {}
    # Getting the type of 'np' (line 138)
    np_250960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 18), 'np', False)
    # Obtaining the member 'resize' of a type (line 138)
    resize_250961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 18), np_250960, 'resize')
    # Calling resize(args, kwargs) (line 138)
    resize_call_result_250966 = invoke(stypy.reporting.localization.Localization(__file__, 138, 18), resize_250961, *[x_scale_250962, shape_250964], **kwargs_250965)
    
    # Assigning a type to the variable 'x_scale' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'x_scale', resize_call_result_250966)
    # SSA join for if statement (line 137)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'x_scale' (line 140)
    x_scale_250967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 7), 'x_scale')
    # Obtaining the member 'shape' of a type (line 140)
    shape_250968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 7), x_scale_250967, 'shape')
    # Getting the type of 'x0' (line 140)
    x0_250969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 24), 'x0')
    # Obtaining the member 'shape' of a type (line 140)
    shape_250970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 24), x0_250969, 'shape')
    # Applying the binary operator '!=' (line 140)
    result_ne_250971 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 7), '!=', shape_250968, shape_250970)
    
    # Testing the type of an if condition (line 140)
    if_condition_250972 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 4), result_ne_250971)
    # Assigning a type to the variable 'if_condition_250972' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'if_condition_250972', if_condition_250972)
    # SSA begins for if statement (line 140)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 141)
    # Processing the call arguments (line 141)
    str_250974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 25), 'str', 'Inconsistent shapes between `x_scale` and `x0`.')
    # Processing the call keyword arguments (line 141)
    kwargs_250975 = {}
    # Getting the type of 'ValueError' (line 141)
    ValueError_250973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 141)
    ValueError_call_result_250976 = invoke(stypy.reporting.localization.Localization(__file__, 141, 14), ValueError_250973, *[str_250974], **kwargs_250975)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 141, 8), ValueError_call_result_250976, 'raise parameter', BaseException)
    # SSA join for if statement (line 140)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x_scale' (line 143)
    x_scale_250977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 11), 'x_scale')
    # Assigning a type to the variable 'stypy_return_type' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type', x_scale_250977)
    
    # ################# End of 'check_x_scale(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_x_scale' in the type store
    # Getting the type of 'stypy_return_type' (line 123)
    stypy_return_type_250978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_250978)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_x_scale'
    return stypy_return_type_250978

# Assigning a type to the variable 'check_x_scale' (line 123)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 0), 'check_x_scale', check_x_scale)

@norecursion
def check_jac_sparsity(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_jac_sparsity'
    module_type_store = module_type_store.open_function_context('check_jac_sparsity', 146, 0, False)
    
    # Passed parameters checking function
    check_jac_sparsity.stypy_localization = localization
    check_jac_sparsity.stypy_type_of_self = None
    check_jac_sparsity.stypy_type_store = module_type_store
    check_jac_sparsity.stypy_function_name = 'check_jac_sparsity'
    check_jac_sparsity.stypy_param_names_list = ['jac_sparsity', 'm', 'n']
    check_jac_sparsity.stypy_varargs_param_name = None
    check_jac_sparsity.stypy_kwargs_param_name = None
    check_jac_sparsity.stypy_call_defaults = defaults
    check_jac_sparsity.stypy_call_varargs = varargs
    check_jac_sparsity.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_jac_sparsity', ['jac_sparsity', 'm', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_jac_sparsity', localization, ['jac_sparsity', 'm', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_jac_sparsity(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 147)
    # Getting the type of 'jac_sparsity' (line 147)
    jac_sparsity_250979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 7), 'jac_sparsity')
    # Getting the type of 'None' (line 147)
    None_250980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 23), 'None')
    
    (may_be_250981, more_types_in_union_250982) = may_be_none(jac_sparsity_250979, None_250980)

    if may_be_250981:

        if more_types_in_union_250982:
            # Runtime conditional SSA (line 147)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'None' (line 148)
        None_250983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'stypy_return_type', None_250983)

        if more_types_in_union_250982:
            # SSA join for if statement (line 147)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    
    # Call to issparse(...): (line 150)
    # Processing the call arguments (line 150)
    # Getting the type of 'jac_sparsity' (line 150)
    jac_sparsity_250985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 20), 'jac_sparsity', False)
    # Processing the call keyword arguments (line 150)
    kwargs_250986 = {}
    # Getting the type of 'issparse' (line 150)
    issparse_250984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 11), 'issparse', False)
    # Calling issparse(args, kwargs) (line 150)
    issparse_call_result_250987 = invoke(stypy.reporting.localization.Localization(__file__, 150, 11), issparse_250984, *[jac_sparsity_250985], **kwargs_250986)
    
    # Applying the 'not' unary operator (line 150)
    result_not__250988 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 7), 'not', issparse_call_result_250987)
    
    # Testing the type of an if condition (line 150)
    if_condition_250989 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 4), result_not__250988)
    # Assigning a type to the variable 'if_condition_250989' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'if_condition_250989', if_condition_250989)
    # SSA begins for if statement (line 150)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 151):
    
    # Assigning a Call to a Name (line 151):
    
    # Call to atleast_2d(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'jac_sparsity' (line 151)
    jac_sparsity_250992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 37), 'jac_sparsity', False)
    # Processing the call keyword arguments (line 151)
    kwargs_250993 = {}
    # Getting the type of 'np' (line 151)
    np_250990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 23), 'np', False)
    # Obtaining the member 'atleast_2d' of a type (line 151)
    atleast_2d_250991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 23), np_250990, 'atleast_2d')
    # Calling atleast_2d(args, kwargs) (line 151)
    atleast_2d_call_result_250994 = invoke(stypy.reporting.localization.Localization(__file__, 151, 23), atleast_2d_250991, *[jac_sparsity_250992], **kwargs_250993)
    
    # Assigning a type to the variable 'jac_sparsity' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'jac_sparsity', atleast_2d_call_result_250994)
    # SSA join for if statement (line 150)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'jac_sparsity' (line 153)
    jac_sparsity_250995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 7), 'jac_sparsity')
    # Obtaining the member 'shape' of a type (line 153)
    shape_250996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 7), jac_sparsity_250995, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 153)
    tuple_250997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 153)
    # Adding element type (line 153)
    # Getting the type of 'm' (line 153)
    m_250998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 30), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 30), tuple_250997, m_250998)
    # Adding element type (line 153)
    # Getting the type of 'n' (line 153)
    n_250999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 33), 'n')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 30), tuple_250997, n_250999)
    
    # Applying the binary operator '!=' (line 153)
    result_ne_251000 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 7), '!=', shape_250996, tuple_250997)
    
    # Testing the type of an if condition (line 153)
    if_condition_251001 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 4), result_ne_251000)
    # Assigning a type to the variable 'if_condition_251001' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'if_condition_251001', if_condition_251001)
    # SSA begins for if statement (line 153)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 154)
    # Processing the call arguments (line 154)
    str_251003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 25), 'str', '`jac_sparsity` has wrong shape.')
    # Processing the call keyword arguments (line 154)
    kwargs_251004 = {}
    # Getting the type of 'ValueError' (line 154)
    ValueError_251002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 154)
    ValueError_call_result_251005 = invoke(stypy.reporting.localization.Localization(__file__, 154, 14), ValueError_251002, *[str_251003], **kwargs_251004)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 154, 8), ValueError_call_result_251005, 'raise parameter', BaseException)
    # SSA join for if statement (line 153)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 156)
    tuple_251006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 156)
    # Adding element type (line 156)
    # Getting the type of 'jac_sparsity' (line 156)
    jac_sparsity_251007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 11), 'jac_sparsity')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 11), tuple_251006, jac_sparsity_251007)
    # Adding element type (line 156)
    
    # Call to group_columns(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'jac_sparsity' (line 156)
    jac_sparsity_251009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 39), 'jac_sparsity', False)
    # Processing the call keyword arguments (line 156)
    kwargs_251010 = {}
    # Getting the type of 'group_columns' (line 156)
    group_columns_251008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 25), 'group_columns', False)
    # Calling group_columns(args, kwargs) (line 156)
    group_columns_call_result_251011 = invoke(stypy.reporting.localization.Localization(__file__, 156, 25), group_columns_251008, *[jac_sparsity_251009], **kwargs_251010)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 11), tuple_251006, group_columns_call_result_251011)
    
    # Assigning a type to the variable 'stypy_return_type' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'stypy_return_type', tuple_251006)
    
    # ################# End of 'check_jac_sparsity(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_jac_sparsity' in the type store
    # Getting the type of 'stypy_return_type' (line 146)
    stypy_return_type_251012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_251012)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_jac_sparsity'
    return stypy_return_type_251012

# Assigning a type to the variable 'check_jac_sparsity' (line 146)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 0), 'check_jac_sparsity', check_jac_sparsity)

@norecursion
def huber(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'huber'
    module_type_store = module_type_store.open_function_context('huber', 162, 0, False)
    
    # Passed parameters checking function
    huber.stypy_localization = localization
    huber.stypy_type_of_self = None
    huber.stypy_type_store = module_type_store
    huber.stypy_function_name = 'huber'
    huber.stypy_param_names_list = ['z', 'rho', 'cost_only']
    huber.stypy_varargs_param_name = None
    huber.stypy_kwargs_param_name = None
    huber.stypy_call_defaults = defaults
    huber.stypy_call_varargs = varargs
    huber.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'huber', ['z', 'rho', 'cost_only'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'huber', localization, ['z', 'rho', 'cost_only'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'huber(...)' code ##################

    
    # Assigning a Compare to a Name (line 163):
    
    # Assigning a Compare to a Name (line 163):
    
    # Getting the type of 'z' (line 163)
    z_251013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 11), 'z')
    int_251014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 16), 'int')
    # Applying the binary operator '<=' (line 163)
    result_le_251015 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 11), '<=', z_251013, int_251014)
    
    # Assigning a type to the variable 'mask' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'mask', result_le_251015)
    
    # Assigning a Subscript to a Subscript (line 164):
    
    # Assigning a Subscript to a Subscript (line 164):
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 164)
    mask_251016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 21), 'mask')
    # Getting the type of 'z' (line 164)
    z_251017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 19), 'z')
    # Obtaining the member '__getitem__' of a type (line 164)
    getitem___251018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 19), z_251017, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 164)
    subscript_call_result_251019 = invoke(stypy.reporting.localization.Localization(__file__, 164, 19), getitem___251018, mask_251016)
    
    # Getting the type of 'rho' (line 164)
    rho_251020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'rho')
    
    # Obtaining an instance of the builtin type 'tuple' (line 164)
    tuple_251021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 8), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 164)
    # Adding element type (line 164)
    int_251022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 8), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 8), tuple_251021, int_251022)
    # Adding element type (line 164)
    # Getting the type of 'mask' (line 164)
    mask_251023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 11), 'mask')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 8), tuple_251021, mask_251023)
    
    # Storing an element on a container (line 164)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 4), rho_251020, (tuple_251021, subscript_call_result_251019))
    
    # Assigning a BinOp to a Subscript (line 165):
    
    # Assigning a BinOp to a Subscript (line 165):
    int_251024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 20), 'int')
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'mask' (line 165)
    mask_251025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 27), 'mask')
    # Applying the '~' unary operator (line 165)
    result_inv_251026 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 26), '~', mask_251025)
    
    # Getting the type of 'z' (line 165)
    z_251027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 24), 'z')
    # Obtaining the member '__getitem__' of a type (line 165)
    getitem___251028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 24), z_251027, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 165)
    subscript_call_result_251029 = invoke(stypy.reporting.localization.Localization(__file__, 165, 24), getitem___251028, result_inv_251026)
    
    float_251030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 34), 'float')
    # Applying the binary operator '**' (line 165)
    result_pow_251031 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 24), '**', subscript_call_result_251029, float_251030)
    
    # Applying the binary operator '*' (line 165)
    result_mul_251032 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 20), '*', int_251024, result_pow_251031)
    
    int_251033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 40), 'int')
    # Applying the binary operator '-' (line 165)
    result_sub_251034 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 20), '-', result_mul_251032, int_251033)
    
    # Getting the type of 'rho' (line 165)
    rho_251035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'rho')
    
    # Obtaining an instance of the builtin type 'tuple' (line 165)
    tuple_251036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 8), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 165)
    # Adding element type (line 165)
    int_251037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 8), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 8), tuple_251036, int_251037)
    # Adding element type (line 165)
    
    # Getting the type of 'mask' (line 165)
    mask_251038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'mask')
    # Applying the '~' unary operator (line 165)
    result_inv_251039 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 11), '~', mask_251038)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 8), tuple_251036, result_inv_251039)
    
    # Storing an element on a container (line 165)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 4), rho_251035, (tuple_251036, result_sub_251034))
    
    # Getting the type of 'cost_only' (line 166)
    cost_only_251040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 7), 'cost_only')
    # Testing the type of an if condition (line 166)
    if_condition_251041 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 166, 4), cost_only_251040)
    # Assigning a type to the variable 'if_condition_251041' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'if_condition_251041', if_condition_251041)
    # SSA begins for if statement (line 166)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 166)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Subscript (line 168):
    
    # Assigning a Num to a Subscript (line 168):
    int_251042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 19), 'int')
    # Getting the type of 'rho' (line 168)
    rho_251043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'rho')
    
    # Obtaining an instance of the builtin type 'tuple' (line 168)
    tuple_251044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 8), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 168)
    # Adding element type (line 168)
    int_251045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 8), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), tuple_251044, int_251045)
    # Adding element type (line 168)
    # Getting the type of 'mask' (line 168)
    mask_251046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 11), 'mask')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), tuple_251044, mask_251046)
    
    # Storing an element on a container (line 168)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 4), rho_251043, (tuple_251044, int_251042))
    
    # Assigning a BinOp to a Subscript (line 169):
    
    # Assigning a BinOp to a Subscript (line 169):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'mask' (line 169)
    mask_251047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 23), 'mask')
    # Applying the '~' unary operator (line 169)
    result_inv_251048 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 22), '~', mask_251047)
    
    # Getting the type of 'z' (line 169)
    z_251049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 20), 'z')
    # Obtaining the member '__getitem__' of a type (line 169)
    getitem___251050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 20), z_251049, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 169)
    subscript_call_result_251051 = invoke(stypy.reporting.localization.Localization(__file__, 169, 20), getitem___251050, result_inv_251048)
    
    float_251052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 30), 'float')
    # Applying the binary operator '**' (line 169)
    result_pow_251053 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 20), '**', subscript_call_result_251051, float_251052)
    
    # Getting the type of 'rho' (line 169)
    rho_251054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'rho')
    
    # Obtaining an instance of the builtin type 'tuple' (line 169)
    tuple_251055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 8), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 169)
    # Adding element type (line 169)
    int_251056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 8), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 8), tuple_251055, int_251056)
    # Adding element type (line 169)
    
    # Getting the type of 'mask' (line 169)
    mask_251057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'mask')
    # Applying the '~' unary operator (line 169)
    result_inv_251058 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 11), '~', mask_251057)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 8), tuple_251055, result_inv_251058)
    
    # Storing an element on a container (line 169)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 4), rho_251054, (tuple_251055, result_pow_251053))
    
    # Assigning a Num to a Subscript (line 170):
    
    # Assigning a Num to a Subscript (line 170):
    int_251059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 19), 'int')
    # Getting the type of 'rho' (line 170)
    rho_251060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'rho')
    
    # Obtaining an instance of the builtin type 'tuple' (line 170)
    tuple_251061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 8), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 170)
    # Adding element type (line 170)
    int_251062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 8), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 8), tuple_251061, int_251062)
    # Adding element type (line 170)
    # Getting the type of 'mask' (line 170)
    mask_251063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 11), 'mask')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 8), tuple_251061, mask_251063)
    
    # Storing an element on a container (line 170)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 4), rho_251060, (tuple_251061, int_251059))
    
    # Assigning a BinOp to a Subscript (line 171):
    
    # Assigning a BinOp to a Subscript (line 171):
    float_251064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 20), 'float')
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'mask' (line 171)
    mask_251065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 30), 'mask')
    # Applying the '~' unary operator (line 171)
    result_inv_251066 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 29), '~', mask_251065)
    
    # Getting the type of 'z' (line 171)
    z_251067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 27), 'z')
    # Obtaining the member '__getitem__' of a type (line 171)
    getitem___251068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 27), z_251067, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 171)
    subscript_call_result_251069 = invoke(stypy.reporting.localization.Localization(__file__, 171, 27), getitem___251068, result_inv_251066)
    
    float_251070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 37), 'float')
    # Applying the binary operator '**' (line 171)
    result_pow_251071 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 27), '**', subscript_call_result_251069, float_251070)
    
    # Applying the binary operator '*' (line 171)
    result_mul_251072 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 20), '*', float_251064, result_pow_251071)
    
    # Getting the type of 'rho' (line 171)
    rho_251073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'rho')
    
    # Obtaining an instance of the builtin type 'tuple' (line 171)
    tuple_251074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 8), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 171)
    # Adding element type (line 171)
    int_251075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 8), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 8), tuple_251074, int_251075)
    # Adding element type (line 171)
    
    # Getting the type of 'mask' (line 171)
    mask_251076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'mask')
    # Applying the '~' unary operator (line 171)
    result_inv_251077 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 11), '~', mask_251076)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 8), tuple_251074, result_inv_251077)
    
    # Storing an element on a container (line 171)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 4), rho_251073, (tuple_251074, result_mul_251072))
    
    # ################# End of 'huber(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'huber' in the type store
    # Getting the type of 'stypy_return_type' (line 162)
    stypy_return_type_251078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_251078)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'huber'
    return stypy_return_type_251078

# Assigning a type to the variable 'huber' (line 162)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 0), 'huber', huber)

@norecursion
def soft_l1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'soft_l1'
    module_type_store = module_type_store.open_function_context('soft_l1', 174, 0, False)
    
    # Passed parameters checking function
    soft_l1.stypy_localization = localization
    soft_l1.stypy_type_of_self = None
    soft_l1.stypy_type_store = module_type_store
    soft_l1.stypy_function_name = 'soft_l1'
    soft_l1.stypy_param_names_list = ['z', 'rho', 'cost_only']
    soft_l1.stypy_varargs_param_name = None
    soft_l1.stypy_kwargs_param_name = None
    soft_l1.stypy_call_defaults = defaults
    soft_l1.stypy_call_varargs = varargs
    soft_l1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'soft_l1', ['z', 'rho', 'cost_only'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'soft_l1', localization, ['z', 'rho', 'cost_only'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'soft_l1(...)' code ##################

    
    # Assigning a BinOp to a Name (line 175):
    
    # Assigning a BinOp to a Name (line 175):
    int_251079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 8), 'int')
    # Getting the type of 'z' (line 175)
    z_251080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'z')
    # Applying the binary operator '+' (line 175)
    result_add_251081 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 8), '+', int_251079, z_251080)
    
    # Assigning a type to the variable 't' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 't', result_add_251081)
    
    # Assigning a BinOp to a Subscript (line 176):
    
    # Assigning a BinOp to a Subscript (line 176):
    int_251082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 13), 'int')
    # Getting the type of 't' (line 176)
    t_251083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 18), 't')
    float_251084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 21), 'float')
    # Applying the binary operator '**' (line 176)
    result_pow_251085 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 18), '**', t_251083, float_251084)
    
    int_251086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 27), 'int')
    # Applying the binary operator '-' (line 176)
    result_sub_251087 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 18), '-', result_pow_251085, int_251086)
    
    # Applying the binary operator '*' (line 176)
    result_mul_251088 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 13), '*', int_251082, result_sub_251087)
    
    # Getting the type of 'rho' (line 176)
    rho_251089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'rho')
    int_251090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 8), 'int')
    # Storing an element on a container (line 176)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 4), rho_251089, (int_251090, result_mul_251088))
    
    # Getting the type of 'cost_only' (line 177)
    cost_only_251091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 7), 'cost_only')
    # Testing the type of an if condition (line 177)
    if_condition_251092 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 177, 4), cost_only_251091)
    # Assigning a type to the variable 'if_condition_251092' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'if_condition_251092', if_condition_251092)
    # SSA begins for if statement (line 177)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 177)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Subscript (line 179):
    
    # Assigning a BinOp to a Subscript (line 179):
    # Getting the type of 't' (line 179)
    t_251093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 13), 't')
    float_251094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 16), 'float')
    # Applying the binary operator '**' (line 179)
    result_pow_251095 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 13), '**', t_251093, float_251094)
    
    # Getting the type of 'rho' (line 179)
    rho_251096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'rho')
    int_251097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 8), 'int')
    # Storing an element on a container (line 179)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 4), rho_251096, (int_251097, result_pow_251095))
    
    # Assigning a BinOp to a Subscript (line 180):
    
    # Assigning a BinOp to a Subscript (line 180):
    float_251098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 13), 'float')
    # Getting the type of 't' (line 180)
    t_251099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 20), 't')
    float_251100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 23), 'float')
    # Applying the binary operator '**' (line 180)
    result_pow_251101 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 20), '**', t_251099, float_251100)
    
    # Applying the binary operator '*' (line 180)
    result_mul_251102 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 13), '*', float_251098, result_pow_251101)
    
    # Getting the type of 'rho' (line 180)
    rho_251103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'rho')
    int_251104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 8), 'int')
    # Storing an element on a container (line 180)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 4), rho_251103, (int_251104, result_mul_251102))
    
    # ################# End of 'soft_l1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'soft_l1' in the type store
    # Getting the type of 'stypy_return_type' (line 174)
    stypy_return_type_251105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_251105)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'soft_l1'
    return stypy_return_type_251105

# Assigning a type to the variable 'soft_l1' (line 174)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 0), 'soft_l1', soft_l1)

@norecursion
def cauchy(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'cauchy'
    module_type_store = module_type_store.open_function_context('cauchy', 183, 0, False)
    
    # Passed parameters checking function
    cauchy.stypy_localization = localization
    cauchy.stypy_type_of_self = None
    cauchy.stypy_type_store = module_type_store
    cauchy.stypy_function_name = 'cauchy'
    cauchy.stypy_param_names_list = ['z', 'rho', 'cost_only']
    cauchy.stypy_varargs_param_name = None
    cauchy.stypy_kwargs_param_name = None
    cauchy.stypy_call_defaults = defaults
    cauchy.stypy_call_varargs = varargs
    cauchy.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cauchy', ['z', 'rho', 'cost_only'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cauchy', localization, ['z', 'rho', 'cost_only'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cauchy(...)' code ##################

    
    # Assigning a Call to a Subscript (line 184):
    
    # Assigning a Call to a Subscript (line 184):
    
    # Call to log1p(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'z' (line 184)
    z_251108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 22), 'z', False)
    # Processing the call keyword arguments (line 184)
    kwargs_251109 = {}
    # Getting the type of 'np' (line 184)
    np_251106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 13), 'np', False)
    # Obtaining the member 'log1p' of a type (line 184)
    log1p_251107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 13), np_251106, 'log1p')
    # Calling log1p(args, kwargs) (line 184)
    log1p_call_result_251110 = invoke(stypy.reporting.localization.Localization(__file__, 184, 13), log1p_251107, *[z_251108], **kwargs_251109)
    
    # Getting the type of 'rho' (line 184)
    rho_251111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'rho')
    int_251112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 8), 'int')
    # Storing an element on a container (line 184)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 4), rho_251111, (int_251112, log1p_call_result_251110))
    
    # Getting the type of 'cost_only' (line 185)
    cost_only_251113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 7), 'cost_only')
    # Testing the type of an if condition (line 185)
    if_condition_251114 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 4), cost_only_251113)
    # Assigning a type to the variable 'if_condition_251114' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'if_condition_251114', if_condition_251114)
    # SSA begins for if statement (line 185)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 185)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 187):
    
    # Assigning a BinOp to a Name (line 187):
    int_251115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 8), 'int')
    # Getting the type of 'z' (line 187)
    z_251116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'z')
    # Applying the binary operator '+' (line 187)
    result_add_251117 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 8), '+', int_251115, z_251116)
    
    # Assigning a type to the variable 't' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 't', result_add_251117)
    
    # Assigning a BinOp to a Subscript (line 188):
    
    # Assigning a BinOp to a Subscript (line 188):
    int_251118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 13), 'int')
    # Getting the type of 't' (line 188)
    t_251119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 17), 't')
    # Applying the binary operator 'div' (line 188)
    result_div_251120 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 13), 'div', int_251118, t_251119)
    
    # Getting the type of 'rho' (line 188)
    rho_251121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'rho')
    int_251122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 8), 'int')
    # Storing an element on a container (line 188)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 4), rho_251121, (int_251122, result_div_251120))
    
    # Assigning a BinOp to a Subscript (line 189):
    
    # Assigning a BinOp to a Subscript (line 189):
    int_251123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 13), 'int')
    # Getting the type of 't' (line 189)
    t_251124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 18), 't')
    int_251125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 21), 'int')
    # Applying the binary operator '**' (line 189)
    result_pow_251126 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 18), '**', t_251124, int_251125)
    
    # Applying the binary operator 'div' (line 189)
    result_div_251127 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 13), 'div', int_251123, result_pow_251126)
    
    # Getting the type of 'rho' (line 189)
    rho_251128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'rho')
    int_251129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 8), 'int')
    # Storing an element on a container (line 189)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 4), rho_251128, (int_251129, result_div_251127))
    
    # ################# End of 'cauchy(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cauchy' in the type store
    # Getting the type of 'stypy_return_type' (line 183)
    stypy_return_type_251130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_251130)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cauchy'
    return stypy_return_type_251130

# Assigning a type to the variable 'cauchy' (line 183)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), 'cauchy', cauchy)

@norecursion
def arctan(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'arctan'
    module_type_store = module_type_store.open_function_context('arctan', 192, 0, False)
    
    # Passed parameters checking function
    arctan.stypy_localization = localization
    arctan.stypy_type_of_self = None
    arctan.stypy_type_store = module_type_store
    arctan.stypy_function_name = 'arctan'
    arctan.stypy_param_names_list = ['z', 'rho', 'cost_only']
    arctan.stypy_varargs_param_name = None
    arctan.stypy_kwargs_param_name = None
    arctan.stypy_call_defaults = defaults
    arctan.stypy_call_varargs = varargs
    arctan.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'arctan', ['z', 'rho', 'cost_only'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'arctan', localization, ['z', 'rho', 'cost_only'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'arctan(...)' code ##################

    
    # Assigning a Call to a Subscript (line 193):
    
    # Assigning a Call to a Subscript (line 193):
    
    # Call to arctan(...): (line 193)
    # Processing the call arguments (line 193)
    # Getting the type of 'z' (line 193)
    z_251133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 23), 'z', False)
    # Processing the call keyword arguments (line 193)
    kwargs_251134 = {}
    # Getting the type of 'np' (line 193)
    np_251131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 13), 'np', False)
    # Obtaining the member 'arctan' of a type (line 193)
    arctan_251132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 13), np_251131, 'arctan')
    # Calling arctan(args, kwargs) (line 193)
    arctan_call_result_251135 = invoke(stypy.reporting.localization.Localization(__file__, 193, 13), arctan_251132, *[z_251133], **kwargs_251134)
    
    # Getting the type of 'rho' (line 193)
    rho_251136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'rho')
    int_251137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 8), 'int')
    # Storing an element on a container (line 193)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 4), rho_251136, (int_251137, arctan_call_result_251135))
    
    # Getting the type of 'cost_only' (line 194)
    cost_only_251138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 7), 'cost_only')
    # Testing the type of an if condition (line 194)
    if_condition_251139 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 4), cost_only_251138)
    # Assigning a type to the variable 'if_condition_251139' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'if_condition_251139', if_condition_251139)
    # SSA begins for if statement (line 194)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 194)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 196):
    
    # Assigning a BinOp to a Name (line 196):
    int_251140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 8), 'int')
    # Getting the type of 'z' (line 196)
    z_251141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'z')
    int_251142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 15), 'int')
    # Applying the binary operator '**' (line 196)
    result_pow_251143 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 12), '**', z_251141, int_251142)
    
    # Applying the binary operator '+' (line 196)
    result_add_251144 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 8), '+', int_251140, result_pow_251143)
    
    # Assigning a type to the variable 't' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 't', result_add_251144)
    
    # Assigning a BinOp to a Subscript (line 197):
    
    # Assigning a BinOp to a Subscript (line 197):
    int_251145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 13), 'int')
    # Getting the type of 't' (line 197)
    t_251146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 17), 't')
    # Applying the binary operator 'div' (line 197)
    result_div_251147 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 13), 'div', int_251145, t_251146)
    
    # Getting the type of 'rho' (line 197)
    rho_251148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'rho')
    int_251149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 8), 'int')
    # Storing an element on a container (line 197)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 4), rho_251148, (int_251149, result_div_251147))
    
    # Assigning a BinOp to a Subscript (line 198):
    
    # Assigning a BinOp to a Subscript (line 198):
    int_251150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 13), 'int')
    # Getting the type of 'z' (line 198)
    z_251151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 18), 'z')
    # Applying the binary operator '*' (line 198)
    result_mul_251152 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 13), '*', int_251150, z_251151)
    
    # Getting the type of 't' (line 198)
    t_251153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 22), 't')
    int_251154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 25), 'int')
    # Applying the binary operator '**' (line 198)
    result_pow_251155 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 22), '**', t_251153, int_251154)
    
    # Applying the binary operator 'div' (line 198)
    result_div_251156 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 20), 'div', result_mul_251152, result_pow_251155)
    
    # Getting the type of 'rho' (line 198)
    rho_251157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'rho')
    int_251158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 8), 'int')
    # Storing an element on a container (line 198)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 4), rho_251157, (int_251158, result_div_251156))
    
    # ################# End of 'arctan(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'arctan' in the type store
    # Getting the type of 'stypy_return_type' (line 192)
    stypy_return_type_251159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_251159)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'arctan'
    return stypy_return_type_251159

# Assigning a type to the variable 'arctan' (line 192)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 0), 'arctan', arctan)

# Assigning a Call to a Name (line 201):

# Assigning a Call to a Name (line 201):

# Call to dict(...): (line 201)
# Processing the call keyword arguments (line 201)
# Getting the type of 'None' (line 201)
None_251161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 33), 'None', False)
keyword_251162 = None_251161
# Getting the type of 'huber' (line 201)
huber_251163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 45), 'huber', False)
keyword_251164 = huber_251163
# Getting the type of 'soft_l1' (line 201)
soft_l1_251165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 60), 'soft_l1', False)
keyword_251166 = soft_l1_251165
# Getting the type of 'cauchy' (line 202)
cauchy_251167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 33), 'cauchy', False)
keyword_251168 = cauchy_251167
# Getting the type of 'arctan' (line 202)
arctan_251169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 48), 'arctan', False)
keyword_251170 = arctan_251169
kwargs_251171 = {'huber': keyword_251164, 'cauchy': keyword_251168, 'linear': keyword_251162, 'arctan': keyword_251170, 'soft_l1': keyword_251166}
# Getting the type of 'dict' (line 201)
dict_251160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 21), 'dict', False)
# Calling dict(args, kwargs) (line 201)
dict_call_result_251172 = invoke(stypy.reporting.localization.Localization(__file__, 201, 21), dict_251160, *[], **kwargs_251171)

# Assigning a type to the variable 'IMPLEMENTED_LOSSES' (line 201)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 0), 'IMPLEMENTED_LOSSES', dict_call_result_251172)

@norecursion
def construct_loss_function(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'construct_loss_function'
    module_type_store = module_type_store.open_function_context('construct_loss_function', 205, 0, False)
    
    # Passed parameters checking function
    construct_loss_function.stypy_localization = localization
    construct_loss_function.stypy_type_of_self = None
    construct_loss_function.stypy_type_store = module_type_store
    construct_loss_function.stypy_function_name = 'construct_loss_function'
    construct_loss_function.stypy_param_names_list = ['m', 'loss', 'f_scale']
    construct_loss_function.stypy_varargs_param_name = None
    construct_loss_function.stypy_kwargs_param_name = None
    construct_loss_function.stypy_call_defaults = defaults
    construct_loss_function.stypy_call_varargs = varargs
    construct_loss_function.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'construct_loss_function', ['m', 'loss', 'f_scale'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'construct_loss_function', localization, ['m', 'loss', 'f_scale'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'construct_loss_function(...)' code ##################

    
    
    # Getting the type of 'loss' (line 206)
    loss_251173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 7), 'loss')
    str_251174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 15), 'str', 'linear')
    # Applying the binary operator '==' (line 206)
    result_eq_251175 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 7), '==', loss_251173, str_251174)
    
    # Testing the type of an if condition (line 206)
    if_condition_251176 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 4), result_eq_251175)
    # Assigning a type to the variable 'if_condition_251176' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'if_condition_251176', if_condition_251176)
    # SSA begins for if statement (line 206)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'None' (line 207)
    None_251177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'stypy_return_type', None_251177)
    # SSA join for if statement (line 206)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to callable(...): (line 209)
    # Processing the call arguments (line 209)
    # Getting the type of 'loss' (line 209)
    loss_251179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 20), 'loss', False)
    # Processing the call keyword arguments (line 209)
    kwargs_251180 = {}
    # Getting the type of 'callable' (line 209)
    callable_251178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 11), 'callable', False)
    # Calling callable(args, kwargs) (line 209)
    callable_call_result_251181 = invoke(stypy.reporting.localization.Localization(__file__, 209, 11), callable_251178, *[loss_251179], **kwargs_251180)
    
    # Applying the 'not' unary operator (line 209)
    result_not__251182 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 7), 'not', callable_call_result_251181)
    
    # Testing the type of an if condition (line 209)
    if_condition_251183 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 209, 4), result_not__251182)
    # Assigning a type to the variable 'if_condition_251183' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'if_condition_251183', if_condition_251183)
    # SSA begins for if statement (line 209)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 210):
    
    # Assigning a Subscript to a Name (line 210):
    
    # Obtaining the type of the subscript
    # Getting the type of 'loss' (line 210)
    loss_251184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 34), 'loss')
    # Getting the type of 'IMPLEMENTED_LOSSES' (line 210)
    IMPLEMENTED_LOSSES_251185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 15), 'IMPLEMENTED_LOSSES')
    # Obtaining the member '__getitem__' of a type (line 210)
    getitem___251186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 15), IMPLEMENTED_LOSSES_251185, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 210)
    subscript_call_result_251187 = invoke(stypy.reporting.localization.Localization(__file__, 210, 15), getitem___251186, loss_251184)
    
    # Assigning a type to the variable 'loss' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'loss', subscript_call_result_251187)
    
    # Assigning a Call to a Name (line 211):
    
    # Assigning a Call to a Name (line 211):
    
    # Call to empty(...): (line 211)
    # Processing the call arguments (line 211)
    
    # Obtaining an instance of the builtin type 'tuple' (line 211)
    tuple_251190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 211)
    # Adding element type (line 211)
    int_251191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 24), tuple_251190, int_251191)
    # Adding element type (line 211)
    # Getting the type of 'm' (line 211)
    m_251192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 27), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 24), tuple_251190, m_251192)
    
    # Processing the call keyword arguments (line 211)
    kwargs_251193 = {}
    # Getting the type of 'np' (line 211)
    np_251188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 14), 'np', False)
    # Obtaining the member 'empty' of a type (line 211)
    empty_251189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 14), np_251188, 'empty')
    # Calling empty(args, kwargs) (line 211)
    empty_call_result_251194 = invoke(stypy.reporting.localization.Localization(__file__, 211, 14), empty_251189, *[tuple_251190], **kwargs_251193)
    
    # Assigning a type to the variable 'rho' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'rho', empty_call_result_251194)

    @norecursion
    def loss_function(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 213)
        False_251195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 39), 'False')
        defaults = [False_251195]
        # Create a new context for function 'loss_function'
        module_type_store = module_type_store.open_function_context('loss_function', 213, 8, False)
        
        # Passed parameters checking function
        loss_function.stypy_localization = localization
        loss_function.stypy_type_of_self = None
        loss_function.stypy_type_store = module_type_store
        loss_function.stypy_function_name = 'loss_function'
        loss_function.stypy_param_names_list = ['f', 'cost_only']
        loss_function.stypy_varargs_param_name = None
        loss_function.stypy_kwargs_param_name = None
        loss_function.stypy_call_defaults = defaults
        loss_function.stypy_call_varargs = varargs
        loss_function.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'loss_function', ['f', 'cost_only'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'loss_function', localization, ['f', 'cost_only'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'loss_function(...)' code ##################

        
        # Assigning a BinOp to a Name (line 214):
        
        # Assigning a BinOp to a Name (line 214):
        # Getting the type of 'f' (line 214)
        f_251196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 17), 'f')
        # Getting the type of 'f_scale' (line 214)
        f_scale_251197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 21), 'f_scale')
        # Applying the binary operator 'div' (line 214)
        result_div_251198 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 17), 'div', f_251196, f_scale_251197)
        
        int_251199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 33), 'int')
        # Applying the binary operator '**' (line 214)
        result_pow_251200 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 16), '**', result_div_251198, int_251199)
        
        # Assigning a type to the variable 'z' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'z', result_pow_251200)
        
        # Call to loss(...): (line 215)
        # Processing the call arguments (line 215)
        # Getting the type of 'z' (line 215)
        z_251202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 17), 'z', False)
        # Getting the type of 'rho' (line 215)
        rho_251203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 20), 'rho', False)
        # Processing the call keyword arguments (line 215)
        # Getting the type of 'cost_only' (line 215)
        cost_only_251204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 35), 'cost_only', False)
        keyword_251205 = cost_only_251204
        kwargs_251206 = {'cost_only': keyword_251205}
        # Getting the type of 'loss' (line 215)
        loss_251201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'loss', False)
        # Calling loss(args, kwargs) (line 215)
        loss_call_result_251207 = invoke(stypy.reporting.localization.Localization(__file__, 215, 12), loss_251201, *[z_251202, rho_251203], **kwargs_251206)
        
        
        # Getting the type of 'cost_only' (line 216)
        cost_only_251208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 15), 'cost_only')
        # Testing the type of an if condition (line 216)
        if_condition_251209 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 216, 12), cost_only_251208)
        # Assigning a type to the variable 'if_condition_251209' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'if_condition_251209', if_condition_251209)
        # SSA begins for if statement (line 216)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        float_251210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 23), 'float')
        # Getting the type of 'f_scale' (line 217)
        f_scale_251211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 29), 'f_scale')
        int_251212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 40), 'int')
        # Applying the binary operator '**' (line 217)
        result_pow_251213 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 29), '**', f_scale_251211, int_251212)
        
        # Applying the binary operator '*' (line 217)
        result_mul_251214 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 23), '*', float_251210, result_pow_251213)
        
        
        # Call to sum(...): (line 217)
        # Processing the call arguments (line 217)
        
        # Obtaining the type of the subscript
        int_251217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 55), 'int')
        # Getting the type of 'rho' (line 217)
        rho_251218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 51), 'rho', False)
        # Obtaining the member '__getitem__' of a type (line 217)
        getitem___251219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 51), rho_251218, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 217)
        subscript_call_result_251220 = invoke(stypy.reporting.localization.Localization(__file__, 217, 51), getitem___251219, int_251217)
        
        # Processing the call keyword arguments (line 217)
        kwargs_251221 = {}
        # Getting the type of 'np' (line 217)
        np_251215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 44), 'np', False)
        # Obtaining the member 'sum' of a type (line 217)
        sum_251216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 44), np_251215, 'sum')
        # Calling sum(args, kwargs) (line 217)
        sum_call_result_251222 = invoke(stypy.reporting.localization.Localization(__file__, 217, 44), sum_251216, *[subscript_call_result_251220], **kwargs_251221)
        
        # Applying the binary operator '*' (line 217)
        result_mul_251223 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 42), '*', result_mul_251214, sum_call_result_251222)
        
        # Assigning a type to the variable 'stypy_return_type' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 16), 'stypy_return_type', result_mul_251223)
        # SSA join for if statement (line 216)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'rho' (line 218)
        rho_251224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'rho')
        
        # Obtaining the type of the subscript
        int_251225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 16), 'int')
        # Getting the type of 'rho' (line 218)
        rho_251226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'rho')
        # Obtaining the member '__getitem__' of a type (line 218)
        getitem___251227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 12), rho_251226, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 218)
        subscript_call_result_251228 = invoke(stypy.reporting.localization.Localization(__file__, 218, 12), getitem___251227, int_251225)
        
        # Getting the type of 'f_scale' (line 218)
        f_scale_251229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 22), 'f_scale')
        int_251230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 33), 'int')
        # Applying the binary operator '**' (line 218)
        result_pow_251231 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 22), '**', f_scale_251229, int_251230)
        
        # Applying the binary operator '*=' (line 218)
        result_imul_251232 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 12), '*=', subscript_call_result_251228, result_pow_251231)
        # Getting the type of 'rho' (line 218)
        rho_251233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'rho')
        int_251234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 16), 'int')
        # Storing an element on a container (line 218)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 12), rho_251233, (int_251234, result_imul_251232))
        
        
        # Getting the type of 'rho' (line 219)
        rho_251235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'rho')
        
        # Obtaining the type of the subscript
        int_251236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 16), 'int')
        # Getting the type of 'rho' (line 219)
        rho_251237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'rho')
        # Obtaining the member '__getitem__' of a type (line 219)
        getitem___251238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 12), rho_251237, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 219)
        subscript_call_result_251239 = invoke(stypy.reporting.localization.Localization(__file__, 219, 12), getitem___251238, int_251236)
        
        # Getting the type of 'f_scale' (line 219)
        f_scale_251240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 22), 'f_scale')
        int_251241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 33), 'int')
        # Applying the binary operator '**' (line 219)
        result_pow_251242 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 22), '**', f_scale_251240, int_251241)
        
        # Applying the binary operator 'div=' (line 219)
        result_div_251243 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 12), 'div=', subscript_call_result_251239, result_pow_251242)
        # Getting the type of 'rho' (line 219)
        rho_251244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'rho')
        int_251245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 16), 'int')
        # Storing an element on a container (line 219)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 12), rho_251244, (int_251245, result_div_251243))
        
        # Getting the type of 'rho' (line 220)
        rho_251246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 19), 'rho')
        # Assigning a type to the variable 'stypy_return_type' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'stypy_return_type', rho_251246)
        
        # ################# End of 'loss_function(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'loss_function' in the type store
        # Getting the type of 'stypy_return_type' (line 213)
        stypy_return_type_251247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251247)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'loss_function'
        return stypy_return_type_251247

    # Assigning a type to the variable 'loss_function' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'loss_function', loss_function)
    # SSA branch for the else part of an if statement (line 209)
    module_type_store.open_ssa_branch('else')

    @norecursion
    def loss_function(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 222)
        False_251248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 39), 'False')
        defaults = [False_251248]
        # Create a new context for function 'loss_function'
        module_type_store = module_type_store.open_function_context('loss_function', 222, 8, False)
        
        # Passed parameters checking function
        loss_function.stypy_localization = localization
        loss_function.stypy_type_of_self = None
        loss_function.stypy_type_store = module_type_store
        loss_function.stypy_function_name = 'loss_function'
        loss_function.stypy_param_names_list = ['f', 'cost_only']
        loss_function.stypy_varargs_param_name = None
        loss_function.stypy_kwargs_param_name = None
        loss_function.stypy_call_defaults = defaults
        loss_function.stypy_call_varargs = varargs
        loss_function.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'loss_function', ['f', 'cost_only'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'loss_function', localization, ['f', 'cost_only'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'loss_function(...)' code ##################

        
        # Assigning a BinOp to a Name (line 223):
        
        # Assigning a BinOp to a Name (line 223):
        # Getting the type of 'f' (line 223)
        f_251249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 17), 'f')
        # Getting the type of 'f_scale' (line 223)
        f_scale_251250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 21), 'f_scale')
        # Applying the binary operator 'div' (line 223)
        result_div_251251 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 17), 'div', f_251249, f_scale_251250)
        
        int_251252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 33), 'int')
        # Applying the binary operator '**' (line 223)
        result_pow_251253 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 16), '**', result_div_251251, int_251252)
        
        # Assigning a type to the variable 'z' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'z', result_pow_251253)
        
        # Assigning a Call to a Name (line 224):
        
        # Assigning a Call to a Name (line 224):
        
        # Call to loss(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'z' (line 224)
        z_251255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 23), 'z', False)
        # Processing the call keyword arguments (line 224)
        kwargs_251256 = {}
        # Getting the type of 'loss' (line 224)
        loss_251254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 18), 'loss', False)
        # Calling loss(args, kwargs) (line 224)
        loss_call_result_251257 = invoke(stypy.reporting.localization.Localization(__file__, 224, 18), loss_251254, *[z_251255], **kwargs_251256)
        
        # Assigning a type to the variable 'rho' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'rho', loss_call_result_251257)
        
        # Getting the type of 'cost_only' (line 225)
        cost_only_251258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 15), 'cost_only')
        # Testing the type of an if condition (line 225)
        if_condition_251259 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 225, 12), cost_only_251258)
        # Assigning a type to the variable 'if_condition_251259' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'if_condition_251259', if_condition_251259)
        # SSA begins for if statement (line 225)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        float_251260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 23), 'float')
        # Getting the type of 'f_scale' (line 226)
        f_scale_251261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 29), 'f_scale')
        int_251262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 40), 'int')
        # Applying the binary operator '**' (line 226)
        result_pow_251263 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 29), '**', f_scale_251261, int_251262)
        
        # Applying the binary operator '*' (line 226)
        result_mul_251264 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 23), '*', float_251260, result_pow_251263)
        
        
        # Call to sum(...): (line 226)
        # Processing the call arguments (line 226)
        
        # Obtaining the type of the subscript
        int_251267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 55), 'int')
        # Getting the type of 'rho' (line 226)
        rho_251268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 51), 'rho', False)
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___251269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 51), rho_251268, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_251270 = invoke(stypy.reporting.localization.Localization(__file__, 226, 51), getitem___251269, int_251267)
        
        # Processing the call keyword arguments (line 226)
        kwargs_251271 = {}
        # Getting the type of 'np' (line 226)
        np_251265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 44), 'np', False)
        # Obtaining the member 'sum' of a type (line 226)
        sum_251266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 44), np_251265, 'sum')
        # Calling sum(args, kwargs) (line 226)
        sum_call_result_251272 = invoke(stypy.reporting.localization.Localization(__file__, 226, 44), sum_251266, *[subscript_call_result_251270], **kwargs_251271)
        
        # Applying the binary operator '*' (line 226)
        result_mul_251273 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 42), '*', result_mul_251264, sum_call_result_251272)
        
        # Assigning a type to the variable 'stypy_return_type' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 16), 'stypy_return_type', result_mul_251273)
        # SSA join for if statement (line 225)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'rho' (line 227)
        rho_251274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'rho')
        
        # Obtaining the type of the subscript
        int_251275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 16), 'int')
        # Getting the type of 'rho' (line 227)
        rho_251276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'rho')
        # Obtaining the member '__getitem__' of a type (line 227)
        getitem___251277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 12), rho_251276, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 227)
        subscript_call_result_251278 = invoke(stypy.reporting.localization.Localization(__file__, 227, 12), getitem___251277, int_251275)
        
        # Getting the type of 'f_scale' (line 227)
        f_scale_251279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 22), 'f_scale')
        int_251280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 33), 'int')
        # Applying the binary operator '**' (line 227)
        result_pow_251281 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 22), '**', f_scale_251279, int_251280)
        
        # Applying the binary operator '*=' (line 227)
        result_imul_251282 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 12), '*=', subscript_call_result_251278, result_pow_251281)
        # Getting the type of 'rho' (line 227)
        rho_251283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'rho')
        int_251284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 16), 'int')
        # Storing an element on a container (line 227)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 12), rho_251283, (int_251284, result_imul_251282))
        
        
        # Getting the type of 'rho' (line 228)
        rho_251285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'rho')
        
        # Obtaining the type of the subscript
        int_251286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 16), 'int')
        # Getting the type of 'rho' (line 228)
        rho_251287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'rho')
        # Obtaining the member '__getitem__' of a type (line 228)
        getitem___251288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 12), rho_251287, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 228)
        subscript_call_result_251289 = invoke(stypy.reporting.localization.Localization(__file__, 228, 12), getitem___251288, int_251286)
        
        # Getting the type of 'f_scale' (line 228)
        f_scale_251290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 22), 'f_scale')
        int_251291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 33), 'int')
        # Applying the binary operator '**' (line 228)
        result_pow_251292 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 22), '**', f_scale_251290, int_251291)
        
        # Applying the binary operator 'div=' (line 228)
        result_div_251293 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 12), 'div=', subscript_call_result_251289, result_pow_251292)
        # Getting the type of 'rho' (line 228)
        rho_251294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'rho')
        int_251295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 16), 'int')
        # Storing an element on a container (line 228)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 12), rho_251294, (int_251295, result_div_251293))
        
        # Getting the type of 'rho' (line 229)
        rho_251296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 19), 'rho')
        # Assigning a type to the variable 'stypy_return_type' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'stypy_return_type', rho_251296)
        
        # ################# End of 'loss_function(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'loss_function' in the type store
        # Getting the type of 'stypy_return_type' (line 222)
        stypy_return_type_251297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251297)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'loss_function'
        return stypy_return_type_251297

    # Assigning a type to the variable 'loss_function' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'loss_function', loss_function)
    # SSA join for if statement (line 209)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'loss_function' (line 231)
    loss_function_251298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 11), 'loss_function')
    # Assigning a type to the variable 'stypy_return_type' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'stypy_return_type', loss_function_251298)
    
    # ################# End of 'construct_loss_function(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'construct_loss_function' in the type store
    # Getting the type of 'stypy_return_type' (line 205)
    stypy_return_type_251299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_251299)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'construct_loss_function'
    return stypy_return_type_251299

# Assigning a type to the variable 'construct_loss_function' (line 205)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 0), 'construct_loss_function', construct_loss_function)

@norecursion
def least_squares(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_251300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 21), 'str', '2-point')
    
    # Obtaining an instance of the builtin type 'tuple' (line 235)
    tuple_251301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 235)
    # Adding element type (line 235)
    
    # Getting the type of 'np' (line 235)
    np_251302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 41), 'np')
    # Obtaining the member 'inf' of a type (line 235)
    inf_251303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 41), np_251302, 'inf')
    # Applying the 'usub' unary operator (line 235)
    result___neg___251304 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 40), 'usub', inf_251303)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 40), tuple_251301, result___neg___251304)
    # Adding element type (line 235)
    # Getting the type of 'np' (line 235)
    np_251305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 49), 'np')
    # Obtaining the member 'inf' of a type (line 235)
    inf_251306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 49), np_251305, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 40), tuple_251301, inf_251306)
    
    str_251307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 65), 'str', 'trf')
    float_251308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 13), 'float')
    float_251309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 24), 'float')
    float_251310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 35), 'float')
    float_251311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 49), 'float')
    str_251312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 59), 'str', 'linear')
    float_251313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 16), 'float')
    # Getting the type of 'None' (line 237)
    None_251314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 31), 'None')
    # Getting the type of 'None' (line 237)
    None_251315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 47), 'None')
    
    # Obtaining an instance of the builtin type 'dict' (line 237)
    dict_251316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 64), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 237)
    
    # Getting the type of 'None' (line 238)
    None_251317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 21), 'None')
    # Getting the type of 'None' (line 238)
    None_251318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 36), 'None')
    int_251319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 50), 'int')
    
    # Obtaining an instance of the builtin type 'tuple' (line 238)
    tuple_251320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 58), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 238)
    
    
    # Obtaining an instance of the builtin type 'dict' (line 238)
    dict_251321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 69), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 238)
    
    defaults = [str_251300, tuple_251301, str_251307, float_251308, float_251309, float_251310, float_251311, str_251312, float_251313, None_251314, None_251315, dict_251316, None_251317, None_251318, int_251319, tuple_251320, dict_251321]
    # Create a new context for function 'least_squares'
    module_type_store = module_type_store.open_function_context('least_squares', 234, 0, False)
    
    # Passed parameters checking function
    least_squares.stypy_localization = localization
    least_squares.stypy_type_of_self = None
    least_squares.stypy_type_store = module_type_store
    least_squares.stypy_function_name = 'least_squares'
    least_squares.stypy_param_names_list = ['fun', 'x0', 'jac', 'bounds', 'method', 'ftol', 'xtol', 'gtol', 'x_scale', 'loss', 'f_scale', 'diff_step', 'tr_solver', 'tr_options', 'jac_sparsity', 'max_nfev', 'verbose', 'args', 'kwargs']
    least_squares.stypy_varargs_param_name = None
    least_squares.stypy_kwargs_param_name = None
    least_squares.stypy_call_defaults = defaults
    least_squares.stypy_call_varargs = varargs
    least_squares.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'least_squares', ['fun', 'x0', 'jac', 'bounds', 'method', 'ftol', 'xtol', 'gtol', 'x_scale', 'loss', 'f_scale', 'diff_step', 'tr_solver', 'tr_options', 'jac_sparsity', 'max_nfev', 'verbose', 'args', 'kwargs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'least_squares', localization, ['fun', 'x0', 'jac', 'bounds', 'method', 'ftol', 'xtol', 'gtol', 'x_scale', 'loss', 'f_scale', 'diff_step', 'tr_solver', 'tr_options', 'jac_sparsity', 'max_nfev', 'verbose', 'args', 'kwargs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'least_squares(...)' code ##################

    str_251322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, (-1)), 'str', 'Solve a nonlinear least-squares problem with bounds on the variables.\n\n    Given the residuals f(x) (an m-dimensional real function of n real\n    variables) and the loss function rho(s) (a scalar function), `least_squares`\n    finds a local minimum of the cost function F(x)::\n\n        minimize F(x) = 0.5 * sum(rho(f_i(x)**2), i = 0, ..., m - 1)\n        subject to lb <= x <= ub\n\n    The purpose of the loss function rho(s) is to reduce the influence of\n    outliers on the solution.\n\n    Parameters\n    ----------\n    fun : callable\n        Function which computes the vector of residuals, with the signature\n        ``fun(x, *args, **kwargs)``, i.e., the minimization proceeds with\n        respect to its first argument. The argument ``x`` passed to this\n        function is an ndarray of shape (n,) (never a scalar, even for n=1).\n        It must return a 1-d array_like of shape (m,) or a scalar. If the\n        argument ``x`` is complex or the function ``fun`` returns complex\n        residuals, it must be wrapped in a real function of real arguments,\n        as shown at the end of the Examples section.\n    x0 : array_like with shape (n,) or float\n        Initial guess on independent variables. If float, it will be treated\n        as a 1-d array with one element.\n    jac : {\'2-point\', \'3-point\', \'cs\', callable}, optional\n        Method of computing the Jacobian matrix (an m-by-n matrix, where\n        element (i, j) is the partial derivative of f[i] with respect to\n        x[j]). The keywords select a finite difference scheme for numerical\n        estimation. The scheme \'3-point\' is more accurate, but requires\n        twice as much operations compared to \'2-point\' (default). The\n        scheme \'cs\' uses complex steps, and while potentially the most\n        accurate, it is applicable only when `fun` correctly handles\n        complex inputs and can be analytically continued to the complex\n        plane. Method \'lm\' always uses the \'2-point\' scheme. If callable,\n        it is used as ``jac(x, *args, **kwargs)`` and should return a\n        good approximation (or the exact value) for the Jacobian as an\n        array_like (np.atleast_2d is applied), a sparse matrix or a\n        `scipy.sparse.linalg.LinearOperator`.\n    bounds : 2-tuple of array_like, optional\n        Lower and upper bounds on independent variables. Defaults to no bounds.\n        Each array must match the size of `x0` or be a scalar, in the latter\n        case a bound will be the same for all variables. Use ``np.inf`` with\n        an appropriate sign to disable bounds on all or some variables.\n    method : {\'trf\', \'dogbox\', \'lm\'}, optional\n        Algorithm to perform minimization.\n\n            * \'trf\' : Trust Region Reflective algorithm, particularly suitable\n              for large sparse problems with bounds. Generally robust method.\n            * \'dogbox\' : dogleg algorithm with rectangular trust regions,\n              typical use case is small problems with bounds. Not recommended\n              for problems with rank-deficient Jacobian.\n            * \'lm\' : Levenberg-Marquardt algorithm as implemented in MINPACK.\n              Doesn\'t handle bounds and sparse Jacobians. Usually the most\n              efficient method for small unconstrained problems.\n\n        Default is \'trf\'. See Notes for more information.\n    ftol : float, optional\n        Tolerance for termination by the change of the cost function. Default\n        is 1e-8. The optimization process is stopped when  ``dF < ftol * F``,\n        and there was an adequate agreement between a local quadratic model and\n        the true model in the last step.\n    xtol : float, optional\n        Tolerance for termination by the change of the independent variables.\n        Default is 1e-8. The exact condition depends on the `method` used:\n\n            * For \'trf\' and \'dogbox\' : ``norm(dx) < xtol * (xtol + norm(x))``\n            * For \'lm\' : ``Delta < xtol * norm(xs)``, where ``Delta`` is\n              a trust-region radius and ``xs`` is the value of ``x``\n              scaled according to `x_scale` parameter (see below).\n\n    gtol : float, optional\n        Tolerance for termination by the norm of the gradient. Default is 1e-8.\n        The exact condition depends on a `method` used:\n\n            * For \'trf\' : ``norm(g_scaled, ord=np.inf) < gtol``, where\n              ``g_scaled`` is the value of the gradient scaled to account for\n              the presence of the bounds [STIR]_.\n            * For \'dogbox\' : ``norm(g_free, ord=np.inf) < gtol``, where\n              ``g_free`` is the gradient with respect to the variables which\n              are not in the optimal state on the boundary.\n            * For \'lm\' : the maximum absolute value of the cosine of angles\n              between columns of the Jacobian and the residual vector is less\n              than `gtol`, or the residual vector is zero.\n\n    x_scale : array_like or \'jac\', optional\n        Characteristic scale of each variable. Setting `x_scale` is equivalent\n        to reformulating the problem in scaled variables ``xs = x / x_scale``.\n        An alternative view is that the size of a trust region along j-th\n        dimension is proportional to ``x_scale[j]``. Improved convergence may\n        be achieved by setting `x_scale` such that a step of a given size\n        along any of the scaled variables has a similar effect on the cost\n        function. If set to \'jac\', the scale is iteratively updated using the\n        inverse norms of the columns of the Jacobian matrix (as described in\n        [JJMore]_).\n    loss : str or callable, optional\n        Determines the loss function. The following keyword values are allowed:\n\n            * \'linear\' (default) : ``rho(z) = z``. Gives a standard\n              least-squares problem.\n            * \'soft_l1\' : ``rho(z) = 2 * ((1 + z)**0.5 - 1)``. The smooth\n              approximation of l1 (absolute value) loss. Usually a good\n              choice for robust least squares.\n            * \'huber\' : ``rho(z) = z if z <= 1 else 2*z**0.5 - 1``. Works\n              similarly to \'soft_l1\'.\n            * \'cauchy\' : ``rho(z) = ln(1 + z)``. Severely weakens outliers\n              influence, but may cause difficulties in optimization process.\n            * \'arctan\' : ``rho(z) = arctan(z)``. Limits a maximum loss on\n              a single residual, has properties similar to \'cauchy\'.\n\n        If callable, it must take a 1-d ndarray ``z=f**2`` and return an\n        array_like with shape (3, m) where row 0 contains function values,\n        row 1 contains first derivatives and row 2 contains second\n        derivatives. Method \'lm\' supports only \'linear\' loss.\n    f_scale : float, optional\n        Value of soft margin between inlier and outlier residuals, default\n        is 1.0. The loss function is evaluated as follows\n        ``rho_(f**2) = C**2 * rho(f**2 / C**2)``, where ``C`` is `f_scale`,\n        and ``rho`` is determined by `loss` parameter. This parameter has\n        no effect with ``loss=\'linear\'``, but for other `loss` values it is\n        of crucial importance.\n    max_nfev : None or int, optional\n        Maximum number of function evaluations before the termination.\n        If None (default), the value is chosen automatically:\n\n            * For \'trf\' and \'dogbox\' : 100 * n.\n            * For \'lm\' :  100 * n if `jac` is callable and 100 * n * (n + 1)\n              otherwise (because \'lm\' counts function calls in Jacobian\n              estimation).\n\n    diff_step : None or array_like, optional\n        Determines the relative step size for the finite difference\n        approximation of the Jacobian. The actual step is computed as\n        ``x * diff_step``. If None (default), then `diff_step` is taken to be\n        a conventional "optimal" power of machine epsilon for the finite\n        difference scheme used [NR]_.\n    tr_solver : {None, \'exact\', \'lsmr\'}, optional\n        Method for solving trust-region subproblems, relevant only for \'trf\'\n        and \'dogbox\' methods.\n\n            * \'exact\' is suitable for not very large problems with dense\n              Jacobian matrices. The computational complexity per iteration is\n              comparable to a singular value decomposition of the Jacobian\n              matrix.\n            * \'lsmr\' is suitable for problems with sparse and large Jacobian\n              matrices. It uses the iterative procedure\n              `scipy.sparse.linalg.lsmr` for finding a solution of a linear\n              least-squares problem and only requires matrix-vector product\n              evaluations.\n\n        If None (default) the solver is chosen based on the type of Jacobian\n        returned on the first iteration.\n    tr_options : dict, optional\n        Keyword options passed to trust-region solver.\n\n            * ``tr_solver=\'exact\'``: `tr_options` are ignored.\n            * ``tr_solver=\'lsmr\'``: options for `scipy.sparse.linalg.lsmr`.\n              Additionally  ``method=\'trf\'`` supports  \'regularize\' option\n              (bool, default is True) which adds a regularization term to the\n              normal equation, which improves convergence if the Jacobian is\n              rank-deficient [Byrd]_ (eq. 3.4).\n\n    jac_sparsity : {None, array_like, sparse matrix}, optional\n        Defines the sparsity structure of the Jacobian matrix for finite\n        difference estimation, its shape must be (m, n). If the Jacobian has\n        only few non-zero elements in *each* row, providing the sparsity\n        structure will greatly speed up the computations [Curtis]_. A zero\n        entry means that a corresponding element in the Jacobian is identically\n        zero. If provided, forces the use of \'lsmr\' trust-region solver.\n        If None (default) then dense differencing will be used. Has no effect\n        for \'lm\' method.\n    verbose : {0, 1, 2}, optional\n        Level of algorithm\'s verbosity:\n\n            * 0 (default) : work silently.\n            * 1 : display a termination report.\n            * 2 : display progress during iterations (not supported by \'lm\'\n              method).\n\n    args, kwargs : tuple and dict, optional\n        Additional arguments passed to `fun` and `jac`. Both empty by default.\n        The calling signature is ``fun(x, *args, **kwargs)`` and the same for\n        `jac`.\n\n    Returns\n    -------\n    `OptimizeResult` with the following fields defined:\n    x : ndarray, shape (n,)\n        Solution found.\n    cost : float\n        Value of the cost function at the solution.\n    fun : ndarray, shape (m,)\n        Vector of residuals at the solution.\n    jac : ndarray, sparse matrix or LinearOperator, shape (m, n)\n        Modified Jacobian matrix at the solution, in the sense that J^T J\n        is a Gauss-Newton approximation of the Hessian of the cost function.\n        The type is the same as the one used by the algorithm.\n    grad : ndarray, shape (m,)\n        Gradient of the cost function at the solution.\n    optimality : float\n        First-order optimality measure. In unconstrained problems, it is always\n        the uniform norm of the gradient. In constrained problems, it is the\n        quantity which was compared with `gtol` during iterations.\n    active_mask : ndarray of int, shape (n,)\n        Each component shows whether a corresponding constraint is active\n        (that is, whether a variable is at the bound):\n\n            *  0 : a constraint is not active.\n            * -1 : a lower bound is active.\n            *  1 : an upper bound is active.\n\n        Might be somewhat arbitrary for \'trf\' method as it generates a sequence\n        of strictly feasible iterates and `active_mask` is determined within a\n        tolerance threshold.\n    nfev : int\n        Number of function evaluations done. Methods \'trf\' and \'dogbox\' do not\n        count function calls for numerical Jacobian approximation, as opposed\n        to \'lm\' method.\n    njev : int or None\n        Number of Jacobian evaluations done. If numerical Jacobian\n        approximation is used in \'lm\' method, it is set to None.\n    status : int\n        The reason for algorithm termination:\n\n            * -1 : improper input parameters status returned from MINPACK.\n            *  0 : the maximum number of function evaluations is exceeded.\n            *  1 : `gtol` termination condition is satisfied.\n            *  2 : `ftol` termination condition is satisfied.\n            *  3 : `xtol` termination condition is satisfied.\n            *  4 : Both `ftol` and `xtol` termination conditions are satisfied.\n\n    message : str\n        Verbal description of the termination reason.\n    success : bool\n        True if one of the convergence criteria is satisfied (`status` > 0).\n\n    See Also\n    --------\n    leastsq : A legacy wrapper for the MINPACK implementation of the\n              Levenberg-Marquadt algorithm.\n    curve_fit : Least-squares minimization applied to a curve fitting problem.\n\n    Notes\n    -----\n    Method \'lm\' (Levenberg-Marquardt) calls a wrapper over least-squares\n    algorithms implemented in MINPACK (lmder, lmdif). It runs the\n    Levenberg-Marquardt algorithm formulated as a trust-region type algorithm.\n    The implementation is based on paper [JJMore]_, it is very robust and\n    efficient with a lot of smart tricks. It should be your first choice\n    for unconstrained problems. Note that it doesn\'t support bounds. Also\n    it doesn\'t work when m < n.\n\n    Method \'trf\' (Trust Region Reflective) is motivated by the process of\n    solving a system of equations, which constitute the first-order optimality\n    condition for a bound-constrained minimization problem as formulated in\n    [STIR]_. The algorithm iteratively solves trust-region subproblems\n    augmented by a special diagonal quadratic term and with trust-region shape\n    determined by the distance from the bounds and the direction of the\n    gradient. This enhancements help to avoid making steps directly into bounds\n    and efficiently explore the whole space of variables. To further improve\n    convergence, the algorithm considers search directions reflected from the\n    bounds. To obey theoretical requirements, the algorithm keeps iterates\n    strictly feasible. With dense Jacobians trust-region subproblems are\n    solved by an exact method very similar to the one described in [JJMore]_\n    (and implemented in MINPACK). The difference from the MINPACK\n    implementation is that a singular value decomposition of a Jacobian\n    matrix is done once per iteration, instead of a QR decomposition and series\n    of Givens rotation eliminations. For large sparse Jacobians a 2-d subspace\n    approach of solving trust-region subproblems is used [STIR]_, [Byrd]_.\n    The subspace is spanned by a scaled gradient and an approximate\n    Gauss-Newton solution delivered by `scipy.sparse.linalg.lsmr`. When no\n    constraints are imposed the algorithm is very similar to MINPACK and has\n    generally comparable performance. The algorithm works quite robust in\n    unbounded and bounded problems, thus it is chosen as a default algorithm.\n\n    Method \'dogbox\' operates in a trust-region framework, but considers\n    rectangular trust regions as opposed to conventional ellipsoids [Voglis]_.\n    The intersection of a current trust region and initial bounds is again\n    rectangular, so on each iteration a quadratic minimization problem subject\n    to bound constraints is solved approximately by Powell\'s dogleg method\n    [NumOpt]_. The required Gauss-Newton step can be computed exactly for\n    dense Jacobians or approximately by `scipy.sparse.linalg.lsmr` for large\n    sparse Jacobians. The algorithm is likely to exhibit slow convergence when\n    the rank of Jacobian is less than the number of variables. The algorithm\n    often outperforms \'trf\' in bounded problems with a small number of\n    variables.\n\n    Robust loss functions are implemented as described in [BA]_. The idea\n    is to modify a residual vector and a Jacobian matrix on each iteration\n    such that computed gradient and Gauss-Newton Hessian approximation match\n    the true gradient and Hessian approximation of the cost function. Then\n    the algorithm proceeds in a normal way, i.e. robust loss functions are\n    implemented as a simple wrapper over standard least-squares algorithms.\n\n    .. versionadded:: 0.17.0\n\n    References\n    ----------\n    .. [STIR] M. A. Branch, T. F. Coleman, and Y. Li, "A Subspace, Interior,\n              and Conjugate Gradient Method for Large-Scale Bound-Constrained\n              Minimization Problems," SIAM Journal on Scientific Computing,\n              Vol. 21, Number 1, pp 1-23, 1999.\n    .. [NR] William H. Press et. al., "Numerical Recipes. The Art of Scientific\n            Computing. 3rd edition", Sec. 5.7.\n    .. [Byrd] R. H. Byrd, R. B. Schnabel and G. A. Shultz, "Approximate\n              solution of the trust region problem by minimization over\n              two-dimensional subspaces", Math. Programming, 40, pp. 247-263,\n              1988.\n    .. [Curtis] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of\n                sparse Jacobian matrices", Journal of the Institute of\n                Mathematics and its Applications, 13, pp. 117-120, 1974.\n    .. [JJMore] J. J. More, "The Levenberg-Marquardt Algorithm: Implementation\n                and Theory," Numerical Analysis, ed. G. A. Watson, Lecture\n                Notes in Mathematics 630, Springer Verlag, pp. 105-116, 1977.\n    .. [Voglis] C. Voglis and I. E. Lagaris, "A Rectangular Trust Region\n                Dogleg Approach for Unconstrained and Bound Constrained\n                Nonlinear Optimization", WSEAS International Conference on\n                Applied Mathematics, Corfu, Greece, 2004.\n    .. [NumOpt] J. Nocedal and S. J. Wright, "Numerical optimization,\n                2nd edition", Chapter 4.\n    .. [BA] B. Triggs et. al., "Bundle Adjustment - A Modern Synthesis",\n            Proceedings of the International Workshop on Vision Algorithms:\n            Theory and Practice, pp. 298-372, 1999.\n\n    Examples\n    --------\n    In this example we find a minimum of the Rosenbrock function without bounds\n    on independed variables.\n\n    >>> def fun_rosenbrock(x):\n    ...     return np.array([10 * (x[1] - x[0]**2), (1 - x[0])])\n\n    Notice that we only provide the vector of the residuals. The algorithm\n    constructs the cost function as a sum of squares of the residuals, which\n    gives the Rosenbrock function. The exact minimum is at ``x = [1.0, 1.0]``.\n\n    >>> from scipy.optimize import least_squares\n    >>> x0_rosenbrock = np.array([2, 2])\n    >>> res_1 = least_squares(fun_rosenbrock, x0_rosenbrock)\n    >>> res_1.x\n    array([ 1.,  1.])\n    >>> res_1.cost\n    9.8669242910846867e-30\n    >>> res_1.optimality\n    8.8928864934219529e-14\n\n    We now constrain the variables, in such a way that the previous solution\n    becomes infeasible. Specifically, we require that ``x[1] >= 1.5``, and\n    ``x[0]`` left unconstrained. To this end, we specify the `bounds` parameter\n    to `least_squares` in the form ``bounds=([-np.inf, 1.5], np.inf)``.\n\n    We also provide the analytic Jacobian:\n\n    >>> def jac_rosenbrock(x):\n    ...     return np.array([\n    ...         [-20 * x[0], 10],\n    ...         [-1, 0]])\n\n    Putting this all together, we see that the new solution lies on the bound:\n\n    >>> res_2 = least_squares(fun_rosenbrock, x0_rosenbrock, jac_rosenbrock,\n    ...                       bounds=([-np.inf, 1.5], np.inf))\n    >>> res_2.x\n    array([ 1.22437075,  1.5       ])\n    >>> res_2.cost\n    0.025213093946805685\n    >>> res_2.optimality\n    1.5885401433157753e-07\n\n    Now we solve a system of equations (i.e., the cost function should be zero\n    at a minimum) for a Broyden tridiagonal vector-valued function of 100000\n    variables:\n\n    >>> def fun_broyden(x):\n    ...     f = (3 - x) * x + 1\n    ...     f[1:] -= x[:-1]\n    ...     f[:-1] -= 2 * x[1:]\n    ...     return f\n\n    The corresponding Jacobian matrix is sparse. We tell the algorithm to\n    estimate it by finite differences and provide the sparsity structure of\n    Jacobian to significantly speed up this process.\n\n    >>> from scipy.sparse import lil_matrix\n    >>> def sparsity_broyden(n):\n    ...     sparsity = lil_matrix((n, n), dtype=int)\n    ...     i = np.arange(n)\n    ...     sparsity[i, i] = 1\n    ...     i = np.arange(1, n)\n    ...     sparsity[i, i - 1] = 1\n    ...     i = np.arange(n - 1)\n    ...     sparsity[i, i + 1] = 1\n    ...     return sparsity\n    ...\n    >>> n = 100000\n    >>> x0_broyden = -np.ones(n)\n    ...\n    >>> res_3 = least_squares(fun_broyden, x0_broyden,\n    ...                       jac_sparsity=sparsity_broyden(n))\n    >>> res_3.cost\n    4.5687069299604613e-23\n    >>> res_3.optimality\n    1.1650454296851518e-11\n\n    Let\'s also solve a curve fitting problem using robust loss function to\n    take care of outliers in the data. Define the model function as\n    ``y = a + b * exp(c * t)``, where t is a predictor variable, y is an\n    observation and a, b, c are parameters to estimate.\n\n    First, define the function which generates the data with noise and\n    outliers, define the model parameters, and generate data:\n\n    >>> def gen_data(t, a, b, c, noise=0, n_outliers=0, random_state=0):\n    ...     y = a + b * np.exp(t * c)\n    ...\n    ...     rnd = np.random.RandomState(random_state)\n    ...     error = noise * rnd.randn(t.size)\n    ...     outliers = rnd.randint(0, t.size, n_outliers)\n    ...     error[outliers] *= 10\n    ...\n    ...     return y + error\n    ...\n    >>> a = 0.5\n    >>> b = 2.0\n    >>> c = -1\n    >>> t_min = 0\n    >>> t_max = 10\n    >>> n_points = 15\n    ...\n    >>> t_train = np.linspace(t_min, t_max, n_points)\n    >>> y_train = gen_data(t_train, a, b, c, noise=0.1, n_outliers=3)\n\n    Define function for computing residuals and initial estimate of\n    parameters.\n\n    >>> def fun(x, t, y):\n    ...     return x[0] + x[1] * np.exp(x[2] * t) - y\n    ...\n    >>> x0 = np.array([1.0, 1.0, 0.0])\n\n    Compute a standard least-squares solution:\n\n    >>> res_lsq = least_squares(fun, x0, args=(t_train, y_train))\n\n    Now compute two solutions with two different robust loss functions. The\n    parameter `f_scale` is set to 0.1, meaning that inlier residuals should\n    not significantly exceed 0.1 (the noise level used).\n\n    >>> res_soft_l1 = least_squares(fun, x0, loss=\'soft_l1\', f_scale=0.1,\n    ...                             args=(t_train, y_train))\n    >>> res_log = least_squares(fun, x0, loss=\'cauchy\', f_scale=0.1,\n    ...                         args=(t_train, y_train))\n\n    And finally plot all the curves. We see that by selecting an appropriate\n    `loss`  we can get estimates close to optimal even in the presence of\n    strong outliers. But keep in mind that generally it is recommended to try\n    \'soft_l1\' or \'huber\' losses first (if at all necessary) as the other two\n    options may cause difficulties in optimization process.\n\n    >>> t_test = np.linspace(t_min, t_max, n_points * 10)\n    >>> y_true = gen_data(t_test, a, b, c)\n    >>> y_lsq = gen_data(t_test, *res_lsq.x)\n    >>> y_soft_l1 = gen_data(t_test, *res_soft_l1.x)\n    >>> y_log = gen_data(t_test, *res_log.x)\n    ...\n    >>> import matplotlib.pyplot as plt\n    >>> plt.plot(t_train, y_train, \'o\')\n    >>> plt.plot(t_test, y_true, \'k\', linewidth=2, label=\'true\')\n    >>> plt.plot(t_test, y_lsq, label=\'linear loss\')\n    >>> plt.plot(t_test, y_soft_l1, label=\'soft_l1 loss\')\n    >>> plt.plot(t_test, y_log, label=\'cauchy loss\')\n    >>> plt.xlabel("t")\n    >>> plt.ylabel("y")\n    >>> plt.legend()\n    >>> plt.show()\n\n    In the next example, we show how complex-valued residual functions of\n    complex variables can be optimized with ``least_squares()``. Consider the\n    following function:\n\n    >>> def f(z):\n    ...     return z - (0.5 + 0.5j)\n\n    We wrap it into a function of real variables that returns real residuals\n    by simply handling the real and imaginary parts as independent variables:\n\n    >>> def f_wrap(x):\n    ...     fx = f(x[0] + 1j*x[1])\n    ...     return np.array([fx.real, fx.imag])\n\n    Thus, instead of the original m-dimensional complex function of n complex\n    variables we optimize a 2m-dimensional real function of 2n real variables:\n\n    >>> from scipy.optimize import least_squares\n    >>> res_wrapped = least_squares(f_wrap, (0.1, 0.1), bounds=([0, 0], [1, 1]))\n    >>> z = res_wrapped.x[0] + res_wrapped.x[1]*1j\n    >>> z\n    (0.49999999999925893+0.49999999999925893j)\n\n    ')
    
    
    # Getting the type of 'method' (line 740)
    method_251323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 7), 'method')
    
    # Obtaining an instance of the builtin type 'list' (line 740)
    list_251324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 740)
    # Adding element type (line 740)
    str_251325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 22), 'str', 'trf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 740, 21), list_251324, str_251325)
    # Adding element type (line 740)
    str_251326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 29), 'str', 'dogbox')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 740, 21), list_251324, str_251326)
    # Adding element type (line 740)
    str_251327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 39), 'str', 'lm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 740, 21), list_251324, str_251327)
    
    # Applying the binary operator 'notin' (line 740)
    result_contains_251328 = python_operator(stypy.reporting.localization.Localization(__file__, 740, 7), 'notin', method_251323, list_251324)
    
    # Testing the type of an if condition (line 740)
    if_condition_251329 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 740, 4), result_contains_251328)
    # Assigning a type to the variable 'if_condition_251329' (line 740)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 740, 4), 'if_condition_251329', if_condition_251329)
    # SSA begins for if statement (line 740)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 741)
    # Processing the call arguments (line 741)
    str_251331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 25), 'str', "`method` must be 'trf', 'dogbox' or 'lm'.")
    # Processing the call keyword arguments (line 741)
    kwargs_251332 = {}
    # Getting the type of 'ValueError' (line 741)
    ValueError_251330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 741)
    ValueError_call_result_251333 = invoke(stypy.reporting.localization.Localization(__file__, 741, 14), ValueError_251330, *[str_251331], **kwargs_251332)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 741, 8), ValueError_call_result_251333, 'raise parameter', BaseException)
    # SSA join for if statement (line 740)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'jac' (line 743)
    jac_251334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 7), 'jac')
    
    # Obtaining an instance of the builtin type 'list' (line 743)
    list_251335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 743)
    # Adding element type (line 743)
    str_251336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 19), 'str', '2-point')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 743, 18), list_251335, str_251336)
    # Adding element type (line 743)
    str_251337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 30), 'str', '3-point')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 743, 18), list_251335, str_251337)
    # Adding element type (line 743)
    str_251338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 41), 'str', 'cs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 743, 18), list_251335, str_251338)
    
    # Applying the binary operator 'notin' (line 743)
    result_contains_251339 = python_operator(stypy.reporting.localization.Localization(__file__, 743, 7), 'notin', jac_251334, list_251335)
    
    
    
    # Call to callable(...): (line 743)
    # Processing the call arguments (line 743)
    # Getting the type of 'jac' (line 743)
    jac_251341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 64), 'jac', False)
    # Processing the call keyword arguments (line 743)
    kwargs_251342 = {}
    # Getting the type of 'callable' (line 743)
    callable_251340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 55), 'callable', False)
    # Calling callable(args, kwargs) (line 743)
    callable_call_result_251343 = invoke(stypy.reporting.localization.Localization(__file__, 743, 55), callable_251340, *[jac_251341], **kwargs_251342)
    
    # Applying the 'not' unary operator (line 743)
    result_not__251344 = python_operator(stypy.reporting.localization.Localization(__file__, 743, 51), 'not', callable_call_result_251343)
    
    # Applying the binary operator 'and' (line 743)
    result_and_keyword_251345 = python_operator(stypy.reporting.localization.Localization(__file__, 743, 7), 'and', result_contains_251339, result_not__251344)
    
    # Testing the type of an if condition (line 743)
    if_condition_251346 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 743, 4), result_and_keyword_251345)
    # Assigning a type to the variable 'if_condition_251346' (line 743)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 743, 4), 'if_condition_251346', if_condition_251346)
    # SSA begins for if statement (line 743)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 744)
    # Processing the call arguments (line 744)
    str_251348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 25), 'str', "`jac` must be '2-point', '3-point', 'cs' or callable.")
    # Processing the call keyword arguments (line 744)
    kwargs_251349 = {}
    # Getting the type of 'ValueError' (line 744)
    ValueError_251347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 744)
    ValueError_call_result_251350 = invoke(stypy.reporting.localization.Localization(__file__, 744, 14), ValueError_251347, *[str_251348], **kwargs_251349)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 744, 8), ValueError_call_result_251350, 'raise parameter', BaseException)
    # SSA join for if statement (line 743)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'tr_solver' (line 747)
    tr_solver_251351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 7), 'tr_solver')
    
    # Obtaining an instance of the builtin type 'list' (line 747)
    list_251352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 747)
    # Adding element type (line 747)
    # Getting the type of 'None' (line 747)
    None_251353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 25), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 747, 24), list_251352, None_251353)
    # Adding element type (line 747)
    str_251354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 31), 'str', 'exact')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 747, 24), list_251352, str_251354)
    # Adding element type (line 747)
    str_251355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 40), 'str', 'lsmr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 747, 24), list_251352, str_251355)
    
    # Applying the binary operator 'notin' (line 747)
    result_contains_251356 = python_operator(stypy.reporting.localization.Localization(__file__, 747, 7), 'notin', tr_solver_251351, list_251352)
    
    # Testing the type of an if condition (line 747)
    if_condition_251357 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 747, 4), result_contains_251356)
    # Assigning a type to the variable 'if_condition_251357' (line 747)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 4), 'if_condition_251357', if_condition_251357)
    # SSA begins for if statement (line 747)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 748)
    # Processing the call arguments (line 748)
    str_251359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 25), 'str', "`tr_solver` must be None, 'exact' or 'lsmr'.")
    # Processing the call keyword arguments (line 748)
    kwargs_251360 = {}
    # Getting the type of 'ValueError' (line 748)
    ValueError_251358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 748)
    ValueError_call_result_251361 = invoke(stypy.reporting.localization.Localization(__file__, 748, 14), ValueError_251358, *[str_251359], **kwargs_251360)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 748, 8), ValueError_call_result_251361, 'raise parameter', BaseException)
    # SSA join for if statement (line 747)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'loss' (line 750)
    loss_251362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 7), 'loss')
    # Getting the type of 'IMPLEMENTED_LOSSES' (line 750)
    IMPLEMENTED_LOSSES_251363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 19), 'IMPLEMENTED_LOSSES')
    # Applying the binary operator 'notin' (line 750)
    result_contains_251364 = python_operator(stypy.reporting.localization.Localization(__file__, 750, 7), 'notin', loss_251362, IMPLEMENTED_LOSSES_251363)
    
    
    
    # Call to callable(...): (line 750)
    # Processing the call arguments (line 750)
    # Getting the type of 'loss' (line 750)
    loss_251366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 55), 'loss', False)
    # Processing the call keyword arguments (line 750)
    kwargs_251367 = {}
    # Getting the type of 'callable' (line 750)
    callable_251365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 46), 'callable', False)
    # Calling callable(args, kwargs) (line 750)
    callable_call_result_251368 = invoke(stypy.reporting.localization.Localization(__file__, 750, 46), callable_251365, *[loss_251366], **kwargs_251367)
    
    # Applying the 'not' unary operator (line 750)
    result_not__251369 = python_operator(stypy.reporting.localization.Localization(__file__, 750, 42), 'not', callable_call_result_251368)
    
    # Applying the binary operator 'and' (line 750)
    result_and_keyword_251370 = python_operator(stypy.reporting.localization.Localization(__file__, 750, 7), 'and', result_contains_251364, result_not__251369)
    
    # Testing the type of an if condition (line 750)
    if_condition_251371 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 750, 4), result_and_keyword_251370)
    # Assigning a type to the variable 'if_condition_251371' (line 750)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 4), 'if_condition_251371', if_condition_251371)
    # SSA begins for if statement (line 750)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 751)
    # Processing the call arguments (line 751)
    
    # Call to format(...): (line 751)
    # Processing the call arguments (line 751)
    
    # Call to keys(...): (line 752)
    # Processing the call keyword arguments (line 752)
    kwargs_251377 = {}
    # Getting the type of 'IMPLEMENTED_LOSSES' (line 752)
    IMPLEMENTED_LOSSES_251375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 33), 'IMPLEMENTED_LOSSES', False)
    # Obtaining the member 'keys' of a type (line 752)
    keys_251376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 752, 33), IMPLEMENTED_LOSSES_251375, 'keys')
    # Calling keys(args, kwargs) (line 752)
    keys_call_result_251378 = invoke(stypy.reporting.localization.Localization(__file__, 752, 33), keys_251376, *[], **kwargs_251377)
    
    # Processing the call keyword arguments (line 751)
    kwargs_251379 = {}
    str_251373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 25), 'str', '`loss` must be one of {0} or a callable.')
    # Obtaining the member 'format' of a type (line 751)
    format_251374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 25), str_251373, 'format')
    # Calling format(args, kwargs) (line 751)
    format_call_result_251380 = invoke(stypy.reporting.localization.Localization(__file__, 751, 25), format_251374, *[keys_call_result_251378], **kwargs_251379)
    
    # Processing the call keyword arguments (line 751)
    kwargs_251381 = {}
    # Getting the type of 'ValueError' (line 751)
    ValueError_251372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 751)
    ValueError_call_result_251382 = invoke(stypy.reporting.localization.Localization(__file__, 751, 14), ValueError_251372, *[format_call_result_251380], **kwargs_251381)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 751, 8), ValueError_call_result_251382, 'raise parameter', BaseException)
    # SSA join for if statement (line 750)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'method' (line 754)
    method_251383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 7), 'method')
    str_251384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 17), 'str', 'lm')
    # Applying the binary operator '==' (line 754)
    result_eq_251385 = python_operator(stypy.reporting.localization.Localization(__file__, 754, 7), '==', method_251383, str_251384)
    
    
    # Getting the type of 'loss' (line 754)
    loss_251386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 26), 'loss')
    str_251387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 34), 'str', 'linear')
    # Applying the binary operator '!=' (line 754)
    result_ne_251388 = python_operator(stypy.reporting.localization.Localization(__file__, 754, 26), '!=', loss_251386, str_251387)
    
    # Applying the binary operator 'and' (line 754)
    result_and_keyword_251389 = python_operator(stypy.reporting.localization.Localization(__file__, 754, 7), 'and', result_eq_251385, result_ne_251388)
    
    # Testing the type of an if condition (line 754)
    if_condition_251390 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 754, 4), result_and_keyword_251389)
    # Assigning a type to the variable 'if_condition_251390' (line 754)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 754, 4), 'if_condition_251390', if_condition_251390)
    # SSA begins for if statement (line 754)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 755)
    # Processing the call arguments (line 755)
    str_251392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 25), 'str', "method='lm' supports only 'linear' loss function.")
    # Processing the call keyword arguments (line 755)
    kwargs_251393 = {}
    # Getting the type of 'ValueError' (line 755)
    ValueError_251391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 755)
    ValueError_call_result_251394 = invoke(stypy.reporting.localization.Localization(__file__, 755, 14), ValueError_251391, *[str_251392], **kwargs_251393)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 755, 8), ValueError_call_result_251394, 'raise parameter', BaseException)
    # SSA join for if statement (line 754)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'verbose' (line 757)
    verbose_251395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 7), 'verbose')
    
    # Obtaining an instance of the builtin type 'list' (line 757)
    list_251396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 757)
    # Adding element type (line 757)
    int_251397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 757, 22), list_251396, int_251397)
    # Adding element type (line 757)
    int_251398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 757, 22), list_251396, int_251398)
    # Adding element type (line 757)
    int_251399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 757, 22), list_251396, int_251399)
    
    # Applying the binary operator 'notin' (line 757)
    result_contains_251400 = python_operator(stypy.reporting.localization.Localization(__file__, 757, 7), 'notin', verbose_251395, list_251396)
    
    # Testing the type of an if condition (line 757)
    if_condition_251401 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 757, 4), result_contains_251400)
    # Assigning a type to the variable 'if_condition_251401' (line 757)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 757, 4), 'if_condition_251401', if_condition_251401)
    # SSA begins for if statement (line 757)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 758)
    # Processing the call arguments (line 758)
    str_251403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 25), 'str', '`verbose` must be in [0, 1, 2].')
    # Processing the call keyword arguments (line 758)
    kwargs_251404 = {}
    # Getting the type of 'ValueError' (line 758)
    ValueError_251402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 758)
    ValueError_call_result_251405 = invoke(stypy.reporting.localization.Localization(__file__, 758, 14), ValueError_251402, *[str_251403], **kwargs_251404)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 758, 8), ValueError_call_result_251405, 'raise parameter', BaseException)
    # SSA join for if statement (line 757)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 760)
    # Processing the call arguments (line 760)
    # Getting the type of 'bounds' (line 760)
    bounds_251407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 11), 'bounds', False)
    # Processing the call keyword arguments (line 760)
    kwargs_251408 = {}
    # Getting the type of 'len' (line 760)
    len_251406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 7), 'len', False)
    # Calling len(args, kwargs) (line 760)
    len_call_result_251409 = invoke(stypy.reporting.localization.Localization(__file__, 760, 7), len_251406, *[bounds_251407], **kwargs_251408)
    
    int_251410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 22), 'int')
    # Applying the binary operator '!=' (line 760)
    result_ne_251411 = python_operator(stypy.reporting.localization.Localization(__file__, 760, 7), '!=', len_call_result_251409, int_251410)
    
    # Testing the type of an if condition (line 760)
    if_condition_251412 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 760, 4), result_ne_251411)
    # Assigning a type to the variable 'if_condition_251412' (line 760)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 4), 'if_condition_251412', if_condition_251412)
    # SSA begins for if statement (line 760)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 761)
    # Processing the call arguments (line 761)
    str_251414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 25), 'str', '`bounds` must contain 2 elements.')
    # Processing the call keyword arguments (line 761)
    kwargs_251415 = {}
    # Getting the type of 'ValueError' (line 761)
    ValueError_251413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 761)
    ValueError_call_result_251416 = invoke(stypy.reporting.localization.Localization(__file__, 761, 14), ValueError_251413, *[str_251414], **kwargs_251415)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 761, 8), ValueError_call_result_251416, 'raise parameter', BaseException)
    # SSA join for if statement (line 760)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'max_nfev' (line 763)
    max_nfev_251417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 7), 'max_nfev')
    # Getting the type of 'None' (line 763)
    None_251418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 23), 'None')
    # Applying the binary operator 'isnot' (line 763)
    result_is_not_251419 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 7), 'isnot', max_nfev_251417, None_251418)
    
    
    # Getting the type of 'max_nfev' (line 763)
    max_nfev_251420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 32), 'max_nfev')
    int_251421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 44), 'int')
    # Applying the binary operator '<=' (line 763)
    result_le_251422 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 32), '<=', max_nfev_251420, int_251421)
    
    # Applying the binary operator 'and' (line 763)
    result_and_keyword_251423 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 7), 'and', result_is_not_251419, result_le_251422)
    
    # Testing the type of an if condition (line 763)
    if_condition_251424 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 763, 4), result_and_keyword_251423)
    # Assigning a type to the variable 'if_condition_251424' (line 763)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 4), 'if_condition_251424', if_condition_251424)
    # SSA begins for if statement (line 763)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 764)
    # Processing the call arguments (line 764)
    str_251426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 25), 'str', '`max_nfev` must be None or positive integer.')
    # Processing the call keyword arguments (line 764)
    kwargs_251427 = {}
    # Getting the type of 'ValueError' (line 764)
    ValueError_251425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 764)
    ValueError_call_result_251428 = invoke(stypy.reporting.localization.Localization(__file__, 764, 14), ValueError_251425, *[str_251426], **kwargs_251427)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 764, 8), ValueError_call_result_251428, 'raise parameter', BaseException)
    # SSA join for if statement (line 763)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to iscomplexobj(...): (line 766)
    # Processing the call arguments (line 766)
    # Getting the type of 'x0' (line 766)
    x0_251431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 23), 'x0', False)
    # Processing the call keyword arguments (line 766)
    kwargs_251432 = {}
    # Getting the type of 'np' (line 766)
    np_251429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 7), 'np', False)
    # Obtaining the member 'iscomplexobj' of a type (line 766)
    iscomplexobj_251430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 766, 7), np_251429, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 766)
    iscomplexobj_call_result_251433 = invoke(stypy.reporting.localization.Localization(__file__, 766, 7), iscomplexobj_251430, *[x0_251431], **kwargs_251432)
    
    # Testing the type of an if condition (line 766)
    if_condition_251434 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 766, 4), iscomplexobj_call_result_251433)
    # Assigning a type to the variable 'if_condition_251434' (line 766)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 4), 'if_condition_251434', if_condition_251434)
    # SSA begins for if statement (line 766)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 767)
    # Processing the call arguments (line 767)
    str_251436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 25), 'str', '`x0` must be real.')
    # Processing the call keyword arguments (line 767)
    kwargs_251437 = {}
    # Getting the type of 'ValueError' (line 767)
    ValueError_251435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 767)
    ValueError_call_result_251438 = invoke(stypy.reporting.localization.Localization(__file__, 767, 14), ValueError_251435, *[str_251436], **kwargs_251437)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 767, 8), ValueError_call_result_251438, 'raise parameter', BaseException)
    # SSA join for if statement (line 766)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 769):
    
    # Assigning a Call to a Name (line 769):
    
    # Call to astype(...): (line 769)
    # Processing the call arguments (line 769)
    # Getting the type of 'float' (line 769)
    float_251445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 34), 'float', False)
    # Processing the call keyword arguments (line 769)
    kwargs_251446 = {}
    
    # Call to atleast_1d(...): (line 769)
    # Processing the call arguments (line 769)
    # Getting the type of 'x0' (line 769)
    x0_251441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 23), 'x0', False)
    # Processing the call keyword arguments (line 769)
    kwargs_251442 = {}
    # Getting the type of 'np' (line 769)
    np_251439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 9), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 769)
    atleast_1d_251440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 769, 9), np_251439, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 769)
    atleast_1d_call_result_251443 = invoke(stypy.reporting.localization.Localization(__file__, 769, 9), atleast_1d_251440, *[x0_251441], **kwargs_251442)
    
    # Obtaining the member 'astype' of a type (line 769)
    astype_251444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 769, 9), atleast_1d_call_result_251443, 'astype')
    # Calling astype(args, kwargs) (line 769)
    astype_call_result_251447 = invoke(stypy.reporting.localization.Localization(__file__, 769, 9), astype_251444, *[float_251445], **kwargs_251446)
    
    # Assigning a type to the variable 'x0' (line 769)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 4), 'x0', astype_call_result_251447)
    
    
    # Getting the type of 'x0' (line 771)
    x0_251448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 7), 'x0')
    # Obtaining the member 'ndim' of a type (line 771)
    ndim_251449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 7), x0_251448, 'ndim')
    int_251450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 17), 'int')
    # Applying the binary operator '>' (line 771)
    result_gt_251451 = python_operator(stypy.reporting.localization.Localization(__file__, 771, 7), '>', ndim_251449, int_251450)
    
    # Testing the type of an if condition (line 771)
    if_condition_251452 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 771, 4), result_gt_251451)
    # Assigning a type to the variable 'if_condition_251452' (line 771)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 4), 'if_condition_251452', if_condition_251452)
    # SSA begins for if statement (line 771)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 772)
    # Processing the call arguments (line 772)
    str_251454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 25), 'str', '`x0` must have at most 1 dimension.')
    # Processing the call keyword arguments (line 772)
    kwargs_251455 = {}
    # Getting the type of 'ValueError' (line 772)
    ValueError_251453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 772)
    ValueError_call_result_251456 = invoke(stypy.reporting.localization.Localization(__file__, 772, 14), ValueError_251453, *[str_251454], **kwargs_251455)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 772, 8), ValueError_call_result_251456, 'raise parameter', BaseException)
    # SSA join for if statement (line 771)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 774):
    
    # Assigning a Subscript to a Name (line 774):
    
    # Obtaining the type of the subscript
    int_251457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 4), 'int')
    
    # Call to prepare_bounds(...): (line 774)
    # Processing the call arguments (line 774)
    # Getting the type of 'bounds' (line 774)
    bounds_251459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 28), 'bounds', False)
    
    # Obtaining the type of the subscript
    int_251460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 45), 'int')
    # Getting the type of 'x0' (line 774)
    x0_251461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 36), 'x0', False)
    # Obtaining the member 'shape' of a type (line 774)
    shape_251462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 36), x0_251461, 'shape')
    # Obtaining the member '__getitem__' of a type (line 774)
    getitem___251463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 36), shape_251462, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 774)
    subscript_call_result_251464 = invoke(stypy.reporting.localization.Localization(__file__, 774, 36), getitem___251463, int_251460)
    
    # Processing the call keyword arguments (line 774)
    kwargs_251465 = {}
    # Getting the type of 'prepare_bounds' (line 774)
    prepare_bounds_251458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 13), 'prepare_bounds', False)
    # Calling prepare_bounds(args, kwargs) (line 774)
    prepare_bounds_call_result_251466 = invoke(stypy.reporting.localization.Localization(__file__, 774, 13), prepare_bounds_251458, *[bounds_251459, subscript_call_result_251464], **kwargs_251465)
    
    # Obtaining the member '__getitem__' of a type (line 774)
    getitem___251467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 4), prepare_bounds_call_result_251466, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 774)
    subscript_call_result_251468 = invoke(stypy.reporting.localization.Localization(__file__, 774, 4), getitem___251467, int_251457)
    
    # Assigning a type to the variable 'tuple_var_assignment_250504' (line 774)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 774, 4), 'tuple_var_assignment_250504', subscript_call_result_251468)
    
    # Assigning a Subscript to a Name (line 774):
    
    # Obtaining the type of the subscript
    int_251469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 4), 'int')
    
    # Call to prepare_bounds(...): (line 774)
    # Processing the call arguments (line 774)
    # Getting the type of 'bounds' (line 774)
    bounds_251471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 28), 'bounds', False)
    
    # Obtaining the type of the subscript
    int_251472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 45), 'int')
    # Getting the type of 'x0' (line 774)
    x0_251473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 36), 'x0', False)
    # Obtaining the member 'shape' of a type (line 774)
    shape_251474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 36), x0_251473, 'shape')
    # Obtaining the member '__getitem__' of a type (line 774)
    getitem___251475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 36), shape_251474, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 774)
    subscript_call_result_251476 = invoke(stypy.reporting.localization.Localization(__file__, 774, 36), getitem___251475, int_251472)
    
    # Processing the call keyword arguments (line 774)
    kwargs_251477 = {}
    # Getting the type of 'prepare_bounds' (line 774)
    prepare_bounds_251470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 13), 'prepare_bounds', False)
    # Calling prepare_bounds(args, kwargs) (line 774)
    prepare_bounds_call_result_251478 = invoke(stypy.reporting.localization.Localization(__file__, 774, 13), prepare_bounds_251470, *[bounds_251471, subscript_call_result_251476], **kwargs_251477)
    
    # Obtaining the member '__getitem__' of a type (line 774)
    getitem___251479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 4), prepare_bounds_call_result_251478, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 774)
    subscript_call_result_251480 = invoke(stypy.reporting.localization.Localization(__file__, 774, 4), getitem___251479, int_251469)
    
    # Assigning a type to the variable 'tuple_var_assignment_250505' (line 774)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 774, 4), 'tuple_var_assignment_250505', subscript_call_result_251480)
    
    # Assigning a Name to a Name (line 774):
    # Getting the type of 'tuple_var_assignment_250504' (line 774)
    tuple_var_assignment_250504_251481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 4), 'tuple_var_assignment_250504')
    # Assigning a type to the variable 'lb' (line 774)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 774, 4), 'lb', tuple_var_assignment_250504_251481)
    
    # Assigning a Name to a Name (line 774):
    # Getting the type of 'tuple_var_assignment_250505' (line 774)
    tuple_var_assignment_250505_251482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 4), 'tuple_var_assignment_250505')
    # Assigning a type to the variable 'ub' (line 774)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 774, 8), 'ub', tuple_var_assignment_250505_251482)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'method' (line 776)
    method_251483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 7), 'method')
    str_251484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 17), 'str', 'lm')
    # Applying the binary operator '==' (line 776)
    result_eq_251485 = python_operator(stypy.reporting.localization.Localization(__file__, 776, 7), '==', method_251483, str_251484)
    
    
    
    # Call to all(...): (line 776)
    # Processing the call arguments (line 776)
    
    # Getting the type of 'lb' (line 776)
    lb_251488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 38), 'lb', False)
    
    # Getting the type of 'np' (line 776)
    np_251489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 45), 'np', False)
    # Obtaining the member 'inf' of a type (line 776)
    inf_251490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 45), np_251489, 'inf')
    # Applying the 'usub' unary operator (line 776)
    result___neg___251491 = python_operator(stypy.reporting.localization.Localization(__file__, 776, 44), 'usub', inf_251490)
    
    # Applying the binary operator '==' (line 776)
    result_eq_251492 = python_operator(stypy.reporting.localization.Localization(__file__, 776, 38), '==', lb_251488, result___neg___251491)
    
    
    # Getting the type of 'ub' (line 776)
    ub_251493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 56), 'ub', False)
    # Getting the type of 'np' (line 776)
    np_251494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 62), 'np', False)
    # Obtaining the member 'inf' of a type (line 776)
    inf_251495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 62), np_251494, 'inf')
    # Applying the binary operator '==' (line 776)
    result_eq_251496 = python_operator(stypy.reporting.localization.Localization(__file__, 776, 56), '==', ub_251493, inf_251495)
    
    # Applying the binary operator '&' (line 776)
    result_and__251497 = python_operator(stypy.reporting.localization.Localization(__file__, 776, 37), '&', result_eq_251492, result_eq_251496)
    
    # Processing the call keyword arguments (line 776)
    kwargs_251498 = {}
    # Getting the type of 'np' (line 776)
    np_251486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 30), 'np', False)
    # Obtaining the member 'all' of a type (line 776)
    all_251487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 30), np_251486, 'all')
    # Calling all(args, kwargs) (line 776)
    all_call_result_251499 = invoke(stypy.reporting.localization.Localization(__file__, 776, 30), all_251487, *[result_and__251497], **kwargs_251498)
    
    # Applying the 'not' unary operator (line 776)
    result_not__251500 = python_operator(stypy.reporting.localization.Localization(__file__, 776, 26), 'not', all_call_result_251499)
    
    # Applying the binary operator 'and' (line 776)
    result_and_keyword_251501 = python_operator(stypy.reporting.localization.Localization(__file__, 776, 7), 'and', result_eq_251485, result_not__251500)
    
    # Testing the type of an if condition (line 776)
    if_condition_251502 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 776, 4), result_and_keyword_251501)
    # Assigning a type to the variable 'if_condition_251502' (line 776)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 4), 'if_condition_251502', if_condition_251502)
    # SSA begins for if statement (line 776)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 777)
    # Processing the call arguments (line 777)
    str_251504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 25), 'str', "Method 'lm' doesn't support bounds.")
    # Processing the call keyword arguments (line 777)
    kwargs_251505 = {}
    # Getting the type of 'ValueError' (line 777)
    ValueError_251503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 777)
    ValueError_call_result_251506 = invoke(stypy.reporting.localization.Localization(__file__, 777, 14), ValueError_251503, *[str_251504], **kwargs_251505)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 777, 8), ValueError_call_result_251506, 'raise parameter', BaseException)
    # SSA join for if statement (line 776)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'lb' (line 779)
    lb_251507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 7), 'lb')
    # Obtaining the member 'shape' of a type (line 779)
    shape_251508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 7), lb_251507, 'shape')
    # Getting the type of 'x0' (line 779)
    x0_251509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 19), 'x0')
    # Obtaining the member 'shape' of a type (line 779)
    shape_251510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 19), x0_251509, 'shape')
    # Applying the binary operator '!=' (line 779)
    result_ne_251511 = python_operator(stypy.reporting.localization.Localization(__file__, 779, 7), '!=', shape_251508, shape_251510)
    
    
    # Getting the type of 'ub' (line 779)
    ub_251512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 31), 'ub')
    # Obtaining the member 'shape' of a type (line 779)
    shape_251513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 31), ub_251512, 'shape')
    # Getting the type of 'x0' (line 779)
    x0_251514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 43), 'x0')
    # Obtaining the member 'shape' of a type (line 779)
    shape_251515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 43), x0_251514, 'shape')
    # Applying the binary operator '!=' (line 779)
    result_ne_251516 = python_operator(stypy.reporting.localization.Localization(__file__, 779, 31), '!=', shape_251513, shape_251515)
    
    # Applying the binary operator 'or' (line 779)
    result_or_keyword_251517 = python_operator(stypy.reporting.localization.Localization(__file__, 779, 7), 'or', result_ne_251511, result_ne_251516)
    
    # Testing the type of an if condition (line 779)
    if_condition_251518 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 779, 4), result_or_keyword_251517)
    # Assigning a type to the variable 'if_condition_251518' (line 779)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 4), 'if_condition_251518', if_condition_251518)
    # SSA begins for if statement (line 779)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 780)
    # Processing the call arguments (line 780)
    str_251520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 25), 'str', 'Inconsistent shapes between bounds and `x0`.')
    # Processing the call keyword arguments (line 780)
    kwargs_251521 = {}
    # Getting the type of 'ValueError' (line 780)
    ValueError_251519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 780)
    ValueError_call_result_251522 = invoke(stypy.reporting.localization.Localization(__file__, 780, 14), ValueError_251519, *[str_251520], **kwargs_251521)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 780, 8), ValueError_call_result_251522, 'raise parameter', BaseException)
    # SSA join for if statement (line 779)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to any(...): (line 782)
    # Processing the call arguments (line 782)
    
    # Getting the type of 'lb' (line 782)
    lb_251525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 14), 'lb', False)
    # Getting the type of 'ub' (line 782)
    ub_251526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 20), 'ub', False)
    # Applying the binary operator '>=' (line 782)
    result_ge_251527 = python_operator(stypy.reporting.localization.Localization(__file__, 782, 14), '>=', lb_251525, ub_251526)
    
    # Processing the call keyword arguments (line 782)
    kwargs_251528 = {}
    # Getting the type of 'np' (line 782)
    np_251523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 7), 'np', False)
    # Obtaining the member 'any' of a type (line 782)
    any_251524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 7), np_251523, 'any')
    # Calling any(args, kwargs) (line 782)
    any_call_result_251529 = invoke(stypy.reporting.localization.Localization(__file__, 782, 7), any_251524, *[result_ge_251527], **kwargs_251528)
    
    # Testing the type of an if condition (line 782)
    if_condition_251530 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 782, 4), any_call_result_251529)
    # Assigning a type to the variable 'if_condition_251530' (line 782)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 782, 4), 'if_condition_251530', if_condition_251530)
    # SSA begins for if statement (line 782)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 783)
    # Processing the call arguments (line 783)
    str_251532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 25), 'str', 'Each lower bound must be strictly less than each upper bound.')
    # Processing the call keyword arguments (line 783)
    kwargs_251533 = {}
    # Getting the type of 'ValueError' (line 783)
    ValueError_251531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 783)
    ValueError_call_result_251534 = invoke(stypy.reporting.localization.Localization(__file__, 783, 14), ValueError_251531, *[str_251532], **kwargs_251533)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 783, 8), ValueError_call_result_251534, 'raise parameter', BaseException)
    # SSA join for if statement (line 782)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to in_bounds(...): (line 786)
    # Processing the call arguments (line 786)
    # Getting the type of 'x0' (line 786)
    x0_251536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 21), 'x0', False)
    # Getting the type of 'lb' (line 786)
    lb_251537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 25), 'lb', False)
    # Getting the type of 'ub' (line 786)
    ub_251538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 29), 'ub', False)
    # Processing the call keyword arguments (line 786)
    kwargs_251539 = {}
    # Getting the type of 'in_bounds' (line 786)
    in_bounds_251535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 11), 'in_bounds', False)
    # Calling in_bounds(args, kwargs) (line 786)
    in_bounds_call_result_251540 = invoke(stypy.reporting.localization.Localization(__file__, 786, 11), in_bounds_251535, *[x0_251536, lb_251537, ub_251538], **kwargs_251539)
    
    # Applying the 'not' unary operator (line 786)
    result_not__251541 = python_operator(stypy.reporting.localization.Localization(__file__, 786, 7), 'not', in_bounds_call_result_251540)
    
    # Testing the type of an if condition (line 786)
    if_condition_251542 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 786, 4), result_not__251541)
    # Assigning a type to the variable 'if_condition_251542' (line 786)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 786, 4), 'if_condition_251542', if_condition_251542)
    # SSA begins for if statement (line 786)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 787)
    # Processing the call arguments (line 787)
    str_251544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 787, 25), 'str', '`x0` is infeasible.')
    # Processing the call keyword arguments (line 787)
    kwargs_251545 = {}
    # Getting the type of 'ValueError' (line 787)
    ValueError_251543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 787)
    ValueError_call_result_251546 = invoke(stypy.reporting.localization.Localization(__file__, 787, 14), ValueError_251543, *[str_251544], **kwargs_251545)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 787, 8), ValueError_call_result_251546, 'raise parameter', BaseException)
    # SSA join for if statement (line 786)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 789):
    
    # Assigning a Call to a Name (line 789):
    
    # Call to check_x_scale(...): (line 789)
    # Processing the call arguments (line 789)
    # Getting the type of 'x_scale' (line 789)
    x_scale_251548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 28), 'x_scale', False)
    # Getting the type of 'x0' (line 789)
    x0_251549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 37), 'x0', False)
    # Processing the call keyword arguments (line 789)
    kwargs_251550 = {}
    # Getting the type of 'check_x_scale' (line 789)
    check_x_scale_251547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 14), 'check_x_scale', False)
    # Calling check_x_scale(args, kwargs) (line 789)
    check_x_scale_call_result_251551 = invoke(stypy.reporting.localization.Localization(__file__, 789, 14), check_x_scale_251547, *[x_scale_251548, x0_251549], **kwargs_251550)
    
    # Assigning a type to the variable 'x_scale' (line 789)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 789, 4), 'x_scale', check_x_scale_call_result_251551)
    
    # Assigning a Call to a Tuple (line 791):
    
    # Assigning a Subscript to a Name (line 791):
    
    # Obtaining the type of the subscript
    int_251552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 791, 4), 'int')
    
    # Call to check_tolerance(...): (line 791)
    # Processing the call arguments (line 791)
    # Getting the type of 'ftol' (line 791)
    ftol_251554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 39), 'ftol', False)
    # Getting the type of 'xtol' (line 791)
    xtol_251555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 45), 'xtol', False)
    # Getting the type of 'gtol' (line 791)
    gtol_251556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 51), 'gtol', False)
    # Processing the call keyword arguments (line 791)
    kwargs_251557 = {}
    # Getting the type of 'check_tolerance' (line 791)
    check_tolerance_251553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 23), 'check_tolerance', False)
    # Calling check_tolerance(args, kwargs) (line 791)
    check_tolerance_call_result_251558 = invoke(stypy.reporting.localization.Localization(__file__, 791, 23), check_tolerance_251553, *[ftol_251554, xtol_251555, gtol_251556], **kwargs_251557)
    
    # Obtaining the member '__getitem__' of a type (line 791)
    getitem___251559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 4), check_tolerance_call_result_251558, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 791)
    subscript_call_result_251560 = invoke(stypy.reporting.localization.Localization(__file__, 791, 4), getitem___251559, int_251552)
    
    # Assigning a type to the variable 'tuple_var_assignment_250506' (line 791)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 791, 4), 'tuple_var_assignment_250506', subscript_call_result_251560)
    
    # Assigning a Subscript to a Name (line 791):
    
    # Obtaining the type of the subscript
    int_251561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 791, 4), 'int')
    
    # Call to check_tolerance(...): (line 791)
    # Processing the call arguments (line 791)
    # Getting the type of 'ftol' (line 791)
    ftol_251563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 39), 'ftol', False)
    # Getting the type of 'xtol' (line 791)
    xtol_251564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 45), 'xtol', False)
    # Getting the type of 'gtol' (line 791)
    gtol_251565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 51), 'gtol', False)
    # Processing the call keyword arguments (line 791)
    kwargs_251566 = {}
    # Getting the type of 'check_tolerance' (line 791)
    check_tolerance_251562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 23), 'check_tolerance', False)
    # Calling check_tolerance(args, kwargs) (line 791)
    check_tolerance_call_result_251567 = invoke(stypy.reporting.localization.Localization(__file__, 791, 23), check_tolerance_251562, *[ftol_251563, xtol_251564, gtol_251565], **kwargs_251566)
    
    # Obtaining the member '__getitem__' of a type (line 791)
    getitem___251568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 4), check_tolerance_call_result_251567, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 791)
    subscript_call_result_251569 = invoke(stypy.reporting.localization.Localization(__file__, 791, 4), getitem___251568, int_251561)
    
    # Assigning a type to the variable 'tuple_var_assignment_250507' (line 791)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 791, 4), 'tuple_var_assignment_250507', subscript_call_result_251569)
    
    # Assigning a Subscript to a Name (line 791):
    
    # Obtaining the type of the subscript
    int_251570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 791, 4), 'int')
    
    # Call to check_tolerance(...): (line 791)
    # Processing the call arguments (line 791)
    # Getting the type of 'ftol' (line 791)
    ftol_251572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 39), 'ftol', False)
    # Getting the type of 'xtol' (line 791)
    xtol_251573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 45), 'xtol', False)
    # Getting the type of 'gtol' (line 791)
    gtol_251574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 51), 'gtol', False)
    # Processing the call keyword arguments (line 791)
    kwargs_251575 = {}
    # Getting the type of 'check_tolerance' (line 791)
    check_tolerance_251571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 23), 'check_tolerance', False)
    # Calling check_tolerance(args, kwargs) (line 791)
    check_tolerance_call_result_251576 = invoke(stypy.reporting.localization.Localization(__file__, 791, 23), check_tolerance_251571, *[ftol_251572, xtol_251573, gtol_251574], **kwargs_251575)
    
    # Obtaining the member '__getitem__' of a type (line 791)
    getitem___251577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 4), check_tolerance_call_result_251576, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 791)
    subscript_call_result_251578 = invoke(stypy.reporting.localization.Localization(__file__, 791, 4), getitem___251577, int_251570)
    
    # Assigning a type to the variable 'tuple_var_assignment_250508' (line 791)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 791, 4), 'tuple_var_assignment_250508', subscript_call_result_251578)
    
    # Assigning a Name to a Name (line 791):
    # Getting the type of 'tuple_var_assignment_250506' (line 791)
    tuple_var_assignment_250506_251579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 4), 'tuple_var_assignment_250506')
    # Assigning a type to the variable 'ftol' (line 791)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 791, 4), 'ftol', tuple_var_assignment_250506_251579)
    
    # Assigning a Name to a Name (line 791):
    # Getting the type of 'tuple_var_assignment_250507' (line 791)
    tuple_var_assignment_250507_251580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 4), 'tuple_var_assignment_250507')
    # Assigning a type to the variable 'xtol' (line 791)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 791, 10), 'xtol', tuple_var_assignment_250507_251580)
    
    # Assigning a Name to a Name (line 791):
    # Getting the type of 'tuple_var_assignment_250508' (line 791)
    tuple_var_assignment_250508_251581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 4), 'tuple_var_assignment_250508')
    # Assigning a type to the variable 'gtol' (line 791)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 791, 16), 'gtol', tuple_var_assignment_250508_251581)

    @norecursion
    def fun_wrapped(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun_wrapped'
        module_type_store = module_type_store.open_function_context('fun_wrapped', 793, 4, False)
        
        # Passed parameters checking function
        fun_wrapped.stypy_localization = localization
        fun_wrapped.stypy_type_of_self = None
        fun_wrapped.stypy_type_store = module_type_store
        fun_wrapped.stypy_function_name = 'fun_wrapped'
        fun_wrapped.stypy_param_names_list = ['x']
        fun_wrapped.stypy_varargs_param_name = None
        fun_wrapped.stypy_kwargs_param_name = None
        fun_wrapped.stypy_call_defaults = defaults
        fun_wrapped.stypy_call_varargs = varargs
        fun_wrapped.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'fun_wrapped', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun_wrapped', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun_wrapped(...)' code ##################

        
        # Call to atleast_1d(...): (line 794)
        # Processing the call arguments (line 794)
        
        # Call to fun(...): (line 794)
        # Processing the call arguments (line 794)
        # Getting the type of 'x' (line 794)
        x_251585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 33), 'x', False)
        # Getting the type of 'args' (line 794)
        args_251586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 37), 'args', False)
        # Processing the call keyword arguments (line 794)
        # Getting the type of 'kwargs' (line 794)
        kwargs_251587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 45), 'kwargs', False)
        kwargs_251588 = {'kwargs_251587': kwargs_251587}
        # Getting the type of 'fun' (line 794)
        fun_251584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 29), 'fun', False)
        # Calling fun(args, kwargs) (line 794)
        fun_call_result_251589 = invoke(stypy.reporting.localization.Localization(__file__, 794, 29), fun_251584, *[x_251585, args_251586], **kwargs_251588)
        
        # Processing the call keyword arguments (line 794)
        kwargs_251590 = {}
        # Getting the type of 'np' (line 794)
        np_251582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 15), 'np', False)
        # Obtaining the member 'atleast_1d' of a type (line 794)
        atleast_1d_251583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 15), np_251582, 'atleast_1d')
        # Calling atleast_1d(args, kwargs) (line 794)
        atleast_1d_call_result_251591 = invoke(stypy.reporting.localization.Localization(__file__, 794, 15), atleast_1d_251583, *[fun_call_result_251589], **kwargs_251590)
        
        # Assigning a type to the variable 'stypy_return_type' (line 794)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 794, 8), 'stypy_return_type', atleast_1d_call_result_251591)
        
        # ################# End of 'fun_wrapped(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun_wrapped' in the type store
        # Getting the type of 'stypy_return_type' (line 793)
        stypy_return_type_251592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251592)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun_wrapped'
        return stypy_return_type_251592

    # Assigning a type to the variable 'fun_wrapped' (line 793)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 793, 4), 'fun_wrapped', fun_wrapped)
    
    
    # Getting the type of 'method' (line 796)
    method_251593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 7), 'method')
    str_251594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 796, 17), 'str', 'trf')
    # Applying the binary operator '==' (line 796)
    result_eq_251595 = python_operator(stypy.reporting.localization.Localization(__file__, 796, 7), '==', method_251593, str_251594)
    
    # Testing the type of an if condition (line 796)
    if_condition_251596 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 796, 4), result_eq_251595)
    # Assigning a type to the variable 'if_condition_251596' (line 796)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 796, 4), 'if_condition_251596', if_condition_251596)
    # SSA begins for if statement (line 796)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 797):
    
    # Assigning a Call to a Name (line 797):
    
    # Call to make_strictly_feasible(...): (line 797)
    # Processing the call arguments (line 797)
    # Getting the type of 'x0' (line 797)
    x0_251598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 36), 'x0', False)
    # Getting the type of 'lb' (line 797)
    lb_251599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 40), 'lb', False)
    # Getting the type of 'ub' (line 797)
    ub_251600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 44), 'ub', False)
    # Processing the call keyword arguments (line 797)
    kwargs_251601 = {}
    # Getting the type of 'make_strictly_feasible' (line 797)
    make_strictly_feasible_251597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 13), 'make_strictly_feasible', False)
    # Calling make_strictly_feasible(args, kwargs) (line 797)
    make_strictly_feasible_call_result_251602 = invoke(stypy.reporting.localization.Localization(__file__, 797, 13), make_strictly_feasible_251597, *[x0_251598, lb_251599, ub_251600], **kwargs_251601)
    
    # Assigning a type to the variable 'x0' (line 797)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 8), 'x0', make_strictly_feasible_call_result_251602)
    # SSA join for if statement (line 796)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 799):
    
    # Assigning a Call to a Name (line 799):
    
    # Call to fun_wrapped(...): (line 799)
    # Processing the call arguments (line 799)
    # Getting the type of 'x0' (line 799)
    x0_251604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 21), 'x0', False)
    # Processing the call keyword arguments (line 799)
    kwargs_251605 = {}
    # Getting the type of 'fun_wrapped' (line 799)
    fun_wrapped_251603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 9), 'fun_wrapped', False)
    # Calling fun_wrapped(args, kwargs) (line 799)
    fun_wrapped_call_result_251606 = invoke(stypy.reporting.localization.Localization(__file__, 799, 9), fun_wrapped_251603, *[x0_251604], **kwargs_251605)
    
    # Assigning a type to the variable 'f0' (line 799)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 799, 4), 'f0', fun_wrapped_call_result_251606)
    
    
    # Getting the type of 'f0' (line 801)
    f0_251607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 7), 'f0')
    # Obtaining the member 'ndim' of a type (line 801)
    ndim_251608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 801, 7), f0_251607, 'ndim')
    int_251609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 801, 18), 'int')
    # Applying the binary operator '!=' (line 801)
    result_ne_251610 = python_operator(stypy.reporting.localization.Localization(__file__, 801, 7), '!=', ndim_251608, int_251609)
    
    # Testing the type of an if condition (line 801)
    if_condition_251611 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 801, 4), result_ne_251610)
    # Assigning a type to the variable 'if_condition_251611' (line 801)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 801, 4), 'if_condition_251611', if_condition_251611)
    # SSA begins for if statement (line 801)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 802)
    # Processing the call arguments (line 802)
    str_251613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 802, 25), 'str', '`fun` must return at most 1-d array_like.')
    # Processing the call keyword arguments (line 802)
    kwargs_251614 = {}
    # Getting the type of 'ValueError' (line 802)
    ValueError_251612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 802)
    ValueError_call_result_251615 = invoke(stypy.reporting.localization.Localization(__file__, 802, 14), ValueError_251612, *[str_251613], **kwargs_251614)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 802, 8), ValueError_call_result_251615, 'raise parameter', BaseException)
    # SSA join for if statement (line 801)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to all(...): (line 804)
    # Processing the call arguments (line 804)
    
    # Call to isfinite(...): (line 804)
    # Processing the call arguments (line 804)
    # Getting the type of 'f0' (line 804)
    f0_251620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 30), 'f0', False)
    # Processing the call keyword arguments (line 804)
    kwargs_251621 = {}
    # Getting the type of 'np' (line 804)
    np_251618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 18), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 804)
    isfinite_251619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 804, 18), np_251618, 'isfinite')
    # Calling isfinite(args, kwargs) (line 804)
    isfinite_call_result_251622 = invoke(stypy.reporting.localization.Localization(__file__, 804, 18), isfinite_251619, *[f0_251620], **kwargs_251621)
    
    # Processing the call keyword arguments (line 804)
    kwargs_251623 = {}
    # Getting the type of 'np' (line 804)
    np_251616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 11), 'np', False)
    # Obtaining the member 'all' of a type (line 804)
    all_251617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 804, 11), np_251616, 'all')
    # Calling all(args, kwargs) (line 804)
    all_call_result_251624 = invoke(stypy.reporting.localization.Localization(__file__, 804, 11), all_251617, *[isfinite_call_result_251622], **kwargs_251623)
    
    # Applying the 'not' unary operator (line 804)
    result_not__251625 = python_operator(stypy.reporting.localization.Localization(__file__, 804, 7), 'not', all_call_result_251624)
    
    # Testing the type of an if condition (line 804)
    if_condition_251626 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 804, 4), result_not__251625)
    # Assigning a type to the variable 'if_condition_251626' (line 804)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 804, 4), 'if_condition_251626', if_condition_251626)
    # SSA begins for if statement (line 804)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 805)
    # Processing the call arguments (line 805)
    str_251628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, 25), 'str', 'Residuals are not finite in the initial point.')
    # Processing the call keyword arguments (line 805)
    kwargs_251629 = {}
    # Getting the type of 'ValueError' (line 805)
    ValueError_251627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 805)
    ValueError_call_result_251630 = invoke(stypy.reporting.localization.Localization(__file__, 805, 14), ValueError_251627, *[str_251628], **kwargs_251629)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 805, 8), ValueError_call_result_251630, 'raise parameter', BaseException)
    # SSA join for if statement (line 804)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 807):
    
    # Assigning a Attribute to a Name (line 807):
    # Getting the type of 'x0' (line 807)
    x0_251631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 8), 'x0')
    # Obtaining the member 'size' of a type (line 807)
    size_251632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 807, 8), x0_251631, 'size')
    # Assigning a type to the variable 'n' (line 807)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 807, 4), 'n', size_251632)
    
    # Assigning a Attribute to a Name (line 808):
    
    # Assigning a Attribute to a Name (line 808):
    # Getting the type of 'f0' (line 808)
    f0_251633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 8), 'f0')
    # Obtaining the member 'size' of a type (line 808)
    size_251634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 808, 8), f0_251633, 'size')
    # Assigning a type to the variable 'm' (line 808)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 808, 4), 'm', size_251634)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'method' (line 810)
    method_251635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 7), 'method')
    str_251636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 810, 17), 'str', 'lm')
    # Applying the binary operator '==' (line 810)
    result_eq_251637 = python_operator(stypy.reporting.localization.Localization(__file__, 810, 7), '==', method_251635, str_251636)
    
    
    # Getting the type of 'm' (line 810)
    m_251638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 26), 'm')
    # Getting the type of 'n' (line 810)
    n_251639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 30), 'n')
    # Applying the binary operator '<' (line 810)
    result_lt_251640 = python_operator(stypy.reporting.localization.Localization(__file__, 810, 26), '<', m_251638, n_251639)
    
    # Applying the binary operator 'and' (line 810)
    result_and_keyword_251641 = python_operator(stypy.reporting.localization.Localization(__file__, 810, 7), 'and', result_eq_251637, result_lt_251640)
    
    # Testing the type of an if condition (line 810)
    if_condition_251642 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 810, 4), result_and_keyword_251641)
    # Assigning a type to the variable 'if_condition_251642' (line 810)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 4), 'if_condition_251642', if_condition_251642)
    # SSA begins for if statement (line 810)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 811)
    # Processing the call arguments (line 811)
    str_251644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 811, 25), 'str', "Method 'lm' doesn't work when the number of residuals is less than the number of variables.")
    # Processing the call keyword arguments (line 811)
    kwargs_251645 = {}
    # Getting the type of 'ValueError' (line 811)
    ValueError_251643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 811)
    ValueError_call_result_251646 = invoke(stypy.reporting.localization.Localization(__file__, 811, 14), ValueError_251643, *[str_251644], **kwargs_251645)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 811, 8), ValueError_call_result_251646, 'raise parameter', BaseException)
    # SSA join for if statement (line 810)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 814):
    
    # Assigning a Call to a Name (line 814):
    
    # Call to construct_loss_function(...): (line 814)
    # Processing the call arguments (line 814)
    # Getting the type of 'm' (line 814)
    m_251648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 44), 'm', False)
    # Getting the type of 'loss' (line 814)
    loss_251649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 47), 'loss', False)
    # Getting the type of 'f_scale' (line 814)
    f_scale_251650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 53), 'f_scale', False)
    # Processing the call keyword arguments (line 814)
    kwargs_251651 = {}
    # Getting the type of 'construct_loss_function' (line 814)
    construct_loss_function_251647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 20), 'construct_loss_function', False)
    # Calling construct_loss_function(args, kwargs) (line 814)
    construct_loss_function_call_result_251652 = invoke(stypy.reporting.localization.Localization(__file__, 814, 20), construct_loss_function_251647, *[m_251648, loss_251649, f_scale_251650], **kwargs_251651)
    
    # Assigning a type to the variable 'loss_function' (line 814)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 814, 4), 'loss_function', construct_loss_function_call_result_251652)
    
    
    # Call to callable(...): (line 815)
    # Processing the call arguments (line 815)
    # Getting the type of 'loss' (line 815)
    loss_251654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 16), 'loss', False)
    # Processing the call keyword arguments (line 815)
    kwargs_251655 = {}
    # Getting the type of 'callable' (line 815)
    callable_251653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 7), 'callable', False)
    # Calling callable(args, kwargs) (line 815)
    callable_call_result_251656 = invoke(stypy.reporting.localization.Localization(__file__, 815, 7), callable_251653, *[loss_251654], **kwargs_251655)
    
    # Testing the type of an if condition (line 815)
    if_condition_251657 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 815, 4), callable_call_result_251656)
    # Assigning a type to the variable 'if_condition_251657' (line 815)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 4), 'if_condition_251657', if_condition_251657)
    # SSA begins for if statement (line 815)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 816):
    
    # Assigning a Call to a Name (line 816):
    
    # Call to loss_function(...): (line 816)
    # Processing the call arguments (line 816)
    # Getting the type of 'f0' (line 816)
    f0_251659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 28), 'f0', False)
    # Processing the call keyword arguments (line 816)
    kwargs_251660 = {}
    # Getting the type of 'loss_function' (line 816)
    loss_function_251658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 14), 'loss_function', False)
    # Calling loss_function(args, kwargs) (line 816)
    loss_function_call_result_251661 = invoke(stypy.reporting.localization.Localization(__file__, 816, 14), loss_function_251658, *[f0_251659], **kwargs_251660)
    
    # Assigning a type to the variable 'rho' (line 816)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 816, 8), 'rho', loss_function_call_result_251661)
    
    
    # Getting the type of 'rho' (line 817)
    rho_251662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 11), 'rho')
    # Obtaining the member 'shape' of a type (line 817)
    shape_251663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 817, 11), rho_251662, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 817)
    tuple_251664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 817)
    # Adding element type (line 817)
    int_251665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 817, 25), tuple_251664, int_251665)
    # Adding element type (line 817)
    # Getting the type of 'm' (line 817)
    m_251666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 28), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 817, 25), tuple_251664, m_251666)
    
    # Applying the binary operator '!=' (line 817)
    result_ne_251667 = python_operator(stypy.reporting.localization.Localization(__file__, 817, 11), '!=', shape_251663, tuple_251664)
    
    # Testing the type of an if condition (line 817)
    if_condition_251668 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 817, 8), result_ne_251667)
    # Assigning a type to the variable 'if_condition_251668' (line 817)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 817, 8), 'if_condition_251668', if_condition_251668)
    # SSA begins for if statement (line 817)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 818)
    # Processing the call arguments (line 818)
    str_251670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 29), 'str', 'The return value of `loss` callable has wrong shape.')
    # Processing the call keyword arguments (line 818)
    kwargs_251671 = {}
    # Getting the type of 'ValueError' (line 818)
    ValueError_251669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 818)
    ValueError_call_result_251672 = invoke(stypy.reporting.localization.Localization(__file__, 818, 18), ValueError_251669, *[str_251670], **kwargs_251671)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 818, 12), ValueError_call_result_251672, 'raise parameter', BaseException)
    # SSA join for if statement (line 817)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 820):
    
    # Assigning a BinOp to a Name (line 820):
    float_251673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 820, 23), 'float')
    
    # Call to sum(...): (line 820)
    # Processing the call arguments (line 820)
    
    # Obtaining the type of the subscript
    int_251676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 820, 40), 'int')
    # Getting the type of 'rho' (line 820)
    rho_251677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 36), 'rho', False)
    # Obtaining the member '__getitem__' of a type (line 820)
    getitem___251678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 820, 36), rho_251677, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 820)
    subscript_call_result_251679 = invoke(stypy.reporting.localization.Localization(__file__, 820, 36), getitem___251678, int_251676)
    
    # Processing the call keyword arguments (line 820)
    kwargs_251680 = {}
    # Getting the type of 'np' (line 820)
    np_251674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 29), 'np', False)
    # Obtaining the member 'sum' of a type (line 820)
    sum_251675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 820, 29), np_251674, 'sum')
    # Calling sum(args, kwargs) (line 820)
    sum_call_result_251681 = invoke(stypy.reporting.localization.Localization(__file__, 820, 29), sum_251675, *[subscript_call_result_251679], **kwargs_251680)
    
    # Applying the binary operator '*' (line 820)
    result_mul_251682 = python_operator(stypy.reporting.localization.Localization(__file__, 820, 23), '*', float_251673, sum_call_result_251681)
    
    # Assigning a type to the variable 'initial_cost' (line 820)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 820, 8), 'initial_cost', result_mul_251682)
    # SSA branch for the else part of an if statement (line 815)
    module_type_store.open_ssa_branch('else')
    
    # Type idiom detected: calculating its left and rigth part (line 821)
    # Getting the type of 'loss_function' (line 821)
    loss_function_251683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 9), 'loss_function')
    # Getting the type of 'None' (line 821)
    None_251684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 30), 'None')
    
    (may_be_251685, more_types_in_union_251686) = may_not_be_none(loss_function_251683, None_251684)

    if may_be_251685:

        if more_types_in_union_251686:
            # Runtime conditional SSA (line 821)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 822):
        
        # Assigning a Call to a Name (line 822):
        
        # Call to loss_function(...): (line 822)
        # Processing the call arguments (line 822)
        # Getting the type of 'f0' (line 822)
        f0_251688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 37), 'f0', False)
        # Processing the call keyword arguments (line 822)
        # Getting the type of 'True' (line 822)
        True_251689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 51), 'True', False)
        keyword_251690 = True_251689
        kwargs_251691 = {'cost_only': keyword_251690}
        # Getting the type of 'loss_function' (line 822)
        loss_function_251687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 23), 'loss_function', False)
        # Calling loss_function(args, kwargs) (line 822)
        loss_function_call_result_251692 = invoke(stypy.reporting.localization.Localization(__file__, 822, 23), loss_function_251687, *[f0_251688], **kwargs_251691)
        
        # Assigning a type to the variable 'initial_cost' (line 822)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 822, 8), 'initial_cost', loss_function_call_result_251692)

        if more_types_in_union_251686:
            # Runtime conditional SSA for else branch (line 821)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_251685) or more_types_in_union_251686):
        
        # Assigning a BinOp to a Name (line 824):
        
        # Assigning a BinOp to a Name (line 824):
        float_251693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 824, 23), 'float')
        
        # Call to dot(...): (line 824)
        # Processing the call arguments (line 824)
        # Getting the type of 'f0' (line 824)
        f0_251696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 36), 'f0', False)
        # Getting the type of 'f0' (line 824)
        f0_251697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 40), 'f0', False)
        # Processing the call keyword arguments (line 824)
        kwargs_251698 = {}
        # Getting the type of 'np' (line 824)
        np_251694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 29), 'np', False)
        # Obtaining the member 'dot' of a type (line 824)
        dot_251695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 824, 29), np_251694, 'dot')
        # Calling dot(args, kwargs) (line 824)
        dot_call_result_251699 = invoke(stypy.reporting.localization.Localization(__file__, 824, 29), dot_251695, *[f0_251696, f0_251697], **kwargs_251698)
        
        # Applying the binary operator '*' (line 824)
        result_mul_251700 = python_operator(stypy.reporting.localization.Localization(__file__, 824, 23), '*', float_251693, dot_call_result_251699)
        
        # Assigning a type to the variable 'initial_cost' (line 824)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 824, 8), 'initial_cost', result_mul_251700)

        if (may_be_251685 and more_types_in_union_251686):
            # SSA join for if statement (line 821)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 815)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to callable(...): (line 826)
    # Processing the call arguments (line 826)
    # Getting the type of 'jac' (line 826)
    jac_251702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 16), 'jac', False)
    # Processing the call keyword arguments (line 826)
    kwargs_251703 = {}
    # Getting the type of 'callable' (line 826)
    callable_251701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 7), 'callable', False)
    # Calling callable(args, kwargs) (line 826)
    callable_call_result_251704 = invoke(stypy.reporting.localization.Localization(__file__, 826, 7), callable_251701, *[jac_251702], **kwargs_251703)
    
    # Testing the type of an if condition (line 826)
    if_condition_251705 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 826, 4), callable_call_result_251704)
    # Assigning a type to the variable 'if_condition_251705' (line 826)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 826, 4), 'if_condition_251705', if_condition_251705)
    # SSA begins for if statement (line 826)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 827):
    
    # Assigning a Call to a Name (line 827):
    
    # Call to jac(...): (line 827)
    # Processing the call arguments (line 827)
    # Getting the type of 'x0' (line 827)
    x0_251707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 17), 'x0', False)
    # Getting the type of 'args' (line 827)
    args_251708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 22), 'args', False)
    # Processing the call keyword arguments (line 827)
    # Getting the type of 'kwargs' (line 827)
    kwargs_251709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 30), 'kwargs', False)
    kwargs_251710 = {'kwargs_251709': kwargs_251709}
    # Getting the type of 'jac' (line 827)
    jac_251706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 13), 'jac', False)
    # Calling jac(args, kwargs) (line 827)
    jac_call_result_251711 = invoke(stypy.reporting.localization.Localization(__file__, 827, 13), jac_251706, *[x0_251707, args_251708], **kwargs_251710)
    
    # Assigning a type to the variable 'J0' (line 827)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 827, 8), 'J0', jac_call_result_251711)
    
    
    # Call to issparse(...): (line 829)
    # Processing the call arguments (line 829)
    # Getting the type of 'J0' (line 829)
    J0_251713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 20), 'J0', False)
    # Processing the call keyword arguments (line 829)
    kwargs_251714 = {}
    # Getting the type of 'issparse' (line 829)
    issparse_251712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 11), 'issparse', False)
    # Calling issparse(args, kwargs) (line 829)
    issparse_call_result_251715 = invoke(stypy.reporting.localization.Localization(__file__, 829, 11), issparse_251712, *[J0_251713], **kwargs_251714)
    
    # Testing the type of an if condition (line 829)
    if_condition_251716 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 829, 8), issparse_call_result_251715)
    # Assigning a type to the variable 'if_condition_251716' (line 829)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 829, 8), 'if_condition_251716', if_condition_251716)
    # SSA begins for if statement (line 829)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 830):
    
    # Assigning a Call to a Name (line 830):
    
    # Call to csr_matrix(...): (line 830)
    # Processing the call arguments (line 830)
    # Getting the type of 'J0' (line 830)
    J0_251718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 28), 'J0', False)
    # Processing the call keyword arguments (line 830)
    kwargs_251719 = {}
    # Getting the type of 'csr_matrix' (line 830)
    csr_matrix_251717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 17), 'csr_matrix', False)
    # Calling csr_matrix(args, kwargs) (line 830)
    csr_matrix_call_result_251720 = invoke(stypy.reporting.localization.Localization(__file__, 830, 17), csr_matrix_251717, *[J0_251718], **kwargs_251719)
    
    # Assigning a type to the variable 'J0' (line 830)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 830, 12), 'J0', csr_matrix_call_result_251720)

    @norecursion
    def jac_wrapped(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 832)
        None_251721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 33), 'None')
        defaults = [None_251721]
        # Create a new context for function 'jac_wrapped'
        module_type_store = module_type_store.open_function_context('jac_wrapped', 832, 12, False)
        
        # Passed parameters checking function
        jac_wrapped.stypy_localization = localization
        jac_wrapped.stypy_type_of_self = None
        jac_wrapped.stypy_type_store = module_type_store
        jac_wrapped.stypy_function_name = 'jac_wrapped'
        jac_wrapped.stypy_param_names_list = ['x', '_']
        jac_wrapped.stypy_varargs_param_name = None
        jac_wrapped.stypy_kwargs_param_name = None
        jac_wrapped.stypy_call_defaults = defaults
        jac_wrapped.stypy_call_varargs = varargs
        jac_wrapped.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'jac_wrapped', ['x', '_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'jac_wrapped', localization, ['x', '_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'jac_wrapped(...)' code ##################

        
        # Call to csr_matrix(...): (line 833)
        # Processing the call arguments (line 833)
        
        # Call to jac(...): (line 833)
        # Processing the call arguments (line 833)
        # Getting the type of 'x' (line 833)
        x_251724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 38), 'x', False)
        # Getting the type of 'args' (line 833)
        args_251725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 42), 'args', False)
        # Processing the call keyword arguments (line 833)
        # Getting the type of 'kwargs' (line 833)
        kwargs_251726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 50), 'kwargs', False)
        kwargs_251727 = {'kwargs_251726': kwargs_251726}
        # Getting the type of 'jac' (line 833)
        jac_251723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 34), 'jac', False)
        # Calling jac(args, kwargs) (line 833)
        jac_call_result_251728 = invoke(stypy.reporting.localization.Localization(__file__, 833, 34), jac_251723, *[x_251724, args_251725], **kwargs_251727)
        
        # Processing the call keyword arguments (line 833)
        kwargs_251729 = {}
        # Getting the type of 'csr_matrix' (line 833)
        csr_matrix_251722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 23), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 833)
        csr_matrix_call_result_251730 = invoke(stypy.reporting.localization.Localization(__file__, 833, 23), csr_matrix_251722, *[jac_call_result_251728], **kwargs_251729)
        
        # Assigning a type to the variable 'stypy_return_type' (line 833)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 833, 16), 'stypy_return_type', csr_matrix_call_result_251730)
        
        # ################# End of 'jac_wrapped(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'jac_wrapped' in the type store
        # Getting the type of 'stypy_return_type' (line 832)
        stypy_return_type_251731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 12), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251731)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'jac_wrapped'
        return stypy_return_type_251731

    # Assigning a type to the variable 'jac_wrapped' (line 832)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 832, 12), 'jac_wrapped', jac_wrapped)
    # SSA branch for the else part of an if statement (line 829)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isinstance(...): (line 835)
    # Processing the call arguments (line 835)
    # Getting the type of 'J0' (line 835)
    J0_251733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 24), 'J0', False)
    # Getting the type of 'LinearOperator' (line 835)
    LinearOperator_251734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 28), 'LinearOperator', False)
    # Processing the call keyword arguments (line 835)
    kwargs_251735 = {}
    # Getting the type of 'isinstance' (line 835)
    isinstance_251732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 13), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 835)
    isinstance_call_result_251736 = invoke(stypy.reporting.localization.Localization(__file__, 835, 13), isinstance_251732, *[J0_251733, LinearOperator_251734], **kwargs_251735)
    
    # Testing the type of an if condition (line 835)
    if_condition_251737 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 835, 13), isinstance_call_result_251736)
    # Assigning a type to the variable 'if_condition_251737' (line 835)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 835, 13), 'if_condition_251737', if_condition_251737)
    # SSA begins for if statement (line 835)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

    @norecursion
    def jac_wrapped(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 836)
        None_251738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 33), 'None')
        defaults = [None_251738]
        # Create a new context for function 'jac_wrapped'
        module_type_store = module_type_store.open_function_context('jac_wrapped', 836, 12, False)
        
        # Passed parameters checking function
        jac_wrapped.stypy_localization = localization
        jac_wrapped.stypy_type_of_self = None
        jac_wrapped.stypy_type_store = module_type_store
        jac_wrapped.stypy_function_name = 'jac_wrapped'
        jac_wrapped.stypy_param_names_list = ['x', '_']
        jac_wrapped.stypy_varargs_param_name = None
        jac_wrapped.stypy_kwargs_param_name = None
        jac_wrapped.stypy_call_defaults = defaults
        jac_wrapped.stypy_call_varargs = varargs
        jac_wrapped.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'jac_wrapped', ['x', '_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'jac_wrapped', localization, ['x', '_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'jac_wrapped(...)' code ##################

        
        # Call to jac(...): (line 837)
        # Processing the call arguments (line 837)
        # Getting the type of 'x' (line 837)
        x_251740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 27), 'x', False)
        # Getting the type of 'args' (line 837)
        args_251741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 31), 'args', False)
        # Processing the call keyword arguments (line 837)
        # Getting the type of 'kwargs' (line 837)
        kwargs_251742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 39), 'kwargs', False)
        kwargs_251743 = {'kwargs_251742': kwargs_251742}
        # Getting the type of 'jac' (line 837)
        jac_251739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 23), 'jac', False)
        # Calling jac(args, kwargs) (line 837)
        jac_call_result_251744 = invoke(stypy.reporting.localization.Localization(__file__, 837, 23), jac_251739, *[x_251740, args_251741], **kwargs_251743)
        
        # Assigning a type to the variable 'stypy_return_type' (line 837)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 837, 16), 'stypy_return_type', jac_call_result_251744)
        
        # ################# End of 'jac_wrapped(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'jac_wrapped' in the type store
        # Getting the type of 'stypy_return_type' (line 836)
        stypy_return_type_251745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 12), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251745)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'jac_wrapped'
        return stypy_return_type_251745

    # Assigning a type to the variable 'jac_wrapped' (line 836)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 836, 12), 'jac_wrapped', jac_wrapped)
    # SSA branch for the else part of an if statement (line 835)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 840):
    
    # Assigning a Call to a Name (line 840):
    
    # Call to atleast_2d(...): (line 840)
    # Processing the call arguments (line 840)
    # Getting the type of 'J0' (line 840)
    J0_251748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 31), 'J0', False)
    # Processing the call keyword arguments (line 840)
    kwargs_251749 = {}
    # Getting the type of 'np' (line 840)
    np_251746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 17), 'np', False)
    # Obtaining the member 'atleast_2d' of a type (line 840)
    atleast_2d_251747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 840, 17), np_251746, 'atleast_2d')
    # Calling atleast_2d(args, kwargs) (line 840)
    atleast_2d_call_result_251750 = invoke(stypy.reporting.localization.Localization(__file__, 840, 17), atleast_2d_251747, *[J0_251748], **kwargs_251749)
    
    # Assigning a type to the variable 'J0' (line 840)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 840, 12), 'J0', atleast_2d_call_result_251750)

    @norecursion
    def jac_wrapped(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 842)
        None_251751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 33), 'None')
        defaults = [None_251751]
        # Create a new context for function 'jac_wrapped'
        module_type_store = module_type_store.open_function_context('jac_wrapped', 842, 12, False)
        
        # Passed parameters checking function
        jac_wrapped.stypy_localization = localization
        jac_wrapped.stypy_type_of_self = None
        jac_wrapped.stypy_type_store = module_type_store
        jac_wrapped.stypy_function_name = 'jac_wrapped'
        jac_wrapped.stypy_param_names_list = ['x', '_']
        jac_wrapped.stypy_varargs_param_name = None
        jac_wrapped.stypy_kwargs_param_name = None
        jac_wrapped.stypy_call_defaults = defaults
        jac_wrapped.stypy_call_varargs = varargs
        jac_wrapped.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'jac_wrapped', ['x', '_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'jac_wrapped', localization, ['x', '_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'jac_wrapped(...)' code ##################

        
        # Call to atleast_2d(...): (line 843)
        # Processing the call arguments (line 843)
        
        # Call to jac(...): (line 843)
        # Processing the call arguments (line 843)
        # Getting the type of 'x' (line 843)
        x_251755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 41), 'x', False)
        # Getting the type of 'args' (line 843)
        args_251756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 45), 'args', False)
        # Processing the call keyword arguments (line 843)
        # Getting the type of 'kwargs' (line 843)
        kwargs_251757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 53), 'kwargs', False)
        kwargs_251758 = {'kwargs_251757': kwargs_251757}
        # Getting the type of 'jac' (line 843)
        jac_251754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 37), 'jac', False)
        # Calling jac(args, kwargs) (line 843)
        jac_call_result_251759 = invoke(stypy.reporting.localization.Localization(__file__, 843, 37), jac_251754, *[x_251755, args_251756], **kwargs_251758)
        
        # Processing the call keyword arguments (line 843)
        kwargs_251760 = {}
        # Getting the type of 'np' (line 843)
        np_251752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 23), 'np', False)
        # Obtaining the member 'atleast_2d' of a type (line 843)
        atleast_2d_251753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 23), np_251752, 'atleast_2d')
        # Calling atleast_2d(args, kwargs) (line 843)
        atleast_2d_call_result_251761 = invoke(stypy.reporting.localization.Localization(__file__, 843, 23), atleast_2d_251753, *[jac_call_result_251759], **kwargs_251760)
        
        # Assigning a type to the variable 'stypy_return_type' (line 843)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 843, 16), 'stypy_return_type', atleast_2d_call_result_251761)
        
        # ################# End of 'jac_wrapped(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'jac_wrapped' in the type store
        # Getting the type of 'stypy_return_type' (line 842)
        stypy_return_type_251762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 12), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251762)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'jac_wrapped'
        return stypy_return_type_251762

    # Assigning a type to the variable 'jac_wrapped' (line 842)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 842, 12), 'jac_wrapped', jac_wrapped)
    # SSA join for if statement (line 835)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 829)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 826)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'method' (line 846)
    method_251763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 11), 'method')
    str_251764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 846, 21), 'str', 'lm')
    # Applying the binary operator '==' (line 846)
    result_eq_251765 = python_operator(stypy.reporting.localization.Localization(__file__, 846, 11), '==', method_251763, str_251764)
    
    # Testing the type of an if condition (line 846)
    if_condition_251766 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 846, 8), result_eq_251765)
    # Assigning a type to the variable 'if_condition_251766' (line 846)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 846, 8), 'if_condition_251766', if_condition_251766)
    # SSA begins for if statement (line 846)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 847)
    # Getting the type of 'jac_sparsity' (line 847)
    jac_sparsity_251767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 12), 'jac_sparsity')
    # Getting the type of 'None' (line 847)
    None_251768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 35), 'None')
    
    (may_be_251769, more_types_in_union_251770) = may_not_be_none(jac_sparsity_251767, None_251768)

    if may_be_251769:

        if more_types_in_union_251770:
            # Runtime conditional SSA (line 847)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to ValueError(...): (line 848)
        # Processing the call arguments (line 848)
        str_251772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 33), 'str', "method='lm' does not support `jac_sparsity`.")
        # Processing the call keyword arguments (line 848)
        kwargs_251773 = {}
        # Getting the type of 'ValueError' (line 848)
        ValueError_251771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 848)
        ValueError_call_result_251774 = invoke(stypy.reporting.localization.Localization(__file__, 848, 22), ValueError_251771, *[str_251772], **kwargs_251773)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 848, 16), ValueError_call_result_251774, 'raise parameter', BaseException)

        if more_types_in_union_251770:
            # SSA join for if statement (line 847)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'jac' (line 851)
    jac_251775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 15), 'jac')
    str_251776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 851, 22), 'str', '2-point')
    # Applying the binary operator '!=' (line 851)
    result_ne_251777 = python_operator(stypy.reporting.localization.Localization(__file__, 851, 15), '!=', jac_251775, str_251776)
    
    # Testing the type of an if condition (line 851)
    if_condition_251778 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 851, 12), result_ne_251777)
    # Assigning a type to the variable 'if_condition_251778' (line 851)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 851, 12), 'if_condition_251778', if_condition_251778)
    # SSA begins for if statement (line 851)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 852)
    # Processing the call arguments (line 852)
    
    # Call to format(...): (line 852)
    # Processing the call arguments (line 852)
    # Getting the type of 'jac' (line 853)
    jac_251782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 47), 'jac', False)
    # Processing the call keyword arguments (line 852)
    kwargs_251783 = {}
    str_251780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 852, 21), 'str', "jac='{0}' works equivalently to '2-point' for method='lm'.")
    # Obtaining the member 'format' of a type (line 852)
    format_251781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 852, 21), str_251780, 'format')
    # Calling format(args, kwargs) (line 852)
    format_call_result_251784 = invoke(stypy.reporting.localization.Localization(__file__, 852, 21), format_251781, *[jac_251782], **kwargs_251783)
    
    # Processing the call keyword arguments (line 852)
    kwargs_251785 = {}
    # Getting the type of 'warn' (line 852)
    warn_251779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 16), 'warn', False)
    # Calling warn(args, kwargs) (line 852)
    warn_call_result_251786 = invoke(stypy.reporting.localization.Localization(__file__, 852, 16), warn_251779, *[format_call_result_251784], **kwargs_251785)
    
    # SSA join for if statement (line 851)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Multiple assignment of 2 elements.
    
    # Assigning a Name to a Name (line 855):
    # Getting the type of 'None' (line 855)
    None_251787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 31), 'None')
    # Assigning a type to the variable 'jac_wrapped' (line 855)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 855, 17), 'jac_wrapped', None_251787)
    
    # Assigning a Name to a Name (line 855):
    # Getting the type of 'jac_wrapped' (line 855)
    jac_wrapped_251788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 17), 'jac_wrapped')
    # Assigning a type to the variable 'J0' (line 855)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 855, 12), 'J0', jac_wrapped_251788)
    # SSA branch for the else part of an if statement (line 846)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'jac_sparsity' (line 857)
    jac_sparsity_251789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 15), 'jac_sparsity')
    # Getting the type of 'None' (line 857)
    None_251790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 35), 'None')
    # Applying the binary operator 'isnot' (line 857)
    result_is_not_251791 = python_operator(stypy.reporting.localization.Localization(__file__, 857, 15), 'isnot', jac_sparsity_251789, None_251790)
    
    
    # Getting the type of 'tr_solver' (line 857)
    tr_solver_251792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 44), 'tr_solver')
    str_251793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 857, 57), 'str', 'exact')
    # Applying the binary operator '==' (line 857)
    result_eq_251794 = python_operator(stypy.reporting.localization.Localization(__file__, 857, 44), '==', tr_solver_251792, str_251793)
    
    # Applying the binary operator 'and' (line 857)
    result_and_keyword_251795 = python_operator(stypy.reporting.localization.Localization(__file__, 857, 15), 'and', result_is_not_251791, result_eq_251794)
    
    # Testing the type of an if condition (line 857)
    if_condition_251796 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 857, 12), result_and_keyword_251795)
    # Assigning a type to the variable 'if_condition_251796' (line 857)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 857, 12), 'if_condition_251796', if_condition_251796)
    # SSA begins for if statement (line 857)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 858)
    # Processing the call arguments (line 858)
    str_251798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 858, 33), 'str', "tr_solver='exact' is incompatible with `jac_sparsity`.")
    # Processing the call keyword arguments (line 858)
    kwargs_251799 = {}
    # Getting the type of 'ValueError' (line 858)
    ValueError_251797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 858, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 858)
    ValueError_call_result_251800 = invoke(stypy.reporting.localization.Localization(__file__, 858, 22), ValueError_251797, *[str_251798], **kwargs_251799)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 858, 16), ValueError_call_result_251800, 'raise parameter', BaseException)
    # SSA join for if statement (line 857)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 861):
    
    # Assigning a Call to a Name (line 861):
    
    # Call to check_jac_sparsity(...): (line 861)
    # Processing the call arguments (line 861)
    # Getting the type of 'jac_sparsity' (line 861)
    jac_sparsity_251802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 46), 'jac_sparsity', False)
    # Getting the type of 'm' (line 861)
    m_251803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 60), 'm', False)
    # Getting the type of 'n' (line 861)
    n_251804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 63), 'n', False)
    # Processing the call keyword arguments (line 861)
    kwargs_251805 = {}
    # Getting the type of 'check_jac_sparsity' (line 861)
    check_jac_sparsity_251801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 27), 'check_jac_sparsity', False)
    # Calling check_jac_sparsity(args, kwargs) (line 861)
    check_jac_sparsity_call_result_251806 = invoke(stypy.reporting.localization.Localization(__file__, 861, 27), check_jac_sparsity_251801, *[jac_sparsity_251802, m_251803, n_251804], **kwargs_251805)
    
    # Assigning a type to the variable 'jac_sparsity' (line 861)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 861, 12), 'jac_sparsity', check_jac_sparsity_call_result_251806)

    @norecursion
    def jac_wrapped(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'jac_wrapped'
        module_type_store = module_type_store.open_function_context('jac_wrapped', 863, 12, False)
        
        # Passed parameters checking function
        jac_wrapped.stypy_localization = localization
        jac_wrapped.stypy_type_of_self = None
        jac_wrapped.stypy_type_store = module_type_store
        jac_wrapped.stypy_function_name = 'jac_wrapped'
        jac_wrapped.stypy_param_names_list = ['x', 'f']
        jac_wrapped.stypy_varargs_param_name = None
        jac_wrapped.stypy_kwargs_param_name = None
        jac_wrapped.stypy_call_defaults = defaults
        jac_wrapped.stypy_call_varargs = varargs
        jac_wrapped.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'jac_wrapped', ['x', 'f'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'jac_wrapped', localization, ['x', 'f'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'jac_wrapped(...)' code ##################

        
        # Assigning a Call to a Name (line 864):
        
        # Assigning a Call to a Name (line 864):
        
        # Call to approx_derivative(...): (line 864)
        # Processing the call arguments (line 864)
        # Getting the type of 'fun' (line 864)
        fun_251808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 38), 'fun', False)
        # Getting the type of 'x' (line 864)
        x_251809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 43), 'x', False)
        # Processing the call keyword arguments (line 864)
        # Getting the type of 'diff_step' (line 864)
        diff_step_251810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 55), 'diff_step', False)
        keyword_251811 = diff_step_251810
        # Getting the type of 'jac' (line 864)
        jac_251812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 73), 'jac', False)
        keyword_251813 = jac_251812
        # Getting the type of 'f' (line 865)
        f_251814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 41), 'f', False)
        keyword_251815 = f_251814
        # Getting the type of 'bounds' (line 865)
        bounds_251816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 51), 'bounds', False)
        keyword_251817 = bounds_251816
        # Getting the type of 'args' (line 865)
        args_251818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 64), 'args', False)
        keyword_251819 = args_251818
        # Getting the type of 'kwargs' (line 866)
        kwargs_251820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 45), 'kwargs', False)
        keyword_251821 = kwargs_251820
        # Getting the type of 'jac_sparsity' (line 866)
        jac_sparsity_251822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 62), 'jac_sparsity', False)
        keyword_251823 = jac_sparsity_251822
        kwargs_251824 = {'f0': keyword_251815, 'args': keyword_251819, 'bounds': keyword_251817, 'sparsity': keyword_251823, 'kwargs': keyword_251821, 'method': keyword_251813, 'rel_step': keyword_251811}
        # Getting the type of 'approx_derivative' (line 864)
        approx_derivative_251807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 20), 'approx_derivative', False)
        # Calling approx_derivative(args, kwargs) (line 864)
        approx_derivative_call_result_251825 = invoke(stypy.reporting.localization.Localization(__file__, 864, 20), approx_derivative_251807, *[fun_251808, x_251809], **kwargs_251824)
        
        # Assigning a type to the variable 'J' (line 864)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 864, 16), 'J', approx_derivative_call_result_251825)
        
        
        # Getting the type of 'J' (line 867)
        J_251826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 19), 'J')
        # Obtaining the member 'ndim' of a type (line 867)
        ndim_251827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 867, 19), J_251826, 'ndim')
        int_251828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 867, 29), 'int')
        # Applying the binary operator '!=' (line 867)
        result_ne_251829 = python_operator(stypy.reporting.localization.Localization(__file__, 867, 19), '!=', ndim_251827, int_251828)
        
        # Testing the type of an if condition (line 867)
        if_condition_251830 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 867, 16), result_ne_251829)
        # Assigning a type to the variable 'if_condition_251830' (line 867)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 867, 16), 'if_condition_251830', if_condition_251830)
        # SSA begins for if statement (line 867)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 868):
        
        # Assigning a Call to a Name (line 868):
        
        # Call to atleast_2d(...): (line 868)
        # Processing the call arguments (line 868)
        # Getting the type of 'J' (line 868)
        J_251833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 38), 'J', False)
        # Processing the call keyword arguments (line 868)
        kwargs_251834 = {}
        # Getting the type of 'np' (line 868)
        np_251831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 24), 'np', False)
        # Obtaining the member 'atleast_2d' of a type (line 868)
        atleast_2d_251832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 868, 24), np_251831, 'atleast_2d')
        # Calling atleast_2d(args, kwargs) (line 868)
        atleast_2d_call_result_251835 = invoke(stypy.reporting.localization.Localization(__file__, 868, 24), atleast_2d_251832, *[J_251833], **kwargs_251834)
        
        # Assigning a type to the variable 'J' (line 868)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 20), 'J', atleast_2d_call_result_251835)
        # SSA join for if statement (line 867)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'J' (line 870)
        J_251836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 23), 'J')
        # Assigning a type to the variable 'stypy_return_type' (line 870)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 870, 16), 'stypy_return_type', J_251836)
        
        # ################# End of 'jac_wrapped(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'jac_wrapped' in the type store
        # Getting the type of 'stypy_return_type' (line 863)
        stypy_return_type_251837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 12), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251837)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'jac_wrapped'
        return stypy_return_type_251837

    # Assigning a type to the variable 'jac_wrapped' (line 863)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 863, 12), 'jac_wrapped', jac_wrapped)
    
    # Assigning a Call to a Name (line 872):
    
    # Assigning a Call to a Name (line 872):
    
    # Call to jac_wrapped(...): (line 872)
    # Processing the call arguments (line 872)
    # Getting the type of 'x0' (line 872)
    x0_251839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 29), 'x0', False)
    # Getting the type of 'f0' (line 872)
    f0_251840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 33), 'f0', False)
    # Processing the call keyword arguments (line 872)
    kwargs_251841 = {}
    # Getting the type of 'jac_wrapped' (line 872)
    jac_wrapped_251838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 17), 'jac_wrapped', False)
    # Calling jac_wrapped(args, kwargs) (line 872)
    jac_wrapped_call_result_251842 = invoke(stypy.reporting.localization.Localization(__file__, 872, 17), jac_wrapped_251838, *[x0_251839, f0_251840], **kwargs_251841)
    
    # Assigning a type to the variable 'J0' (line 872)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 872, 12), 'J0', jac_wrapped_call_result_251842)
    # SSA join for if statement (line 846)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 826)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 874)
    # Getting the type of 'J0' (line 874)
    J0_251843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 4), 'J0')
    # Getting the type of 'None' (line 874)
    None_251844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 17), 'None')
    
    (may_be_251845, more_types_in_union_251846) = may_not_be_none(J0_251843, None_251844)

    if may_be_251845:

        if more_types_in_union_251846:
            # Runtime conditional SSA (line 874)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Getting the type of 'J0' (line 875)
        J0_251847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 11), 'J0')
        # Obtaining the member 'shape' of a type (line 875)
        shape_251848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 875, 11), J0_251847, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 875)
        tuple_251849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 875, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 875)
        # Adding element type (line 875)
        # Getting the type of 'm' (line 875)
        m_251850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 24), 'm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 875, 24), tuple_251849, m_251850)
        # Adding element type (line 875)
        # Getting the type of 'n' (line 875)
        n_251851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 27), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 875, 24), tuple_251849, n_251851)
        
        # Applying the binary operator '!=' (line 875)
        result_ne_251852 = python_operator(stypy.reporting.localization.Localization(__file__, 875, 11), '!=', shape_251848, tuple_251849)
        
        # Testing the type of an if condition (line 875)
        if_condition_251853 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 875, 8), result_ne_251852)
        # Assigning a type to the variable 'if_condition_251853' (line 875)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 875, 8), 'if_condition_251853', if_condition_251853)
        # SSA begins for if statement (line 875)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 876)
        # Processing the call arguments (line 876)
        
        # Call to format(...): (line 877)
        # Processing the call arguments (line 877)
        
        # Obtaining an instance of the builtin type 'tuple' (line 878)
        tuple_251857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 878, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 878)
        # Adding element type (line 878)
        # Getting the type of 'm' (line 878)
        m_251858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 38), 'm', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 878, 38), tuple_251857, m_251858)
        # Adding element type (line 878)
        # Getting the type of 'n' (line 878)
        n_251859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 41), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 878, 38), tuple_251857, n_251859)
        
        # Getting the type of 'J0' (line 878)
        J0_251860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 45), 'J0', False)
        # Obtaining the member 'shape' of a type (line 878)
        shape_251861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 878, 45), J0_251860, 'shape')
        # Processing the call keyword arguments (line 877)
        kwargs_251862 = {}
        str_251855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 877, 16), 'str', 'The return value of `jac` has wrong shape: expected {0}, actual {1}.')
        # Obtaining the member 'format' of a type (line 877)
        format_251856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 16), str_251855, 'format')
        # Calling format(args, kwargs) (line 877)
        format_call_result_251863 = invoke(stypy.reporting.localization.Localization(__file__, 877, 16), format_251856, *[tuple_251857, shape_251861], **kwargs_251862)
        
        # Processing the call keyword arguments (line 876)
        kwargs_251864 = {}
        # Getting the type of 'ValueError' (line 876)
        ValueError_251854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 876)
        ValueError_call_result_251865 = invoke(stypy.reporting.localization.Localization(__file__, 876, 18), ValueError_251854, *[format_call_result_251863], **kwargs_251864)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 876, 12), ValueError_call_result_251865, 'raise parameter', BaseException)
        # SSA join for if statement (line 875)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to isinstance(...): (line 880)
        # Processing the call arguments (line 880)
        # Getting the type of 'J0' (line 880)
        J0_251867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 26), 'J0', False)
        # Getting the type of 'np' (line 880)
        np_251868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 30), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 880)
        ndarray_251869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 880, 30), np_251868, 'ndarray')
        # Processing the call keyword arguments (line 880)
        kwargs_251870 = {}
        # Getting the type of 'isinstance' (line 880)
        isinstance_251866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 880)
        isinstance_call_result_251871 = invoke(stypy.reporting.localization.Localization(__file__, 880, 15), isinstance_251866, *[J0_251867, ndarray_251869], **kwargs_251870)
        
        # Applying the 'not' unary operator (line 880)
        result_not__251872 = python_operator(stypy.reporting.localization.Localization(__file__, 880, 11), 'not', isinstance_call_result_251871)
        
        # Testing the type of an if condition (line 880)
        if_condition_251873 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 880, 8), result_not__251872)
        # Assigning a type to the variable 'if_condition_251873' (line 880)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 880, 8), 'if_condition_251873', if_condition_251873)
        # SSA begins for if statement (line 880)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'method' (line 881)
        method_251874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 15), 'method')
        str_251875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 881, 25), 'str', 'lm')
        # Applying the binary operator '==' (line 881)
        result_eq_251876 = python_operator(stypy.reporting.localization.Localization(__file__, 881, 15), '==', method_251874, str_251875)
        
        # Testing the type of an if condition (line 881)
        if_condition_251877 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 881, 12), result_eq_251876)
        # Assigning a type to the variable 'if_condition_251877' (line 881)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 881, 12), 'if_condition_251877', if_condition_251877)
        # SSA begins for if statement (line 881)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 882)
        # Processing the call arguments (line 882)
        str_251879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 882, 33), 'str', "method='lm' works only with dense Jacobian matrices.")
        # Processing the call keyword arguments (line 882)
        kwargs_251880 = {}
        # Getting the type of 'ValueError' (line 882)
        ValueError_251878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 882)
        ValueError_call_result_251881 = invoke(stypy.reporting.localization.Localization(__file__, 882, 22), ValueError_251878, *[str_251879], **kwargs_251880)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 882, 16), ValueError_call_result_251881, 'raise parameter', BaseException)
        # SSA join for if statement (line 881)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'tr_solver' (line 885)
        tr_solver_251882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 15), 'tr_solver')
        str_251883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 885, 28), 'str', 'exact')
        # Applying the binary operator '==' (line 885)
        result_eq_251884 = python_operator(stypy.reporting.localization.Localization(__file__, 885, 15), '==', tr_solver_251882, str_251883)
        
        # Testing the type of an if condition (line 885)
        if_condition_251885 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 885, 12), result_eq_251884)
        # Assigning a type to the variable 'if_condition_251885' (line 885)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 885, 12), 'if_condition_251885', if_condition_251885)
        # SSA begins for if statement (line 885)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 886)
        # Processing the call arguments (line 886)
        str_251887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 887, 20), 'str', "tr_solver='exact' works only with dense Jacobian matrices.")
        # Processing the call keyword arguments (line 886)
        kwargs_251888 = {}
        # Getting the type of 'ValueError' (line 886)
        ValueError_251886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 886)
        ValueError_call_result_251889 = invoke(stypy.reporting.localization.Localization(__file__, 886, 22), ValueError_251886, *[str_251887], **kwargs_251888)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 886, 16), ValueError_call_result_251889, 'raise parameter', BaseException)
        # SSA join for if statement (line 885)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 880)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BoolOp to a Name (line 890):
        
        # Assigning a BoolOp to a Name (line 890):
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 890)
        # Processing the call arguments (line 890)
        # Getting the type of 'x_scale' (line 890)
        x_scale_251891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 31), 'x_scale', False)
        # Getting the type of 'string_types' (line 890)
        string_types_251892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 40), 'string_types', False)
        # Processing the call keyword arguments (line 890)
        kwargs_251893 = {}
        # Getting the type of 'isinstance' (line 890)
        isinstance_251890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 890)
        isinstance_call_result_251894 = invoke(stypy.reporting.localization.Localization(__file__, 890, 20), isinstance_251890, *[x_scale_251891, string_types_251892], **kwargs_251893)
        
        
        # Getting the type of 'x_scale' (line 890)
        x_scale_251895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 58), 'x_scale')
        str_251896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 890, 69), 'str', 'jac')
        # Applying the binary operator '==' (line 890)
        result_eq_251897 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 58), '==', x_scale_251895, str_251896)
        
        # Applying the binary operator 'and' (line 890)
        result_and_keyword_251898 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 20), 'and', isinstance_call_result_251894, result_eq_251897)
        
        # Assigning a type to the variable 'jac_scale' (line 890)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 890, 8), 'jac_scale', result_and_keyword_251898)
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 891)
        # Processing the call arguments (line 891)
        # Getting the type of 'J0' (line 891)
        J0_251900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 22), 'J0', False)
        # Getting the type of 'LinearOperator' (line 891)
        LinearOperator_251901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 26), 'LinearOperator', False)
        # Processing the call keyword arguments (line 891)
        kwargs_251902 = {}
        # Getting the type of 'isinstance' (line 891)
        isinstance_251899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 891)
        isinstance_call_result_251903 = invoke(stypy.reporting.localization.Localization(__file__, 891, 11), isinstance_251899, *[J0_251900, LinearOperator_251901], **kwargs_251902)
        
        # Getting the type of 'jac_scale' (line 891)
        jac_scale_251904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 46), 'jac_scale')
        # Applying the binary operator 'and' (line 891)
        result_and_keyword_251905 = python_operator(stypy.reporting.localization.Localization(__file__, 891, 11), 'and', isinstance_call_result_251903, jac_scale_251904)
        
        # Testing the type of an if condition (line 891)
        if_condition_251906 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 891, 8), result_and_keyword_251905)
        # Assigning a type to the variable 'if_condition_251906' (line 891)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 891, 8), 'if_condition_251906', if_condition_251906)
        # SSA begins for if statement (line 891)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 892)
        # Processing the call arguments (line 892)
        str_251908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 892, 29), 'str', "x_scale='jac' can't be used when `jac` returns LinearOperator.")
        # Processing the call keyword arguments (line 892)
        kwargs_251909 = {}
        # Getting the type of 'ValueError' (line 892)
        ValueError_251907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 892)
        ValueError_call_result_251910 = invoke(stypy.reporting.localization.Localization(__file__, 892, 18), ValueError_251907, *[str_251908], **kwargs_251909)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 892, 12), ValueError_call_result_251910, 'raise parameter', BaseException)
        # SSA join for if statement (line 891)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 895)
        # Getting the type of 'tr_solver' (line 895)
        tr_solver_251911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 11), 'tr_solver')
        # Getting the type of 'None' (line 895)
        None_251912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 24), 'None')
        
        (may_be_251913, more_types_in_union_251914) = may_be_none(tr_solver_251911, None_251912)

        if may_be_251913:

            if more_types_in_union_251914:
                # Runtime conditional SSA (line 895)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Call to isinstance(...): (line 896)
            # Processing the call arguments (line 896)
            # Getting the type of 'J0' (line 896)
            J0_251916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 26), 'J0', False)
            # Getting the type of 'np' (line 896)
            np_251917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 30), 'np', False)
            # Obtaining the member 'ndarray' of a type (line 896)
            ndarray_251918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 896, 30), np_251917, 'ndarray')
            # Processing the call keyword arguments (line 896)
            kwargs_251919 = {}
            # Getting the type of 'isinstance' (line 896)
            isinstance_251915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 896)
            isinstance_call_result_251920 = invoke(stypy.reporting.localization.Localization(__file__, 896, 15), isinstance_251915, *[J0_251916, ndarray_251918], **kwargs_251919)
            
            # Testing the type of an if condition (line 896)
            if_condition_251921 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 896, 12), isinstance_call_result_251920)
            # Assigning a type to the variable 'if_condition_251921' (line 896)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 896, 12), 'if_condition_251921', if_condition_251921)
            # SSA begins for if statement (line 896)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Name (line 897):
            
            # Assigning a Str to a Name (line 897):
            str_251922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 897, 28), 'str', 'exact')
            # Assigning a type to the variable 'tr_solver' (line 897)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 897, 16), 'tr_solver', str_251922)
            # SSA branch for the else part of an if statement (line 896)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Str to a Name (line 899):
            
            # Assigning a Str to a Name (line 899):
            str_251923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 28), 'str', 'lsmr')
            # Assigning a type to the variable 'tr_solver' (line 899)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 16), 'tr_solver', str_251923)
            # SSA join for if statement (line 896)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_251914:
                # SSA join for if statement (line 895)
                module_type_store = module_type_store.join_ssa_context()


        

        if more_types_in_union_251846:
            # SSA join for if statement (line 874)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'method' (line 901)
    method_251924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 7), 'method')
    str_251925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 901, 17), 'str', 'lm')
    # Applying the binary operator '==' (line 901)
    result_eq_251926 = python_operator(stypy.reporting.localization.Localization(__file__, 901, 7), '==', method_251924, str_251925)
    
    # Testing the type of an if condition (line 901)
    if_condition_251927 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 901, 4), result_eq_251926)
    # Assigning a type to the variable 'if_condition_251927' (line 901)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 901, 4), 'if_condition_251927', if_condition_251927)
    # SSA begins for if statement (line 901)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 902):
    
    # Assigning a Call to a Name (line 902):
    
    # Call to call_minpack(...): (line 902)
    # Processing the call arguments (line 902)
    # Getting the type of 'fun_wrapped' (line 902)
    fun_wrapped_251929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 30), 'fun_wrapped', False)
    # Getting the type of 'x0' (line 902)
    x0_251930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 43), 'x0', False)
    # Getting the type of 'jac_wrapped' (line 902)
    jac_wrapped_251931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 47), 'jac_wrapped', False)
    # Getting the type of 'ftol' (line 902)
    ftol_251932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 60), 'ftol', False)
    # Getting the type of 'xtol' (line 902)
    xtol_251933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 66), 'xtol', False)
    # Getting the type of 'gtol' (line 902)
    gtol_251934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 72), 'gtol', False)
    # Getting the type of 'max_nfev' (line 903)
    max_nfev_251935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 30), 'max_nfev', False)
    # Getting the type of 'x_scale' (line 903)
    x_scale_251936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 40), 'x_scale', False)
    # Getting the type of 'diff_step' (line 903)
    diff_step_251937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 49), 'diff_step', False)
    # Processing the call keyword arguments (line 902)
    kwargs_251938 = {}
    # Getting the type of 'call_minpack' (line 902)
    call_minpack_251928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 17), 'call_minpack', False)
    # Calling call_minpack(args, kwargs) (line 902)
    call_minpack_call_result_251939 = invoke(stypy.reporting.localization.Localization(__file__, 902, 17), call_minpack_251928, *[fun_wrapped_251929, x0_251930, jac_wrapped_251931, ftol_251932, xtol_251933, gtol_251934, max_nfev_251935, x_scale_251936, diff_step_251937], **kwargs_251938)
    
    # Assigning a type to the variable 'result' (line 902)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 902, 8), 'result', call_minpack_call_result_251939)
    # SSA branch for the else part of an if statement (line 901)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'method' (line 905)
    method_251940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 9), 'method')
    str_251941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 905, 19), 'str', 'trf')
    # Applying the binary operator '==' (line 905)
    result_eq_251942 = python_operator(stypy.reporting.localization.Localization(__file__, 905, 9), '==', method_251940, str_251941)
    
    # Testing the type of an if condition (line 905)
    if_condition_251943 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 905, 9), result_eq_251942)
    # Assigning a type to the variable 'if_condition_251943' (line 905)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 905, 9), 'if_condition_251943', if_condition_251943)
    # SSA begins for if statement (line 905)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 906):
    
    # Assigning a Call to a Name (line 906):
    
    # Call to trf(...): (line 906)
    # Processing the call arguments (line 906)
    # Getting the type of 'fun_wrapped' (line 906)
    fun_wrapped_251945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 21), 'fun_wrapped', False)
    # Getting the type of 'jac_wrapped' (line 906)
    jac_wrapped_251946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 34), 'jac_wrapped', False)
    # Getting the type of 'x0' (line 906)
    x0_251947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 47), 'x0', False)
    # Getting the type of 'f0' (line 906)
    f0_251948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 51), 'f0', False)
    # Getting the type of 'J0' (line 906)
    J0_251949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 55), 'J0', False)
    # Getting the type of 'lb' (line 906)
    lb_251950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 59), 'lb', False)
    # Getting the type of 'ub' (line 906)
    ub_251951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 63), 'ub', False)
    # Getting the type of 'ftol' (line 906)
    ftol_251952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 67), 'ftol', False)
    # Getting the type of 'xtol' (line 906)
    xtol_251953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 73), 'xtol', False)
    # Getting the type of 'gtol' (line 907)
    gtol_251954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 21), 'gtol', False)
    # Getting the type of 'max_nfev' (line 907)
    max_nfev_251955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 27), 'max_nfev', False)
    # Getting the type of 'x_scale' (line 907)
    x_scale_251956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 37), 'x_scale', False)
    # Getting the type of 'loss_function' (line 907)
    loss_function_251957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 46), 'loss_function', False)
    # Getting the type of 'tr_solver' (line 907)
    tr_solver_251958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 61), 'tr_solver', False)
    
    # Call to copy(...): (line 908)
    # Processing the call keyword arguments (line 908)
    kwargs_251961 = {}
    # Getting the type of 'tr_options' (line 908)
    tr_options_251959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 21), 'tr_options', False)
    # Obtaining the member 'copy' of a type (line 908)
    copy_251960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 908, 21), tr_options_251959, 'copy')
    # Calling copy(args, kwargs) (line 908)
    copy_call_result_251962 = invoke(stypy.reporting.localization.Localization(__file__, 908, 21), copy_251960, *[], **kwargs_251961)
    
    # Getting the type of 'verbose' (line 908)
    verbose_251963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 40), 'verbose', False)
    # Processing the call keyword arguments (line 906)
    kwargs_251964 = {}
    # Getting the type of 'trf' (line 906)
    trf_251944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 17), 'trf', False)
    # Calling trf(args, kwargs) (line 906)
    trf_call_result_251965 = invoke(stypy.reporting.localization.Localization(__file__, 906, 17), trf_251944, *[fun_wrapped_251945, jac_wrapped_251946, x0_251947, f0_251948, J0_251949, lb_251950, ub_251951, ftol_251952, xtol_251953, gtol_251954, max_nfev_251955, x_scale_251956, loss_function_251957, tr_solver_251958, copy_call_result_251962, verbose_251963], **kwargs_251964)
    
    # Assigning a type to the variable 'result' (line 906)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 906, 8), 'result', trf_call_result_251965)
    # SSA branch for the else part of an if statement (line 905)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'method' (line 910)
    method_251966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 910, 9), 'method')
    str_251967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 910, 19), 'str', 'dogbox')
    # Applying the binary operator '==' (line 910)
    result_eq_251968 = python_operator(stypy.reporting.localization.Localization(__file__, 910, 9), '==', method_251966, str_251967)
    
    # Testing the type of an if condition (line 910)
    if_condition_251969 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 910, 9), result_eq_251968)
    # Assigning a type to the variable 'if_condition_251969' (line 910)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 910, 9), 'if_condition_251969', if_condition_251969)
    # SSA begins for if statement (line 910)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'tr_solver' (line 911)
    tr_solver_251970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 11), 'tr_solver')
    str_251971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 911, 24), 'str', 'lsmr')
    # Applying the binary operator '==' (line 911)
    result_eq_251972 = python_operator(stypy.reporting.localization.Localization(__file__, 911, 11), '==', tr_solver_251970, str_251971)
    
    
    str_251973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 911, 35), 'str', 'regularize')
    # Getting the type of 'tr_options' (line 911)
    tr_options_251974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 51), 'tr_options')
    # Applying the binary operator 'in' (line 911)
    result_contains_251975 = python_operator(stypy.reporting.localization.Localization(__file__, 911, 35), 'in', str_251973, tr_options_251974)
    
    # Applying the binary operator 'and' (line 911)
    result_and_keyword_251976 = python_operator(stypy.reporting.localization.Localization(__file__, 911, 11), 'and', result_eq_251972, result_contains_251975)
    
    # Testing the type of an if condition (line 911)
    if_condition_251977 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 911, 8), result_and_keyword_251976)
    # Assigning a type to the variable 'if_condition_251977' (line 911)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 911, 8), 'if_condition_251977', if_condition_251977)
    # SSA begins for if statement (line 911)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 912)
    # Processing the call arguments (line 912)
    str_251979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 912, 17), 'str', "The keyword 'regularize' in `tr_options` is not relevant for 'dogbox' method.")
    # Processing the call keyword arguments (line 912)
    kwargs_251980 = {}
    # Getting the type of 'warn' (line 912)
    warn_251978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 12), 'warn', False)
    # Calling warn(args, kwargs) (line 912)
    warn_call_result_251981 = invoke(stypy.reporting.localization.Localization(__file__, 912, 12), warn_251978, *[str_251979], **kwargs_251980)
    
    
    # Assigning a Call to a Name (line 914):
    
    # Assigning a Call to a Name (line 914):
    
    # Call to copy(...): (line 914)
    # Processing the call keyword arguments (line 914)
    kwargs_251984 = {}
    # Getting the type of 'tr_options' (line 914)
    tr_options_251982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 25), 'tr_options', False)
    # Obtaining the member 'copy' of a type (line 914)
    copy_251983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 914, 25), tr_options_251982, 'copy')
    # Calling copy(args, kwargs) (line 914)
    copy_call_result_251985 = invoke(stypy.reporting.localization.Localization(__file__, 914, 25), copy_251983, *[], **kwargs_251984)
    
    # Assigning a type to the variable 'tr_options' (line 914)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 914, 12), 'tr_options', copy_call_result_251985)
    # Deleting a member
    # Getting the type of 'tr_options' (line 915)
    tr_options_251986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 16), 'tr_options')
    
    # Obtaining the type of the subscript
    str_251987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 915, 27), 'str', 'regularize')
    # Getting the type of 'tr_options' (line 915)
    tr_options_251988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 16), 'tr_options')
    # Obtaining the member '__getitem__' of a type (line 915)
    getitem___251989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 915, 16), tr_options_251988, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 915)
    subscript_call_result_251990 = invoke(stypy.reporting.localization.Localization(__file__, 915, 16), getitem___251989, str_251987)
    
    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 915, 12), tr_options_251986, subscript_call_result_251990)
    # SSA join for if statement (line 911)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 917):
    
    # Assigning a Call to a Name (line 917):
    
    # Call to dogbox(...): (line 917)
    # Processing the call arguments (line 917)
    # Getting the type of 'fun_wrapped' (line 917)
    fun_wrapped_251992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 24), 'fun_wrapped', False)
    # Getting the type of 'jac_wrapped' (line 917)
    jac_wrapped_251993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 37), 'jac_wrapped', False)
    # Getting the type of 'x0' (line 917)
    x0_251994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 50), 'x0', False)
    # Getting the type of 'f0' (line 917)
    f0_251995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 54), 'f0', False)
    # Getting the type of 'J0' (line 917)
    J0_251996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 58), 'J0', False)
    # Getting the type of 'lb' (line 917)
    lb_251997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 62), 'lb', False)
    # Getting the type of 'ub' (line 917)
    ub_251998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 66), 'ub', False)
    # Getting the type of 'ftol' (line 917)
    ftol_251999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 70), 'ftol', False)
    # Getting the type of 'xtol' (line 918)
    xtol_252000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 24), 'xtol', False)
    # Getting the type of 'gtol' (line 918)
    gtol_252001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 30), 'gtol', False)
    # Getting the type of 'max_nfev' (line 918)
    max_nfev_252002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 36), 'max_nfev', False)
    # Getting the type of 'x_scale' (line 918)
    x_scale_252003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 46), 'x_scale', False)
    # Getting the type of 'loss_function' (line 918)
    loss_function_252004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 55), 'loss_function', False)
    # Getting the type of 'tr_solver' (line 919)
    tr_solver_252005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 24), 'tr_solver', False)
    # Getting the type of 'tr_options' (line 919)
    tr_options_252006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 35), 'tr_options', False)
    # Getting the type of 'verbose' (line 919)
    verbose_252007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 47), 'verbose', False)
    # Processing the call keyword arguments (line 917)
    kwargs_252008 = {}
    # Getting the type of 'dogbox' (line 917)
    dogbox_251991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 17), 'dogbox', False)
    # Calling dogbox(args, kwargs) (line 917)
    dogbox_call_result_252009 = invoke(stypy.reporting.localization.Localization(__file__, 917, 17), dogbox_251991, *[fun_wrapped_251992, jac_wrapped_251993, x0_251994, f0_251995, J0_251996, lb_251997, ub_251998, ftol_251999, xtol_252000, gtol_252001, max_nfev_252002, x_scale_252003, loss_function_252004, tr_solver_252005, tr_options_252006, verbose_252007], **kwargs_252008)
    
    # Assigning a type to the variable 'result' (line 917)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 917, 8), 'result', dogbox_call_result_252009)
    # SSA join for if statement (line 910)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 905)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 901)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Attribute (line 921):
    
    # Assigning a Subscript to a Attribute (line 921):
    
    # Obtaining the type of the subscript
    # Getting the type of 'result' (line 921)
    result_252010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 42), 'result')
    # Obtaining the member 'status' of a type (line 921)
    status_252011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 921, 42), result_252010, 'status')
    # Getting the type of 'TERMINATION_MESSAGES' (line 921)
    TERMINATION_MESSAGES_252012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 21), 'TERMINATION_MESSAGES')
    # Obtaining the member '__getitem__' of a type (line 921)
    getitem___252013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 921, 21), TERMINATION_MESSAGES_252012, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 921)
    subscript_call_result_252014 = invoke(stypy.reporting.localization.Localization(__file__, 921, 21), getitem___252013, status_252011)
    
    # Getting the type of 'result' (line 921)
    result_252015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 4), 'result')
    # Setting the type of the member 'message' of a type (line 921)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 921, 4), result_252015, 'message', subscript_call_result_252014)
    
    # Assigning a Compare to a Attribute (line 922):
    
    # Assigning a Compare to a Attribute (line 922):
    
    # Getting the type of 'result' (line 922)
    result_252016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 21), 'result')
    # Obtaining the member 'status' of a type (line 922)
    status_252017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 21), result_252016, 'status')
    int_252018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 37), 'int')
    # Applying the binary operator '>' (line 922)
    result_gt_252019 = python_operator(stypy.reporting.localization.Localization(__file__, 922, 21), '>', status_252017, int_252018)
    
    # Getting the type of 'result' (line 922)
    result_252020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 4), 'result')
    # Setting the type of the member 'success' of a type (line 922)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 4), result_252020, 'success', result_gt_252019)
    
    
    # Getting the type of 'verbose' (line 924)
    verbose_252021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 7), 'verbose')
    int_252022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 924, 18), 'int')
    # Applying the binary operator '>=' (line 924)
    result_ge_252023 = python_operator(stypy.reporting.localization.Localization(__file__, 924, 7), '>=', verbose_252021, int_252022)
    
    # Testing the type of an if condition (line 924)
    if_condition_252024 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 924, 4), result_ge_252023)
    # Assigning a type to the variable 'if_condition_252024' (line 924)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 4), 'if_condition_252024', if_condition_252024)
    # SSA begins for if statement (line 924)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 925)
    # Processing the call arguments (line 925)
    # Getting the type of 'result' (line 925)
    result_252026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 14), 'result', False)
    # Obtaining the member 'message' of a type (line 925)
    message_252027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 925, 14), result_252026, 'message')
    # Processing the call keyword arguments (line 925)
    kwargs_252028 = {}
    # Getting the type of 'print' (line 925)
    print_252025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 8), 'print', False)
    # Calling print(args, kwargs) (line 925)
    print_call_result_252029 = invoke(stypy.reporting.localization.Localization(__file__, 925, 8), print_252025, *[message_252027], **kwargs_252028)
    
    
    # Call to print(...): (line 926)
    # Processing the call arguments (line 926)
    
    # Call to format(...): (line 926)
    # Processing the call arguments (line 926)
    # Getting the type of 'result' (line 928)
    result_252033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 22), 'result', False)
    # Obtaining the member 'nfev' of a type (line 928)
    nfev_252034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 928, 22), result_252033, 'nfev')
    # Getting the type of 'initial_cost' (line 928)
    initial_cost_252035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 35), 'initial_cost', False)
    # Getting the type of 'result' (line 928)
    result_252036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 49), 'result', False)
    # Obtaining the member 'cost' of a type (line 928)
    cost_252037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 928, 49), result_252036, 'cost')
    # Getting the type of 'result' (line 929)
    result_252038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 22), 'result', False)
    # Obtaining the member 'optimality' of a type (line 929)
    optimality_252039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 929, 22), result_252038, 'optimality')
    # Processing the call keyword arguments (line 926)
    kwargs_252040 = {}
    str_252031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 926, 14), 'str', 'Function evaluations {0}, initial cost {1:.4e}, final cost {2:.4e}, first-order optimality {3:.2e}.')
    # Obtaining the member 'format' of a type (line 926)
    format_252032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 926, 14), str_252031, 'format')
    # Calling format(args, kwargs) (line 926)
    format_call_result_252041 = invoke(stypy.reporting.localization.Localization(__file__, 926, 14), format_252032, *[nfev_252034, initial_cost_252035, cost_252037, optimality_252039], **kwargs_252040)
    
    # Processing the call keyword arguments (line 926)
    kwargs_252042 = {}
    # Getting the type of 'print' (line 926)
    print_252030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 8), 'print', False)
    # Calling print(args, kwargs) (line 926)
    print_call_result_252043 = invoke(stypy.reporting.localization.Localization(__file__, 926, 8), print_252030, *[format_call_result_252041], **kwargs_252042)
    
    # SSA join for if statement (line 924)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'result' (line 931)
    result_252044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 931)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 931, 4), 'stypy_return_type', result_252044)
    
    # ################# End of 'least_squares(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'least_squares' in the type store
    # Getting the type of 'stypy_return_type' (line 234)
    stypy_return_type_252045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_252045)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'least_squares'
    return stypy_return_type_252045

# Assigning a type to the variable 'least_squares' (line 234)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 0), 'least_squares', least_squares)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
