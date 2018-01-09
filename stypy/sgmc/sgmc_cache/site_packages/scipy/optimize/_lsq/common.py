
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Functions used by least-squares algorithms.'''
2: from __future__ import division, print_function, absolute_import
3: 
4: from math import copysign
5: 
6: import numpy as np
7: from numpy.linalg import norm
8: 
9: from scipy.linalg import cho_factor, cho_solve, LinAlgError
10: from scipy.sparse import issparse
11: from scipy.sparse.linalg import LinearOperator, aslinearoperator
12: 
13: 
14: EPS = np.finfo(float).eps
15: 
16: 
17: # Functions related to a trust-region problem.
18: 
19: 
20: def intersect_trust_region(x, s, Delta):
21:     '''Find the intersection of a line with the boundary of a trust region.
22:     
23:     This function solves the quadratic equation with respect to t
24:     ||(x + s*t)||**2 = Delta**2.
25:     
26:     Returns
27:     -------
28:     t_neg, t_pos : tuple of float
29:         Negative and positive roots.
30:     
31:     Raises
32:     ------
33:     ValueError
34:         If `s` is zero or `x` is not within the trust region.
35:     '''
36:     a = np.dot(s, s)
37:     if a == 0:
38:         raise ValueError("`s` is zero.")
39: 
40:     b = np.dot(x, s)
41: 
42:     c = np.dot(x, x) - Delta**2
43:     if c > 0:
44:         raise ValueError("`x` is not within the trust region.")
45: 
46:     d = np.sqrt(b*b - a*c)  # Root from one fourth of the discriminant.
47: 
48:     # Computations below avoid loss of significance, see "Numerical Recipes".
49:     q = -(b + copysign(d, b))
50:     t1 = q / a
51:     t2 = c / q
52: 
53:     if t1 < t2:
54:         return t1, t2
55:     else:
56:         return t2, t1
57: 
58: 
59: def solve_lsq_trust_region(n, m, uf, s, V, Delta, initial_alpha=None,
60:                            rtol=0.01, max_iter=10):
61:     '''Solve a trust-region problem arising in least-squares minimization.
62:     
63:     This function implements a method described by J. J. More [1]_ and used
64:     in MINPACK, but it relies on a single SVD of Jacobian instead of series
65:     of Cholesky decompositions. Before running this function, compute:
66:     ``U, s, VT = svd(J, full_matrices=False)``.
67:     
68:     Parameters
69:     ----------
70:     n : int
71:         Number of variables.
72:     m : int
73:         Number of residuals.
74:     uf : ndarray
75:         Computed as U.T.dot(f).
76:     s : ndarray
77:         Singular values of J.
78:     V : ndarray
79:         Transpose of VT.
80:     Delta : float
81:         Radius of a trust region.
82:     initial_alpha : float, optional
83:         Initial guess for alpha, which might be available from a previous
84:         iteration. If None, determined automatically.
85:     rtol : float, optional
86:         Stopping tolerance for the root-finding procedure. Namely, the
87:         solution ``p`` will satisfy ``abs(norm(p) - Delta) < rtol * Delta``.
88:     max_iter : int, optional
89:         Maximum allowed number of iterations for the root-finding procedure.
90:     
91:     Returns
92:     -------
93:     p : ndarray, shape (n,)
94:         Found solution of a trust-region problem.
95:     alpha : float
96:         Positive value such that (J.T*J + alpha*I)*p = -J.T*f.
97:         Sometimes called Levenberg-Marquardt parameter.
98:     n_iter : int
99:         Number of iterations made by root-finding procedure. Zero means
100:         that Gauss-Newton step was selected as the solution.
101:     
102:     References
103:     ----------
104:     .. [1] More, J. J., "The Levenberg-Marquardt Algorithm: Implementation
105:            and Theory," Numerical Analysis, ed. G. A. Watson, Lecture Notes
106:            in Mathematics 630, Springer Verlag, pp. 105-116, 1977.
107:     '''
108:     def phi_and_derivative(alpha, suf, s, Delta):
109:         '''Function of which to find zero.
110:         
111:         It is defined as "norm of regularized (by alpha) least-squares
112:         solution minus `Delta`". Refer to [1]_.
113:         '''
114:         denom = s**2 + alpha
115:         p_norm = norm(suf / denom)
116:         phi = p_norm - Delta
117:         phi_prime = -np.sum(suf ** 2 / denom**3) / p_norm
118:         return phi, phi_prime
119: 
120:     suf = s * uf
121: 
122:     # Check if J has full rank and try Gauss-Newton step.
123:     if m >= n:
124:         threshold = EPS * m * s[0]
125:         full_rank = s[-1] > threshold
126:     else:
127:         full_rank = False
128: 
129:     if full_rank:
130:         p = -V.dot(uf / s)
131:         if norm(p) <= Delta:
132:             return p, 0.0, 0
133: 
134:     alpha_upper = norm(suf) / Delta
135: 
136:     if full_rank:
137:         phi, phi_prime = phi_and_derivative(0.0, suf, s, Delta)
138:         alpha_lower = -phi / phi_prime
139:     else:
140:         alpha_lower = 0.0
141: 
142:     if initial_alpha is None or not full_rank and initial_alpha == 0:
143:         alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper)**0.5)
144:     else:
145:         alpha = initial_alpha
146: 
147:     for it in range(max_iter):
148:         if alpha < alpha_lower or alpha > alpha_upper:
149:             alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper)**0.5)
150: 
151:         phi, phi_prime = phi_and_derivative(alpha, suf, s, Delta)
152: 
153:         if phi < 0:
154:             alpha_upper = alpha
155: 
156:         ratio = phi / phi_prime
157:         alpha_lower = max(alpha_lower, alpha - ratio)
158:         alpha -= (phi + Delta) * ratio / Delta
159: 
160:         if np.abs(phi) < rtol * Delta:
161:             break
162: 
163:     p = -V.dot(suf / (s**2 + alpha))
164: 
165:     # Make the norm of p equal to Delta, p is changed only slightly during
166:     # this. It is done to prevent p lie outside the trust region (which can
167:     # cause problems later).
168:     p *= Delta / norm(p)
169: 
170:     return p, alpha, it + 1
171: 
172: 
173: def solve_trust_region_2d(B, g, Delta):
174:     '''Solve a general trust-region problem in 2 dimensions.
175:     
176:     The problem is reformulated as a 4-th order algebraic equation,
177:     the solution of which is found by numpy.roots.
178:     
179:     Parameters
180:     ----------
181:     B : ndarray, shape (2, 2)
182:         Symmetric matrix, defines a quadratic term of the function.
183:     g : ndarray, shape (2,)
184:         Defines a linear term of the function.
185:     Delta : float
186:         Radius of a trust region.
187:     
188:     Returns
189:     -------
190:     p : ndarray, shape (2,)
191:         Found solution.
192:     newton_step : bool
193:         Whether the returned solution is the Newton step which lies within
194:         the trust region.
195:     '''
196:     try:
197:         R, lower = cho_factor(B)
198:         p = -cho_solve((R, lower), g)
199:         if np.dot(p, p) <= Delta**2:
200:             return p, True
201:     except LinAlgError:
202:         pass
203: 
204:     a = B[0, 0] * Delta**2
205:     b = B[0, 1] * Delta**2
206:     c = B[1, 1] * Delta**2
207: 
208:     d = g[0] * Delta
209:     f = g[1] * Delta
210: 
211:     coeffs = np.array(
212:         [-b + d, 2 * (a - c + f), 6 * b, 2 * (-a + c + f), -b - d])
213:     t = np.roots(coeffs)  # Can handle leading zeros.
214:     t = np.real(t[np.isreal(t)])
215: 
216:     p = Delta * np.vstack((2 * t / (1 + t**2), (1 - t**2) / (1 + t**2)))
217:     value = 0.5 * np.sum(p * B.dot(p), axis=0) + np.dot(g, p)
218:     i = np.argmin(value)
219:     p = p[:, i]
220: 
221:     return p, False
222: 
223: 
224: def update_tr_radius(Delta, actual_reduction, predicted_reduction,
225:                      step_norm, bound_hit):
226:     '''Update the radius of a trust region based on the cost reduction.
227: 
228:     Returns
229:     -------
230:     Delta : float
231:         New radius.
232:     ratio : float
233:         Ratio between actual and predicted reductions. Zero if predicted
234:         reduction is zero.
235:     '''
236:     if predicted_reduction > 0:
237:         ratio = actual_reduction / predicted_reduction
238:     else:
239:         ratio = 0
240: 
241:     if ratio < 0.25:
242:         Delta = 0.25 * step_norm
243:     elif ratio > 0.75 and bound_hit:
244:         Delta *= 2.0
245: 
246:     return Delta, ratio
247: 
248: 
249: # Construction and minimization of quadratic functions.
250: 
251: 
252: def build_quadratic_1d(J, g, s, diag=None, s0=None):
253:     '''Parameterize a multivariate quadratic function along a line.
254:     
255:     The resulting univariate quadratic function is given as follows:
256:     ::
257:         f(t) = 0.5 * (s0 + s*t).T * (J.T*J + diag) * (s0 + s*t) +
258:                g.T * (s0 + s*t)
259:     
260:     Parameters
261:     ----------
262:     J : ndarray, sparse matrix or LinearOperator shape (m, n)
263:         Jacobian matrix, affects the quadratic term.
264:     g : ndarray, shape (n,)
265:         Gradient, defines the linear term.
266:     s : ndarray, shape (n,)
267:         Direction vector of a line.
268:     diag : None or ndarray with shape (n,), optional
269:         Addition diagonal part, affects the quadratic term.
270:         If None, assumed to be 0.
271:     s0 : None or ndarray with shape (n,), optional
272:         Initial point. If None, assumed to be 0.
273:     
274:     Returns
275:     -------
276:     a : float
277:         Coefficient for t**2.
278:     b : float
279:         Coefficient for t.
280:     c : float
281:         Free term. Returned only if `s0` is provided.
282:     '''
283:     v = J.dot(s)
284:     a = np.dot(v, v)
285:     if diag is not None:
286:         a += np.dot(s * diag, s)
287:     a *= 0.5
288: 
289:     b = np.dot(g, s)
290: 
291:     if s0 is not None:
292:         u = J.dot(s0)
293:         b += np.dot(u, v)
294:         c = 0.5 * np.dot(u, u) + np.dot(g, s0)
295:         if diag is not None:
296:             b += np.dot(s0 * diag, s)
297:             c += 0.5 * np.dot(s0 * diag, s0)
298:         return a, b, c
299:     else:
300:         return a, b
301: 
302: 
303: def minimize_quadratic_1d(a, b, lb, ub, c=0):
304:     '''Minimize a 1-d quadratic function subject to bounds.
305:     
306:     The free term `c` is 0 by default. Bounds must be finite.
307:     
308:     Returns
309:     -------
310:     t : float
311:         Minimum point.
312:     y : float
313:         Minimum value.
314:     '''
315:     t = [lb, ub]
316:     if a != 0:
317:         extremum = -0.5 * b / a
318:         if lb < extremum < ub:
319:             t.append(extremum)
320:     t = np.asarray(t)
321:     y = a * t**2 + b * t + c
322:     min_index = np.argmin(y)
323:     return t[min_index], y[min_index]
324: 
325: 
326: def evaluate_quadratic(J, g, s, diag=None):
327:     '''Compute values of a quadratic function arising in least squares.
328:     
329:     The function is 0.5 * s.T * (J.T * J + diag) * s + g.T * s.
330:     
331:     Parameters
332:     ----------
333:     J : ndarray, sparse matrix or LinearOperator, shape (m, n)
334:         Jacobian matrix, affects the quadratic term.
335:     g : ndarray, shape (n,)
336:         Gradient, defines the linear term.
337:     s : ndarray, shape (k, n) or (n,)
338:         Array containing steps as rows.
339:     diag : ndarray, shape (n,), optional
340:         Addition diagonal part, affects the quadratic term.
341:         If None, assumed to be 0.
342:     
343:     Returns
344:     -------
345:     values : ndarray with shape (k,) or float
346:         Values of the function. If `s` was 2-dimensional then ndarray is
347:         returned, otherwise float is returned.
348:     '''
349:     if s.ndim == 1:
350:         Js = J.dot(s)
351:         q = np.dot(Js, Js)
352:         if diag is not None:
353:             q += np.dot(s * diag, s)
354:     else:
355:         Js = J.dot(s.T)
356:         q = np.sum(Js**2, axis=0)
357:         if diag is not None:
358:             q += np.sum(diag * s**2, axis=1)
359: 
360:     l = np.dot(s, g)
361: 
362:     return 0.5 * q + l
363: 
364: 
365: # Utility functions to work with bound constraints.
366: 
367: 
368: def in_bounds(x, lb, ub):
369:     '''Check if a point lies within bounds.'''
370:     return np.all((x >= lb) & (x <= ub))
371: 
372: 
373: def step_size_to_bound(x, s, lb, ub):
374:     '''Compute a min_step size required to reach a bound.
375:     
376:     The function computes a positive scalar t, such that x + s * t is on
377:     the bound.
378:     
379:     Returns
380:     -------
381:     step : float
382:         Computed step. Non-negative value.
383:     hits : ndarray of int with shape of x
384:         Each element indicates whether a corresponding variable reaches the
385:         bound:
386:              
387:              *  0 - the bound was not hit.
388:              * -1 - the lower bound was hit.
389:              *  1 - the upper bound was hit.
390:     '''
391:     non_zero = np.nonzero(s)
392:     s_non_zero = s[non_zero]
393:     steps = np.empty_like(x)
394:     steps.fill(np.inf)
395:     with np.errstate(over='ignore'):
396:         steps[non_zero] = np.maximum((lb - x)[non_zero] / s_non_zero,
397:                                      (ub - x)[non_zero] / s_non_zero)
398:     min_step = np.min(steps)
399:     return min_step, np.equal(steps, min_step) * np.sign(s).astype(int)
400: 
401: 
402: def find_active_constraints(x, lb, ub, rtol=1e-10):
403:     '''Determine which constraints are active in a given point.
404:     
405:     The threshold is computed using `rtol` and the absolute value of the
406:     closest bound.
407:     
408:     Returns
409:     -------
410:     active : ndarray of int with shape of x
411:         Each component shows whether the corresponding constraint is active:
412:              
413:              *  0 - a constraint is not active.
414:              * -1 - a lower bound is active.
415:              *  1 - a upper bound is active.
416:     '''
417:     active = np.zeros_like(x, dtype=int)
418: 
419:     if rtol == 0:
420:         active[x <= lb] = -1
421:         active[x >= ub] = 1
422:         return active
423: 
424:     lower_dist = x - lb
425:     upper_dist = ub - x
426: 
427:     lower_threshold = rtol * np.maximum(1, np.abs(lb))
428:     upper_threshold = rtol * np.maximum(1, np.abs(ub))
429: 
430:     lower_active = (np.isfinite(lb) &
431:                     (lower_dist <= np.minimum(upper_dist, lower_threshold)))
432:     active[lower_active] = -1
433: 
434:     upper_active = (np.isfinite(ub) &
435:                     (upper_dist <= np.minimum(lower_dist, upper_threshold)))
436:     active[upper_active] = 1
437: 
438:     return active
439: 
440: 
441: def make_strictly_feasible(x, lb, ub, rstep=1e-10):
442:     '''Shift a point to the interior of a feasible region.
443:     
444:     Each element of the returned vector is at least at a relative distance
445:     `rstep` from the closest bound. If ``rstep=0`` then `np.nextafter` is used.
446:     '''
447:     x_new = x.copy()
448: 
449:     active = find_active_constraints(x, lb, ub, rstep)
450:     lower_mask = np.equal(active, -1)
451:     upper_mask = np.equal(active, 1)
452: 
453:     if rstep == 0:
454:         x_new[lower_mask] = np.nextafter(lb[lower_mask], ub[lower_mask])
455:         x_new[upper_mask] = np.nextafter(ub[upper_mask], lb[upper_mask])
456:     else:
457:         x_new[lower_mask] = (lb[lower_mask] +
458:                              rstep * np.maximum(1, np.abs(lb[lower_mask])))
459:         x_new[upper_mask] = (ub[upper_mask] -
460:                              rstep * np.maximum(1, np.abs(ub[upper_mask])))
461: 
462:     tight_bounds = (x_new < lb) | (x_new > ub)
463:     x_new[tight_bounds] = 0.5 * (lb[tight_bounds] + ub[tight_bounds])
464: 
465:     return x_new
466: 
467:  
468: def CL_scaling_vector(x, g, lb, ub):
469:     '''Compute Coleman-Li scaling vector and its derivatives.
470:     
471:     Components of a vector v are defined as follows:
472:     ::
473:                | ub[i] - x[i], if g[i] < 0 and ub[i] < np.inf
474:         v[i] = | x[i] - lb[i], if g[i] > 0 and lb[i] > -np.inf
475:                | 1,           otherwise
476:     
477:     According to this definition v[i] >= 0 for all i. It differs from the
478:     definition in paper [1]_ (eq. (2.2)), where the absolute value of v is
479:     used. Both definitions are equivalent down the line.
480:     Derivatives of v with respect to x take value 1, -1 or 0 depending on a
481:     case.
482:     
483:     Returns
484:     -------
485:     v : ndarray with shape of x
486:         Scaling vector.
487:     dv : ndarray with shape of x
488:         Derivatives of v[i] with respect to x[i], diagonal elements of v's
489:         Jacobian.
490:     
491:     References
492:     ----------
493:     .. [1] M.A. Branch, T.F. Coleman, and Y. Li, "A Subspace, Interior,
494:            and Conjugate Gradient Method for Large-Scale Bound-Constrained
495:            Minimization Problems," SIAM Journal on Scientific Computing,
496:            Vol. 21, Number 1, pp 1-23, 1999.
497:     '''
498:     v = np.ones_like(x)
499:     dv = np.zeros_like(x)
500: 
501:     mask = (g < 0) & np.isfinite(ub)
502:     v[mask] = ub[mask] - x[mask]
503:     dv[mask] = -1
504: 
505:     mask = (g > 0) & np.isfinite(lb)
506:     v[mask] = x[mask] - lb[mask]
507:     dv[mask] = 1
508: 
509:     return v, dv
510: 
511: 
512: def reflective_transformation(y, lb, ub):
513:     '''Compute reflective transformation and its gradient.'''
514:     if in_bounds(y, lb, ub):
515:         return y, np.ones_like(y)
516: 
517:     lb_finite = np.isfinite(lb)
518:     ub_finite = np.isfinite(ub)
519: 
520:     x = y.copy()
521:     g_negative = np.zeros_like(y, dtype=bool)
522: 
523:     mask = lb_finite & ~ub_finite
524:     x[mask] = np.maximum(y[mask], 2 * lb[mask] - y[mask])
525:     g_negative[mask] = y[mask] < lb[mask]
526: 
527:     mask = ~lb_finite & ub_finite
528:     x[mask] = np.minimum(y[mask], 2 * ub[mask] - y[mask])
529:     g_negative[mask] = y[mask] > ub[mask]
530: 
531:     mask = lb_finite & ub_finite
532:     d = ub - lb
533:     t = np.remainder(y[mask] - lb[mask], 2 * d[mask])
534:     x[mask] = lb[mask] + np.minimum(t, 2 * d[mask] - t)
535:     g_negative[mask] = t > d[mask]
536: 
537:     g = np.ones_like(y)
538:     g[g_negative] = -1
539: 
540:     return x, g
541: 
542: 
543: # Functions to display algorithm's progress.
544: 
545: 
546: def print_header_nonlinear():
547:     print("{0:^15}{1:^15}{2:^15}{3:^15}{4:^15}{5:^15}"
548:           .format("Iteration", "Total nfev", "Cost", "Cost reduction",
549:                   "Step norm", "Optimality"))
550: 
551: 
552: def print_iteration_nonlinear(iteration, nfev, cost, cost_reduction,
553:                               step_norm, optimality):
554:     if cost_reduction is None:
555:         cost_reduction = " " * 15
556:     else:
557:         cost_reduction = "{0:^15.2e}".format(cost_reduction)
558: 
559:     if step_norm is None:
560:         step_norm = " " * 15
561:     else:
562:         step_norm = "{0:^15.2e}".format(step_norm)
563: 
564:     print("{0:^15}{1:^15}{2:^15.4e}{3}{4}{5:^15.2e}"
565:           .format(iteration, nfev, cost, cost_reduction,
566:                   step_norm, optimality))
567: 
568: 
569: def print_header_linear():
570:     print("{0:^15}{1:^15}{2:^15}{3:^15}{4:^15}"
571:           .format("Iteration", "Cost", "Cost reduction", "Step norm",
572:                   "Optimality"))
573: 
574: 
575: def print_iteration_linear(iteration, cost, cost_reduction, step_norm,
576:                            optimality):
577:     if cost_reduction is None:
578:         cost_reduction = " " * 15
579:     else:
580:         cost_reduction = "{0:^15.2e}".format(cost_reduction)
581: 
582:     if step_norm is None:
583:         step_norm = " " * 15
584:     else:
585:         step_norm = "{0:^15.2e}".format(step_norm)
586: 
587:     print("{0:^15}{1:^15.4e}{2}{3}{4:^15.2e}".format(
588:         iteration, cost, cost_reduction, step_norm, optimality))
589: 
590: 
591: # Simple helper functions.
592: 
593: 
594: def compute_grad(J, f):
595:     '''Compute gradient of the least-squares cost function.'''
596:     if isinstance(J, LinearOperator):
597:         return J.rmatvec(f)
598:     else:
599:         return J.T.dot(f)
600: 
601: 
602: def compute_jac_scale(J, scale_inv_old=None):
603:     '''Compute variables scale based on the Jacobian matrix.'''
604:     if issparse(J):
605:         scale_inv = np.asarray(J.power(2).sum(axis=0)).ravel()**0.5
606:     else:
607:         scale_inv = np.sum(J**2, axis=0)**0.5
608: 
609:     if scale_inv_old is None:
610:         scale_inv[scale_inv == 0] = 1
611:     else:
612:         scale_inv = np.maximum(scale_inv, scale_inv_old)
613: 
614:     return 1 / scale_inv, scale_inv
615: 
616: 
617: def left_multiplied_operator(J, d):
618:     '''Return diag(d) J as LinearOperator.'''
619:     J = aslinearoperator(J)
620: 
621:     def matvec(x):
622:         return d * J.matvec(x)
623: 
624:     def matmat(X):
625:         return d * J.matmat(X)
626: 
627:     def rmatvec(x):
628:         return J.rmatvec(x.ravel() * d)
629: 
630:     return LinearOperator(J.shape, matvec=matvec, matmat=matmat,
631:                           rmatvec=rmatvec)
632: 
633: 
634: def right_multiplied_operator(J, d):
635:     '''Return J diag(d) as LinearOperator.'''
636:     J = aslinearoperator(J)
637: 
638:     def matvec(x):
639:         return J.matvec(np.ravel(x) * d)
640: 
641:     def matmat(X):
642:         return J.matmat(X * d[:, np.newaxis])
643: 
644:     def rmatvec(x):
645:         return d * J.rmatvec(x)
646: 
647:     return LinearOperator(J.shape, matvec=matvec, matmat=matmat,
648:                           rmatvec=rmatvec)
649: 
650: 
651: def regularized_lsq_operator(J, diag):
652:     '''Return a matrix arising in regularized least squares as LinearOperator.
653:     
654:     The matrix is
655:         [ J ]
656:         [ D ]
657:     where D is diagonal matrix with elements from `diag`.
658:     '''
659:     J = aslinearoperator(J)
660:     m, n = J.shape
661: 
662:     def matvec(x):
663:         return np.hstack((J.matvec(x), diag * x))
664: 
665:     def rmatvec(x):
666:         x1 = x[:m]
667:         x2 = x[m:]
668:         return J.rmatvec(x1) + diag * x2
669: 
670:     return LinearOperator((m + n, n), matvec=matvec, rmatvec=rmatvec)
671: 
672: 
673: def right_multiply(J, d, copy=True):
674:     '''Compute J diag(d).
675:     
676:     If `copy` is False, `J` is modified in place (unless being LinearOperator).
677:     '''
678:     if copy and not isinstance(J, LinearOperator):
679:         J = J.copy()
680: 
681:     if issparse(J):
682:         J.data *= d.take(J.indices, mode='clip')  # scikit-learn recipe.
683:     elif isinstance(J, LinearOperator):
684:         J = right_multiplied_operator(J, d)
685:     else:
686:         J *= d
687: 
688:     return J
689: 
690: 
691: def left_multiply(J, d, copy=True):
692:     '''Compute diag(d) J.
693:     
694:     If `copy` is False, `J` is modified in place (unless being LinearOperator).
695:     '''
696:     if copy and not isinstance(J, LinearOperator):
697:         J = J.copy()
698: 
699:     if issparse(J):
700:         J.data *= np.repeat(d, np.diff(J.indptr))  # scikit-learn recipe.
701:     elif isinstance(J, LinearOperator):
702:         J = left_multiplied_operator(J, d)
703:     else:
704:         J *= d[:, np.newaxis]
705: 
706:     return J
707: 
708: 
709: def check_termination(dF, F, dx_norm, x_norm, ratio, ftol, xtol):
710:     '''Check termination condition for nonlinear least squares.'''
711:     ftol_satisfied = dF < ftol * F and ratio > 0.25
712:     xtol_satisfied = dx_norm < xtol * (xtol + x_norm)
713: 
714:     if ftol_satisfied and xtol_satisfied:
715:         return 4
716:     elif ftol_satisfied:
717:         return 2
718:     elif xtol_satisfied:
719:         return 3
720:     else:
721:         return None
722: 
723: 
724: def scale_for_robust_loss_function(J, f, rho):
725:     '''Scale Jacobian and residuals for a robust loss function.
726:     
727:     Arrays are modified in place.
728:     '''
729:     J_scale = rho[1] + 2 * rho[2] * f**2
730:     J_scale[J_scale < EPS] = EPS
731:     J_scale **= 0.5
732: 
733:     f *= rho[1] / J_scale
734: 
735:     return left_multiply(J, J_scale, copy=False), f
736: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_247677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Functions used by least-squares algorithms.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from math import copysign' statement (line 4)
try:
    from math import copysign

except:
    copysign = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'math', None, module_type_store, ['copysign'], [copysign])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_247678 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_247678) is not StypyTypeError):

    if (import_247678 != 'pyd_module'):
        __import__(import_247678)
        sys_modules_247679 = sys.modules[import_247678]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_247679.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_247678)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.linalg import norm' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_247680 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.linalg')

if (type(import_247680) is not StypyTypeError):

    if (import_247680 != 'pyd_module'):
        __import__(import_247680)
        sys_modules_247681 = sys.modules[import_247680]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.linalg', sys_modules_247681.module_type_store, module_type_store, ['norm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_247681, sys_modules_247681.module_type_store, module_type_store)
    else:
        from numpy.linalg import norm

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.linalg', None, module_type_store, ['norm'], [norm])

else:
    # Assigning a type to the variable 'numpy.linalg' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.linalg', import_247680)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.linalg import cho_factor, cho_solve, LinAlgError' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_247682 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg')

if (type(import_247682) is not StypyTypeError):

    if (import_247682 != 'pyd_module'):
        __import__(import_247682)
        sys_modules_247683 = sys.modules[import_247682]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg', sys_modules_247683.module_type_store, module_type_store, ['cho_factor', 'cho_solve', 'LinAlgError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_247683, sys_modules_247683.module_type_store, module_type_store)
    else:
        from scipy.linalg import cho_factor, cho_solve, LinAlgError

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg', None, module_type_store, ['cho_factor', 'cho_solve', 'LinAlgError'], [cho_factor, cho_solve, LinAlgError])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg', import_247682)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.sparse import issparse' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_247684 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse')

if (type(import_247684) is not StypyTypeError):

    if (import_247684 != 'pyd_module'):
        __import__(import_247684)
        sys_modules_247685 = sys.modules[import_247684]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse', sys_modules_247685.module_type_store, module_type_store, ['issparse'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_247685, sys_modules_247685.module_type_store, module_type_store)
    else:
        from scipy.sparse import issparse

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse', None, module_type_store, ['issparse'], [issparse])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse', import_247684)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.sparse.linalg import LinearOperator, aslinearoperator' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_247686 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg')

if (type(import_247686) is not StypyTypeError):

    if (import_247686 != 'pyd_module'):
        __import__(import_247686)
        sys_modules_247687 = sys.modules[import_247686]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg', sys_modules_247687.module_type_store, module_type_store, ['LinearOperator', 'aslinearoperator'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_247687, sys_modules_247687.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import LinearOperator, aslinearoperator

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg', None, module_type_store, ['LinearOperator', 'aslinearoperator'], [LinearOperator, aslinearoperator])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.linalg', import_247686)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')


# Assigning a Attribute to a Name (line 14):

# Assigning a Attribute to a Name (line 14):

# Call to finfo(...): (line 14)
# Processing the call arguments (line 14)
# Getting the type of 'float' (line 14)
float_247690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 15), 'float', False)
# Processing the call keyword arguments (line 14)
kwargs_247691 = {}
# Getting the type of 'np' (line 14)
np_247688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 6), 'np', False)
# Obtaining the member 'finfo' of a type (line 14)
finfo_247689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 6), np_247688, 'finfo')
# Calling finfo(args, kwargs) (line 14)
finfo_call_result_247692 = invoke(stypy.reporting.localization.Localization(__file__, 14, 6), finfo_247689, *[float_247690], **kwargs_247691)

# Obtaining the member 'eps' of a type (line 14)
eps_247693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 6), finfo_call_result_247692, 'eps')
# Assigning a type to the variable 'EPS' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'EPS', eps_247693)

@norecursion
def intersect_trust_region(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'intersect_trust_region'
    module_type_store = module_type_store.open_function_context('intersect_trust_region', 20, 0, False)
    
    # Passed parameters checking function
    intersect_trust_region.stypy_localization = localization
    intersect_trust_region.stypy_type_of_self = None
    intersect_trust_region.stypy_type_store = module_type_store
    intersect_trust_region.stypy_function_name = 'intersect_trust_region'
    intersect_trust_region.stypy_param_names_list = ['x', 's', 'Delta']
    intersect_trust_region.stypy_varargs_param_name = None
    intersect_trust_region.stypy_kwargs_param_name = None
    intersect_trust_region.stypy_call_defaults = defaults
    intersect_trust_region.stypy_call_varargs = varargs
    intersect_trust_region.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'intersect_trust_region', ['x', 's', 'Delta'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'intersect_trust_region', localization, ['x', 's', 'Delta'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'intersect_trust_region(...)' code ##################

    str_247694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, (-1)), 'str', 'Find the intersection of a line with the boundary of a trust region.\n    \n    This function solves the quadratic equation with respect to t\n    ||(x + s*t)||**2 = Delta**2.\n    \n    Returns\n    -------\n    t_neg, t_pos : tuple of float\n        Negative and positive roots.\n    \n    Raises\n    ------\n    ValueError\n        If `s` is zero or `x` is not within the trust region.\n    ')
    
    # Assigning a Call to a Name (line 36):
    
    # Assigning a Call to a Name (line 36):
    
    # Call to dot(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 's' (line 36)
    s_247697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 's', False)
    # Getting the type of 's' (line 36)
    s_247698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 18), 's', False)
    # Processing the call keyword arguments (line 36)
    kwargs_247699 = {}
    # Getting the type of 'np' (line 36)
    np_247695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'np', False)
    # Obtaining the member 'dot' of a type (line 36)
    dot_247696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), np_247695, 'dot')
    # Calling dot(args, kwargs) (line 36)
    dot_call_result_247700 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), dot_247696, *[s_247697, s_247698], **kwargs_247699)
    
    # Assigning a type to the variable 'a' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'a', dot_call_result_247700)
    
    
    # Getting the type of 'a' (line 37)
    a_247701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 7), 'a')
    int_247702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 12), 'int')
    # Applying the binary operator '==' (line 37)
    result_eq_247703 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 7), '==', a_247701, int_247702)
    
    # Testing the type of an if condition (line 37)
    if_condition_247704 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 4), result_eq_247703)
    # Assigning a type to the variable 'if_condition_247704' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'if_condition_247704', if_condition_247704)
    # SSA begins for if statement (line 37)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 38)
    # Processing the call arguments (line 38)
    str_247706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 25), 'str', '`s` is zero.')
    # Processing the call keyword arguments (line 38)
    kwargs_247707 = {}
    # Getting the type of 'ValueError' (line 38)
    ValueError_247705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 38)
    ValueError_call_result_247708 = invoke(stypy.reporting.localization.Localization(__file__, 38, 14), ValueError_247705, *[str_247706], **kwargs_247707)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 38, 8), ValueError_call_result_247708, 'raise parameter', BaseException)
    # SSA join for if statement (line 37)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 40):
    
    # Assigning a Call to a Name (line 40):
    
    # Call to dot(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'x' (line 40)
    x_247711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'x', False)
    # Getting the type of 's' (line 40)
    s_247712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 18), 's', False)
    # Processing the call keyword arguments (line 40)
    kwargs_247713 = {}
    # Getting the type of 'np' (line 40)
    np_247709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'np', False)
    # Obtaining the member 'dot' of a type (line 40)
    dot_247710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), np_247709, 'dot')
    # Calling dot(args, kwargs) (line 40)
    dot_call_result_247714 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), dot_247710, *[x_247711, s_247712], **kwargs_247713)
    
    # Assigning a type to the variable 'b' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'b', dot_call_result_247714)
    
    # Assigning a BinOp to a Name (line 42):
    
    # Assigning a BinOp to a Name (line 42):
    
    # Call to dot(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'x' (line 42)
    x_247717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 15), 'x', False)
    # Getting the type of 'x' (line 42)
    x_247718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 18), 'x', False)
    # Processing the call keyword arguments (line 42)
    kwargs_247719 = {}
    # Getting the type of 'np' (line 42)
    np_247715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'np', False)
    # Obtaining the member 'dot' of a type (line 42)
    dot_247716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), np_247715, 'dot')
    # Calling dot(args, kwargs) (line 42)
    dot_call_result_247720 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), dot_247716, *[x_247717, x_247718], **kwargs_247719)
    
    # Getting the type of 'Delta' (line 42)
    Delta_247721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 23), 'Delta')
    int_247722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 30), 'int')
    # Applying the binary operator '**' (line 42)
    result_pow_247723 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 23), '**', Delta_247721, int_247722)
    
    # Applying the binary operator '-' (line 42)
    result_sub_247724 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 8), '-', dot_call_result_247720, result_pow_247723)
    
    # Assigning a type to the variable 'c' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'c', result_sub_247724)
    
    
    # Getting the type of 'c' (line 43)
    c_247725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 7), 'c')
    int_247726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 11), 'int')
    # Applying the binary operator '>' (line 43)
    result_gt_247727 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 7), '>', c_247725, int_247726)
    
    # Testing the type of an if condition (line 43)
    if_condition_247728 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 4), result_gt_247727)
    # Assigning a type to the variable 'if_condition_247728' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'if_condition_247728', if_condition_247728)
    # SSA begins for if statement (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 44)
    # Processing the call arguments (line 44)
    str_247730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 25), 'str', '`x` is not within the trust region.')
    # Processing the call keyword arguments (line 44)
    kwargs_247731 = {}
    # Getting the type of 'ValueError' (line 44)
    ValueError_247729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 44)
    ValueError_call_result_247732 = invoke(stypy.reporting.localization.Localization(__file__, 44, 14), ValueError_247729, *[str_247730], **kwargs_247731)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 44, 8), ValueError_call_result_247732, 'raise parameter', BaseException)
    # SSA join for if statement (line 43)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 46):
    
    # Assigning a Call to a Name (line 46):
    
    # Call to sqrt(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'b' (line 46)
    b_247735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'b', False)
    # Getting the type of 'b' (line 46)
    b_247736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 18), 'b', False)
    # Applying the binary operator '*' (line 46)
    result_mul_247737 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 16), '*', b_247735, b_247736)
    
    # Getting the type of 'a' (line 46)
    a_247738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'a', False)
    # Getting the type of 'c' (line 46)
    c_247739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 24), 'c', False)
    # Applying the binary operator '*' (line 46)
    result_mul_247740 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 22), '*', a_247738, c_247739)
    
    # Applying the binary operator '-' (line 46)
    result_sub_247741 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 16), '-', result_mul_247737, result_mul_247740)
    
    # Processing the call keyword arguments (line 46)
    kwargs_247742 = {}
    # Getting the type of 'np' (line 46)
    np_247733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 46)
    sqrt_247734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), np_247733, 'sqrt')
    # Calling sqrt(args, kwargs) (line 46)
    sqrt_call_result_247743 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), sqrt_247734, *[result_sub_247741], **kwargs_247742)
    
    # Assigning a type to the variable 'd' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'd', sqrt_call_result_247743)
    
    # Assigning a UnaryOp to a Name (line 49):
    
    # Assigning a UnaryOp to a Name (line 49):
    
    # Getting the type of 'b' (line 49)
    b_247744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 10), 'b')
    
    # Call to copysign(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'd' (line 49)
    d_247746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 23), 'd', False)
    # Getting the type of 'b' (line 49)
    b_247747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 26), 'b', False)
    # Processing the call keyword arguments (line 49)
    kwargs_247748 = {}
    # Getting the type of 'copysign' (line 49)
    copysign_247745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 14), 'copysign', False)
    # Calling copysign(args, kwargs) (line 49)
    copysign_call_result_247749 = invoke(stypy.reporting.localization.Localization(__file__, 49, 14), copysign_247745, *[d_247746, b_247747], **kwargs_247748)
    
    # Applying the binary operator '+' (line 49)
    result_add_247750 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 10), '+', b_247744, copysign_call_result_247749)
    
    # Applying the 'usub' unary operator (line 49)
    result___neg___247751 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 8), 'usub', result_add_247750)
    
    # Assigning a type to the variable 'q' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'q', result___neg___247751)
    
    # Assigning a BinOp to a Name (line 50):
    
    # Assigning a BinOp to a Name (line 50):
    # Getting the type of 'q' (line 50)
    q_247752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 9), 'q')
    # Getting the type of 'a' (line 50)
    a_247753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 13), 'a')
    # Applying the binary operator 'div' (line 50)
    result_div_247754 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 9), 'div', q_247752, a_247753)
    
    # Assigning a type to the variable 't1' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 't1', result_div_247754)
    
    # Assigning a BinOp to a Name (line 51):
    
    # Assigning a BinOp to a Name (line 51):
    # Getting the type of 'c' (line 51)
    c_247755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 9), 'c')
    # Getting the type of 'q' (line 51)
    q_247756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 13), 'q')
    # Applying the binary operator 'div' (line 51)
    result_div_247757 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 9), 'div', c_247755, q_247756)
    
    # Assigning a type to the variable 't2' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 't2', result_div_247757)
    
    
    # Getting the type of 't1' (line 53)
    t1_247758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 7), 't1')
    # Getting the type of 't2' (line 53)
    t2_247759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 't2')
    # Applying the binary operator '<' (line 53)
    result_lt_247760 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 7), '<', t1_247758, t2_247759)
    
    # Testing the type of an if condition (line 53)
    if_condition_247761 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 4), result_lt_247760)
    # Assigning a type to the variable 'if_condition_247761' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'if_condition_247761', if_condition_247761)
    # SSA begins for if statement (line 53)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 54)
    tuple_247762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 54)
    # Adding element type (line 54)
    # Getting the type of 't1' (line 54)
    t1_247763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 15), 't1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 15), tuple_247762, t1_247763)
    # Adding element type (line 54)
    # Getting the type of 't2' (line 54)
    t2_247764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 19), 't2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 15), tuple_247762, t2_247764)
    
    # Assigning a type to the variable 'stypy_return_type' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'stypy_return_type', tuple_247762)
    # SSA branch for the else part of an if statement (line 53)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 56)
    tuple_247765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 56)
    # Adding element type (line 56)
    # Getting the type of 't2' (line 56)
    t2_247766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 15), 't2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 15), tuple_247765, t2_247766)
    # Adding element type (line 56)
    # Getting the type of 't1' (line 56)
    t1_247767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 19), 't1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 15), tuple_247765, t1_247767)
    
    # Assigning a type to the variable 'stypy_return_type' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'stypy_return_type', tuple_247765)
    # SSA join for if statement (line 53)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'intersect_trust_region(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'intersect_trust_region' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_247768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_247768)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'intersect_trust_region'
    return stypy_return_type_247768

# Assigning a type to the variable 'intersect_trust_region' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'intersect_trust_region', intersect_trust_region)

@norecursion
def solve_lsq_trust_region(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 59)
    None_247769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 64), 'None')
    float_247770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 32), 'float')
    int_247771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 47), 'int')
    defaults = [None_247769, float_247770, int_247771]
    # Create a new context for function 'solve_lsq_trust_region'
    module_type_store = module_type_store.open_function_context('solve_lsq_trust_region', 59, 0, False)
    
    # Passed parameters checking function
    solve_lsq_trust_region.stypy_localization = localization
    solve_lsq_trust_region.stypy_type_of_self = None
    solve_lsq_trust_region.stypy_type_store = module_type_store
    solve_lsq_trust_region.stypy_function_name = 'solve_lsq_trust_region'
    solve_lsq_trust_region.stypy_param_names_list = ['n', 'm', 'uf', 's', 'V', 'Delta', 'initial_alpha', 'rtol', 'max_iter']
    solve_lsq_trust_region.stypy_varargs_param_name = None
    solve_lsq_trust_region.stypy_kwargs_param_name = None
    solve_lsq_trust_region.stypy_call_defaults = defaults
    solve_lsq_trust_region.stypy_call_varargs = varargs
    solve_lsq_trust_region.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'solve_lsq_trust_region', ['n', 'm', 'uf', 's', 'V', 'Delta', 'initial_alpha', 'rtol', 'max_iter'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'solve_lsq_trust_region', localization, ['n', 'm', 'uf', 's', 'V', 'Delta', 'initial_alpha', 'rtol', 'max_iter'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'solve_lsq_trust_region(...)' code ##################

    str_247772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, (-1)), 'str', 'Solve a trust-region problem arising in least-squares minimization.\n    \n    This function implements a method described by J. J. More [1]_ and used\n    in MINPACK, but it relies on a single SVD of Jacobian instead of series\n    of Cholesky decompositions. Before running this function, compute:\n    ``U, s, VT = svd(J, full_matrices=False)``.\n    \n    Parameters\n    ----------\n    n : int\n        Number of variables.\n    m : int\n        Number of residuals.\n    uf : ndarray\n        Computed as U.T.dot(f).\n    s : ndarray\n        Singular values of J.\n    V : ndarray\n        Transpose of VT.\n    Delta : float\n        Radius of a trust region.\n    initial_alpha : float, optional\n        Initial guess for alpha, which might be available from a previous\n        iteration. If None, determined automatically.\n    rtol : float, optional\n        Stopping tolerance for the root-finding procedure. Namely, the\n        solution ``p`` will satisfy ``abs(norm(p) - Delta) < rtol * Delta``.\n    max_iter : int, optional\n        Maximum allowed number of iterations for the root-finding procedure.\n    \n    Returns\n    -------\n    p : ndarray, shape (n,)\n        Found solution of a trust-region problem.\n    alpha : float\n        Positive value such that (J.T*J + alpha*I)*p = -J.T*f.\n        Sometimes called Levenberg-Marquardt parameter.\n    n_iter : int\n        Number of iterations made by root-finding procedure. Zero means\n        that Gauss-Newton step was selected as the solution.\n    \n    References\n    ----------\n    .. [1] More, J. J., "The Levenberg-Marquardt Algorithm: Implementation\n           and Theory," Numerical Analysis, ed. G. A. Watson, Lecture Notes\n           in Mathematics 630, Springer Verlag, pp. 105-116, 1977.\n    ')

    @norecursion
    def phi_and_derivative(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'phi_and_derivative'
        module_type_store = module_type_store.open_function_context('phi_and_derivative', 108, 4, False)
        
        # Passed parameters checking function
        phi_and_derivative.stypy_localization = localization
        phi_and_derivative.stypy_type_of_self = None
        phi_and_derivative.stypy_type_store = module_type_store
        phi_and_derivative.stypy_function_name = 'phi_and_derivative'
        phi_and_derivative.stypy_param_names_list = ['alpha', 'suf', 's', 'Delta']
        phi_and_derivative.stypy_varargs_param_name = None
        phi_and_derivative.stypy_kwargs_param_name = None
        phi_and_derivative.stypy_call_defaults = defaults
        phi_and_derivative.stypy_call_varargs = varargs
        phi_and_derivative.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'phi_and_derivative', ['alpha', 'suf', 's', 'Delta'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'phi_and_derivative', localization, ['alpha', 'suf', 's', 'Delta'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'phi_and_derivative(...)' code ##################

        str_247773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, (-1)), 'str', 'Function of which to find zero.\n        \n        It is defined as "norm of regularized (by alpha) least-squares\n        solution minus `Delta`". Refer to [1]_.\n        ')
        
        # Assigning a BinOp to a Name (line 114):
        
        # Assigning a BinOp to a Name (line 114):
        # Getting the type of 's' (line 114)
        s_247774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 's')
        int_247775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 19), 'int')
        # Applying the binary operator '**' (line 114)
        result_pow_247776 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 16), '**', s_247774, int_247775)
        
        # Getting the type of 'alpha' (line 114)
        alpha_247777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 23), 'alpha')
        # Applying the binary operator '+' (line 114)
        result_add_247778 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 16), '+', result_pow_247776, alpha_247777)
        
        # Assigning a type to the variable 'denom' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'denom', result_add_247778)
        
        # Assigning a Call to a Name (line 115):
        
        # Assigning a Call to a Name (line 115):
        
        # Call to norm(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'suf' (line 115)
        suf_247780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 22), 'suf', False)
        # Getting the type of 'denom' (line 115)
        denom_247781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 28), 'denom', False)
        # Applying the binary operator 'div' (line 115)
        result_div_247782 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 22), 'div', suf_247780, denom_247781)
        
        # Processing the call keyword arguments (line 115)
        kwargs_247783 = {}
        # Getting the type of 'norm' (line 115)
        norm_247779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 17), 'norm', False)
        # Calling norm(args, kwargs) (line 115)
        norm_call_result_247784 = invoke(stypy.reporting.localization.Localization(__file__, 115, 17), norm_247779, *[result_div_247782], **kwargs_247783)
        
        # Assigning a type to the variable 'p_norm' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'p_norm', norm_call_result_247784)
        
        # Assigning a BinOp to a Name (line 116):
        
        # Assigning a BinOp to a Name (line 116):
        # Getting the type of 'p_norm' (line 116)
        p_norm_247785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 14), 'p_norm')
        # Getting the type of 'Delta' (line 116)
        Delta_247786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 23), 'Delta')
        # Applying the binary operator '-' (line 116)
        result_sub_247787 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 14), '-', p_norm_247785, Delta_247786)
        
        # Assigning a type to the variable 'phi' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'phi', result_sub_247787)
        
        # Assigning a BinOp to a Name (line 117):
        
        # Assigning a BinOp to a Name (line 117):
        
        
        # Call to sum(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'suf' (line 117)
        suf_247790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 28), 'suf', False)
        int_247791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 35), 'int')
        # Applying the binary operator '**' (line 117)
        result_pow_247792 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 28), '**', suf_247790, int_247791)
        
        # Getting the type of 'denom' (line 117)
        denom_247793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 39), 'denom', False)
        int_247794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 46), 'int')
        # Applying the binary operator '**' (line 117)
        result_pow_247795 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 39), '**', denom_247793, int_247794)
        
        # Applying the binary operator 'div' (line 117)
        result_div_247796 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 28), 'div', result_pow_247792, result_pow_247795)
        
        # Processing the call keyword arguments (line 117)
        kwargs_247797 = {}
        # Getting the type of 'np' (line 117)
        np_247788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 21), 'np', False)
        # Obtaining the member 'sum' of a type (line 117)
        sum_247789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 21), np_247788, 'sum')
        # Calling sum(args, kwargs) (line 117)
        sum_call_result_247798 = invoke(stypy.reporting.localization.Localization(__file__, 117, 21), sum_247789, *[result_div_247796], **kwargs_247797)
        
        # Applying the 'usub' unary operator (line 117)
        result___neg___247799 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 20), 'usub', sum_call_result_247798)
        
        # Getting the type of 'p_norm' (line 117)
        p_norm_247800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 51), 'p_norm')
        # Applying the binary operator 'div' (line 117)
        result_div_247801 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 20), 'div', result___neg___247799, p_norm_247800)
        
        # Assigning a type to the variable 'phi_prime' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'phi_prime', result_div_247801)
        
        # Obtaining an instance of the builtin type 'tuple' (line 118)
        tuple_247802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 118)
        # Adding element type (line 118)
        # Getting the type of 'phi' (line 118)
        phi_247803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'phi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 15), tuple_247802, phi_247803)
        # Adding element type (line 118)
        # Getting the type of 'phi_prime' (line 118)
        phi_prime_247804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 20), 'phi_prime')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 15), tuple_247802, phi_prime_247804)
        
        # Assigning a type to the variable 'stypy_return_type' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'stypy_return_type', tuple_247802)
        
        # ################# End of 'phi_and_derivative(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'phi_and_derivative' in the type store
        # Getting the type of 'stypy_return_type' (line 108)
        stypy_return_type_247805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_247805)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'phi_and_derivative'
        return stypy_return_type_247805

    # Assigning a type to the variable 'phi_and_derivative' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'phi_and_derivative', phi_and_derivative)
    
    # Assigning a BinOp to a Name (line 120):
    
    # Assigning a BinOp to a Name (line 120):
    # Getting the type of 's' (line 120)
    s_247806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 10), 's')
    # Getting the type of 'uf' (line 120)
    uf_247807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 14), 'uf')
    # Applying the binary operator '*' (line 120)
    result_mul_247808 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 10), '*', s_247806, uf_247807)
    
    # Assigning a type to the variable 'suf' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'suf', result_mul_247808)
    
    
    # Getting the type of 'm' (line 123)
    m_247809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 7), 'm')
    # Getting the type of 'n' (line 123)
    n_247810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'n')
    # Applying the binary operator '>=' (line 123)
    result_ge_247811 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 7), '>=', m_247809, n_247810)
    
    # Testing the type of an if condition (line 123)
    if_condition_247812 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 4), result_ge_247811)
    # Assigning a type to the variable 'if_condition_247812' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'if_condition_247812', if_condition_247812)
    # SSA begins for if statement (line 123)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 124):
    
    # Assigning a BinOp to a Name (line 124):
    # Getting the type of 'EPS' (line 124)
    EPS_247813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'EPS')
    # Getting the type of 'm' (line 124)
    m_247814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 26), 'm')
    # Applying the binary operator '*' (line 124)
    result_mul_247815 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 20), '*', EPS_247813, m_247814)
    
    
    # Obtaining the type of the subscript
    int_247816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 32), 'int')
    # Getting the type of 's' (line 124)
    s_247817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 30), 's')
    # Obtaining the member '__getitem__' of a type (line 124)
    getitem___247818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 30), s_247817, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 124)
    subscript_call_result_247819 = invoke(stypy.reporting.localization.Localization(__file__, 124, 30), getitem___247818, int_247816)
    
    # Applying the binary operator '*' (line 124)
    result_mul_247820 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 28), '*', result_mul_247815, subscript_call_result_247819)
    
    # Assigning a type to the variable 'threshold' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'threshold', result_mul_247820)
    
    # Assigning a Compare to a Name (line 125):
    
    # Assigning a Compare to a Name (line 125):
    
    
    # Obtaining the type of the subscript
    int_247821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 22), 'int')
    # Getting the type of 's' (line 125)
    s_247822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 20), 's')
    # Obtaining the member '__getitem__' of a type (line 125)
    getitem___247823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 20), s_247822, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 125)
    subscript_call_result_247824 = invoke(stypy.reporting.localization.Localization(__file__, 125, 20), getitem___247823, int_247821)
    
    # Getting the type of 'threshold' (line 125)
    threshold_247825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 28), 'threshold')
    # Applying the binary operator '>' (line 125)
    result_gt_247826 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 20), '>', subscript_call_result_247824, threshold_247825)
    
    # Assigning a type to the variable 'full_rank' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'full_rank', result_gt_247826)
    # SSA branch for the else part of an if statement (line 123)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 127):
    
    # Assigning a Name to a Name (line 127):
    # Getting the type of 'False' (line 127)
    False_247827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 20), 'False')
    # Assigning a type to the variable 'full_rank' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'full_rank', False_247827)
    # SSA join for if statement (line 123)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'full_rank' (line 129)
    full_rank_247828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 7), 'full_rank')
    # Testing the type of an if condition (line 129)
    if_condition_247829 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 129, 4), full_rank_247828)
    # Assigning a type to the variable 'if_condition_247829' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'if_condition_247829', if_condition_247829)
    # SSA begins for if statement (line 129)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a UnaryOp to a Name (line 130):
    
    # Assigning a UnaryOp to a Name (line 130):
    
    
    # Call to dot(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'uf' (line 130)
    uf_247832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 19), 'uf', False)
    # Getting the type of 's' (line 130)
    s_247833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 24), 's', False)
    # Applying the binary operator 'div' (line 130)
    result_div_247834 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 19), 'div', uf_247832, s_247833)
    
    # Processing the call keyword arguments (line 130)
    kwargs_247835 = {}
    # Getting the type of 'V' (line 130)
    V_247830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 13), 'V', False)
    # Obtaining the member 'dot' of a type (line 130)
    dot_247831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 13), V_247830, 'dot')
    # Calling dot(args, kwargs) (line 130)
    dot_call_result_247836 = invoke(stypy.reporting.localization.Localization(__file__, 130, 13), dot_247831, *[result_div_247834], **kwargs_247835)
    
    # Applying the 'usub' unary operator (line 130)
    result___neg___247837 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 12), 'usub', dot_call_result_247836)
    
    # Assigning a type to the variable 'p' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'p', result___neg___247837)
    
    
    
    # Call to norm(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'p' (line 131)
    p_247839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'p', False)
    # Processing the call keyword arguments (line 131)
    kwargs_247840 = {}
    # Getting the type of 'norm' (line 131)
    norm_247838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 11), 'norm', False)
    # Calling norm(args, kwargs) (line 131)
    norm_call_result_247841 = invoke(stypy.reporting.localization.Localization(__file__, 131, 11), norm_247838, *[p_247839], **kwargs_247840)
    
    # Getting the type of 'Delta' (line 131)
    Delta_247842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 22), 'Delta')
    # Applying the binary operator '<=' (line 131)
    result_le_247843 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 11), '<=', norm_call_result_247841, Delta_247842)
    
    # Testing the type of an if condition (line 131)
    if_condition_247844 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 131, 8), result_le_247843)
    # Assigning a type to the variable 'if_condition_247844' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'if_condition_247844', if_condition_247844)
    # SSA begins for if statement (line 131)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 132)
    tuple_247845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 132)
    # Adding element type (line 132)
    # Getting the type of 'p' (line 132)
    p_247846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 19), 'p')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 19), tuple_247845, p_247846)
    # Adding element type (line 132)
    float_247847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 22), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 19), tuple_247845, float_247847)
    # Adding element type (line 132)
    int_247848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 19), tuple_247845, int_247848)
    
    # Assigning a type to the variable 'stypy_return_type' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'stypy_return_type', tuple_247845)
    # SSA join for if statement (line 131)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 129)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 134):
    
    # Assigning a BinOp to a Name (line 134):
    
    # Call to norm(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'suf' (line 134)
    suf_247850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 23), 'suf', False)
    # Processing the call keyword arguments (line 134)
    kwargs_247851 = {}
    # Getting the type of 'norm' (line 134)
    norm_247849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 18), 'norm', False)
    # Calling norm(args, kwargs) (line 134)
    norm_call_result_247852 = invoke(stypy.reporting.localization.Localization(__file__, 134, 18), norm_247849, *[suf_247850], **kwargs_247851)
    
    # Getting the type of 'Delta' (line 134)
    Delta_247853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 30), 'Delta')
    # Applying the binary operator 'div' (line 134)
    result_div_247854 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 18), 'div', norm_call_result_247852, Delta_247853)
    
    # Assigning a type to the variable 'alpha_upper' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'alpha_upper', result_div_247854)
    
    # Getting the type of 'full_rank' (line 136)
    full_rank_247855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 7), 'full_rank')
    # Testing the type of an if condition (line 136)
    if_condition_247856 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 136, 4), full_rank_247855)
    # Assigning a type to the variable 'if_condition_247856' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'if_condition_247856', if_condition_247856)
    # SSA begins for if statement (line 136)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 137):
    
    # Assigning a Subscript to a Name (line 137):
    
    # Obtaining the type of the subscript
    int_247857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 8), 'int')
    
    # Call to phi_and_derivative(...): (line 137)
    # Processing the call arguments (line 137)
    float_247859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 44), 'float')
    # Getting the type of 'suf' (line 137)
    suf_247860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 49), 'suf', False)
    # Getting the type of 's' (line 137)
    s_247861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 54), 's', False)
    # Getting the type of 'Delta' (line 137)
    Delta_247862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 57), 'Delta', False)
    # Processing the call keyword arguments (line 137)
    kwargs_247863 = {}
    # Getting the type of 'phi_and_derivative' (line 137)
    phi_and_derivative_247858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 25), 'phi_and_derivative', False)
    # Calling phi_and_derivative(args, kwargs) (line 137)
    phi_and_derivative_call_result_247864 = invoke(stypy.reporting.localization.Localization(__file__, 137, 25), phi_and_derivative_247858, *[float_247859, suf_247860, s_247861, Delta_247862], **kwargs_247863)
    
    # Obtaining the member '__getitem__' of a type (line 137)
    getitem___247865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), phi_and_derivative_call_result_247864, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 137)
    subscript_call_result_247866 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), getitem___247865, int_247857)
    
    # Assigning a type to the variable 'tuple_var_assignment_247669' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'tuple_var_assignment_247669', subscript_call_result_247866)
    
    # Assigning a Subscript to a Name (line 137):
    
    # Obtaining the type of the subscript
    int_247867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 8), 'int')
    
    # Call to phi_and_derivative(...): (line 137)
    # Processing the call arguments (line 137)
    float_247869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 44), 'float')
    # Getting the type of 'suf' (line 137)
    suf_247870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 49), 'suf', False)
    # Getting the type of 's' (line 137)
    s_247871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 54), 's', False)
    # Getting the type of 'Delta' (line 137)
    Delta_247872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 57), 'Delta', False)
    # Processing the call keyword arguments (line 137)
    kwargs_247873 = {}
    # Getting the type of 'phi_and_derivative' (line 137)
    phi_and_derivative_247868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 25), 'phi_and_derivative', False)
    # Calling phi_and_derivative(args, kwargs) (line 137)
    phi_and_derivative_call_result_247874 = invoke(stypy.reporting.localization.Localization(__file__, 137, 25), phi_and_derivative_247868, *[float_247869, suf_247870, s_247871, Delta_247872], **kwargs_247873)
    
    # Obtaining the member '__getitem__' of a type (line 137)
    getitem___247875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), phi_and_derivative_call_result_247874, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 137)
    subscript_call_result_247876 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), getitem___247875, int_247867)
    
    # Assigning a type to the variable 'tuple_var_assignment_247670' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'tuple_var_assignment_247670', subscript_call_result_247876)
    
    # Assigning a Name to a Name (line 137):
    # Getting the type of 'tuple_var_assignment_247669' (line 137)
    tuple_var_assignment_247669_247877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'tuple_var_assignment_247669')
    # Assigning a type to the variable 'phi' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'phi', tuple_var_assignment_247669_247877)
    
    # Assigning a Name to a Name (line 137):
    # Getting the type of 'tuple_var_assignment_247670' (line 137)
    tuple_var_assignment_247670_247878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'tuple_var_assignment_247670')
    # Assigning a type to the variable 'phi_prime' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 13), 'phi_prime', tuple_var_assignment_247670_247878)
    
    # Assigning a BinOp to a Name (line 138):
    
    # Assigning a BinOp to a Name (line 138):
    
    # Getting the type of 'phi' (line 138)
    phi_247879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 23), 'phi')
    # Applying the 'usub' unary operator (line 138)
    result___neg___247880 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 22), 'usub', phi_247879)
    
    # Getting the type of 'phi_prime' (line 138)
    phi_prime_247881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 29), 'phi_prime')
    # Applying the binary operator 'div' (line 138)
    result_div_247882 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 22), 'div', result___neg___247880, phi_prime_247881)
    
    # Assigning a type to the variable 'alpha_lower' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'alpha_lower', result_div_247882)
    # SSA branch for the else part of an if statement (line 136)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 140):
    
    # Assigning a Num to a Name (line 140):
    float_247883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 22), 'float')
    # Assigning a type to the variable 'alpha_lower' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'alpha_lower', float_247883)
    # SSA join for if statement (line 136)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'initial_alpha' (line 142)
    initial_alpha_247884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 7), 'initial_alpha')
    # Getting the type of 'None' (line 142)
    None_247885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 24), 'None')
    # Applying the binary operator 'is' (line 142)
    result_is__247886 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 7), 'is', initial_alpha_247884, None_247885)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'full_rank' (line 142)
    full_rank_247887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 36), 'full_rank')
    # Applying the 'not' unary operator (line 142)
    result_not__247888 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 32), 'not', full_rank_247887)
    
    
    # Getting the type of 'initial_alpha' (line 142)
    initial_alpha_247889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 50), 'initial_alpha')
    int_247890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 67), 'int')
    # Applying the binary operator '==' (line 142)
    result_eq_247891 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 50), '==', initial_alpha_247889, int_247890)
    
    # Applying the binary operator 'and' (line 142)
    result_and_keyword_247892 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 32), 'and', result_not__247888, result_eq_247891)
    
    # Applying the binary operator 'or' (line 142)
    result_or_keyword_247893 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 7), 'or', result_is__247886, result_and_keyword_247892)
    
    # Testing the type of an if condition (line 142)
    if_condition_247894 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 4), result_or_keyword_247893)
    # Assigning a type to the variable 'if_condition_247894' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'if_condition_247894', if_condition_247894)
    # SSA begins for if statement (line 142)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 143):
    
    # Assigning a Call to a Name (line 143):
    
    # Call to max(...): (line 143)
    # Processing the call arguments (line 143)
    float_247896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 20), 'float')
    # Getting the type of 'alpha_upper' (line 143)
    alpha_upper_247897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 28), 'alpha_upper', False)
    # Applying the binary operator '*' (line 143)
    result_mul_247898 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 20), '*', float_247896, alpha_upper_247897)
    
    # Getting the type of 'alpha_lower' (line 143)
    alpha_lower_247899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 42), 'alpha_lower', False)
    # Getting the type of 'alpha_upper' (line 143)
    alpha_upper_247900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 56), 'alpha_upper', False)
    # Applying the binary operator '*' (line 143)
    result_mul_247901 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 42), '*', alpha_lower_247899, alpha_upper_247900)
    
    float_247902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 70), 'float')
    # Applying the binary operator '**' (line 143)
    result_pow_247903 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 41), '**', result_mul_247901, float_247902)
    
    # Processing the call keyword arguments (line 143)
    kwargs_247904 = {}
    # Getting the type of 'max' (line 143)
    max_247895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'max', False)
    # Calling max(args, kwargs) (line 143)
    max_call_result_247905 = invoke(stypy.reporting.localization.Localization(__file__, 143, 16), max_247895, *[result_mul_247898, result_pow_247903], **kwargs_247904)
    
    # Assigning a type to the variable 'alpha' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'alpha', max_call_result_247905)
    # SSA branch for the else part of an if statement (line 142)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 145):
    
    # Assigning a Name to a Name (line 145):
    # Getting the type of 'initial_alpha' (line 145)
    initial_alpha_247906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'initial_alpha')
    # Assigning a type to the variable 'alpha' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'alpha', initial_alpha_247906)
    # SSA join for if statement (line 142)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 'max_iter' (line 147)
    max_iter_247908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 20), 'max_iter', False)
    # Processing the call keyword arguments (line 147)
    kwargs_247909 = {}
    # Getting the type of 'range' (line 147)
    range_247907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 14), 'range', False)
    # Calling range(args, kwargs) (line 147)
    range_call_result_247910 = invoke(stypy.reporting.localization.Localization(__file__, 147, 14), range_247907, *[max_iter_247908], **kwargs_247909)
    
    # Testing the type of a for loop iterable (line 147)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 147, 4), range_call_result_247910)
    # Getting the type of the for loop variable (line 147)
    for_loop_var_247911 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 147, 4), range_call_result_247910)
    # Assigning a type to the variable 'it' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'it', for_loop_var_247911)
    # SSA begins for a for statement (line 147)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'alpha' (line 148)
    alpha_247912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 11), 'alpha')
    # Getting the type of 'alpha_lower' (line 148)
    alpha_lower_247913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 19), 'alpha_lower')
    # Applying the binary operator '<' (line 148)
    result_lt_247914 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 11), '<', alpha_247912, alpha_lower_247913)
    
    
    # Getting the type of 'alpha' (line 148)
    alpha_247915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 34), 'alpha')
    # Getting the type of 'alpha_upper' (line 148)
    alpha_upper_247916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 42), 'alpha_upper')
    # Applying the binary operator '>' (line 148)
    result_gt_247917 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 34), '>', alpha_247915, alpha_upper_247916)
    
    # Applying the binary operator 'or' (line 148)
    result_or_keyword_247918 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 11), 'or', result_lt_247914, result_gt_247917)
    
    # Testing the type of an if condition (line 148)
    if_condition_247919 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 8), result_or_keyword_247918)
    # Assigning a type to the variable 'if_condition_247919' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'if_condition_247919', if_condition_247919)
    # SSA begins for if statement (line 148)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 149):
    
    # Assigning a Call to a Name (line 149):
    
    # Call to max(...): (line 149)
    # Processing the call arguments (line 149)
    float_247921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 24), 'float')
    # Getting the type of 'alpha_upper' (line 149)
    alpha_upper_247922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 32), 'alpha_upper', False)
    # Applying the binary operator '*' (line 149)
    result_mul_247923 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 24), '*', float_247921, alpha_upper_247922)
    
    # Getting the type of 'alpha_lower' (line 149)
    alpha_lower_247924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 46), 'alpha_lower', False)
    # Getting the type of 'alpha_upper' (line 149)
    alpha_upper_247925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 60), 'alpha_upper', False)
    # Applying the binary operator '*' (line 149)
    result_mul_247926 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 46), '*', alpha_lower_247924, alpha_upper_247925)
    
    float_247927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 74), 'float')
    # Applying the binary operator '**' (line 149)
    result_pow_247928 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 45), '**', result_mul_247926, float_247927)
    
    # Processing the call keyword arguments (line 149)
    kwargs_247929 = {}
    # Getting the type of 'max' (line 149)
    max_247920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'max', False)
    # Calling max(args, kwargs) (line 149)
    max_call_result_247930 = invoke(stypy.reporting.localization.Localization(__file__, 149, 20), max_247920, *[result_mul_247923, result_pow_247928], **kwargs_247929)
    
    # Assigning a type to the variable 'alpha' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'alpha', max_call_result_247930)
    # SSA join for if statement (line 148)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 151):
    
    # Assigning a Subscript to a Name (line 151):
    
    # Obtaining the type of the subscript
    int_247931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 8), 'int')
    
    # Call to phi_and_derivative(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'alpha' (line 151)
    alpha_247933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 44), 'alpha', False)
    # Getting the type of 'suf' (line 151)
    suf_247934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 51), 'suf', False)
    # Getting the type of 's' (line 151)
    s_247935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 56), 's', False)
    # Getting the type of 'Delta' (line 151)
    Delta_247936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 59), 'Delta', False)
    # Processing the call keyword arguments (line 151)
    kwargs_247937 = {}
    # Getting the type of 'phi_and_derivative' (line 151)
    phi_and_derivative_247932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 25), 'phi_and_derivative', False)
    # Calling phi_and_derivative(args, kwargs) (line 151)
    phi_and_derivative_call_result_247938 = invoke(stypy.reporting.localization.Localization(__file__, 151, 25), phi_and_derivative_247932, *[alpha_247933, suf_247934, s_247935, Delta_247936], **kwargs_247937)
    
    # Obtaining the member '__getitem__' of a type (line 151)
    getitem___247939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), phi_and_derivative_call_result_247938, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 151)
    subscript_call_result_247940 = invoke(stypy.reporting.localization.Localization(__file__, 151, 8), getitem___247939, int_247931)
    
    # Assigning a type to the variable 'tuple_var_assignment_247671' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'tuple_var_assignment_247671', subscript_call_result_247940)
    
    # Assigning a Subscript to a Name (line 151):
    
    # Obtaining the type of the subscript
    int_247941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 8), 'int')
    
    # Call to phi_and_derivative(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'alpha' (line 151)
    alpha_247943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 44), 'alpha', False)
    # Getting the type of 'suf' (line 151)
    suf_247944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 51), 'suf', False)
    # Getting the type of 's' (line 151)
    s_247945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 56), 's', False)
    # Getting the type of 'Delta' (line 151)
    Delta_247946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 59), 'Delta', False)
    # Processing the call keyword arguments (line 151)
    kwargs_247947 = {}
    # Getting the type of 'phi_and_derivative' (line 151)
    phi_and_derivative_247942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 25), 'phi_and_derivative', False)
    # Calling phi_and_derivative(args, kwargs) (line 151)
    phi_and_derivative_call_result_247948 = invoke(stypy.reporting.localization.Localization(__file__, 151, 25), phi_and_derivative_247942, *[alpha_247943, suf_247944, s_247945, Delta_247946], **kwargs_247947)
    
    # Obtaining the member '__getitem__' of a type (line 151)
    getitem___247949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), phi_and_derivative_call_result_247948, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 151)
    subscript_call_result_247950 = invoke(stypy.reporting.localization.Localization(__file__, 151, 8), getitem___247949, int_247941)
    
    # Assigning a type to the variable 'tuple_var_assignment_247672' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'tuple_var_assignment_247672', subscript_call_result_247950)
    
    # Assigning a Name to a Name (line 151):
    # Getting the type of 'tuple_var_assignment_247671' (line 151)
    tuple_var_assignment_247671_247951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'tuple_var_assignment_247671')
    # Assigning a type to the variable 'phi' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'phi', tuple_var_assignment_247671_247951)
    
    # Assigning a Name to a Name (line 151):
    # Getting the type of 'tuple_var_assignment_247672' (line 151)
    tuple_var_assignment_247672_247952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'tuple_var_assignment_247672')
    # Assigning a type to the variable 'phi_prime' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 13), 'phi_prime', tuple_var_assignment_247672_247952)
    
    
    # Getting the type of 'phi' (line 153)
    phi_247953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 11), 'phi')
    int_247954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 17), 'int')
    # Applying the binary operator '<' (line 153)
    result_lt_247955 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 11), '<', phi_247953, int_247954)
    
    # Testing the type of an if condition (line 153)
    if_condition_247956 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 8), result_lt_247955)
    # Assigning a type to the variable 'if_condition_247956' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'if_condition_247956', if_condition_247956)
    # SSA begins for if statement (line 153)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 154):
    
    # Assigning a Name to a Name (line 154):
    # Getting the type of 'alpha' (line 154)
    alpha_247957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 26), 'alpha')
    # Assigning a type to the variable 'alpha_upper' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'alpha_upper', alpha_247957)
    # SSA join for if statement (line 153)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 156):
    
    # Assigning a BinOp to a Name (line 156):
    # Getting the type of 'phi' (line 156)
    phi_247958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'phi')
    # Getting the type of 'phi_prime' (line 156)
    phi_prime_247959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 22), 'phi_prime')
    # Applying the binary operator 'div' (line 156)
    result_div_247960 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 16), 'div', phi_247958, phi_prime_247959)
    
    # Assigning a type to the variable 'ratio' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'ratio', result_div_247960)
    
    # Assigning a Call to a Name (line 157):
    
    # Assigning a Call to a Name (line 157):
    
    # Call to max(...): (line 157)
    # Processing the call arguments (line 157)
    # Getting the type of 'alpha_lower' (line 157)
    alpha_lower_247962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 26), 'alpha_lower', False)
    # Getting the type of 'alpha' (line 157)
    alpha_247963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 39), 'alpha', False)
    # Getting the type of 'ratio' (line 157)
    ratio_247964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 47), 'ratio', False)
    # Applying the binary operator '-' (line 157)
    result_sub_247965 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 39), '-', alpha_247963, ratio_247964)
    
    # Processing the call keyword arguments (line 157)
    kwargs_247966 = {}
    # Getting the type of 'max' (line 157)
    max_247961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 22), 'max', False)
    # Calling max(args, kwargs) (line 157)
    max_call_result_247967 = invoke(stypy.reporting.localization.Localization(__file__, 157, 22), max_247961, *[alpha_lower_247962, result_sub_247965], **kwargs_247966)
    
    # Assigning a type to the variable 'alpha_lower' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'alpha_lower', max_call_result_247967)
    
    # Getting the type of 'alpha' (line 158)
    alpha_247968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'alpha')
    # Getting the type of 'phi' (line 158)
    phi_247969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 18), 'phi')
    # Getting the type of 'Delta' (line 158)
    Delta_247970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 24), 'Delta')
    # Applying the binary operator '+' (line 158)
    result_add_247971 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 18), '+', phi_247969, Delta_247970)
    
    # Getting the type of 'ratio' (line 158)
    ratio_247972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 33), 'ratio')
    # Applying the binary operator '*' (line 158)
    result_mul_247973 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 17), '*', result_add_247971, ratio_247972)
    
    # Getting the type of 'Delta' (line 158)
    Delta_247974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 41), 'Delta')
    # Applying the binary operator 'div' (line 158)
    result_div_247975 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 39), 'div', result_mul_247973, Delta_247974)
    
    # Applying the binary operator '-=' (line 158)
    result_isub_247976 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 8), '-=', alpha_247968, result_div_247975)
    # Assigning a type to the variable 'alpha' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'alpha', result_isub_247976)
    
    
    
    
    # Call to abs(...): (line 160)
    # Processing the call arguments (line 160)
    # Getting the type of 'phi' (line 160)
    phi_247979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 18), 'phi', False)
    # Processing the call keyword arguments (line 160)
    kwargs_247980 = {}
    # Getting the type of 'np' (line 160)
    np_247977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 11), 'np', False)
    # Obtaining the member 'abs' of a type (line 160)
    abs_247978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 11), np_247977, 'abs')
    # Calling abs(args, kwargs) (line 160)
    abs_call_result_247981 = invoke(stypy.reporting.localization.Localization(__file__, 160, 11), abs_247978, *[phi_247979], **kwargs_247980)
    
    # Getting the type of 'rtol' (line 160)
    rtol_247982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 25), 'rtol')
    # Getting the type of 'Delta' (line 160)
    Delta_247983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 32), 'Delta')
    # Applying the binary operator '*' (line 160)
    result_mul_247984 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 25), '*', rtol_247982, Delta_247983)
    
    # Applying the binary operator '<' (line 160)
    result_lt_247985 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 11), '<', abs_call_result_247981, result_mul_247984)
    
    # Testing the type of an if condition (line 160)
    if_condition_247986 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 8), result_lt_247985)
    # Assigning a type to the variable 'if_condition_247986' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'if_condition_247986', if_condition_247986)
    # SSA begins for if statement (line 160)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 160)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a UnaryOp to a Name (line 163):
    
    # Assigning a UnaryOp to a Name (line 163):
    
    
    # Call to dot(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'suf' (line 163)
    suf_247989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 15), 'suf', False)
    # Getting the type of 's' (line 163)
    s_247990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 22), 's', False)
    int_247991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 25), 'int')
    # Applying the binary operator '**' (line 163)
    result_pow_247992 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 22), '**', s_247990, int_247991)
    
    # Getting the type of 'alpha' (line 163)
    alpha_247993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 29), 'alpha', False)
    # Applying the binary operator '+' (line 163)
    result_add_247994 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 22), '+', result_pow_247992, alpha_247993)
    
    # Applying the binary operator 'div' (line 163)
    result_div_247995 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 15), 'div', suf_247989, result_add_247994)
    
    # Processing the call keyword arguments (line 163)
    kwargs_247996 = {}
    # Getting the type of 'V' (line 163)
    V_247987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 9), 'V', False)
    # Obtaining the member 'dot' of a type (line 163)
    dot_247988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 9), V_247987, 'dot')
    # Calling dot(args, kwargs) (line 163)
    dot_call_result_247997 = invoke(stypy.reporting.localization.Localization(__file__, 163, 9), dot_247988, *[result_div_247995], **kwargs_247996)
    
    # Applying the 'usub' unary operator (line 163)
    result___neg___247998 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 8), 'usub', dot_call_result_247997)
    
    # Assigning a type to the variable 'p' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'p', result___neg___247998)
    
    # Getting the type of 'p' (line 168)
    p_247999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'p')
    # Getting the type of 'Delta' (line 168)
    Delta_248000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 9), 'Delta')
    
    # Call to norm(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'p' (line 168)
    p_248002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 22), 'p', False)
    # Processing the call keyword arguments (line 168)
    kwargs_248003 = {}
    # Getting the type of 'norm' (line 168)
    norm_248001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 17), 'norm', False)
    # Calling norm(args, kwargs) (line 168)
    norm_call_result_248004 = invoke(stypy.reporting.localization.Localization(__file__, 168, 17), norm_248001, *[p_248002], **kwargs_248003)
    
    # Applying the binary operator 'div' (line 168)
    result_div_248005 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 9), 'div', Delta_248000, norm_call_result_248004)
    
    # Applying the binary operator '*=' (line 168)
    result_imul_248006 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 4), '*=', p_247999, result_div_248005)
    # Assigning a type to the variable 'p' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'p', result_imul_248006)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 170)
    tuple_248007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 170)
    # Adding element type (line 170)
    # Getting the type of 'p' (line 170)
    p_248008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 11), 'p')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 11), tuple_248007, p_248008)
    # Adding element type (line 170)
    # Getting the type of 'alpha' (line 170)
    alpha_248009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 14), 'alpha')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 11), tuple_248007, alpha_248009)
    # Adding element type (line 170)
    # Getting the type of 'it' (line 170)
    it_248010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 21), 'it')
    int_248011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 26), 'int')
    # Applying the binary operator '+' (line 170)
    result_add_248012 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 21), '+', it_248010, int_248011)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 11), tuple_248007, result_add_248012)
    
    # Assigning a type to the variable 'stypy_return_type' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'stypy_return_type', tuple_248007)
    
    # ################# End of 'solve_lsq_trust_region(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'solve_lsq_trust_region' in the type store
    # Getting the type of 'stypy_return_type' (line 59)
    stypy_return_type_248013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_248013)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'solve_lsq_trust_region'
    return stypy_return_type_248013

# Assigning a type to the variable 'solve_lsq_trust_region' (line 59)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'solve_lsq_trust_region', solve_lsq_trust_region)

@norecursion
def solve_trust_region_2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'solve_trust_region_2d'
    module_type_store = module_type_store.open_function_context('solve_trust_region_2d', 173, 0, False)
    
    # Passed parameters checking function
    solve_trust_region_2d.stypy_localization = localization
    solve_trust_region_2d.stypy_type_of_self = None
    solve_trust_region_2d.stypy_type_store = module_type_store
    solve_trust_region_2d.stypy_function_name = 'solve_trust_region_2d'
    solve_trust_region_2d.stypy_param_names_list = ['B', 'g', 'Delta']
    solve_trust_region_2d.stypy_varargs_param_name = None
    solve_trust_region_2d.stypy_kwargs_param_name = None
    solve_trust_region_2d.stypy_call_defaults = defaults
    solve_trust_region_2d.stypy_call_varargs = varargs
    solve_trust_region_2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'solve_trust_region_2d', ['B', 'g', 'Delta'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'solve_trust_region_2d', localization, ['B', 'g', 'Delta'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'solve_trust_region_2d(...)' code ##################

    str_248014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, (-1)), 'str', 'Solve a general trust-region problem in 2 dimensions.\n    \n    The problem is reformulated as a 4-th order algebraic equation,\n    the solution of which is found by numpy.roots.\n    \n    Parameters\n    ----------\n    B : ndarray, shape (2, 2)\n        Symmetric matrix, defines a quadratic term of the function.\n    g : ndarray, shape (2,)\n        Defines a linear term of the function.\n    Delta : float\n        Radius of a trust region.\n    \n    Returns\n    -------\n    p : ndarray, shape (2,)\n        Found solution.\n    newton_step : bool\n        Whether the returned solution is the Newton step which lies within\n        the trust region.\n    ')
    
    
    # SSA begins for try-except statement (line 196)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Tuple (line 197):
    
    # Assigning a Subscript to a Name (line 197):
    
    # Obtaining the type of the subscript
    int_248015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 8), 'int')
    
    # Call to cho_factor(...): (line 197)
    # Processing the call arguments (line 197)
    # Getting the type of 'B' (line 197)
    B_248017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 30), 'B', False)
    # Processing the call keyword arguments (line 197)
    kwargs_248018 = {}
    # Getting the type of 'cho_factor' (line 197)
    cho_factor_248016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 19), 'cho_factor', False)
    # Calling cho_factor(args, kwargs) (line 197)
    cho_factor_call_result_248019 = invoke(stypy.reporting.localization.Localization(__file__, 197, 19), cho_factor_248016, *[B_248017], **kwargs_248018)
    
    # Obtaining the member '__getitem__' of a type (line 197)
    getitem___248020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), cho_factor_call_result_248019, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 197)
    subscript_call_result_248021 = invoke(stypy.reporting.localization.Localization(__file__, 197, 8), getitem___248020, int_248015)
    
    # Assigning a type to the variable 'tuple_var_assignment_247673' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'tuple_var_assignment_247673', subscript_call_result_248021)
    
    # Assigning a Subscript to a Name (line 197):
    
    # Obtaining the type of the subscript
    int_248022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 8), 'int')
    
    # Call to cho_factor(...): (line 197)
    # Processing the call arguments (line 197)
    # Getting the type of 'B' (line 197)
    B_248024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 30), 'B', False)
    # Processing the call keyword arguments (line 197)
    kwargs_248025 = {}
    # Getting the type of 'cho_factor' (line 197)
    cho_factor_248023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 19), 'cho_factor', False)
    # Calling cho_factor(args, kwargs) (line 197)
    cho_factor_call_result_248026 = invoke(stypy.reporting.localization.Localization(__file__, 197, 19), cho_factor_248023, *[B_248024], **kwargs_248025)
    
    # Obtaining the member '__getitem__' of a type (line 197)
    getitem___248027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), cho_factor_call_result_248026, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 197)
    subscript_call_result_248028 = invoke(stypy.reporting.localization.Localization(__file__, 197, 8), getitem___248027, int_248022)
    
    # Assigning a type to the variable 'tuple_var_assignment_247674' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'tuple_var_assignment_247674', subscript_call_result_248028)
    
    # Assigning a Name to a Name (line 197):
    # Getting the type of 'tuple_var_assignment_247673' (line 197)
    tuple_var_assignment_247673_248029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'tuple_var_assignment_247673')
    # Assigning a type to the variable 'R' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'R', tuple_var_assignment_247673_248029)
    
    # Assigning a Name to a Name (line 197):
    # Getting the type of 'tuple_var_assignment_247674' (line 197)
    tuple_var_assignment_247674_248030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'tuple_var_assignment_247674')
    # Assigning a type to the variable 'lower' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 11), 'lower', tuple_var_assignment_247674_248030)
    
    # Assigning a UnaryOp to a Name (line 198):
    
    # Assigning a UnaryOp to a Name (line 198):
    
    
    # Call to cho_solve(...): (line 198)
    # Processing the call arguments (line 198)
    
    # Obtaining an instance of the builtin type 'tuple' (line 198)
    tuple_248032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 198)
    # Adding element type (line 198)
    # Getting the type of 'R' (line 198)
    R_248033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 24), 'R', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 24), tuple_248032, R_248033)
    # Adding element type (line 198)
    # Getting the type of 'lower' (line 198)
    lower_248034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 27), 'lower', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 24), tuple_248032, lower_248034)
    
    # Getting the type of 'g' (line 198)
    g_248035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 35), 'g', False)
    # Processing the call keyword arguments (line 198)
    kwargs_248036 = {}
    # Getting the type of 'cho_solve' (line 198)
    cho_solve_248031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 13), 'cho_solve', False)
    # Calling cho_solve(args, kwargs) (line 198)
    cho_solve_call_result_248037 = invoke(stypy.reporting.localization.Localization(__file__, 198, 13), cho_solve_248031, *[tuple_248032, g_248035], **kwargs_248036)
    
    # Applying the 'usub' unary operator (line 198)
    result___neg___248038 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 12), 'usub', cho_solve_call_result_248037)
    
    # Assigning a type to the variable 'p' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'p', result___neg___248038)
    
    
    
    # Call to dot(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'p' (line 199)
    p_248041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 18), 'p', False)
    # Getting the type of 'p' (line 199)
    p_248042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 21), 'p', False)
    # Processing the call keyword arguments (line 199)
    kwargs_248043 = {}
    # Getting the type of 'np' (line 199)
    np_248039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 11), 'np', False)
    # Obtaining the member 'dot' of a type (line 199)
    dot_248040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 11), np_248039, 'dot')
    # Calling dot(args, kwargs) (line 199)
    dot_call_result_248044 = invoke(stypy.reporting.localization.Localization(__file__, 199, 11), dot_248040, *[p_248041, p_248042], **kwargs_248043)
    
    # Getting the type of 'Delta' (line 199)
    Delta_248045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 27), 'Delta')
    int_248046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 34), 'int')
    # Applying the binary operator '**' (line 199)
    result_pow_248047 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 27), '**', Delta_248045, int_248046)
    
    # Applying the binary operator '<=' (line 199)
    result_le_248048 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 11), '<=', dot_call_result_248044, result_pow_248047)
    
    # Testing the type of an if condition (line 199)
    if_condition_248049 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 199, 8), result_le_248048)
    # Assigning a type to the variable 'if_condition_248049' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'if_condition_248049', if_condition_248049)
    # SSA begins for if statement (line 199)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 200)
    tuple_248050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 200)
    # Adding element type (line 200)
    # Getting the type of 'p' (line 200)
    p_248051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 19), 'p')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 19), tuple_248050, p_248051)
    # Adding element type (line 200)
    # Getting the type of 'True' (line 200)
    True_248052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 22), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 19), tuple_248050, True_248052)
    
    # Assigning a type to the variable 'stypy_return_type' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'stypy_return_type', tuple_248050)
    # SSA join for if statement (line 199)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 196)
    # SSA branch for the except 'LinAlgError' branch of a try statement (line 196)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 196)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 204):
    
    # Assigning a BinOp to a Name (line 204):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 204)
    tuple_248053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 204)
    # Adding element type (line 204)
    int_248054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 10), tuple_248053, int_248054)
    # Adding element type (line 204)
    int_248055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 10), tuple_248053, int_248055)
    
    # Getting the type of 'B' (line 204)
    B_248056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'B')
    # Obtaining the member '__getitem__' of a type (line 204)
    getitem___248057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), B_248056, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 204)
    subscript_call_result_248058 = invoke(stypy.reporting.localization.Localization(__file__, 204, 8), getitem___248057, tuple_248053)
    
    # Getting the type of 'Delta' (line 204)
    Delta_248059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 18), 'Delta')
    int_248060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 25), 'int')
    # Applying the binary operator '**' (line 204)
    result_pow_248061 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 18), '**', Delta_248059, int_248060)
    
    # Applying the binary operator '*' (line 204)
    result_mul_248062 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 8), '*', subscript_call_result_248058, result_pow_248061)
    
    # Assigning a type to the variable 'a' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'a', result_mul_248062)
    
    # Assigning a BinOp to a Name (line 205):
    
    # Assigning a BinOp to a Name (line 205):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 205)
    tuple_248063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 205)
    # Adding element type (line 205)
    int_248064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 10), tuple_248063, int_248064)
    # Adding element type (line 205)
    int_248065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 10), tuple_248063, int_248065)
    
    # Getting the type of 'B' (line 205)
    B_248066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'B')
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___248067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), B_248066, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_248068 = invoke(stypy.reporting.localization.Localization(__file__, 205, 8), getitem___248067, tuple_248063)
    
    # Getting the type of 'Delta' (line 205)
    Delta_248069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 18), 'Delta')
    int_248070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 25), 'int')
    # Applying the binary operator '**' (line 205)
    result_pow_248071 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 18), '**', Delta_248069, int_248070)
    
    # Applying the binary operator '*' (line 205)
    result_mul_248072 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 8), '*', subscript_call_result_248068, result_pow_248071)
    
    # Assigning a type to the variable 'b' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'b', result_mul_248072)
    
    # Assigning a BinOp to a Name (line 206):
    
    # Assigning a BinOp to a Name (line 206):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 206)
    tuple_248073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 206)
    # Adding element type (line 206)
    int_248074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 10), tuple_248073, int_248074)
    # Adding element type (line 206)
    int_248075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 10), tuple_248073, int_248075)
    
    # Getting the type of 'B' (line 206)
    B_248076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'B')
    # Obtaining the member '__getitem__' of a type (line 206)
    getitem___248077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), B_248076, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 206)
    subscript_call_result_248078 = invoke(stypy.reporting.localization.Localization(__file__, 206, 8), getitem___248077, tuple_248073)
    
    # Getting the type of 'Delta' (line 206)
    Delta_248079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 18), 'Delta')
    int_248080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 25), 'int')
    # Applying the binary operator '**' (line 206)
    result_pow_248081 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 18), '**', Delta_248079, int_248080)
    
    # Applying the binary operator '*' (line 206)
    result_mul_248082 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 8), '*', subscript_call_result_248078, result_pow_248081)
    
    # Assigning a type to the variable 'c' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'c', result_mul_248082)
    
    # Assigning a BinOp to a Name (line 208):
    
    # Assigning a BinOp to a Name (line 208):
    
    # Obtaining the type of the subscript
    int_248083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 10), 'int')
    # Getting the type of 'g' (line 208)
    g_248084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'g')
    # Obtaining the member '__getitem__' of a type (line 208)
    getitem___248085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), g_248084, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 208)
    subscript_call_result_248086 = invoke(stypy.reporting.localization.Localization(__file__, 208, 8), getitem___248085, int_248083)
    
    # Getting the type of 'Delta' (line 208)
    Delta_248087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 'Delta')
    # Applying the binary operator '*' (line 208)
    result_mul_248088 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 8), '*', subscript_call_result_248086, Delta_248087)
    
    # Assigning a type to the variable 'd' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'd', result_mul_248088)
    
    # Assigning a BinOp to a Name (line 209):
    
    # Assigning a BinOp to a Name (line 209):
    
    # Obtaining the type of the subscript
    int_248089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 10), 'int')
    # Getting the type of 'g' (line 209)
    g_248090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'g')
    # Obtaining the member '__getitem__' of a type (line 209)
    getitem___248091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 8), g_248090, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 209)
    subscript_call_result_248092 = invoke(stypy.reporting.localization.Localization(__file__, 209, 8), getitem___248091, int_248089)
    
    # Getting the type of 'Delta' (line 209)
    Delta_248093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 15), 'Delta')
    # Applying the binary operator '*' (line 209)
    result_mul_248094 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 8), '*', subscript_call_result_248092, Delta_248093)
    
    # Assigning a type to the variable 'f' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'f', result_mul_248094)
    
    # Assigning a Call to a Name (line 211):
    
    # Assigning a Call to a Name (line 211):
    
    # Call to array(...): (line 211)
    # Processing the call arguments (line 211)
    
    # Obtaining an instance of the builtin type 'list' (line 212)
    list_248097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 212)
    # Adding element type (line 212)
    
    # Getting the type of 'b' (line 212)
    b_248098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 10), 'b', False)
    # Applying the 'usub' unary operator (line 212)
    result___neg___248099 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 9), 'usub', b_248098)
    
    # Getting the type of 'd' (line 212)
    d_248100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 14), 'd', False)
    # Applying the binary operator '+' (line 212)
    result_add_248101 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 9), '+', result___neg___248099, d_248100)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 8), list_248097, result_add_248101)
    # Adding element type (line 212)
    int_248102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 17), 'int')
    # Getting the type of 'a' (line 212)
    a_248103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 22), 'a', False)
    # Getting the type of 'c' (line 212)
    c_248104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 26), 'c', False)
    # Applying the binary operator '-' (line 212)
    result_sub_248105 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 22), '-', a_248103, c_248104)
    
    # Getting the type of 'f' (line 212)
    f_248106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 30), 'f', False)
    # Applying the binary operator '+' (line 212)
    result_add_248107 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 28), '+', result_sub_248105, f_248106)
    
    # Applying the binary operator '*' (line 212)
    result_mul_248108 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 17), '*', int_248102, result_add_248107)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 8), list_248097, result_mul_248108)
    # Adding element type (line 212)
    int_248109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 34), 'int')
    # Getting the type of 'b' (line 212)
    b_248110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 38), 'b', False)
    # Applying the binary operator '*' (line 212)
    result_mul_248111 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 34), '*', int_248109, b_248110)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 8), list_248097, result_mul_248111)
    # Adding element type (line 212)
    int_248112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 41), 'int')
    
    # Getting the type of 'a' (line 212)
    a_248113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 47), 'a', False)
    # Applying the 'usub' unary operator (line 212)
    result___neg___248114 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 46), 'usub', a_248113)
    
    # Getting the type of 'c' (line 212)
    c_248115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 51), 'c', False)
    # Applying the binary operator '+' (line 212)
    result_add_248116 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 46), '+', result___neg___248114, c_248115)
    
    # Getting the type of 'f' (line 212)
    f_248117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 55), 'f', False)
    # Applying the binary operator '+' (line 212)
    result_add_248118 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 53), '+', result_add_248116, f_248117)
    
    # Applying the binary operator '*' (line 212)
    result_mul_248119 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 41), '*', int_248112, result_add_248118)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 8), list_248097, result_mul_248119)
    # Adding element type (line 212)
    
    # Getting the type of 'b' (line 212)
    b_248120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 60), 'b', False)
    # Applying the 'usub' unary operator (line 212)
    result___neg___248121 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 59), 'usub', b_248120)
    
    # Getting the type of 'd' (line 212)
    d_248122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 64), 'd', False)
    # Applying the binary operator '-' (line 212)
    result_sub_248123 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 59), '-', result___neg___248121, d_248122)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 8), list_248097, result_sub_248123)
    
    # Processing the call keyword arguments (line 211)
    kwargs_248124 = {}
    # Getting the type of 'np' (line 211)
    np_248095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 13), 'np', False)
    # Obtaining the member 'array' of a type (line 211)
    array_248096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 13), np_248095, 'array')
    # Calling array(args, kwargs) (line 211)
    array_call_result_248125 = invoke(stypy.reporting.localization.Localization(__file__, 211, 13), array_248096, *[list_248097], **kwargs_248124)
    
    # Assigning a type to the variable 'coeffs' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'coeffs', array_call_result_248125)
    
    # Assigning a Call to a Name (line 213):
    
    # Assigning a Call to a Name (line 213):
    
    # Call to roots(...): (line 213)
    # Processing the call arguments (line 213)
    # Getting the type of 'coeffs' (line 213)
    coeffs_248128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 17), 'coeffs', False)
    # Processing the call keyword arguments (line 213)
    kwargs_248129 = {}
    # Getting the type of 'np' (line 213)
    np_248126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'np', False)
    # Obtaining the member 'roots' of a type (line 213)
    roots_248127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 8), np_248126, 'roots')
    # Calling roots(args, kwargs) (line 213)
    roots_call_result_248130 = invoke(stypy.reporting.localization.Localization(__file__, 213, 8), roots_248127, *[coeffs_248128], **kwargs_248129)
    
    # Assigning a type to the variable 't' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 't', roots_call_result_248130)
    
    # Assigning a Call to a Name (line 214):
    
    # Assigning a Call to a Name (line 214):
    
    # Call to real(...): (line 214)
    # Processing the call arguments (line 214)
    
    # Obtaining the type of the subscript
    
    # Call to isreal(...): (line 214)
    # Processing the call arguments (line 214)
    # Getting the type of 't' (line 214)
    t_248135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 28), 't', False)
    # Processing the call keyword arguments (line 214)
    kwargs_248136 = {}
    # Getting the type of 'np' (line 214)
    np_248133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 18), 'np', False)
    # Obtaining the member 'isreal' of a type (line 214)
    isreal_248134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 18), np_248133, 'isreal')
    # Calling isreal(args, kwargs) (line 214)
    isreal_call_result_248137 = invoke(stypy.reporting.localization.Localization(__file__, 214, 18), isreal_248134, *[t_248135], **kwargs_248136)
    
    # Getting the type of 't' (line 214)
    t_248138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 16), 't', False)
    # Obtaining the member '__getitem__' of a type (line 214)
    getitem___248139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 16), t_248138, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 214)
    subscript_call_result_248140 = invoke(stypy.reporting.localization.Localization(__file__, 214, 16), getitem___248139, isreal_call_result_248137)
    
    # Processing the call keyword arguments (line 214)
    kwargs_248141 = {}
    # Getting the type of 'np' (line 214)
    np_248131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'np', False)
    # Obtaining the member 'real' of a type (line 214)
    real_248132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), np_248131, 'real')
    # Calling real(args, kwargs) (line 214)
    real_call_result_248142 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), real_248132, *[subscript_call_result_248140], **kwargs_248141)
    
    # Assigning a type to the variable 't' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 't', real_call_result_248142)
    
    # Assigning a BinOp to a Name (line 216):
    
    # Assigning a BinOp to a Name (line 216):
    # Getting the type of 'Delta' (line 216)
    Delta_248143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'Delta')
    
    # Call to vstack(...): (line 216)
    # Processing the call arguments (line 216)
    
    # Obtaining an instance of the builtin type 'tuple' (line 216)
    tuple_248146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 216)
    # Adding element type (line 216)
    int_248147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 27), 'int')
    # Getting the type of 't' (line 216)
    t_248148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 31), 't', False)
    # Applying the binary operator '*' (line 216)
    result_mul_248149 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 27), '*', int_248147, t_248148)
    
    int_248150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 36), 'int')
    # Getting the type of 't' (line 216)
    t_248151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 40), 't', False)
    int_248152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 43), 'int')
    # Applying the binary operator '**' (line 216)
    result_pow_248153 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 40), '**', t_248151, int_248152)
    
    # Applying the binary operator '+' (line 216)
    result_add_248154 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 36), '+', int_248150, result_pow_248153)
    
    # Applying the binary operator 'div' (line 216)
    result_div_248155 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 33), 'div', result_mul_248149, result_add_248154)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 27), tuple_248146, result_div_248155)
    # Adding element type (line 216)
    int_248156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 48), 'int')
    # Getting the type of 't' (line 216)
    t_248157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 52), 't', False)
    int_248158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 55), 'int')
    # Applying the binary operator '**' (line 216)
    result_pow_248159 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 52), '**', t_248157, int_248158)
    
    # Applying the binary operator '-' (line 216)
    result_sub_248160 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 48), '-', int_248156, result_pow_248159)
    
    int_248161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 61), 'int')
    # Getting the type of 't' (line 216)
    t_248162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 65), 't', False)
    int_248163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 68), 'int')
    # Applying the binary operator '**' (line 216)
    result_pow_248164 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 65), '**', t_248162, int_248163)
    
    # Applying the binary operator '+' (line 216)
    result_add_248165 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 61), '+', int_248161, result_pow_248164)
    
    # Applying the binary operator 'div' (line 216)
    result_div_248166 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 47), 'div', result_sub_248160, result_add_248165)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 27), tuple_248146, result_div_248166)
    
    # Processing the call keyword arguments (line 216)
    kwargs_248167 = {}
    # Getting the type of 'np' (line 216)
    np_248144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'np', False)
    # Obtaining the member 'vstack' of a type (line 216)
    vstack_248145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 16), np_248144, 'vstack')
    # Calling vstack(args, kwargs) (line 216)
    vstack_call_result_248168 = invoke(stypy.reporting.localization.Localization(__file__, 216, 16), vstack_248145, *[tuple_248146], **kwargs_248167)
    
    # Applying the binary operator '*' (line 216)
    result_mul_248169 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 8), '*', Delta_248143, vstack_call_result_248168)
    
    # Assigning a type to the variable 'p' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'p', result_mul_248169)
    
    # Assigning a BinOp to a Name (line 217):
    
    # Assigning a BinOp to a Name (line 217):
    float_248170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 12), 'float')
    
    # Call to sum(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'p' (line 217)
    p_248173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 25), 'p', False)
    
    # Call to dot(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'p' (line 217)
    p_248176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 35), 'p', False)
    # Processing the call keyword arguments (line 217)
    kwargs_248177 = {}
    # Getting the type of 'B' (line 217)
    B_248174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 29), 'B', False)
    # Obtaining the member 'dot' of a type (line 217)
    dot_248175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 29), B_248174, 'dot')
    # Calling dot(args, kwargs) (line 217)
    dot_call_result_248178 = invoke(stypy.reporting.localization.Localization(__file__, 217, 29), dot_248175, *[p_248176], **kwargs_248177)
    
    # Applying the binary operator '*' (line 217)
    result_mul_248179 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 25), '*', p_248173, dot_call_result_248178)
    
    # Processing the call keyword arguments (line 217)
    int_248180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 44), 'int')
    keyword_248181 = int_248180
    kwargs_248182 = {'axis': keyword_248181}
    # Getting the type of 'np' (line 217)
    np_248171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 18), 'np', False)
    # Obtaining the member 'sum' of a type (line 217)
    sum_248172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 18), np_248171, 'sum')
    # Calling sum(args, kwargs) (line 217)
    sum_call_result_248183 = invoke(stypy.reporting.localization.Localization(__file__, 217, 18), sum_248172, *[result_mul_248179], **kwargs_248182)
    
    # Applying the binary operator '*' (line 217)
    result_mul_248184 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 12), '*', float_248170, sum_call_result_248183)
    
    
    # Call to dot(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'g' (line 217)
    g_248187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 56), 'g', False)
    # Getting the type of 'p' (line 217)
    p_248188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 59), 'p', False)
    # Processing the call keyword arguments (line 217)
    kwargs_248189 = {}
    # Getting the type of 'np' (line 217)
    np_248185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 49), 'np', False)
    # Obtaining the member 'dot' of a type (line 217)
    dot_248186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 49), np_248185, 'dot')
    # Calling dot(args, kwargs) (line 217)
    dot_call_result_248190 = invoke(stypy.reporting.localization.Localization(__file__, 217, 49), dot_248186, *[g_248187, p_248188], **kwargs_248189)
    
    # Applying the binary operator '+' (line 217)
    result_add_248191 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 12), '+', result_mul_248184, dot_call_result_248190)
    
    # Assigning a type to the variable 'value' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'value', result_add_248191)
    
    # Assigning a Call to a Name (line 218):
    
    # Assigning a Call to a Name (line 218):
    
    # Call to argmin(...): (line 218)
    # Processing the call arguments (line 218)
    # Getting the type of 'value' (line 218)
    value_248194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 18), 'value', False)
    # Processing the call keyword arguments (line 218)
    kwargs_248195 = {}
    # Getting the type of 'np' (line 218)
    np_248192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'np', False)
    # Obtaining the member 'argmin' of a type (line 218)
    argmin_248193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), np_248192, 'argmin')
    # Calling argmin(args, kwargs) (line 218)
    argmin_call_result_248196 = invoke(stypy.reporting.localization.Localization(__file__, 218, 8), argmin_248193, *[value_248194], **kwargs_248195)
    
    # Assigning a type to the variable 'i' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'i', argmin_call_result_248196)
    
    # Assigning a Subscript to a Name (line 219):
    
    # Assigning a Subscript to a Name (line 219):
    
    # Obtaining the type of the subscript
    slice_248197 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 219, 8), None, None, None)
    # Getting the type of 'i' (line 219)
    i_248198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 13), 'i')
    # Getting the type of 'p' (line 219)
    p_248199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'p')
    # Obtaining the member '__getitem__' of a type (line 219)
    getitem___248200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 8), p_248199, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 219)
    subscript_call_result_248201 = invoke(stypy.reporting.localization.Localization(__file__, 219, 8), getitem___248200, (slice_248197, i_248198))
    
    # Assigning a type to the variable 'p' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'p', subscript_call_result_248201)
    
    # Obtaining an instance of the builtin type 'tuple' (line 221)
    tuple_248202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 221)
    # Adding element type (line 221)
    # Getting the type of 'p' (line 221)
    p_248203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 11), 'p')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 11), tuple_248202, p_248203)
    # Adding element type (line 221)
    # Getting the type of 'False' (line 221)
    False_248204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 14), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 11), tuple_248202, False_248204)
    
    # Assigning a type to the variable 'stypy_return_type' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'stypy_return_type', tuple_248202)
    
    # ################# End of 'solve_trust_region_2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'solve_trust_region_2d' in the type store
    # Getting the type of 'stypy_return_type' (line 173)
    stypy_return_type_248205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_248205)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'solve_trust_region_2d'
    return stypy_return_type_248205

# Assigning a type to the variable 'solve_trust_region_2d' (line 173)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 0), 'solve_trust_region_2d', solve_trust_region_2d)

@norecursion
def update_tr_radius(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'update_tr_radius'
    module_type_store = module_type_store.open_function_context('update_tr_radius', 224, 0, False)
    
    # Passed parameters checking function
    update_tr_radius.stypy_localization = localization
    update_tr_radius.stypy_type_of_self = None
    update_tr_radius.stypy_type_store = module_type_store
    update_tr_radius.stypy_function_name = 'update_tr_radius'
    update_tr_radius.stypy_param_names_list = ['Delta', 'actual_reduction', 'predicted_reduction', 'step_norm', 'bound_hit']
    update_tr_radius.stypy_varargs_param_name = None
    update_tr_radius.stypy_kwargs_param_name = None
    update_tr_radius.stypy_call_defaults = defaults
    update_tr_radius.stypy_call_varargs = varargs
    update_tr_radius.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'update_tr_radius', ['Delta', 'actual_reduction', 'predicted_reduction', 'step_norm', 'bound_hit'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'update_tr_radius', localization, ['Delta', 'actual_reduction', 'predicted_reduction', 'step_norm', 'bound_hit'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'update_tr_radius(...)' code ##################

    str_248206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, (-1)), 'str', 'Update the radius of a trust region based on the cost reduction.\n\n    Returns\n    -------\n    Delta : float\n        New radius.\n    ratio : float\n        Ratio between actual and predicted reductions. Zero if predicted\n        reduction is zero.\n    ')
    
    
    # Getting the type of 'predicted_reduction' (line 236)
    predicted_reduction_248207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 7), 'predicted_reduction')
    int_248208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 29), 'int')
    # Applying the binary operator '>' (line 236)
    result_gt_248209 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 7), '>', predicted_reduction_248207, int_248208)
    
    # Testing the type of an if condition (line 236)
    if_condition_248210 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 4), result_gt_248209)
    # Assigning a type to the variable 'if_condition_248210' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'if_condition_248210', if_condition_248210)
    # SSA begins for if statement (line 236)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 237):
    
    # Assigning a BinOp to a Name (line 237):
    # Getting the type of 'actual_reduction' (line 237)
    actual_reduction_248211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 16), 'actual_reduction')
    # Getting the type of 'predicted_reduction' (line 237)
    predicted_reduction_248212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 35), 'predicted_reduction')
    # Applying the binary operator 'div' (line 237)
    result_div_248213 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 16), 'div', actual_reduction_248211, predicted_reduction_248212)
    
    # Assigning a type to the variable 'ratio' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'ratio', result_div_248213)
    # SSA branch for the else part of an if statement (line 236)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 239):
    
    # Assigning a Num to a Name (line 239):
    int_248214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 16), 'int')
    # Assigning a type to the variable 'ratio' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'ratio', int_248214)
    # SSA join for if statement (line 236)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'ratio' (line 241)
    ratio_248215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 7), 'ratio')
    float_248216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 15), 'float')
    # Applying the binary operator '<' (line 241)
    result_lt_248217 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 7), '<', ratio_248215, float_248216)
    
    # Testing the type of an if condition (line 241)
    if_condition_248218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 4), result_lt_248217)
    # Assigning a type to the variable 'if_condition_248218' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'if_condition_248218', if_condition_248218)
    # SSA begins for if statement (line 241)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 242):
    
    # Assigning a BinOp to a Name (line 242):
    float_248219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 16), 'float')
    # Getting the type of 'step_norm' (line 242)
    step_norm_248220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 23), 'step_norm')
    # Applying the binary operator '*' (line 242)
    result_mul_248221 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 16), '*', float_248219, step_norm_248220)
    
    # Assigning a type to the variable 'Delta' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'Delta', result_mul_248221)
    # SSA branch for the else part of an if statement (line 241)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'ratio' (line 243)
    ratio_248222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 9), 'ratio')
    float_248223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 17), 'float')
    # Applying the binary operator '>' (line 243)
    result_gt_248224 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 9), '>', ratio_248222, float_248223)
    
    # Getting the type of 'bound_hit' (line 243)
    bound_hit_248225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 26), 'bound_hit')
    # Applying the binary operator 'and' (line 243)
    result_and_keyword_248226 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 9), 'and', result_gt_248224, bound_hit_248225)
    
    # Testing the type of an if condition (line 243)
    if_condition_248227 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 243, 9), result_and_keyword_248226)
    # Assigning a type to the variable 'if_condition_248227' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 9), 'if_condition_248227', if_condition_248227)
    # SSA begins for if statement (line 243)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'Delta' (line 244)
    Delta_248228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'Delta')
    float_248229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 17), 'float')
    # Applying the binary operator '*=' (line 244)
    result_imul_248230 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 8), '*=', Delta_248228, float_248229)
    # Assigning a type to the variable 'Delta' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'Delta', result_imul_248230)
    
    # SSA join for if statement (line 243)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 241)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 246)
    tuple_248231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 246)
    # Adding element type (line 246)
    # Getting the type of 'Delta' (line 246)
    Delta_248232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 11), 'Delta')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 11), tuple_248231, Delta_248232)
    # Adding element type (line 246)
    # Getting the type of 'ratio' (line 246)
    ratio_248233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 18), 'ratio')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 11), tuple_248231, ratio_248233)
    
    # Assigning a type to the variable 'stypy_return_type' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'stypy_return_type', tuple_248231)
    
    # ################# End of 'update_tr_radius(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'update_tr_radius' in the type store
    # Getting the type of 'stypy_return_type' (line 224)
    stypy_return_type_248234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_248234)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'update_tr_radius'
    return stypy_return_type_248234

# Assigning a type to the variable 'update_tr_radius' (line 224)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 0), 'update_tr_radius', update_tr_radius)

@norecursion
def build_quadratic_1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 252)
    None_248235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 37), 'None')
    # Getting the type of 'None' (line 252)
    None_248236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 46), 'None')
    defaults = [None_248235, None_248236]
    # Create a new context for function 'build_quadratic_1d'
    module_type_store = module_type_store.open_function_context('build_quadratic_1d', 252, 0, False)
    
    # Passed parameters checking function
    build_quadratic_1d.stypy_localization = localization
    build_quadratic_1d.stypy_type_of_self = None
    build_quadratic_1d.stypy_type_store = module_type_store
    build_quadratic_1d.stypy_function_name = 'build_quadratic_1d'
    build_quadratic_1d.stypy_param_names_list = ['J', 'g', 's', 'diag', 's0']
    build_quadratic_1d.stypy_varargs_param_name = None
    build_quadratic_1d.stypy_kwargs_param_name = None
    build_quadratic_1d.stypy_call_defaults = defaults
    build_quadratic_1d.stypy_call_varargs = varargs
    build_quadratic_1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'build_quadratic_1d', ['J', 'g', 's', 'diag', 's0'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'build_quadratic_1d', localization, ['J', 'g', 's', 'diag', 's0'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'build_quadratic_1d(...)' code ##################

    str_248237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, (-1)), 'str', 'Parameterize a multivariate quadratic function along a line.\n    \n    The resulting univariate quadratic function is given as follows:\n    ::\n        f(t) = 0.5 * (s0 + s*t).T * (J.T*J + diag) * (s0 + s*t) +\n               g.T * (s0 + s*t)\n    \n    Parameters\n    ----------\n    J : ndarray, sparse matrix or LinearOperator shape (m, n)\n        Jacobian matrix, affects the quadratic term.\n    g : ndarray, shape (n,)\n        Gradient, defines the linear term.\n    s : ndarray, shape (n,)\n        Direction vector of a line.\n    diag : None or ndarray with shape (n,), optional\n        Addition diagonal part, affects the quadratic term.\n        If None, assumed to be 0.\n    s0 : None or ndarray with shape (n,), optional\n        Initial point. If None, assumed to be 0.\n    \n    Returns\n    -------\n    a : float\n        Coefficient for t**2.\n    b : float\n        Coefficient for t.\n    c : float\n        Free term. Returned only if `s0` is provided.\n    ')
    
    # Assigning a Call to a Name (line 283):
    
    # Assigning a Call to a Name (line 283):
    
    # Call to dot(...): (line 283)
    # Processing the call arguments (line 283)
    # Getting the type of 's' (line 283)
    s_248240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 14), 's', False)
    # Processing the call keyword arguments (line 283)
    kwargs_248241 = {}
    # Getting the type of 'J' (line 283)
    J_248238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'J', False)
    # Obtaining the member 'dot' of a type (line 283)
    dot_248239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), J_248238, 'dot')
    # Calling dot(args, kwargs) (line 283)
    dot_call_result_248242 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), dot_248239, *[s_248240], **kwargs_248241)
    
    # Assigning a type to the variable 'v' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'v', dot_call_result_248242)
    
    # Assigning a Call to a Name (line 284):
    
    # Assigning a Call to a Name (line 284):
    
    # Call to dot(...): (line 284)
    # Processing the call arguments (line 284)
    # Getting the type of 'v' (line 284)
    v_248245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 15), 'v', False)
    # Getting the type of 'v' (line 284)
    v_248246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 18), 'v', False)
    # Processing the call keyword arguments (line 284)
    kwargs_248247 = {}
    # Getting the type of 'np' (line 284)
    np_248243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'np', False)
    # Obtaining the member 'dot' of a type (line 284)
    dot_248244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 8), np_248243, 'dot')
    # Calling dot(args, kwargs) (line 284)
    dot_call_result_248248 = invoke(stypy.reporting.localization.Localization(__file__, 284, 8), dot_248244, *[v_248245, v_248246], **kwargs_248247)
    
    # Assigning a type to the variable 'a' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'a', dot_call_result_248248)
    
    # Type idiom detected: calculating its left and rigth part (line 285)
    # Getting the type of 'diag' (line 285)
    diag_248249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'diag')
    # Getting the type of 'None' (line 285)
    None_248250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 19), 'None')
    
    (may_be_248251, more_types_in_union_248252) = may_not_be_none(diag_248249, None_248250)

    if may_be_248251:

        if more_types_in_union_248252:
            # Runtime conditional SSA (line 285)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Getting the type of 'a' (line 286)
        a_248253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'a')
        
        # Call to dot(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 's' (line 286)
        s_248256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 20), 's', False)
        # Getting the type of 'diag' (line 286)
        diag_248257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 24), 'diag', False)
        # Applying the binary operator '*' (line 286)
        result_mul_248258 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 20), '*', s_248256, diag_248257)
        
        # Getting the type of 's' (line 286)
        s_248259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 30), 's', False)
        # Processing the call keyword arguments (line 286)
        kwargs_248260 = {}
        # Getting the type of 'np' (line 286)
        np_248254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 13), 'np', False)
        # Obtaining the member 'dot' of a type (line 286)
        dot_248255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 13), np_248254, 'dot')
        # Calling dot(args, kwargs) (line 286)
        dot_call_result_248261 = invoke(stypy.reporting.localization.Localization(__file__, 286, 13), dot_248255, *[result_mul_248258, s_248259], **kwargs_248260)
        
        # Applying the binary operator '+=' (line 286)
        result_iadd_248262 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 8), '+=', a_248253, dot_call_result_248261)
        # Assigning a type to the variable 'a' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'a', result_iadd_248262)
        

        if more_types_in_union_248252:
            # SSA join for if statement (line 285)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'a' (line 287)
    a_248263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'a')
    float_248264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 9), 'float')
    # Applying the binary operator '*=' (line 287)
    result_imul_248265 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 4), '*=', a_248263, float_248264)
    # Assigning a type to the variable 'a' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'a', result_imul_248265)
    
    
    # Assigning a Call to a Name (line 289):
    
    # Assigning a Call to a Name (line 289):
    
    # Call to dot(...): (line 289)
    # Processing the call arguments (line 289)
    # Getting the type of 'g' (line 289)
    g_248268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 15), 'g', False)
    # Getting the type of 's' (line 289)
    s_248269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 18), 's', False)
    # Processing the call keyword arguments (line 289)
    kwargs_248270 = {}
    # Getting the type of 'np' (line 289)
    np_248266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'np', False)
    # Obtaining the member 'dot' of a type (line 289)
    dot_248267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 8), np_248266, 'dot')
    # Calling dot(args, kwargs) (line 289)
    dot_call_result_248271 = invoke(stypy.reporting.localization.Localization(__file__, 289, 8), dot_248267, *[g_248268, s_248269], **kwargs_248270)
    
    # Assigning a type to the variable 'b' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'b', dot_call_result_248271)
    
    # Type idiom detected: calculating its left and rigth part (line 291)
    # Getting the type of 's0' (line 291)
    s0_248272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 's0')
    # Getting the type of 'None' (line 291)
    None_248273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 17), 'None')
    
    (may_be_248274, more_types_in_union_248275) = may_not_be_none(s0_248272, None_248273)

    if may_be_248274:

        if more_types_in_union_248275:
            # Runtime conditional SSA (line 291)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 292):
        
        # Assigning a Call to a Name (line 292):
        
        # Call to dot(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 's0' (line 292)
        s0_248278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 18), 's0', False)
        # Processing the call keyword arguments (line 292)
        kwargs_248279 = {}
        # Getting the type of 'J' (line 292)
        J_248276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'J', False)
        # Obtaining the member 'dot' of a type (line 292)
        dot_248277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 12), J_248276, 'dot')
        # Calling dot(args, kwargs) (line 292)
        dot_call_result_248280 = invoke(stypy.reporting.localization.Localization(__file__, 292, 12), dot_248277, *[s0_248278], **kwargs_248279)
        
        # Assigning a type to the variable 'u' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'u', dot_call_result_248280)
        
        # Getting the type of 'b' (line 293)
        b_248281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'b')
        
        # Call to dot(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'u' (line 293)
        u_248284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 20), 'u', False)
        # Getting the type of 'v' (line 293)
        v_248285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 23), 'v', False)
        # Processing the call keyword arguments (line 293)
        kwargs_248286 = {}
        # Getting the type of 'np' (line 293)
        np_248282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 13), 'np', False)
        # Obtaining the member 'dot' of a type (line 293)
        dot_248283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 13), np_248282, 'dot')
        # Calling dot(args, kwargs) (line 293)
        dot_call_result_248287 = invoke(stypy.reporting.localization.Localization(__file__, 293, 13), dot_248283, *[u_248284, v_248285], **kwargs_248286)
        
        # Applying the binary operator '+=' (line 293)
        result_iadd_248288 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 8), '+=', b_248281, dot_call_result_248287)
        # Assigning a type to the variable 'b' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'b', result_iadd_248288)
        
        
        # Assigning a BinOp to a Name (line 294):
        
        # Assigning a BinOp to a Name (line 294):
        float_248289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 12), 'float')
        
        # Call to dot(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'u' (line 294)
        u_248292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 25), 'u', False)
        # Getting the type of 'u' (line 294)
        u_248293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 28), 'u', False)
        # Processing the call keyword arguments (line 294)
        kwargs_248294 = {}
        # Getting the type of 'np' (line 294)
        np_248290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 18), 'np', False)
        # Obtaining the member 'dot' of a type (line 294)
        dot_248291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 18), np_248290, 'dot')
        # Calling dot(args, kwargs) (line 294)
        dot_call_result_248295 = invoke(stypy.reporting.localization.Localization(__file__, 294, 18), dot_248291, *[u_248292, u_248293], **kwargs_248294)
        
        # Applying the binary operator '*' (line 294)
        result_mul_248296 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 12), '*', float_248289, dot_call_result_248295)
        
        
        # Call to dot(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'g' (line 294)
        g_248299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 40), 'g', False)
        # Getting the type of 's0' (line 294)
        s0_248300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 43), 's0', False)
        # Processing the call keyword arguments (line 294)
        kwargs_248301 = {}
        # Getting the type of 'np' (line 294)
        np_248297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 33), 'np', False)
        # Obtaining the member 'dot' of a type (line 294)
        dot_248298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 33), np_248297, 'dot')
        # Calling dot(args, kwargs) (line 294)
        dot_call_result_248302 = invoke(stypy.reporting.localization.Localization(__file__, 294, 33), dot_248298, *[g_248299, s0_248300], **kwargs_248301)
        
        # Applying the binary operator '+' (line 294)
        result_add_248303 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 12), '+', result_mul_248296, dot_call_result_248302)
        
        # Assigning a type to the variable 'c' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'c', result_add_248303)
        
        # Type idiom detected: calculating its left and rigth part (line 295)
        # Getting the type of 'diag' (line 295)
        diag_248304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'diag')
        # Getting the type of 'None' (line 295)
        None_248305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 23), 'None')
        
        (may_be_248306, more_types_in_union_248307) = may_not_be_none(diag_248304, None_248305)

        if may_be_248306:

            if more_types_in_union_248307:
                # Runtime conditional SSA (line 295)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Getting the type of 'b' (line 296)
            b_248308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'b')
            
            # Call to dot(...): (line 296)
            # Processing the call arguments (line 296)
            # Getting the type of 's0' (line 296)
            s0_248311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 24), 's0', False)
            # Getting the type of 'diag' (line 296)
            diag_248312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 29), 'diag', False)
            # Applying the binary operator '*' (line 296)
            result_mul_248313 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 24), '*', s0_248311, diag_248312)
            
            # Getting the type of 's' (line 296)
            s_248314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 35), 's', False)
            # Processing the call keyword arguments (line 296)
            kwargs_248315 = {}
            # Getting the type of 'np' (line 296)
            np_248309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 17), 'np', False)
            # Obtaining the member 'dot' of a type (line 296)
            dot_248310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 17), np_248309, 'dot')
            # Calling dot(args, kwargs) (line 296)
            dot_call_result_248316 = invoke(stypy.reporting.localization.Localization(__file__, 296, 17), dot_248310, *[result_mul_248313, s_248314], **kwargs_248315)
            
            # Applying the binary operator '+=' (line 296)
            result_iadd_248317 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 12), '+=', b_248308, dot_call_result_248316)
            # Assigning a type to the variable 'b' (line 296)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'b', result_iadd_248317)
            
            
            # Getting the type of 'c' (line 297)
            c_248318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'c')
            float_248319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 17), 'float')
            
            # Call to dot(...): (line 297)
            # Processing the call arguments (line 297)
            # Getting the type of 's0' (line 297)
            s0_248322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 30), 's0', False)
            # Getting the type of 'diag' (line 297)
            diag_248323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 35), 'diag', False)
            # Applying the binary operator '*' (line 297)
            result_mul_248324 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 30), '*', s0_248322, diag_248323)
            
            # Getting the type of 's0' (line 297)
            s0_248325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 41), 's0', False)
            # Processing the call keyword arguments (line 297)
            kwargs_248326 = {}
            # Getting the type of 'np' (line 297)
            np_248320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 23), 'np', False)
            # Obtaining the member 'dot' of a type (line 297)
            dot_248321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 23), np_248320, 'dot')
            # Calling dot(args, kwargs) (line 297)
            dot_call_result_248327 = invoke(stypy.reporting.localization.Localization(__file__, 297, 23), dot_248321, *[result_mul_248324, s0_248325], **kwargs_248326)
            
            # Applying the binary operator '*' (line 297)
            result_mul_248328 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 17), '*', float_248319, dot_call_result_248327)
            
            # Applying the binary operator '+=' (line 297)
            result_iadd_248329 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 12), '+=', c_248318, result_mul_248328)
            # Assigning a type to the variable 'c' (line 297)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'c', result_iadd_248329)
            

            if more_types_in_union_248307:
                # SSA join for if statement (line 295)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Obtaining an instance of the builtin type 'tuple' (line 298)
        tuple_248330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 298)
        # Adding element type (line 298)
        # Getting the type of 'a' (line 298)
        a_248331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 15), 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 15), tuple_248330, a_248331)
        # Adding element type (line 298)
        # Getting the type of 'b' (line 298)
        b_248332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 18), 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 15), tuple_248330, b_248332)
        # Adding element type (line 298)
        # Getting the type of 'c' (line 298)
        c_248333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 21), 'c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 15), tuple_248330, c_248333)
        
        # Assigning a type to the variable 'stypy_return_type' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'stypy_return_type', tuple_248330)

        if more_types_in_union_248275:
            # Runtime conditional SSA for else branch (line 291)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_248274) or more_types_in_union_248275):
        
        # Obtaining an instance of the builtin type 'tuple' (line 300)
        tuple_248334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 300)
        # Adding element type (line 300)
        # Getting the type of 'a' (line 300)
        a_248335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 15), 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 15), tuple_248334, a_248335)
        # Adding element type (line 300)
        # Getting the type of 'b' (line 300)
        b_248336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 18), 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 15), tuple_248334, b_248336)
        
        # Assigning a type to the variable 'stypy_return_type' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'stypy_return_type', tuple_248334)

        if (may_be_248274 and more_types_in_union_248275):
            # SSA join for if statement (line 291)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'build_quadratic_1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'build_quadratic_1d' in the type store
    # Getting the type of 'stypy_return_type' (line 252)
    stypy_return_type_248337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_248337)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'build_quadratic_1d'
    return stypy_return_type_248337

# Assigning a type to the variable 'build_quadratic_1d' (line 252)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 0), 'build_quadratic_1d', build_quadratic_1d)

@norecursion
def minimize_quadratic_1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_248338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 42), 'int')
    defaults = [int_248338]
    # Create a new context for function 'minimize_quadratic_1d'
    module_type_store = module_type_store.open_function_context('minimize_quadratic_1d', 303, 0, False)
    
    # Passed parameters checking function
    minimize_quadratic_1d.stypy_localization = localization
    minimize_quadratic_1d.stypy_type_of_self = None
    minimize_quadratic_1d.stypy_type_store = module_type_store
    minimize_quadratic_1d.stypy_function_name = 'minimize_quadratic_1d'
    minimize_quadratic_1d.stypy_param_names_list = ['a', 'b', 'lb', 'ub', 'c']
    minimize_quadratic_1d.stypy_varargs_param_name = None
    minimize_quadratic_1d.stypy_kwargs_param_name = None
    minimize_quadratic_1d.stypy_call_defaults = defaults
    minimize_quadratic_1d.stypy_call_varargs = varargs
    minimize_quadratic_1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'minimize_quadratic_1d', ['a', 'b', 'lb', 'ub', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'minimize_quadratic_1d', localization, ['a', 'b', 'lb', 'ub', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'minimize_quadratic_1d(...)' code ##################

    str_248339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, (-1)), 'str', 'Minimize a 1-d quadratic function subject to bounds.\n    \n    The free term `c` is 0 by default. Bounds must be finite.\n    \n    Returns\n    -------\n    t : float\n        Minimum point.\n    y : float\n        Minimum value.\n    ')
    
    # Assigning a List to a Name (line 315):
    
    # Assigning a List to a Name (line 315):
    
    # Obtaining an instance of the builtin type 'list' (line 315)
    list_248340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 315)
    # Adding element type (line 315)
    # Getting the type of 'lb' (line 315)
    lb_248341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 9), 'lb')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 8), list_248340, lb_248341)
    # Adding element type (line 315)
    # Getting the type of 'ub' (line 315)
    ub_248342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 13), 'ub')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 8), list_248340, ub_248342)
    
    # Assigning a type to the variable 't' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 't', list_248340)
    
    
    # Getting the type of 'a' (line 316)
    a_248343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 7), 'a')
    int_248344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 12), 'int')
    # Applying the binary operator '!=' (line 316)
    result_ne_248345 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 7), '!=', a_248343, int_248344)
    
    # Testing the type of an if condition (line 316)
    if_condition_248346 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 316, 4), result_ne_248345)
    # Assigning a type to the variable 'if_condition_248346' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'if_condition_248346', if_condition_248346)
    # SSA begins for if statement (line 316)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 317):
    
    # Assigning a BinOp to a Name (line 317):
    float_248347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 19), 'float')
    # Getting the type of 'b' (line 317)
    b_248348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 26), 'b')
    # Applying the binary operator '*' (line 317)
    result_mul_248349 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 19), '*', float_248347, b_248348)
    
    # Getting the type of 'a' (line 317)
    a_248350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 30), 'a')
    # Applying the binary operator 'div' (line 317)
    result_div_248351 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 28), 'div', result_mul_248349, a_248350)
    
    # Assigning a type to the variable 'extremum' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'extremum', result_div_248351)
    
    
    # Getting the type of 'lb' (line 318)
    lb_248352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 11), 'lb')
    # Getting the type of 'extremum' (line 318)
    extremum_248353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 16), 'extremum')
    # Applying the binary operator '<' (line 318)
    result_lt_248354 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 11), '<', lb_248352, extremum_248353)
    # Getting the type of 'ub' (line 318)
    ub_248355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 27), 'ub')
    # Applying the binary operator '<' (line 318)
    result_lt_248356 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 11), '<', extremum_248353, ub_248355)
    # Applying the binary operator '&' (line 318)
    result_and__248357 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 11), '&', result_lt_248354, result_lt_248356)
    
    # Testing the type of an if condition (line 318)
    if_condition_248358 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 318, 8), result_and__248357)
    # Assigning a type to the variable 'if_condition_248358' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'if_condition_248358', if_condition_248358)
    # SSA begins for if statement (line 318)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 319)
    # Processing the call arguments (line 319)
    # Getting the type of 'extremum' (line 319)
    extremum_248361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 21), 'extremum', False)
    # Processing the call keyword arguments (line 319)
    kwargs_248362 = {}
    # Getting the type of 't' (line 319)
    t_248359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 't', False)
    # Obtaining the member 'append' of a type (line 319)
    append_248360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 12), t_248359, 'append')
    # Calling append(args, kwargs) (line 319)
    append_call_result_248363 = invoke(stypy.reporting.localization.Localization(__file__, 319, 12), append_248360, *[extremum_248361], **kwargs_248362)
    
    # SSA join for if statement (line 318)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 316)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 320):
    
    # Assigning a Call to a Name (line 320):
    
    # Call to asarray(...): (line 320)
    # Processing the call arguments (line 320)
    # Getting the type of 't' (line 320)
    t_248366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 19), 't', False)
    # Processing the call keyword arguments (line 320)
    kwargs_248367 = {}
    # Getting the type of 'np' (line 320)
    np_248364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 320)
    asarray_248365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 8), np_248364, 'asarray')
    # Calling asarray(args, kwargs) (line 320)
    asarray_call_result_248368 = invoke(stypy.reporting.localization.Localization(__file__, 320, 8), asarray_248365, *[t_248366], **kwargs_248367)
    
    # Assigning a type to the variable 't' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 't', asarray_call_result_248368)
    
    # Assigning a BinOp to a Name (line 321):
    
    # Assigning a BinOp to a Name (line 321):
    # Getting the type of 'a' (line 321)
    a_248369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'a')
    # Getting the type of 't' (line 321)
    t_248370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 't')
    int_248371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 15), 'int')
    # Applying the binary operator '**' (line 321)
    result_pow_248372 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 12), '**', t_248370, int_248371)
    
    # Applying the binary operator '*' (line 321)
    result_mul_248373 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 8), '*', a_248369, result_pow_248372)
    
    # Getting the type of 'b' (line 321)
    b_248374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 19), 'b')
    # Getting the type of 't' (line 321)
    t_248375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 23), 't')
    # Applying the binary operator '*' (line 321)
    result_mul_248376 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 19), '*', b_248374, t_248375)
    
    # Applying the binary operator '+' (line 321)
    result_add_248377 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 8), '+', result_mul_248373, result_mul_248376)
    
    # Getting the type of 'c' (line 321)
    c_248378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 27), 'c')
    # Applying the binary operator '+' (line 321)
    result_add_248379 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 25), '+', result_add_248377, c_248378)
    
    # Assigning a type to the variable 'y' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'y', result_add_248379)
    
    # Assigning a Call to a Name (line 322):
    
    # Assigning a Call to a Name (line 322):
    
    # Call to argmin(...): (line 322)
    # Processing the call arguments (line 322)
    # Getting the type of 'y' (line 322)
    y_248382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 26), 'y', False)
    # Processing the call keyword arguments (line 322)
    kwargs_248383 = {}
    # Getting the type of 'np' (line 322)
    np_248380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 16), 'np', False)
    # Obtaining the member 'argmin' of a type (line 322)
    argmin_248381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 16), np_248380, 'argmin')
    # Calling argmin(args, kwargs) (line 322)
    argmin_call_result_248384 = invoke(stypy.reporting.localization.Localization(__file__, 322, 16), argmin_248381, *[y_248382], **kwargs_248383)
    
    # Assigning a type to the variable 'min_index' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'min_index', argmin_call_result_248384)
    
    # Obtaining an instance of the builtin type 'tuple' (line 323)
    tuple_248385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 323)
    # Adding element type (line 323)
    
    # Obtaining the type of the subscript
    # Getting the type of 'min_index' (line 323)
    min_index_248386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 13), 'min_index')
    # Getting the type of 't' (line 323)
    t_248387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 11), 't')
    # Obtaining the member '__getitem__' of a type (line 323)
    getitem___248388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 11), t_248387, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 323)
    subscript_call_result_248389 = invoke(stypy.reporting.localization.Localization(__file__, 323, 11), getitem___248388, min_index_248386)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 11), tuple_248385, subscript_call_result_248389)
    # Adding element type (line 323)
    
    # Obtaining the type of the subscript
    # Getting the type of 'min_index' (line 323)
    min_index_248390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 27), 'min_index')
    # Getting the type of 'y' (line 323)
    y_248391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 25), 'y')
    # Obtaining the member '__getitem__' of a type (line 323)
    getitem___248392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 25), y_248391, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 323)
    subscript_call_result_248393 = invoke(stypy.reporting.localization.Localization(__file__, 323, 25), getitem___248392, min_index_248390)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 11), tuple_248385, subscript_call_result_248393)
    
    # Assigning a type to the variable 'stypy_return_type' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'stypy_return_type', tuple_248385)
    
    # ################# End of 'minimize_quadratic_1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'minimize_quadratic_1d' in the type store
    # Getting the type of 'stypy_return_type' (line 303)
    stypy_return_type_248394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_248394)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'minimize_quadratic_1d'
    return stypy_return_type_248394

# Assigning a type to the variable 'minimize_quadratic_1d' (line 303)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 0), 'minimize_quadratic_1d', minimize_quadratic_1d)

@norecursion
def evaluate_quadratic(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 326)
    None_248395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 37), 'None')
    defaults = [None_248395]
    # Create a new context for function 'evaluate_quadratic'
    module_type_store = module_type_store.open_function_context('evaluate_quadratic', 326, 0, False)
    
    # Passed parameters checking function
    evaluate_quadratic.stypy_localization = localization
    evaluate_quadratic.stypy_type_of_self = None
    evaluate_quadratic.stypy_type_store = module_type_store
    evaluate_quadratic.stypy_function_name = 'evaluate_quadratic'
    evaluate_quadratic.stypy_param_names_list = ['J', 'g', 's', 'diag']
    evaluate_quadratic.stypy_varargs_param_name = None
    evaluate_quadratic.stypy_kwargs_param_name = None
    evaluate_quadratic.stypy_call_defaults = defaults
    evaluate_quadratic.stypy_call_varargs = varargs
    evaluate_quadratic.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'evaluate_quadratic', ['J', 'g', 's', 'diag'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'evaluate_quadratic', localization, ['J', 'g', 's', 'diag'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'evaluate_quadratic(...)' code ##################

    str_248396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, (-1)), 'str', 'Compute values of a quadratic function arising in least squares.\n    \n    The function is 0.5 * s.T * (J.T * J + diag) * s + g.T * s.\n    \n    Parameters\n    ----------\n    J : ndarray, sparse matrix or LinearOperator, shape (m, n)\n        Jacobian matrix, affects the quadratic term.\n    g : ndarray, shape (n,)\n        Gradient, defines the linear term.\n    s : ndarray, shape (k, n) or (n,)\n        Array containing steps as rows.\n    diag : ndarray, shape (n,), optional\n        Addition diagonal part, affects the quadratic term.\n        If None, assumed to be 0.\n    \n    Returns\n    -------\n    values : ndarray with shape (k,) or float\n        Values of the function. If `s` was 2-dimensional then ndarray is\n        returned, otherwise float is returned.\n    ')
    
    
    # Getting the type of 's' (line 349)
    s_248397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 7), 's')
    # Obtaining the member 'ndim' of a type (line 349)
    ndim_248398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 7), s_248397, 'ndim')
    int_248399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 17), 'int')
    # Applying the binary operator '==' (line 349)
    result_eq_248400 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 7), '==', ndim_248398, int_248399)
    
    # Testing the type of an if condition (line 349)
    if_condition_248401 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 349, 4), result_eq_248400)
    # Assigning a type to the variable 'if_condition_248401' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'if_condition_248401', if_condition_248401)
    # SSA begins for if statement (line 349)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 350):
    
    # Assigning a Call to a Name (line 350):
    
    # Call to dot(...): (line 350)
    # Processing the call arguments (line 350)
    # Getting the type of 's' (line 350)
    s_248404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 19), 's', False)
    # Processing the call keyword arguments (line 350)
    kwargs_248405 = {}
    # Getting the type of 'J' (line 350)
    J_248402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 13), 'J', False)
    # Obtaining the member 'dot' of a type (line 350)
    dot_248403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 13), J_248402, 'dot')
    # Calling dot(args, kwargs) (line 350)
    dot_call_result_248406 = invoke(stypy.reporting.localization.Localization(__file__, 350, 13), dot_248403, *[s_248404], **kwargs_248405)
    
    # Assigning a type to the variable 'Js' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'Js', dot_call_result_248406)
    
    # Assigning a Call to a Name (line 351):
    
    # Assigning a Call to a Name (line 351):
    
    # Call to dot(...): (line 351)
    # Processing the call arguments (line 351)
    # Getting the type of 'Js' (line 351)
    Js_248409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 19), 'Js', False)
    # Getting the type of 'Js' (line 351)
    Js_248410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 23), 'Js', False)
    # Processing the call keyword arguments (line 351)
    kwargs_248411 = {}
    # Getting the type of 'np' (line 351)
    np_248407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'np', False)
    # Obtaining the member 'dot' of a type (line 351)
    dot_248408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 12), np_248407, 'dot')
    # Calling dot(args, kwargs) (line 351)
    dot_call_result_248412 = invoke(stypy.reporting.localization.Localization(__file__, 351, 12), dot_248408, *[Js_248409, Js_248410], **kwargs_248411)
    
    # Assigning a type to the variable 'q' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'q', dot_call_result_248412)
    
    # Type idiom detected: calculating its left and rigth part (line 352)
    # Getting the type of 'diag' (line 352)
    diag_248413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'diag')
    # Getting the type of 'None' (line 352)
    None_248414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 23), 'None')
    
    (may_be_248415, more_types_in_union_248416) = may_not_be_none(diag_248413, None_248414)

    if may_be_248415:

        if more_types_in_union_248416:
            # Runtime conditional SSA (line 352)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Getting the type of 'q' (line 353)
        q_248417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'q')
        
        # Call to dot(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 's' (line 353)
        s_248420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 24), 's', False)
        # Getting the type of 'diag' (line 353)
        diag_248421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 28), 'diag', False)
        # Applying the binary operator '*' (line 353)
        result_mul_248422 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 24), '*', s_248420, diag_248421)
        
        # Getting the type of 's' (line 353)
        s_248423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 34), 's', False)
        # Processing the call keyword arguments (line 353)
        kwargs_248424 = {}
        # Getting the type of 'np' (line 353)
        np_248418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 17), 'np', False)
        # Obtaining the member 'dot' of a type (line 353)
        dot_248419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 17), np_248418, 'dot')
        # Calling dot(args, kwargs) (line 353)
        dot_call_result_248425 = invoke(stypy.reporting.localization.Localization(__file__, 353, 17), dot_248419, *[result_mul_248422, s_248423], **kwargs_248424)
        
        # Applying the binary operator '+=' (line 353)
        result_iadd_248426 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 12), '+=', q_248417, dot_call_result_248425)
        # Assigning a type to the variable 'q' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'q', result_iadd_248426)
        

        if more_types_in_union_248416:
            # SSA join for if statement (line 352)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA branch for the else part of an if statement (line 349)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 355):
    
    # Assigning a Call to a Name (line 355):
    
    # Call to dot(...): (line 355)
    # Processing the call arguments (line 355)
    # Getting the type of 's' (line 355)
    s_248429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 19), 's', False)
    # Obtaining the member 'T' of a type (line 355)
    T_248430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 19), s_248429, 'T')
    # Processing the call keyword arguments (line 355)
    kwargs_248431 = {}
    # Getting the type of 'J' (line 355)
    J_248427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 13), 'J', False)
    # Obtaining the member 'dot' of a type (line 355)
    dot_248428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 13), J_248427, 'dot')
    # Calling dot(args, kwargs) (line 355)
    dot_call_result_248432 = invoke(stypy.reporting.localization.Localization(__file__, 355, 13), dot_248428, *[T_248430], **kwargs_248431)
    
    # Assigning a type to the variable 'Js' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'Js', dot_call_result_248432)
    
    # Assigning a Call to a Name (line 356):
    
    # Assigning a Call to a Name (line 356):
    
    # Call to sum(...): (line 356)
    # Processing the call arguments (line 356)
    # Getting the type of 'Js' (line 356)
    Js_248435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 19), 'Js', False)
    int_248436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 23), 'int')
    # Applying the binary operator '**' (line 356)
    result_pow_248437 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 19), '**', Js_248435, int_248436)
    
    # Processing the call keyword arguments (line 356)
    int_248438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 31), 'int')
    keyword_248439 = int_248438
    kwargs_248440 = {'axis': keyword_248439}
    # Getting the type of 'np' (line 356)
    np_248433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'np', False)
    # Obtaining the member 'sum' of a type (line 356)
    sum_248434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 12), np_248433, 'sum')
    # Calling sum(args, kwargs) (line 356)
    sum_call_result_248441 = invoke(stypy.reporting.localization.Localization(__file__, 356, 12), sum_248434, *[result_pow_248437], **kwargs_248440)
    
    # Assigning a type to the variable 'q' (line 356)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'q', sum_call_result_248441)
    
    # Type idiom detected: calculating its left and rigth part (line 357)
    # Getting the type of 'diag' (line 357)
    diag_248442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'diag')
    # Getting the type of 'None' (line 357)
    None_248443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 23), 'None')
    
    (may_be_248444, more_types_in_union_248445) = may_not_be_none(diag_248442, None_248443)

    if may_be_248444:

        if more_types_in_union_248445:
            # Runtime conditional SSA (line 357)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Getting the type of 'q' (line 358)
        q_248446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'q')
        
        # Call to sum(...): (line 358)
        # Processing the call arguments (line 358)
        # Getting the type of 'diag' (line 358)
        diag_248449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 24), 'diag', False)
        # Getting the type of 's' (line 358)
        s_248450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 31), 's', False)
        int_248451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 34), 'int')
        # Applying the binary operator '**' (line 358)
        result_pow_248452 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 31), '**', s_248450, int_248451)
        
        # Applying the binary operator '*' (line 358)
        result_mul_248453 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 24), '*', diag_248449, result_pow_248452)
        
        # Processing the call keyword arguments (line 358)
        int_248454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 42), 'int')
        keyword_248455 = int_248454
        kwargs_248456 = {'axis': keyword_248455}
        # Getting the type of 'np' (line 358)
        np_248447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 17), 'np', False)
        # Obtaining the member 'sum' of a type (line 358)
        sum_248448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 17), np_248447, 'sum')
        # Calling sum(args, kwargs) (line 358)
        sum_call_result_248457 = invoke(stypy.reporting.localization.Localization(__file__, 358, 17), sum_248448, *[result_mul_248453], **kwargs_248456)
        
        # Applying the binary operator '+=' (line 358)
        result_iadd_248458 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 12), '+=', q_248446, sum_call_result_248457)
        # Assigning a type to the variable 'q' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'q', result_iadd_248458)
        

        if more_types_in_union_248445:
            # SSA join for if statement (line 357)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 349)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 360):
    
    # Assigning a Call to a Name (line 360):
    
    # Call to dot(...): (line 360)
    # Processing the call arguments (line 360)
    # Getting the type of 's' (line 360)
    s_248461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 15), 's', False)
    # Getting the type of 'g' (line 360)
    g_248462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 18), 'g', False)
    # Processing the call keyword arguments (line 360)
    kwargs_248463 = {}
    # Getting the type of 'np' (line 360)
    np_248459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'np', False)
    # Obtaining the member 'dot' of a type (line 360)
    dot_248460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 8), np_248459, 'dot')
    # Calling dot(args, kwargs) (line 360)
    dot_call_result_248464 = invoke(stypy.reporting.localization.Localization(__file__, 360, 8), dot_248460, *[s_248461, g_248462], **kwargs_248463)
    
    # Assigning a type to the variable 'l' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'l', dot_call_result_248464)
    float_248465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 11), 'float')
    # Getting the type of 'q' (line 362)
    q_248466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 17), 'q')
    # Applying the binary operator '*' (line 362)
    result_mul_248467 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 11), '*', float_248465, q_248466)
    
    # Getting the type of 'l' (line 362)
    l_248468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 21), 'l')
    # Applying the binary operator '+' (line 362)
    result_add_248469 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 11), '+', result_mul_248467, l_248468)
    
    # Assigning a type to the variable 'stypy_return_type' (line 362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'stypy_return_type', result_add_248469)
    
    # ################# End of 'evaluate_quadratic(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'evaluate_quadratic' in the type store
    # Getting the type of 'stypy_return_type' (line 326)
    stypy_return_type_248470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_248470)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'evaluate_quadratic'
    return stypy_return_type_248470

# Assigning a type to the variable 'evaluate_quadratic' (line 326)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 0), 'evaluate_quadratic', evaluate_quadratic)

@norecursion
def in_bounds(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'in_bounds'
    module_type_store = module_type_store.open_function_context('in_bounds', 368, 0, False)
    
    # Passed parameters checking function
    in_bounds.stypy_localization = localization
    in_bounds.stypy_type_of_self = None
    in_bounds.stypy_type_store = module_type_store
    in_bounds.stypy_function_name = 'in_bounds'
    in_bounds.stypy_param_names_list = ['x', 'lb', 'ub']
    in_bounds.stypy_varargs_param_name = None
    in_bounds.stypy_kwargs_param_name = None
    in_bounds.stypy_call_defaults = defaults
    in_bounds.stypy_call_varargs = varargs
    in_bounds.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'in_bounds', ['x', 'lb', 'ub'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'in_bounds', localization, ['x', 'lb', 'ub'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'in_bounds(...)' code ##################

    str_248471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 4), 'str', 'Check if a point lies within bounds.')
    
    # Call to all(...): (line 370)
    # Processing the call arguments (line 370)
    
    # Getting the type of 'x' (line 370)
    x_248474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 19), 'x', False)
    # Getting the type of 'lb' (line 370)
    lb_248475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 24), 'lb', False)
    # Applying the binary operator '>=' (line 370)
    result_ge_248476 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 19), '>=', x_248474, lb_248475)
    
    
    # Getting the type of 'x' (line 370)
    x_248477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 31), 'x', False)
    # Getting the type of 'ub' (line 370)
    ub_248478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 36), 'ub', False)
    # Applying the binary operator '<=' (line 370)
    result_le_248479 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 31), '<=', x_248477, ub_248478)
    
    # Applying the binary operator '&' (line 370)
    result_and__248480 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 18), '&', result_ge_248476, result_le_248479)
    
    # Processing the call keyword arguments (line 370)
    kwargs_248481 = {}
    # Getting the type of 'np' (line 370)
    np_248472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 11), 'np', False)
    # Obtaining the member 'all' of a type (line 370)
    all_248473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 11), np_248472, 'all')
    # Calling all(args, kwargs) (line 370)
    all_call_result_248482 = invoke(stypy.reporting.localization.Localization(__file__, 370, 11), all_248473, *[result_and__248480], **kwargs_248481)
    
    # Assigning a type to the variable 'stypy_return_type' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'stypy_return_type', all_call_result_248482)
    
    # ################# End of 'in_bounds(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'in_bounds' in the type store
    # Getting the type of 'stypy_return_type' (line 368)
    stypy_return_type_248483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_248483)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'in_bounds'
    return stypy_return_type_248483

# Assigning a type to the variable 'in_bounds' (line 368)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 0), 'in_bounds', in_bounds)

@norecursion
def step_size_to_bound(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'step_size_to_bound'
    module_type_store = module_type_store.open_function_context('step_size_to_bound', 373, 0, False)
    
    # Passed parameters checking function
    step_size_to_bound.stypy_localization = localization
    step_size_to_bound.stypy_type_of_self = None
    step_size_to_bound.stypy_type_store = module_type_store
    step_size_to_bound.stypy_function_name = 'step_size_to_bound'
    step_size_to_bound.stypy_param_names_list = ['x', 's', 'lb', 'ub']
    step_size_to_bound.stypy_varargs_param_name = None
    step_size_to_bound.stypy_kwargs_param_name = None
    step_size_to_bound.stypy_call_defaults = defaults
    step_size_to_bound.stypy_call_varargs = varargs
    step_size_to_bound.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'step_size_to_bound', ['x', 's', 'lb', 'ub'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'step_size_to_bound', localization, ['x', 's', 'lb', 'ub'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'step_size_to_bound(...)' code ##################

    str_248484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, (-1)), 'str', 'Compute a min_step size required to reach a bound.\n    \n    The function computes a positive scalar t, such that x + s * t is on\n    the bound.\n    \n    Returns\n    -------\n    step : float\n        Computed step. Non-negative value.\n    hits : ndarray of int with shape of x\n        Each element indicates whether a corresponding variable reaches the\n        bound:\n             \n             *  0 - the bound was not hit.\n             * -1 - the lower bound was hit.\n             *  1 - the upper bound was hit.\n    ')
    
    # Assigning a Call to a Name (line 391):
    
    # Assigning a Call to a Name (line 391):
    
    # Call to nonzero(...): (line 391)
    # Processing the call arguments (line 391)
    # Getting the type of 's' (line 391)
    s_248487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 26), 's', False)
    # Processing the call keyword arguments (line 391)
    kwargs_248488 = {}
    # Getting the type of 'np' (line 391)
    np_248485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 15), 'np', False)
    # Obtaining the member 'nonzero' of a type (line 391)
    nonzero_248486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 15), np_248485, 'nonzero')
    # Calling nonzero(args, kwargs) (line 391)
    nonzero_call_result_248489 = invoke(stypy.reporting.localization.Localization(__file__, 391, 15), nonzero_248486, *[s_248487], **kwargs_248488)
    
    # Assigning a type to the variable 'non_zero' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'non_zero', nonzero_call_result_248489)
    
    # Assigning a Subscript to a Name (line 392):
    
    # Assigning a Subscript to a Name (line 392):
    
    # Obtaining the type of the subscript
    # Getting the type of 'non_zero' (line 392)
    non_zero_248490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 19), 'non_zero')
    # Getting the type of 's' (line 392)
    s_248491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 17), 's')
    # Obtaining the member '__getitem__' of a type (line 392)
    getitem___248492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 17), s_248491, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 392)
    subscript_call_result_248493 = invoke(stypy.reporting.localization.Localization(__file__, 392, 17), getitem___248492, non_zero_248490)
    
    # Assigning a type to the variable 's_non_zero' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 's_non_zero', subscript_call_result_248493)
    
    # Assigning a Call to a Name (line 393):
    
    # Assigning a Call to a Name (line 393):
    
    # Call to empty_like(...): (line 393)
    # Processing the call arguments (line 393)
    # Getting the type of 'x' (line 393)
    x_248496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 26), 'x', False)
    # Processing the call keyword arguments (line 393)
    kwargs_248497 = {}
    # Getting the type of 'np' (line 393)
    np_248494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'np', False)
    # Obtaining the member 'empty_like' of a type (line 393)
    empty_like_248495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 12), np_248494, 'empty_like')
    # Calling empty_like(args, kwargs) (line 393)
    empty_like_call_result_248498 = invoke(stypy.reporting.localization.Localization(__file__, 393, 12), empty_like_248495, *[x_248496], **kwargs_248497)
    
    # Assigning a type to the variable 'steps' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'steps', empty_like_call_result_248498)
    
    # Call to fill(...): (line 394)
    # Processing the call arguments (line 394)
    # Getting the type of 'np' (line 394)
    np_248501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 15), 'np', False)
    # Obtaining the member 'inf' of a type (line 394)
    inf_248502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 15), np_248501, 'inf')
    # Processing the call keyword arguments (line 394)
    kwargs_248503 = {}
    # Getting the type of 'steps' (line 394)
    steps_248499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'steps', False)
    # Obtaining the member 'fill' of a type (line 394)
    fill_248500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 4), steps_248499, 'fill')
    # Calling fill(args, kwargs) (line 394)
    fill_call_result_248504 = invoke(stypy.reporting.localization.Localization(__file__, 394, 4), fill_248500, *[inf_248502], **kwargs_248503)
    
    
    # Call to errstate(...): (line 395)
    # Processing the call keyword arguments (line 395)
    str_248507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 26), 'str', 'ignore')
    keyword_248508 = str_248507
    kwargs_248509 = {'over': keyword_248508}
    # Getting the type of 'np' (line 395)
    np_248505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 9), 'np', False)
    # Obtaining the member 'errstate' of a type (line 395)
    errstate_248506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 9), np_248505, 'errstate')
    # Calling errstate(args, kwargs) (line 395)
    errstate_call_result_248510 = invoke(stypy.reporting.localization.Localization(__file__, 395, 9), errstate_248506, *[], **kwargs_248509)
    
    with_248511 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 395, 9), errstate_call_result_248510, 'with parameter', '__enter__', '__exit__')

    if with_248511:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 395)
        enter___248512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 9), errstate_call_result_248510, '__enter__')
        with_enter_248513 = invoke(stypy.reporting.localization.Localization(__file__, 395, 9), enter___248512)
        
        # Assigning a Call to a Subscript (line 396):
        
        # Assigning a Call to a Subscript (line 396):
        
        # Call to maximum(...): (line 396)
        # Processing the call arguments (line 396)
        
        # Obtaining the type of the subscript
        # Getting the type of 'non_zero' (line 396)
        non_zero_248516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 46), 'non_zero', False)
        # Getting the type of 'lb' (line 396)
        lb_248517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 38), 'lb', False)
        # Getting the type of 'x' (line 396)
        x_248518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 43), 'x', False)
        # Applying the binary operator '-' (line 396)
        result_sub_248519 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 38), '-', lb_248517, x_248518)
        
        # Obtaining the member '__getitem__' of a type (line 396)
        getitem___248520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 38), result_sub_248519, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 396)
        subscript_call_result_248521 = invoke(stypy.reporting.localization.Localization(__file__, 396, 38), getitem___248520, non_zero_248516)
        
        # Getting the type of 's_non_zero' (line 396)
        s_non_zero_248522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 58), 's_non_zero', False)
        # Applying the binary operator 'div' (line 396)
        result_div_248523 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 37), 'div', subscript_call_result_248521, s_non_zero_248522)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'non_zero' (line 397)
        non_zero_248524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 46), 'non_zero', False)
        # Getting the type of 'ub' (line 397)
        ub_248525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 38), 'ub', False)
        # Getting the type of 'x' (line 397)
        x_248526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 43), 'x', False)
        # Applying the binary operator '-' (line 397)
        result_sub_248527 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 38), '-', ub_248525, x_248526)
        
        # Obtaining the member '__getitem__' of a type (line 397)
        getitem___248528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 38), result_sub_248527, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 397)
        subscript_call_result_248529 = invoke(stypy.reporting.localization.Localization(__file__, 397, 38), getitem___248528, non_zero_248524)
        
        # Getting the type of 's_non_zero' (line 397)
        s_non_zero_248530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 58), 's_non_zero', False)
        # Applying the binary operator 'div' (line 397)
        result_div_248531 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 37), 'div', subscript_call_result_248529, s_non_zero_248530)
        
        # Processing the call keyword arguments (line 396)
        kwargs_248532 = {}
        # Getting the type of 'np' (line 396)
        np_248514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 26), 'np', False)
        # Obtaining the member 'maximum' of a type (line 396)
        maximum_248515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 26), np_248514, 'maximum')
        # Calling maximum(args, kwargs) (line 396)
        maximum_call_result_248533 = invoke(stypy.reporting.localization.Localization(__file__, 396, 26), maximum_248515, *[result_div_248523, result_div_248531], **kwargs_248532)
        
        # Getting the type of 'steps' (line 396)
        steps_248534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'steps')
        # Getting the type of 'non_zero' (line 396)
        non_zero_248535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 14), 'non_zero')
        # Storing an element on a container (line 396)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 8), steps_248534, (non_zero_248535, maximum_call_result_248533))
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 395)
        exit___248536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 9), errstate_call_result_248510, '__exit__')
        with_exit_248537 = invoke(stypy.reporting.localization.Localization(__file__, 395, 9), exit___248536, None, None, None)

    
    # Assigning a Call to a Name (line 398):
    
    # Assigning a Call to a Name (line 398):
    
    # Call to min(...): (line 398)
    # Processing the call arguments (line 398)
    # Getting the type of 'steps' (line 398)
    steps_248540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 22), 'steps', False)
    # Processing the call keyword arguments (line 398)
    kwargs_248541 = {}
    # Getting the type of 'np' (line 398)
    np_248538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 15), 'np', False)
    # Obtaining the member 'min' of a type (line 398)
    min_248539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 15), np_248538, 'min')
    # Calling min(args, kwargs) (line 398)
    min_call_result_248542 = invoke(stypy.reporting.localization.Localization(__file__, 398, 15), min_248539, *[steps_248540], **kwargs_248541)
    
    # Assigning a type to the variable 'min_step' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'min_step', min_call_result_248542)
    
    # Obtaining an instance of the builtin type 'tuple' (line 399)
    tuple_248543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 399)
    # Adding element type (line 399)
    # Getting the type of 'min_step' (line 399)
    min_step_248544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 11), 'min_step')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 11), tuple_248543, min_step_248544)
    # Adding element type (line 399)
    
    # Call to equal(...): (line 399)
    # Processing the call arguments (line 399)
    # Getting the type of 'steps' (line 399)
    steps_248547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 30), 'steps', False)
    # Getting the type of 'min_step' (line 399)
    min_step_248548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 37), 'min_step', False)
    # Processing the call keyword arguments (line 399)
    kwargs_248549 = {}
    # Getting the type of 'np' (line 399)
    np_248545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 21), 'np', False)
    # Obtaining the member 'equal' of a type (line 399)
    equal_248546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 21), np_248545, 'equal')
    # Calling equal(args, kwargs) (line 399)
    equal_call_result_248550 = invoke(stypy.reporting.localization.Localization(__file__, 399, 21), equal_248546, *[steps_248547, min_step_248548], **kwargs_248549)
    
    
    # Call to astype(...): (line 399)
    # Processing the call arguments (line 399)
    # Getting the type of 'int' (line 399)
    int_248557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 67), 'int', False)
    # Processing the call keyword arguments (line 399)
    kwargs_248558 = {}
    
    # Call to sign(...): (line 399)
    # Processing the call arguments (line 399)
    # Getting the type of 's' (line 399)
    s_248553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 57), 's', False)
    # Processing the call keyword arguments (line 399)
    kwargs_248554 = {}
    # Getting the type of 'np' (line 399)
    np_248551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 49), 'np', False)
    # Obtaining the member 'sign' of a type (line 399)
    sign_248552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 49), np_248551, 'sign')
    # Calling sign(args, kwargs) (line 399)
    sign_call_result_248555 = invoke(stypy.reporting.localization.Localization(__file__, 399, 49), sign_248552, *[s_248553], **kwargs_248554)
    
    # Obtaining the member 'astype' of a type (line 399)
    astype_248556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 49), sign_call_result_248555, 'astype')
    # Calling astype(args, kwargs) (line 399)
    astype_call_result_248559 = invoke(stypy.reporting.localization.Localization(__file__, 399, 49), astype_248556, *[int_248557], **kwargs_248558)
    
    # Applying the binary operator '*' (line 399)
    result_mul_248560 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 21), '*', equal_call_result_248550, astype_call_result_248559)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 11), tuple_248543, result_mul_248560)
    
    # Assigning a type to the variable 'stypy_return_type' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'stypy_return_type', tuple_248543)
    
    # ################# End of 'step_size_to_bound(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'step_size_to_bound' in the type store
    # Getting the type of 'stypy_return_type' (line 373)
    stypy_return_type_248561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_248561)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'step_size_to_bound'
    return stypy_return_type_248561

# Assigning a type to the variable 'step_size_to_bound' (line 373)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 0), 'step_size_to_bound', step_size_to_bound)

@norecursion
def find_active_constraints(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_248562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 44), 'float')
    defaults = [float_248562]
    # Create a new context for function 'find_active_constraints'
    module_type_store = module_type_store.open_function_context('find_active_constraints', 402, 0, False)
    
    # Passed parameters checking function
    find_active_constraints.stypy_localization = localization
    find_active_constraints.stypy_type_of_self = None
    find_active_constraints.stypy_type_store = module_type_store
    find_active_constraints.stypy_function_name = 'find_active_constraints'
    find_active_constraints.stypy_param_names_list = ['x', 'lb', 'ub', 'rtol']
    find_active_constraints.stypy_varargs_param_name = None
    find_active_constraints.stypy_kwargs_param_name = None
    find_active_constraints.stypy_call_defaults = defaults
    find_active_constraints.stypy_call_varargs = varargs
    find_active_constraints.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'find_active_constraints', ['x', 'lb', 'ub', 'rtol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'find_active_constraints', localization, ['x', 'lb', 'ub', 'rtol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'find_active_constraints(...)' code ##################

    str_248563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, (-1)), 'str', 'Determine which constraints are active in a given point.\n    \n    The threshold is computed using `rtol` and the absolute value of the\n    closest bound.\n    \n    Returns\n    -------\n    active : ndarray of int with shape of x\n        Each component shows whether the corresponding constraint is active:\n             \n             *  0 - a constraint is not active.\n             * -1 - a lower bound is active.\n             *  1 - a upper bound is active.\n    ')
    
    # Assigning a Call to a Name (line 417):
    
    # Assigning a Call to a Name (line 417):
    
    # Call to zeros_like(...): (line 417)
    # Processing the call arguments (line 417)
    # Getting the type of 'x' (line 417)
    x_248566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 27), 'x', False)
    # Processing the call keyword arguments (line 417)
    # Getting the type of 'int' (line 417)
    int_248567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 36), 'int', False)
    keyword_248568 = int_248567
    kwargs_248569 = {'dtype': keyword_248568}
    # Getting the type of 'np' (line 417)
    np_248564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 13), 'np', False)
    # Obtaining the member 'zeros_like' of a type (line 417)
    zeros_like_248565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 13), np_248564, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 417)
    zeros_like_call_result_248570 = invoke(stypy.reporting.localization.Localization(__file__, 417, 13), zeros_like_248565, *[x_248566], **kwargs_248569)
    
    # Assigning a type to the variable 'active' (line 417)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'active', zeros_like_call_result_248570)
    
    
    # Getting the type of 'rtol' (line 419)
    rtol_248571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 7), 'rtol')
    int_248572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 15), 'int')
    # Applying the binary operator '==' (line 419)
    result_eq_248573 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 7), '==', rtol_248571, int_248572)
    
    # Testing the type of an if condition (line 419)
    if_condition_248574 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 419, 4), result_eq_248573)
    # Assigning a type to the variable 'if_condition_248574' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'if_condition_248574', if_condition_248574)
    # SSA begins for if statement (line 419)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Subscript (line 420):
    
    # Assigning a Num to a Subscript (line 420):
    int_248575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 26), 'int')
    # Getting the type of 'active' (line 420)
    active_248576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'active')
    
    # Getting the type of 'x' (line 420)
    x_248577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 15), 'x')
    # Getting the type of 'lb' (line 420)
    lb_248578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 20), 'lb')
    # Applying the binary operator '<=' (line 420)
    result_le_248579 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 15), '<=', x_248577, lb_248578)
    
    # Storing an element on a container (line 420)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 8), active_248576, (result_le_248579, int_248575))
    
    # Assigning a Num to a Subscript (line 421):
    
    # Assigning a Num to a Subscript (line 421):
    int_248580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 26), 'int')
    # Getting the type of 'active' (line 421)
    active_248581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'active')
    
    # Getting the type of 'x' (line 421)
    x_248582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 15), 'x')
    # Getting the type of 'ub' (line 421)
    ub_248583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 20), 'ub')
    # Applying the binary operator '>=' (line 421)
    result_ge_248584 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 15), '>=', x_248582, ub_248583)
    
    # Storing an element on a container (line 421)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 8), active_248581, (result_ge_248584, int_248580))
    # Getting the type of 'active' (line 422)
    active_248585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 15), 'active')
    # Assigning a type to the variable 'stypy_return_type' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'stypy_return_type', active_248585)
    # SSA join for if statement (line 419)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 424):
    
    # Assigning a BinOp to a Name (line 424):
    # Getting the type of 'x' (line 424)
    x_248586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 17), 'x')
    # Getting the type of 'lb' (line 424)
    lb_248587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 21), 'lb')
    # Applying the binary operator '-' (line 424)
    result_sub_248588 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 17), '-', x_248586, lb_248587)
    
    # Assigning a type to the variable 'lower_dist' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'lower_dist', result_sub_248588)
    
    # Assigning a BinOp to a Name (line 425):
    
    # Assigning a BinOp to a Name (line 425):
    # Getting the type of 'ub' (line 425)
    ub_248589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 17), 'ub')
    # Getting the type of 'x' (line 425)
    x_248590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 22), 'x')
    # Applying the binary operator '-' (line 425)
    result_sub_248591 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 17), '-', ub_248589, x_248590)
    
    # Assigning a type to the variable 'upper_dist' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'upper_dist', result_sub_248591)
    
    # Assigning a BinOp to a Name (line 427):
    
    # Assigning a BinOp to a Name (line 427):
    # Getting the type of 'rtol' (line 427)
    rtol_248592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 22), 'rtol')
    
    # Call to maximum(...): (line 427)
    # Processing the call arguments (line 427)
    int_248595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 40), 'int')
    
    # Call to abs(...): (line 427)
    # Processing the call arguments (line 427)
    # Getting the type of 'lb' (line 427)
    lb_248598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 50), 'lb', False)
    # Processing the call keyword arguments (line 427)
    kwargs_248599 = {}
    # Getting the type of 'np' (line 427)
    np_248596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 43), 'np', False)
    # Obtaining the member 'abs' of a type (line 427)
    abs_248597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 43), np_248596, 'abs')
    # Calling abs(args, kwargs) (line 427)
    abs_call_result_248600 = invoke(stypy.reporting.localization.Localization(__file__, 427, 43), abs_248597, *[lb_248598], **kwargs_248599)
    
    # Processing the call keyword arguments (line 427)
    kwargs_248601 = {}
    # Getting the type of 'np' (line 427)
    np_248593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 29), 'np', False)
    # Obtaining the member 'maximum' of a type (line 427)
    maximum_248594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 29), np_248593, 'maximum')
    # Calling maximum(args, kwargs) (line 427)
    maximum_call_result_248602 = invoke(stypy.reporting.localization.Localization(__file__, 427, 29), maximum_248594, *[int_248595, abs_call_result_248600], **kwargs_248601)
    
    # Applying the binary operator '*' (line 427)
    result_mul_248603 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 22), '*', rtol_248592, maximum_call_result_248602)
    
    # Assigning a type to the variable 'lower_threshold' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'lower_threshold', result_mul_248603)
    
    # Assigning a BinOp to a Name (line 428):
    
    # Assigning a BinOp to a Name (line 428):
    # Getting the type of 'rtol' (line 428)
    rtol_248604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 22), 'rtol')
    
    # Call to maximum(...): (line 428)
    # Processing the call arguments (line 428)
    int_248607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 40), 'int')
    
    # Call to abs(...): (line 428)
    # Processing the call arguments (line 428)
    # Getting the type of 'ub' (line 428)
    ub_248610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 50), 'ub', False)
    # Processing the call keyword arguments (line 428)
    kwargs_248611 = {}
    # Getting the type of 'np' (line 428)
    np_248608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 43), 'np', False)
    # Obtaining the member 'abs' of a type (line 428)
    abs_248609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 43), np_248608, 'abs')
    # Calling abs(args, kwargs) (line 428)
    abs_call_result_248612 = invoke(stypy.reporting.localization.Localization(__file__, 428, 43), abs_248609, *[ub_248610], **kwargs_248611)
    
    # Processing the call keyword arguments (line 428)
    kwargs_248613 = {}
    # Getting the type of 'np' (line 428)
    np_248605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 29), 'np', False)
    # Obtaining the member 'maximum' of a type (line 428)
    maximum_248606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 29), np_248605, 'maximum')
    # Calling maximum(args, kwargs) (line 428)
    maximum_call_result_248614 = invoke(stypy.reporting.localization.Localization(__file__, 428, 29), maximum_248606, *[int_248607, abs_call_result_248612], **kwargs_248613)
    
    # Applying the binary operator '*' (line 428)
    result_mul_248615 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 22), '*', rtol_248604, maximum_call_result_248614)
    
    # Assigning a type to the variable 'upper_threshold' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'upper_threshold', result_mul_248615)
    
    # Assigning a BinOp to a Name (line 430):
    
    # Assigning a BinOp to a Name (line 430):
    
    # Call to isfinite(...): (line 430)
    # Processing the call arguments (line 430)
    # Getting the type of 'lb' (line 430)
    lb_248618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 32), 'lb', False)
    # Processing the call keyword arguments (line 430)
    kwargs_248619 = {}
    # Getting the type of 'np' (line 430)
    np_248616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 20), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 430)
    isfinite_248617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 20), np_248616, 'isfinite')
    # Calling isfinite(args, kwargs) (line 430)
    isfinite_call_result_248620 = invoke(stypy.reporting.localization.Localization(__file__, 430, 20), isfinite_248617, *[lb_248618], **kwargs_248619)
    
    
    # Getting the type of 'lower_dist' (line 431)
    lower_dist_248621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 21), 'lower_dist')
    
    # Call to minimum(...): (line 431)
    # Processing the call arguments (line 431)
    # Getting the type of 'upper_dist' (line 431)
    upper_dist_248624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 46), 'upper_dist', False)
    # Getting the type of 'lower_threshold' (line 431)
    lower_threshold_248625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 58), 'lower_threshold', False)
    # Processing the call keyword arguments (line 431)
    kwargs_248626 = {}
    # Getting the type of 'np' (line 431)
    np_248622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 35), 'np', False)
    # Obtaining the member 'minimum' of a type (line 431)
    minimum_248623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 35), np_248622, 'minimum')
    # Calling minimum(args, kwargs) (line 431)
    minimum_call_result_248627 = invoke(stypy.reporting.localization.Localization(__file__, 431, 35), minimum_248623, *[upper_dist_248624, lower_threshold_248625], **kwargs_248626)
    
    # Applying the binary operator '<=' (line 431)
    result_le_248628 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 21), '<=', lower_dist_248621, minimum_call_result_248627)
    
    # Applying the binary operator '&' (line 430)
    result_and__248629 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 20), '&', isfinite_call_result_248620, result_le_248628)
    
    # Assigning a type to the variable 'lower_active' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'lower_active', result_and__248629)
    
    # Assigning a Num to a Subscript (line 432):
    
    # Assigning a Num to a Subscript (line 432):
    int_248630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 27), 'int')
    # Getting the type of 'active' (line 432)
    active_248631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 4), 'active')
    # Getting the type of 'lower_active' (line 432)
    lower_active_248632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 11), 'lower_active')
    # Storing an element on a container (line 432)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 4), active_248631, (lower_active_248632, int_248630))
    
    # Assigning a BinOp to a Name (line 434):
    
    # Assigning a BinOp to a Name (line 434):
    
    # Call to isfinite(...): (line 434)
    # Processing the call arguments (line 434)
    # Getting the type of 'ub' (line 434)
    ub_248635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 32), 'ub', False)
    # Processing the call keyword arguments (line 434)
    kwargs_248636 = {}
    # Getting the type of 'np' (line 434)
    np_248633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 20), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 434)
    isfinite_248634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 20), np_248633, 'isfinite')
    # Calling isfinite(args, kwargs) (line 434)
    isfinite_call_result_248637 = invoke(stypy.reporting.localization.Localization(__file__, 434, 20), isfinite_248634, *[ub_248635], **kwargs_248636)
    
    
    # Getting the type of 'upper_dist' (line 435)
    upper_dist_248638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 21), 'upper_dist')
    
    # Call to minimum(...): (line 435)
    # Processing the call arguments (line 435)
    # Getting the type of 'lower_dist' (line 435)
    lower_dist_248641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 46), 'lower_dist', False)
    # Getting the type of 'upper_threshold' (line 435)
    upper_threshold_248642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 58), 'upper_threshold', False)
    # Processing the call keyword arguments (line 435)
    kwargs_248643 = {}
    # Getting the type of 'np' (line 435)
    np_248639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 35), 'np', False)
    # Obtaining the member 'minimum' of a type (line 435)
    minimum_248640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 35), np_248639, 'minimum')
    # Calling minimum(args, kwargs) (line 435)
    minimum_call_result_248644 = invoke(stypy.reporting.localization.Localization(__file__, 435, 35), minimum_248640, *[lower_dist_248641, upper_threshold_248642], **kwargs_248643)
    
    # Applying the binary operator '<=' (line 435)
    result_le_248645 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 21), '<=', upper_dist_248638, minimum_call_result_248644)
    
    # Applying the binary operator '&' (line 434)
    result_and__248646 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 20), '&', isfinite_call_result_248637, result_le_248645)
    
    # Assigning a type to the variable 'upper_active' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'upper_active', result_and__248646)
    
    # Assigning a Num to a Subscript (line 436):
    
    # Assigning a Num to a Subscript (line 436):
    int_248647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 27), 'int')
    # Getting the type of 'active' (line 436)
    active_248648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'active')
    # Getting the type of 'upper_active' (line 436)
    upper_active_248649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 11), 'upper_active')
    # Storing an element on a container (line 436)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 4), active_248648, (upper_active_248649, int_248647))
    # Getting the type of 'active' (line 438)
    active_248650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 11), 'active')
    # Assigning a type to the variable 'stypy_return_type' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 4), 'stypy_return_type', active_248650)
    
    # ################# End of 'find_active_constraints(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find_active_constraints' in the type store
    # Getting the type of 'stypy_return_type' (line 402)
    stypy_return_type_248651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_248651)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find_active_constraints'
    return stypy_return_type_248651

# Assigning a type to the variable 'find_active_constraints' (line 402)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 0), 'find_active_constraints', find_active_constraints)

@norecursion
def make_strictly_feasible(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_248652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 44), 'float')
    defaults = [float_248652]
    # Create a new context for function 'make_strictly_feasible'
    module_type_store = module_type_store.open_function_context('make_strictly_feasible', 441, 0, False)
    
    # Passed parameters checking function
    make_strictly_feasible.stypy_localization = localization
    make_strictly_feasible.stypy_type_of_self = None
    make_strictly_feasible.stypy_type_store = module_type_store
    make_strictly_feasible.stypy_function_name = 'make_strictly_feasible'
    make_strictly_feasible.stypy_param_names_list = ['x', 'lb', 'ub', 'rstep']
    make_strictly_feasible.stypy_varargs_param_name = None
    make_strictly_feasible.stypy_kwargs_param_name = None
    make_strictly_feasible.stypy_call_defaults = defaults
    make_strictly_feasible.stypy_call_varargs = varargs
    make_strictly_feasible.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'make_strictly_feasible', ['x', 'lb', 'ub', 'rstep'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'make_strictly_feasible', localization, ['x', 'lb', 'ub', 'rstep'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'make_strictly_feasible(...)' code ##################

    str_248653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, (-1)), 'str', 'Shift a point to the interior of a feasible region.\n    \n    Each element of the returned vector is at least at a relative distance\n    `rstep` from the closest bound. If ``rstep=0`` then `np.nextafter` is used.\n    ')
    
    # Assigning a Call to a Name (line 447):
    
    # Assigning a Call to a Name (line 447):
    
    # Call to copy(...): (line 447)
    # Processing the call keyword arguments (line 447)
    kwargs_248656 = {}
    # Getting the type of 'x' (line 447)
    x_248654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'x', False)
    # Obtaining the member 'copy' of a type (line 447)
    copy_248655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 12), x_248654, 'copy')
    # Calling copy(args, kwargs) (line 447)
    copy_call_result_248657 = invoke(stypy.reporting.localization.Localization(__file__, 447, 12), copy_248655, *[], **kwargs_248656)
    
    # Assigning a type to the variable 'x_new' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'x_new', copy_call_result_248657)
    
    # Assigning a Call to a Name (line 449):
    
    # Assigning a Call to a Name (line 449):
    
    # Call to find_active_constraints(...): (line 449)
    # Processing the call arguments (line 449)
    # Getting the type of 'x' (line 449)
    x_248659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 37), 'x', False)
    # Getting the type of 'lb' (line 449)
    lb_248660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 40), 'lb', False)
    # Getting the type of 'ub' (line 449)
    ub_248661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 44), 'ub', False)
    # Getting the type of 'rstep' (line 449)
    rstep_248662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 48), 'rstep', False)
    # Processing the call keyword arguments (line 449)
    kwargs_248663 = {}
    # Getting the type of 'find_active_constraints' (line 449)
    find_active_constraints_248658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 13), 'find_active_constraints', False)
    # Calling find_active_constraints(args, kwargs) (line 449)
    find_active_constraints_call_result_248664 = invoke(stypy.reporting.localization.Localization(__file__, 449, 13), find_active_constraints_248658, *[x_248659, lb_248660, ub_248661, rstep_248662], **kwargs_248663)
    
    # Assigning a type to the variable 'active' (line 449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 4), 'active', find_active_constraints_call_result_248664)
    
    # Assigning a Call to a Name (line 450):
    
    # Assigning a Call to a Name (line 450):
    
    # Call to equal(...): (line 450)
    # Processing the call arguments (line 450)
    # Getting the type of 'active' (line 450)
    active_248667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 26), 'active', False)
    int_248668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 34), 'int')
    # Processing the call keyword arguments (line 450)
    kwargs_248669 = {}
    # Getting the type of 'np' (line 450)
    np_248665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 17), 'np', False)
    # Obtaining the member 'equal' of a type (line 450)
    equal_248666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 17), np_248665, 'equal')
    # Calling equal(args, kwargs) (line 450)
    equal_call_result_248670 = invoke(stypy.reporting.localization.Localization(__file__, 450, 17), equal_248666, *[active_248667, int_248668], **kwargs_248669)
    
    # Assigning a type to the variable 'lower_mask' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'lower_mask', equal_call_result_248670)
    
    # Assigning a Call to a Name (line 451):
    
    # Assigning a Call to a Name (line 451):
    
    # Call to equal(...): (line 451)
    # Processing the call arguments (line 451)
    # Getting the type of 'active' (line 451)
    active_248673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 26), 'active', False)
    int_248674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 34), 'int')
    # Processing the call keyword arguments (line 451)
    kwargs_248675 = {}
    # Getting the type of 'np' (line 451)
    np_248671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 17), 'np', False)
    # Obtaining the member 'equal' of a type (line 451)
    equal_248672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 17), np_248671, 'equal')
    # Calling equal(args, kwargs) (line 451)
    equal_call_result_248676 = invoke(stypy.reporting.localization.Localization(__file__, 451, 17), equal_248672, *[active_248673, int_248674], **kwargs_248675)
    
    # Assigning a type to the variable 'upper_mask' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'upper_mask', equal_call_result_248676)
    
    
    # Getting the type of 'rstep' (line 453)
    rstep_248677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 7), 'rstep')
    int_248678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 16), 'int')
    # Applying the binary operator '==' (line 453)
    result_eq_248679 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 7), '==', rstep_248677, int_248678)
    
    # Testing the type of an if condition (line 453)
    if_condition_248680 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 453, 4), result_eq_248679)
    # Assigning a type to the variable 'if_condition_248680' (line 453)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'if_condition_248680', if_condition_248680)
    # SSA begins for if statement (line 453)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 454):
    
    # Assigning a Call to a Subscript (line 454):
    
    # Call to nextafter(...): (line 454)
    # Processing the call arguments (line 454)
    
    # Obtaining the type of the subscript
    # Getting the type of 'lower_mask' (line 454)
    lower_mask_248683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 44), 'lower_mask', False)
    # Getting the type of 'lb' (line 454)
    lb_248684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 41), 'lb', False)
    # Obtaining the member '__getitem__' of a type (line 454)
    getitem___248685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 41), lb_248684, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 454)
    subscript_call_result_248686 = invoke(stypy.reporting.localization.Localization(__file__, 454, 41), getitem___248685, lower_mask_248683)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'lower_mask' (line 454)
    lower_mask_248687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 60), 'lower_mask', False)
    # Getting the type of 'ub' (line 454)
    ub_248688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 57), 'ub', False)
    # Obtaining the member '__getitem__' of a type (line 454)
    getitem___248689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 57), ub_248688, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 454)
    subscript_call_result_248690 = invoke(stypy.reporting.localization.Localization(__file__, 454, 57), getitem___248689, lower_mask_248687)
    
    # Processing the call keyword arguments (line 454)
    kwargs_248691 = {}
    # Getting the type of 'np' (line 454)
    np_248681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 28), 'np', False)
    # Obtaining the member 'nextafter' of a type (line 454)
    nextafter_248682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 28), np_248681, 'nextafter')
    # Calling nextafter(args, kwargs) (line 454)
    nextafter_call_result_248692 = invoke(stypy.reporting.localization.Localization(__file__, 454, 28), nextafter_248682, *[subscript_call_result_248686, subscript_call_result_248690], **kwargs_248691)
    
    # Getting the type of 'x_new' (line 454)
    x_new_248693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'x_new')
    # Getting the type of 'lower_mask' (line 454)
    lower_mask_248694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 14), 'lower_mask')
    # Storing an element on a container (line 454)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 454, 8), x_new_248693, (lower_mask_248694, nextafter_call_result_248692))
    
    # Assigning a Call to a Subscript (line 455):
    
    # Assigning a Call to a Subscript (line 455):
    
    # Call to nextafter(...): (line 455)
    # Processing the call arguments (line 455)
    
    # Obtaining the type of the subscript
    # Getting the type of 'upper_mask' (line 455)
    upper_mask_248697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 44), 'upper_mask', False)
    # Getting the type of 'ub' (line 455)
    ub_248698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 41), 'ub', False)
    # Obtaining the member '__getitem__' of a type (line 455)
    getitem___248699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 41), ub_248698, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 455)
    subscript_call_result_248700 = invoke(stypy.reporting.localization.Localization(__file__, 455, 41), getitem___248699, upper_mask_248697)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'upper_mask' (line 455)
    upper_mask_248701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 60), 'upper_mask', False)
    # Getting the type of 'lb' (line 455)
    lb_248702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 57), 'lb', False)
    # Obtaining the member '__getitem__' of a type (line 455)
    getitem___248703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 57), lb_248702, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 455)
    subscript_call_result_248704 = invoke(stypy.reporting.localization.Localization(__file__, 455, 57), getitem___248703, upper_mask_248701)
    
    # Processing the call keyword arguments (line 455)
    kwargs_248705 = {}
    # Getting the type of 'np' (line 455)
    np_248695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 28), 'np', False)
    # Obtaining the member 'nextafter' of a type (line 455)
    nextafter_248696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 28), np_248695, 'nextafter')
    # Calling nextafter(args, kwargs) (line 455)
    nextafter_call_result_248706 = invoke(stypy.reporting.localization.Localization(__file__, 455, 28), nextafter_248696, *[subscript_call_result_248700, subscript_call_result_248704], **kwargs_248705)
    
    # Getting the type of 'x_new' (line 455)
    x_new_248707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'x_new')
    # Getting the type of 'upper_mask' (line 455)
    upper_mask_248708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 14), 'upper_mask')
    # Storing an element on a container (line 455)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 8), x_new_248707, (upper_mask_248708, nextafter_call_result_248706))
    # SSA branch for the else part of an if statement (line 453)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Subscript (line 457):
    
    # Assigning a BinOp to a Subscript (line 457):
    
    # Obtaining the type of the subscript
    # Getting the type of 'lower_mask' (line 457)
    lower_mask_248709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 32), 'lower_mask')
    # Getting the type of 'lb' (line 457)
    lb_248710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 29), 'lb')
    # Obtaining the member '__getitem__' of a type (line 457)
    getitem___248711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 29), lb_248710, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 457)
    subscript_call_result_248712 = invoke(stypy.reporting.localization.Localization(__file__, 457, 29), getitem___248711, lower_mask_248709)
    
    # Getting the type of 'rstep' (line 458)
    rstep_248713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 29), 'rstep')
    
    # Call to maximum(...): (line 458)
    # Processing the call arguments (line 458)
    int_248716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 48), 'int')
    
    # Call to abs(...): (line 458)
    # Processing the call arguments (line 458)
    
    # Obtaining the type of the subscript
    # Getting the type of 'lower_mask' (line 458)
    lower_mask_248719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 61), 'lower_mask', False)
    # Getting the type of 'lb' (line 458)
    lb_248720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 58), 'lb', False)
    # Obtaining the member '__getitem__' of a type (line 458)
    getitem___248721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 58), lb_248720, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 458)
    subscript_call_result_248722 = invoke(stypy.reporting.localization.Localization(__file__, 458, 58), getitem___248721, lower_mask_248719)
    
    # Processing the call keyword arguments (line 458)
    kwargs_248723 = {}
    # Getting the type of 'np' (line 458)
    np_248717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 51), 'np', False)
    # Obtaining the member 'abs' of a type (line 458)
    abs_248718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 51), np_248717, 'abs')
    # Calling abs(args, kwargs) (line 458)
    abs_call_result_248724 = invoke(stypy.reporting.localization.Localization(__file__, 458, 51), abs_248718, *[subscript_call_result_248722], **kwargs_248723)
    
    # Processing the call keyword arguments (line 458)
    kwargs_248725 = {}
    # Getting the type of 'np' (line 458)
    np_248714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 37), 'np', False)
    # Obtaining the member 'maximum' of a type (line 458)
    maximum_248715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 37), np_248714, 'maximum')
    # Calling maximum(args, kwargs) (line 458)
    maximum_call_result_248726 = invoke(stypy.reporting.localization.Localization(__file__, 458, 37), maximum_248715, *[int_248716, abs_call_result_248724], **kwargs_248725)
    
    # Applying the binary operator '*' (line 458)
    result_mul_248727 = python_operator(stypy.reporting.localization.Localization(__file__, 458, 29), '*', rstep_248713, maximum_call_result_248726)
    
    # Applying the binary operator '+' (line 457)
    result_add_248728 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 29), '+', subscript_call_result_248712, result_mul_248727)
    
    # Getting the type of 'x_new' (line 457)
    x_new_248729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'x_new')
    # Getting the type of 'lower_mask' (line 457)
    lower_mask_248730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 14), 'lower_mask')
    # Storing an element on a container (line 457)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 8), x_new_248729, (lower_mask_248730, result_add_248728))
    
    # Assigning a BinOp to a Subscript (line 459):
    
    # Assigning a BinOp to a Subscript (line 459):
    
    # Obtaining the type of the subscript
    # Getting the type of 'upper_mask' (line 459)
    upper_mask_248731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 32), 'upper_mask')
    # Getting the type of 'ub' (line 459)
    ub_248732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 29), 'ub')
    # Obtaining the member '__getitem__' of a type (line 459)
    getitem___248733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 29), ub_248732, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 459)
    subscript_call_result_248734 = invoke(stypy.reporting.localization.Localization(__file__, 459, 29), getitem___248733, upper_mask_248731)
    
    # Getting the type of 'rstep' (line 460)
    rstep_248735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 29), 'rstep')
    
    # Call to maximum(...): (line 460)
    # Processing the call arguments (line 460)
    int_248738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 48), 'int')
    
    # Call to abs(...): (line 460)
    # Processing the call arguments (line 460)
    
    # Obtaining the type of the subscript
    # Getting the type of 'upper_mask' (line 460)
    upper_mask_248741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 61), 'upper_mask', False)
    # Getting the type of 'ub' (line 460)
    ub_248742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 58), 'ub', False)
    # Obtaining the member '__getitem__' of a type (line 460)
    getitem___248743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 58), ub_248742, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 460)
    subscript_call_result_248744 = invoke(stypy.reporting.localization.Localization(__file__, 460, 58), getitem___248743, upper_mask_248741)
    
    # Processing the call keyword arguments (line 460)
    kwargs_248745 = {}
    # Getting the type of 'np' (line 460)
    np_248739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 51), 'np', False)
    # Obtaining the member 'abs' of a type (line 460)
    abs_248740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 51), np_248739, 'abs')
    # Calling abs(args, kwargs) (line 460)
    abs_call_result_248746 = invoke(stypy.reporting.localization.Localization(__file__, 460, 51), abs_248740, *[subscript_call_result_248744], **kwargs_248745)
    
    # Processing the call keyword arguments (line 460)
    kwargs_248747 = {}
    # Getting the type of 'np' (line 460)
    np_248736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 37), 'np', False)
    # Obtaining the member 'maximum' of a type (line 460)
    maximum_248737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 37), np_248736, 'maximum')
    # Calling maximum(args, kwargs) (line 460)
    maximum_call_result_248748 = invoke(stypy.reporting.localization.Localization(__file__, 460, 37), maximum_248737, *[int_248738, abs_call_result_248746], **kwargs_248747)
    
    # Applying the binary operator '*' (line 460)
    result_mul_248749 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 29), '*', rstep_248735, maximum_call_result_248748)
    
    # Applying the binary operator '-' (line 459)
    result_sub_248750 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 29), '-', subscript_call_result_248734, result_mul_248749)
    
    # Getting the type of 'x_new' (line 459)
    x_new_248751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'x_new')
    # Getting the type of 'upper_mask' (line 459)
    upper_mask_248752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 14), 'upper_mask')
    # Storing an element on a container (line 459)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 8), x_new_248751, (upper_mask_248752, result_sub_248750))
    # SSA join for if statement (line 453)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 462):
    
    # Assigning a BinOp to a Name (line 462):
    
    # Getting the type of 'x_new' (line 462)
    x_new_248753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 20), 'x_new')
    # Getting the type of 'lb' (line 462)
    lb_248754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 28), 'lb')
    # Applying the binary operator '<' (line 462)
    result_lt_248755 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 20), '<', x_new_248753, lb_248754)
    
    
    # Getting the type of 'x_new' (line 462)
    x_new_248756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 35), 'x_new')
    # Getting the type of 'ub' (line 462)
    ub_248757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 43), 'ub')
    # Applying the binary operator '>' (line 462)
    result_gt_248758 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 35), '>', x_new_248756, ub_248757)
    
    # Applying the binary operator '|' (line 462)
    result_or__248759 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 19), '|', result_lt_248755, result_gt_248758)
    
    # Assigning a type to the variable 'tight_bounds' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'tight_bounds', result_or__248759)
    
    # Assigning a BinOp to a Subscript (line 463):
    
    # Assigning a BinOp to a Subscript (line 463):
    float_248760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 26), 'float')
    
    # Obtaining the type of the subscript
    # Getting the type of 'tight_bounds' (line 463)
    tight_bounds_248761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 36), 'tight_bounds')
    # Getting the type of 'lb' (line 463)
    lb_248762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 33), 'lb')
    # Obtaining the member '__getitem__' of a type (line 463)
    getitem___248763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 33), lb_248762, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 463)
    subscript_call_result_248764 = invoke(stypy.reporting.localization.Localization(__file__, 463, 33), getitem___248763, tight_bounds_248761)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'tight_bounds' (line 463)
    tight_bounds_248765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 55), 'tight_bounds')
    # Getting the type of 'ub' (line 463)
    ub_248766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 52), 'ub')
    # Obtaining the member '__getitem__' of a type (line 463)
    getitem___248767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 52), ub_248766, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 463)
    subscript_call_result_248768 = invoke(stypy.reporting.localization.Localization(__file__, 463, 52), getitem___248767, tight_bounds_248765)
    
    # Applying the binary operator '+' (line 463)
    result_add_248769 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 33), '+', subscript_call_result_248764, subscript_call_result_248768)
    
    # Applying the binary operator '*' (line 463)
    result_mul_248770 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 26), '*', float_248760, result_add_248769)
    
    # Getting the type of 'x_new' (line 463)
    x_new_248771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'x_new')
    # Getting the type of 'tight_bounds' (line 463)
    tight_bounds_248772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 10), 'tight_bounds')
    # Storing an element on a container (line 463)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 4), x_new_248771, (tight_bounds_248772, result_mul_248770))
    # Getting the type of 'x_new' (line 465)
    x_new_248773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 11), 'x_new')
    # Assigning a type to the variable 'stypy_return_type' (line 465)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 4), 'stypy_return_type', x_new_248773)
    
    # ################# End of 'make_strictly_feasible(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'make_strictly_feasible' in the type store
    # Getting the type of 'stypy_return_type' (line 441)
    stypy_return_type_248774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_248774)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'make_strictly_feasible'
    return stypy_return_type_248774

# Assigning a type to the variable 'make_strictly_feasible' (line 441)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 0), 'make_strictly_feasible', make_strictly_feasible)

@norecursion
def CL_scaling_vector(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'CL_scaling_vector'
    module_type_store = module_type_store.open_function_context('CL_scaling_vector', 468, 0, False)
    
    # Passed parameters checking function
    CL_scaling_vector.stypy_localization = localization
    CL_scaling_vector.stypy_type_of_self = None
    CL_scaling_vector.stypy_type_store = module_type_store
    CL_scaling_vector.stypy_function_name = 'CL_scaling_vector'
    CL_scaling_vector.stypy_param_names_list = ['x', 'g', 'lb', 'ub']
    CL_scaling_vector.stypy_varargs_param_name = None
    CL_scaling_vector.stypy_kwargs_param_name = None
    CL_scaling_vector.stypy_call_defaults = defaults
    CL_scaling_vector.stypy_call_varargs = varargs
    CL_scaling_vector.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'CL_scaling_vector', ['x', 'g', 'lb', 'ub'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'CL_scaling_vector', localization, ['x', 'g', 'lb', 'ub'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'CL_scaling_vector(...)' code ##################

    str_248775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, (-1)), 'str', 'Compute Coleman-Li scaling vector and its derivatives.\n    \n    Components of a vector v are defined as follows:\n    ::\n               | ub[i] - x[i], if g[i] < 0 and ub[i] < np.inf\n        v[i] = | x[i] - lb[i], if g[i] > 0 and lb[i] > -np.inf\n               | 1,           otherwise\n    \n    According to this definition v[i] >= 0 for all i. It differs from the\n    definition in paper [1]_ (eq. (2.2)), where the absolute value of v is\n    used. Both definitions are equivalent down the line.\n    Derivatives of v with respect to x take value 1, -1 or 0 depending on a\n    case.\n    \n    Returns\n    -------\n    v : ndarray with shape of x\n        Scaling vector.\n    dv : ndarray with shape of x\n        Derivatives of v[i] with respect to x[i], diagonal elements of v\'s\n        Jacobian.\n    \n    References\n    ----------\n    .. [1] M.A. Branch, T.F. Coleman, and Y. Li, "A Subspace, Interior,\n           and Conjugate Gradient Method for Large-Scale Bound-Constrained\n           Minimization Problems," SIAM Journal on Scientific Computing,\n           Vol. 21, Number 1, pp 1-23, 1999.\n    ')
    
    # Assigning a Call to a Name (line 498):
    
    # Assigning a Call to a Name (line 498):
    
    # Call to ones_like(...): (line 498)
    # Processing the call arguments (line 498)
    # Getting the type of 'x' (line 498)
    x_248778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 21), 'x', False)
    # Processing the call keyword arguments (line 498)
    kwargs_248779 = {}
    # Getting the type of 'np' (line 498)
    np_248776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'np', False)
    # Obtaining the member 'ones_like' of a type (line 498)
    ones_like_248777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 8), np_248776, 'ones_like')
    # Calling ones_like(args, kwargs) (line 498)
    ones_like_call_result_248780 = invoke(stypy.reporting.localization.Localization(__file__, 498, 8), ones_like_248777, *[x_248778], **kwargs_248779)
    
    # Assigning a type to the variable 'v' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 4), 'v', ones_like_call_result_248780)
    
    # Assigning a Call to a Name (line 499):
    
    # Assigning a Call to a Name (line 499):
    
    # Call to zeros_like(...): (line 499)
    # Processing the call arguments (line 499)
    # Getting the type of 'x' (line 499)
    x_248783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 23), 'x', False)
    # Processing the call keyword arguments (line 499)
    kwargs_248784 = {}
    # Getting the type of 'np' (line 499)
    np_248781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 9), 'np', False)
    # Obtaining the member 'zeros_like' of a type (line 499)
    zeros_like_248782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 9), np_248781, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 499)
    zeros_like_call_result_248785 = invoke(stypy.reporting.localization.Localization(__file__, 499, 9), zeros_like_248782, *[x_248783], **kwargs_248784)
    
    # Assigning a type to the variable 'dv' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 4), 'dv', zeros_like_call_result_248785)
    
    # Assigning a BinOp to a Name (line 501):
    
    # Assigning a BinOp to a Name (line 501):
    
    # Getting the type of 'g' (line 501)
    g_248786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 12), 'g')
    int_248787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 16), 'int')
    # Applying the binary operator '<' (line 501)
    result_lt_248788 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 12), '<', g_248786, int_248787)
    
    
    # Call to isfinite(...): (line 501)
    # Processing the call arguments (line 501)
    # Getting the type of 'ub' (line 501)
    ub_248791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 33), 'ub', False)
    # Processing the call keyword arguments (line 501)
    kwargs_248792 = {}
    # Getting the type of 'np' (line 501)
    np_248789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 21), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 501)
    isfinite_248790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 21), np_248789, 'isfinite')
    # Calling isfinite(args, kwargs) (line 501)
    isfinite_call_result_248793 = invoke(stypy.reporting.localization.Localization(__file__, 501, 21), isfinite_248790, *[ub_248791], **kwargs_248792)
    
    # Applying the binary operator '&' (line 501)
    result_and__248794 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 11), '&', result_lt_248788, isfinite_call_result_248793)
    
    # Assigning a type to the variable 'mask' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'mask', result_and__248794)
    
    # Assigning a BinOp to a Subscript (line 502):
    
    # Assigning a BinOp to a Subscript (line 502):
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 502)
    mask_248795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 17), 'mask')
    # Getting the type of 'ub' (line 502)
    ub_248796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 14), 'ub')
    # Obtaining the member '__getitem__' of a type (line 502)
    getitem___248797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 14), ub_248796, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 502)
    subscript_call_result_248798 = invoke(stypy.reporting.localization.Localization(__file__, 502, 14), getitem___248797, mask_248795)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 502)
    mask_248799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 27), 'mask')
    # Getting the type of 'x' (line 502)
    x_248800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 25), 'x')
    # Obtaining the member '__getitem__' of a type (line 502)
    getitem___248801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 25), x_248800, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 502)
    subscript_call_result_248802 = invoke(stypy.reporting.localization.Localization(__file__, 502, 25), getitem___248801, mask_248799)
    
    # Applying the binary operator '-' (line 502)
    result_sub_248803 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 14), '-', subscript_call_result_248798, subscript_call_result_248802)
    
    # Getting the type of 'v' (line 502)
    v_248804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'v')
    # Getting the type of 'mask' (line 502)
    mask_248805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 6), 'mask')
    # Storing an element on a container (line 502)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 4), v_248804, (mask_248805, result_sub_248803))
    
    # Assigning a Num to a Subscript (line 503):
    
    # Assigning a Num to a Subscript (line 503):
    int_248806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 15), 'int')
    # Getting the type of 'dv' (line 503)
    dv_248807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 4), 'dv')
    # Getting the type of 'mask' (line 503)
    mask_248808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 7), 'mask')
    # Storing an element on a container (line 503)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 4), dv_248807, (mask_248808, int_248806))
    
    # Assigning a BinOp to a Name (line 505):
    
    # Assigning a BinOp to a Name (line 505):
    
    # Getting the type of 'g' (line 505)
    g_248809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'g')
    int_248810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 16), 'int')
    # Applying the binary operator '>' (line 505)
    result_gt_248811 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 12), '>', g_248809, int_248810)
    
    
    # Call to isfinite(...): (line 505)
    # Processing the call arguments (line 505)
    # Getting the type of 'lb' (line 505)
    lb_248814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 33), 'lb', False)
    # Processing the call keyword arguments (line 505)
    kwargs_248815 = {}
    # Getting the type of 'np' (line 505)
    np_248812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 21), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 505)
    isfinite_248813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 21), np_248812, 'isfinite')
    # Calling isfinite(args, kwargs) (line 505)
    isfinite_call_result_248816 = invoke(stypy.reporting.localization.Localization(__file__, 505, 21), isfinite_248813, *[lb_248814], **kwargs_248815)
    
    # Applying the binary operator '&' (line 505)
    result_and__248817 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 11), '&', result_gt_248811, isfinite_call_result_248816)
    
    # Assigning a type to the variable 'mask' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'mask', result_and__248817)
    
    # Assigning a BinOp to a Subscript (line 506):
    
    # Assigning a BinOp to a Subscript (line 506):
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 506)
    mask_248818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 16), 'mask')
    # Getting the type of 'x' (line 506)
    x_248819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 14), 'x')
    # Obtaining the member '__getitem__' of a type (line 506)
    getitem___248820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 14), x_248819, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 506)
    subscript_call_result_248821 = invoke(stypy.reporting.localization.Localization(__file__, 506, 14), getitem___248820, mask_248818)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 506)
    mask_248822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 27), 'mask')
    # Getting the type of 'lb' (line 506)
    lb_248823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 24), 'lb')
    # Obtaining the member '__getitem__' of a type (line 506)
    getitem___248824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 24), lb_248823, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 506)
    subscript_call_result_248825 = invoke(stypy.reporting.localization.Localization(__file__, 506, 24), getitem___248824, mask_248822)
    
    # Applying the binary operator '-' (line 506)
    result_sub_248826 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 14), '-', subscript_call_result_248821, subscript_call_result_248825)
    
    # Getting the type of 'v' (line 506)
    v_248827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'v')
    # Getting the type of 'mask' (line 506)
    mask_248828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 6), 'mask')
    # Storing an element on a container (line 506)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 4), v_248827, (mask_248828, result_sub_248826))
    
    # Assigning a Num to a Subscript (line 507):
    
    # Assigning a Num to a Subscript (line 507):
    int_248829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 15), 'int')
    # Getting the type of 'dv' (line 507)
    dv_248830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'dv')
    # Getting the type of 'mask' (line 507)
    mask_248831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 7), 'mask')
    # Storing an element on a container (line 507)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 4), dv_248830, (mask_248831, int_248829))
    
    # Obtaining an instance of the builtin type 'tuple' (line 509)
    tuple_248832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 509)
    # Adding element type (line 509)
    # Getting the type of 'v' (line 509)
    v_248833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 11), 'v')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 11), tuple_248832, v_248833)
    # Adding element type (line 509)
    # Getting the type of 'dv' (line 509)
    dv_248834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 14), 'dv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 11), tuple_248832, dv_248834)
    
    # Assigning a type to the variable 'stypy_return_type' (line 509)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 4), 'stypy_return_type', tuple_248832)
    
    # ################# End of 'CL_scaling_vector(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'CL_scaling_vector' in the type store
    # Getting the type of 'stypy_return_type' (line 468)
    stypy_return_type_248835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_248835)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'CL_scaling_vector'
    return stypy_return_type_248835

# Assigning a type to the variable 'CL_scaling_vector' (line 468)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 0), 'CL_scaling_vector', CL_scaling_vector)

@norecursion
def reflective_transformation(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'reflective_transformation'
    module_type_store = module_type_store.open_function_context('reflective_transformation', 512, 0, False)
    
    # Passed parameters checking function
    reflective_transformation.stypy_localization = localization
    reflective_transformation.stypy_type_of_self = None
    reflective_transformation.stypy_type_store = module_type_store
    reflective_transformation.stypy_function_name = 'reflective_transformation'
    reflective_transformation.stypy_param_names_list = ['y', 'lb', 'ub']
    reflective_transformation.stypy_varargs_param_name = None
    reflective_transformation.stypy_kwargs_param_name = None
    reflective_transformation.stypy_call_defaults = defaults
    reflective_transformation.stypy_call_varargs = varargs
    reflective_transformation.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'reflective_transformation', ['y', 'lb', 'ub'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'reflective_transformation', localization, ['y', 'lb', 'ub'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'reflective_transformation(...)' code ##################

    str_248836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 4), 'str', 'Compute reflective transformation and its gradient.')
    
    
    # Call to in_bounds(...): (line 514)
    # Processing the call arguments (line 514)
    # Getting the type of 'y' (line 514)
    y_248838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 17), 'y', False)
    # Getting the type of 'lb' (line 514)
    lb_248839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 20), 'lb', False)
    # Getting the type of 'ub' (line 514)
    ub_248840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 24), 'ub', False)
    # Processing the call keyword arguments (line 514)
    kwargs_248841 = {}
    # Getting the type of 'in_bounds' (line 514)
    in_bounds_248837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 7), 'in_bounds', False)
    # Calling in_bounds(args, kwargs) (line 514)
    in_bounds_call_result_248842 = invoke(stypy.reporting.localization.Localization(__file__, 514, 7), in_bounds_248837, *[y_248838, lb_248839, ub_248840], **kwargs_248841)
    
    # Testing the type of an if condition (line 514)
    if_condition_248843 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 514, 4), in_bounds_call_result_248842)
    # Assigning a type to the variable 'if_condition_248843' (line 514)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 4), 'if_condition_248843', if_condition_248843)
    # SSA begins for if statement (line 514)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 515)
    tuple_248844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 515)
    # Adding element type (line 515)
    # Getting the type of 'y' (line 515)
    y_248845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 15), 'y')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 15), tuple_248844, y_248845)
    # Adding element type (line 515)
    
    # Call to ones_like(...): (line 515)
    # Processing the call arguments (line 515)
    # Getting the type of 'y' (line 515)
    y_248848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 31), 'y', False)
    # Processing the call keyword arguments (line 515)
    kwargs_248849 = {}
    # Getting the type of 'np' (line 515)
    np_248846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 18), 'np', False)
    # Obtaining the member 'ones_like' of a type (line 515)
    ones_like_248847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 18), np_248846, 'ones_like')
    # Calling ones_like(args, kwargs) (line 515)
    ones_like_call_result_248850 = invoke(stypy.reporting.localization.Localization(__file__, 515, 18), ones_like_248847, *[y_248848], **kwargs_248849)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 15), tuple_248844, ones_like_call_result_248850)
    
    # Assigning a type to the variable 'stypy_return_type' (line 515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 'stypy_return_type', tuple_248844)
    # SSA join for if statement (line 514)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 517):
    
    # Assigning a Call to a Name (line 517):
    
    # Call to isfinite(...): (line 517)
    # Processing the call arguments (line 517)
    # Getting the type of 'lb' (line 517)
    lb_248853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 28), 'lb', False)
    # Processing the call keyword arguments (line 517)
    kwargs_248854 = {}
    # Getting the type of 'np' (line 517)
    np_248851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 16), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 517)
    isfinite_248852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 16), np_248851, 'isfinite')
    # Calling isfinite(args, kwargs) (line 517)
    isfinite_call_result_248855 = invoke(stypy.reporting.localization.Localization(__file__, 517, 16), isfinite_248852, *[lb_248853], **kwargs_248854)
    
    # Assigning a type to the variable 'lb_finite' (line 517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 4), 'lb_finite', isfinite_call_result_248855)
    
    # Assigning a Call to a Name (line 518):
    
    # Assigning a Call to a Name (line 518):
    
    # Call to isfinite(...): (line 518)
    # Processing the call arguments (line 518)
    # Getting the type of 'ub' (line 518)
    ub_248858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 28), 'ub', False)
    # Processing the call keyword arguments (line 518)
    kwargs_248859 = {}
    # Getting the type of 'np' (line 518)
    np_248856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 16), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 518)
    isfinite_248857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 16), np_248856, 'isfinite')
    # Calling isfinite(args, kwargs) (line 518)
    isfinite_call_result_248860 = invoke(stypy.reporting.localization.Localization(__file__, 518, 16), isfinite_248857, *[ub_248858], **kwargs_248859)
    
    # Assigning a type to the variable 'ub_finite' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 4), 'ub_finite', isfinite_call_result_248860)
    
    # Assigning a Call to a Name (line 520):
    
    # Assigning a Call to a Name (line 520):
    
    # Call to copy(...): (line 520)
    # Processing the call keyword arguments (line 520)
    kwargs_248863 = {}
    # Getting the type of 'y' (line 520)
    y_248861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'y', False)
    # Obtaining the member 'copy' of a type (line 520)
    copy_248862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 8), y_248861, 'copy')
    # Calling copy(args, kwargs) (line 520)
    copy_call_result_248864 = invoke(stypy.reporting.localization.Localization(__file__, 520, 8), copy_248862, *[], **kwargs_248863)
    
    # Assigning a type to the variable 'x' (line 520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 4), 'x', copy_call_result_248864)
    
    # Assigning a Call to a Name (line 521):
    
    # Assigning a Call to a Name (line 521):
    
    # Call to zeros_like(...): (line 521)
    # Processing the call arguments (line 521)
    # Getting the type of 'y' (line 521)
    y_248867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 31), 'y', False)
    # Processing the call keyword arguments (line 521)
    # Getting the type of 'bool' (line 521)
    bool_248868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 40), 'bool', False)
    keyword_248869 = bool_248868
    kwargs_248870 = {'dtype': keyword_248869}
    # Getting the type of 'np' (line 521)
    np_248865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 17), 'np', False)
    # Obtaining the member 'zeros_like' of a type (line 521)
    zeros_like_248866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 17), np_248865, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 521)
    zeros_like_call_result_248871 = invoke(stypy.reporting.localization.Localization(__file__, 521, 17), zeros_like_248866, *[y_248867], **kwargs_248870)
    
    # Assigning a type to the variable 'g_negative' (line 521)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 4), 'g_negative', zeros_like_call_result_248871)
    
    # Assigning a BinOp to a Name (line 523):
    
    # Assigning a BinOp to a Name (line 523):
    # Getting the type of 'lb_finite' (line 523)
    lb_finite_248872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 11), 'lb_finite')
    
    # Getting the type of 'ub_finite' (line 523)
    ub_finite_248873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 24), 'ub_finite')
    # Applying the '~' unary operator (line 523)
    result_inv_248874 = python_operator(stypy.reporting.localization.Localization(__file__, 523, 23), '~', ub_finite_248873)
    
    # Applying the binary operator '&' (line 523)
    result_and__248875 = python_operator(stypy.reporting.localization.Localization(__file__, 523, 11), '&', lb_finite_248872, result_inv_248874)
    
    # Assigning a type to the variable 'mask' (line 523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 4), 'mask', result_and__248875)
    
    # Assigning a Call to a Subscript (line 524):
    
    # Assigning a Call to a Subscript (line 524):
    
    # Call to maximum(...): (line 524)
    # Processing the call arguments (line 524)
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 524)
    mask_248878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 27), 'mask', False)
    # Getting the type of 'y' (line 524)
    y_248879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 25), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 524)
    getitem___248880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 25), y_248879, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 524)
    subscript_call_result_248881 = invoke(stypy.reporting.localization.Localization(__file__, 524, 25), getitem___248880, mask_248878)
    
    int_248882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 34), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 524)
    mask_248883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 41), 'mask', False)
    # Getting the type of 'lb' (line 524)
    lb_248884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 38), 'lb', False)
    # Obtaining the member '__getitem__' of a type (line 524)
    getitem___248885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 38), lb_248884, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 524)
    subscript_call_result_248886 = invoke(stypy.reporting.localization.Localization(__file__, 524, 38), getitem___248885, mask_248883)
    
    # Applying the binary operator '*' (line 524)
    result_mul_248887 = python_operator(stypy.reporting.localization.Localization(__file__, 524, 34), '*', int_248882, subscript_call_result_248886)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 524)
    mask_248888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 51), 'mask', False)
    # Getting the type of 'y' (line 524)
    y_248889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 49), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 524)
    getitem___248890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 49), y_248889, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 524)
    subscript_call_result_248891 = invoke(stypy.reporting.localization.Localization(__file__, 524, 49), getitem___248890, mask_248888)
    
    # Applying the binary operator '-' (line 524)
    result_sub_248892 = python_operator(stypy.reporting.localization.Localization(__file__, 524, 34), '-', result_mul_248887, subscript_call_result_248891)
    
    # Processing the call keyword arguments (line 524)
    kwargs_248893 = {}
    # Getting the type of 'np' (line 524)
    np_248876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 14), 'np', False)
    # Obtaining the member 'maximum' of a type (line 524)
    maximum_248877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 14), np_248876, 'maximum')
    # Calling maximum(args, kwargs) (line 524)
    maximum_call_result_248894 = invoke(stypy.reporting.localization.Localization(__file__, 524, 14), maximum_248877, *[subscript_call_result_248881, result_sub_248892], **kwargs_248893)
    
    # Getting the type of 'x' (line 524)
    x_248895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 4), 'x')
    # Getting the type of 'mask' (line 524)
    mask_248896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 6), 'mask')
    # Storing an element on a container (line 524)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 4), x_248895, (mask_248896, maximum_call_result_248894))
    
    # Assigning a Compare to a Subscript (line 525):
    
    # Assigning a Compare to a Subscript (line 525):
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 525)
    mask_248897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 25), 'mask')
    # Getting the type of 'y' (line 525)
    y_248898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 23), 'y')
    # Obtaining the member '__getitem__' of a type (line 525)
    getitem___248899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 23), y_248898, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 525)
    subscript_call_result_248900 = invoke(stypy.reporting.localization.Localization(__file__, 525, 23), getitem___248899, mask_248897)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 525)
    mask_248901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 36), 'mask')
    # Getting the type of 'lb' (line 525)
    lb_248902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 33), 'lb')
    # Obtaining the member '__getitem__' of a type (line 525)
    getitem___248903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 33), lb_248902, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 525)
    subscript_call_result_248904 = invoke(stypy.reporting.localization.Localization(__file__, 525, 33), getitem___248903, mask_248901)
    
    # Applying the binary operator '<' (line 525)
    result_lt_248905 = python_operator(stypy.reporting.localization.Localization(__file__, 525, 23), '<', subscript_call_result_248900, subscript_call_result_248904)
    
    # Getting the type of 'g_negative' (line 525)
    g_negative_248906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 4), 'g_negative')
    # Getting the type of 'mask' (line 525)
    mask_248907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 15), 'mask')
    # Storing an element on a container (line 525)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 525, 4), g_negative_248906, (mask_248907, result_lt_248905))
    
    # Assigning a BinOp to a Name (line 527):
    
    # Assigning a BinOp to a Name (line 527):
    
    # Getting the type of 'lb_finite' (line 527)
    lb_finite_248908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 12), 'lb_finite')
    # Applying the '~' unary operator (line 527)
    result_inv_248909 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 11), '~', lb_finite_248908)
    
    # Getting the type of 'ub_finite' (line 527)
    ub_finite_248910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 24), 'ub_finite')
    # Applying the binary operator '&' (line 527)
    result_and__248911 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 11), '&', result_inv_248909, ub_finite_248910)
    
    # Assigning a type to the variable 'mask' (line 527)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 4), 'mask', result_and__248911)
    
    # Assigning a Call to a Subscript (line 528):
    
    # Assigning a Call to a Subscript (line 528):
    
    # Call to minimum(...): (line 528)
    # Processing the call arguments (line 528)
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 528)
    mask_248914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 27), 'mask', False)
    # Getting the type of 'y' (line 528)
    y_248915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 25), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 528)
    getitem___248916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 25), y_248915, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 528)
    subscript_call_result_248917 = invoke(stypy.reporting.localization.Localization(__file__, 528, 25), getitem___248916, mask_248914)
    
    int_248918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 34), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 528)
    mask_248919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 41), 'mask', False)
    # Getting the type of 'ub' (line 528)
    ub_248920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 38), 'ub', False)
    # Obtaining the member '__getitem__' of a type (line 528)
    getitem___248921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 38), ub_248920, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 528)
    subscript_call_result_248922 = invoke(stypy.reporting.localization.Localization(__file__, 528, 38), getitem___248921, mask_248919)
    
    # Applying the binary operator '*' (line 528)
    result_mul_248923 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 34), '*', int_248918, subscript_call_result_248922)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 528)
    mask_248924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 51), 'mask', False)
    # Getting the type of 'y' (line 528)
    y_248925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 49), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 528)
    getitem___248926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 49), y_248925, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 528)
    subscript_call_result_248927 = invoke(stypy.reporting.localization.Localization(__file__, 528, 49), getitem___248926, mask_248924)
    
    # Applying the binary operator '-' (line 528)
    result_sub_248928 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 34), '-', result_mul_248923, subscript_call_result_248927)
    
    # Processing the call keyword arguments (line 528)
    kwargs_248929 = {}
    # Getting the type of 'np' (line 528)
    np_248912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 14), 'np', False)
    # Obtaining the member 'minimum' of a type (line 528)
    minimum_248913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 14), np_248912, 'minimum')
    # Calling minimum(args, kwargs) (line 528)
    minimum_call_result_248930 = invoke(stypy.reporting.localization.Localization(__file__, 528, 14), minimum_248913, *[subscript_call_result_248917, result_sub_248928], **kwargs_248929)
    
    # Getting the type of 'x' (line 528)
    x_248931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 4), 'x')
    # Getting the type of 'mask' (line 528)
    mask_248932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 6), 'mask')
    # Storing an element on a container (line 528)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 528, 4), x_248931, (mask_248932, minimum_call_result_248930))
    
    # Assigning a Compare to a Subscript (line 529):
    
    # Assigning a Compare to a Subscript (line 529):
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 529)
    mask_248933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 25), 'mask')
    # Getting the type of 'y' (line 529)
    y_248934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 23), 'y')
    # Obtaining the member '__getitem__' of a type (line 529)
    getitem___248935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 23), y_248934, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 529)
    subscript_call_result_248936 = invoke(stypy.reporting.localization.Localization(__file__, 529, 23), getitem___248935, mask_248933)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 529)
    mask_248937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 36), 'mask')
    # Getting the type of 'ub' (line 529)
    ub_248938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 33), 'ub')
    # Obtaining the member '__getitem__' of a type (line 529)
    getitem___248939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 33), ub_248938, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 529)
    subscript_call_result_248940 = invoke(stypy.reporting.localization.Localization(__file__, 529, 33), getitem___248939, mask_248937)
    
    # Applying the binary operator '>' (line 529)
    result_gt_248941 = python_operator(stypy.reporting.localization.Localization(__file__, 529, 23), '>', subscript_call_result_248936, subscript_call_result_248940)
    
    # Getting the type of 'g_negative' (line 529)
    g_negative_248942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 4), 'g_negative')
    # Getting the type of 'mask' (line 529)
    mask_248943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 15), 'mask')
    # Storing an element on a container (line 529)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 4), g_negative_248942, (mask_248943, result_gt_248941))
    
    # Assigning a BinOp to a Name (line 531):
    
    # Assigning a BinOp to a Name (line 531):
    # Getting the type of 'lb_finite' (line 531)
    lb_finite_248944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 11), 'lb_finite')
    # Getting the type of 'ub_finite' (line 531)
    ub_finite_248945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 23), 'ub_finite')
    # Applying the binary operator '&' (line 531)
    result_and__248946 = python_operator(stypy.reporting.localization.Localization(__file__, 531, 11), '&', lb_finite_248944, ub_finite_248945)
    
    # Assigning a type to the variable 'mask' (line 531)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 4), 'mask', result_and__248946)
    
    # Assigning a BinOp to a Name (line 532):
    
    # Assigning a BinOp to a Name (line 532):
    # Getting the type of 'ub' (line 532)
    ub_248947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'ub')
    # Getting the type of 'lb' (line 532)
    lb_248948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 13), 'lb')
    # Applying the binary operator '-' (line 532)
    result_sub_248949 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 8), '-', ub_248947, lb_248948)
    
    # Assigning a type to the variable 'd' (line 532)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 4), 'd', result_sub_248949)
    
    # Assigning a Call to a Name (line 533):
    
    # Assigning a Call to a Name (line 533):
    
    # Call to remainder(...): (line 533)
    # Processing the call arguments (line 533)
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 533)
    mask_248952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 23), 'mask', False)
    # Getting the type of 'y' (line 533)
    y_248953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 21), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 533)
    getitem___248954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 21), y_248953, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 533)
    subscript_call_result_248955 = invoke(stypy.reporting.localization.Localization(__file__, 533, 21), getitem___248954, mask_248952)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 533)
    mask_248956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 34), 'mask', False)
    # Getting the type of 'lb' (line 533)
    lb_248957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 31), 'lb', False)
    # Obtaining the member '__getitem__' of a type (line 533)
    getitem___248958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 31), lb_248957, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 533)
    subscript_call_result_248959 = invoke(stypy.reporting.localization.Localization(__file__, 533, 31), getitem___248958, mask_248956)
    
    # Applying the binary operator '-' (line 533)
    result_sub_248960 = python_operator(stypy.reporting.localization.Localization(__file__, 533, 21), '-', subscript_call_result_248955, subscript_call_result_248959)
    
    int_248961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 41), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 533)
    mask_248962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 47), 'mask', False)
    # Getting the type of 'd' (line 533)
    d_248963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 45), 'd', False)
    # Obtaining the member '__getitem__' of a type (line 533)
    getitem___248964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 45), d_248963, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 533)
    subscript_call_result_248965 = invoke(stypy.reporting.localization.Localization(__file__, 533, 45), getitem___248964, mask_248962)
    
    # Applying the binary operator '*' (line 533)
    result_mul_248966 = python_operator(stypy.reporting.localization.Localization(__file__, 533, 41), '*', int_248961, subscript_call_result_248965)
    
    # Processing the call keyword arguments (line 533)
    kwargs_248967 = {}
    # Getting the type of 'np' (line 533)
    np_248950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'np', False)
    # Obtaining the member 'remainder' of a type (line 533)
    remainder_248951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 8), np_248950, 'remainder')
    # Calling remainder(args, kwargs) (line 533)
    remainder_call_result_248968 = invoke(stypy.reporting.localization.Localization(__file__, 533, 8), remainder_248951, *[result_sub_248960, result_mul_248966], **kwargs_248967)
    
    # Assigning a type to the variable 't' (line 533)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 4), 't', remainder_call_result_248968)
    
    # Assigning a BinOp to a Subscript (line 534):
    
    # Assigning a BinOp to a Subscript (line 534):
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 534)
    mask_248969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 17), 'mask')
    # Getting the type of 'lb' (line 534)
    lb_248970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 14), 'lb')
    # Obtaining the member '__getitem__' of a type (line 534)
    getitem___248971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 14), lb_248970, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 534)
    subscript_call_result_248972 = invoke(stypy.reporting.localization.Localization(__file__, 534, 14), getitem___248971, mask_248969)
    
    
    # Call to minimum(...): (line 534)
    # Processing the call arguments (line 534)
    # Getting the type of 't' (line 534)
    t_248975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 36), 't', False)
    int_248976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 39), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 534)
    mask_248977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 45), 'mask', False)
    # Getting the type of 'd' (line 534)
    d_248978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 43), 'd', False)
    # Obtaining the member '__getitem__' of a type (line 534)
    getitem___248979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 43), d_248978, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 534)
    subscript_call_result_248980 = invoke(stypy.reporting.localization.Localization(__file__, 534, 43), getitem___248979, mask_248977)
    
    # Applying the binary operator '*' (line 534)
    result_mul_248981 = python_operator(stypy.reporting.localization.Localization(__file__, 534, 39), '*', int_248976, subscript_call_result_248980)
    
    # Getting the type of 't' (line 534)
    t_248982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 53), 't', False)
    # Applying the binary operator '-' (line 534)
    result_sub_248983 = python_operator(stypy.reporting.localization.Localization(__file__, 534, 39), '-', result_mul_248981, t_248982)
    
    # Processing the call keyword arguments (line 534)
    kwargs_248984 = {}
    # Getting the type of 'np' (line 534)
    np_248973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 25), 'np', False)
    # Obtaining the member 'minimum' of a type (line 534)
    minimum_248974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 25), np_248973, 'minimum')
    # Calling minimum(args, kwargs) (line 534)
    minimum_call_result_248985 = invoke(stypy.reporting.localization.Localization(__file__, 534, 25), minimum_248974, *[t_248975, result_sub_248983], **kwargs_248984)
    
    # Applying the binary operator '+' (line 534)
    result_add_248986 = python_operator(stypy.reporting.localization.Localization(__file__, 534, 14), '+', subscript_call_result_248972, minimum_call_result_248985)
    
    # Getting the type of 'x' (line 534)
    x_248987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 4), 'x')
    # Getting the type of 'mask' (line 534)
    mask_248988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 6), 'mask')
    # Storing an element on a container (line 534)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 534, 4), x_248987, (mask_248988, result_add_248986))
    
    # Assigning a Compare to a Subscript (line 535):
    
    # Assigning a Compare to a Subscript (line 535):
    
    # Getting the type of 't' (line 535)
    t_248989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 23), 't')
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 535)
    mask_248990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 29), 'mask')
    # Getting the type of 'd' (line 535)
    d_248991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 27), 'd')
    # Obtaining the member '__getitem__' of a type (line 535)
    getitem___248992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 27), d_248991, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 535)
    subscript_call_result_248993 = invoke(stypy.reporting.localization.Localization(__file__, 535, 27), getitem___248992, mask_248990)
    
    # Applying the binary operator '>' (line 535)
    result_gt_248994 = python_operator(stypy.reporting.localization.Localization(__file__, 535, 23), '>', t_248989, subscript_call_result_248993)
    
    # Getting the type of 'g_negative' (line 535)
    g_negative_248995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'g_negative')
    # Getting the type of 'mask' (line 535)
    mask_248996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 15), 'mask')
    # Storing an element on a container (line 535)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 4), g_negative_248995, (mask_248996, result_gt_248994))
    
    # Assigning a Call to a Name (line 537):
    
    # Assigning a Call to a Name (line 537):
    
    # Call to ones_like(...): (line 537)
    # Processing the call arguments (line 537)
    # Getting the type of 'y' (line 537)
    y_248999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 21), 'y', False)
    # Processing the call keyword arguments (line 537)
    kwargs_249000 = {}
    # Getting the type of 'np' (line 537)
    np_248997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'np', False)
    # Obtaining the member 'ones_like' of a type (line 537)
    ones_like_248998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 8), np_248997, 'ones_like')
    # Calling ones_like(args, kwargs) (line 537)
    ones_like_call_result_249001 = invoke(stypy.reporting.localization.Localization(__file__, 537, 8), ones_like_248998, *[y_248999], **kwargs_249000)
    
    # Assigning a type to the variable 'g' (line 537)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 4), 'g', ones_like_call_result_249001)
    
    # Assigning a Num to a Subscript (line 538):
    
    # Assigning a Num to a Subscript (line 538):
    int_249002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 20), 'int')
    # Getting the type of 'g' (line 538)
    g_249003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 4), 'g')
    # Getting the type of 'g_negative' (line 538)
    g_negative_249004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 6), 'g_negative')
    # Storing an element on a container (line 538)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 538, 4), g_249003, (g_negative_249004, int_249002))
    
    # Obtaining an instance of the builtin type 'tuple' (line 540)
    tuple_249005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 540)
    # Adding element type (line 540)
    # Getting the type of 'x' (line 540)
    x_249006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 11), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 540, 11), tuple_249005, x_249006)
    # Adding element type (line 540)
    # Getting the type of 'g' (line 540)
    g_249007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 14), 'g')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 540, 11), tuple_249005, g_249007)
    
    # Assigning a type to the variable 'stypy_return_type' (line 540)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 4), 'stypy_return_type', tuple_249005)
    
    # ################# End of 'reflective_transformation(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'reflective_transformation' in the type store
    # Getting the type of 'stypy_return_type' (line 512)
    stypy_return_type_249008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_249008)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'reflective_transformation'
    return stypy_return_type_249008

# Assigning a type to the variable 'reflective_transformation' (line 512)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 0), 'reflective_transformation', reflective_transformation)

@norecursion
def print_header_nonlinear(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'print_header_nonlinear'
    module_type_store = module_type_store.open_function_context('print_header_nonlinear', 546, 0, False)
    
    # Passed parameters checking function
    print_header_nonlinear.stypy_localization = localization
    print_header_nonlinear.stypy_type_of_self = None
    print_header_nonlinear.stypy_type_store = module_type_store
    print_header_nonlinear.stypy_function_name = 'print_header_nonlinear'
    print_header_nonlinear.stypy_param_names_list = []
    print_header_nonlinear.stypy_varargs_param_name = None
    print_header_nonlinear.stypy_kwargs_param_name = None
    print_header_nonlinear.stypy_call_defaults = defaults
    print_header_nonlinear.stypy_call_varargs = varargs
    print_header_nonlinear.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'print_header_nonlinear', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'print_header_nonlinear', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'print_header_nonlinear(...)' code ##################

    
    # Call to print(...): (line 547)
    # Processing the call arguments (line 547)
    
    # Call to format(...): (line 547)
    # Processing the call arguments (line 547)
    str_249012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 18), 'str', 'Iteration')
    str_249013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 31), 'str', 'Total nfev')
    str_249014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 45), 'str', 'Cost')
    str_249015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 53), 'str', 'Cost reduction')
    str_249016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 18), 'str', 'Step norm')
    str_249017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 31), 'str', 'Optimality')
    # Processing the call keyword arguments (line 547)
    kwargs_249018 = {}
    str_249010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 10), 'str', '{0:^15}{1:^15}{2:^15}{3:^15}{4:^15}{5:^15}')
    # Obtaining the member 'format' of a type (line 547)
    format_249011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 10), str_249010, 'format')
    # Calling format(args, kwargs) (line 547)
    format_call_result_249019 = invoke(stypy.reporting.localization.Localization(__file__, 547, 10), format_249011, *[str_249012, str_249013, str_249014, str_249015, str_249016, str_249017], **kwargs_249018)
    
    # Processing the call keyword arguments (line 547)
    kwargs_249020 = {}
    # Getting the type of 'print' (line 547)
    print_249009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 4), 'print', False)
    # Calling print(args, kwargs) (line 547)
    print_call_result_249021 = invoke(stypy.reporting.localization.Localization(__file__, 547, 4), print_249009, *[format_call_result_249019], **kwargs_249020)
    
    
    # ################# End of 'print_header_nonlinear(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'print_header_nonlinear' in the type store
    # Getting the type of 'stypy_return_type' (line 546)
    stypy_return_type_249022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_249022)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'print_header_nonlinear'
    return stypy_return_type_249022

# Assigning a type to the variable 'print_header_nonlinear' (line 546)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 0), 'print_header_nonlinear', print_header_nonlinear)

@norecursion
def print_iteration_nonlinear(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'print_iteration_nonlinear'
    module_type_store = module_type_store.open_function_context('print_iteration_nonlinear', 552, 0, False)
    
    # Passed parameters checking function
    print_iteration_nonlinear.stypy_localization = localization
    print_iteration_nonlinear.stypy_type_of_self = None
    print_iteration_nonlinear.stypy_type_store = module_type_store
    print_iteration_nonlinear.stypy_function_name = 'print_iteration_nonlinear'
    print_iteration_nonlinear.stypy_param_names_list = ['iteration', 'nfev', 'cost', 'cost_reduction', 'step_norm', 'optimality']
    print_iteration_nonlinear.stypy_varargs_param_name = None
    print_iteration_nonlinear.stypy_kwargs_param_name = None
    print_iteration_nonlinear.stypy_call_defaults = defaults
    print_iteration_nonlinear.stypy_call_varargs = varargs
    print_iteration_nonlinear.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'print_iteration_nonlinear', ['iteration', 'nfev', 'cost', 'cost_reduction', 'step_norm', 'optimality'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'print_iteration_nonlinear', localization, ['iteration', 'nfev', 'cost', 'cost_reduction', 'step_norm', 'optimality'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'print_iteration_nonlinear(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 554)
    # Getting the type of 'cost_reduction' (line 554)
    cost_reduction_249023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 7), 'cost_reduction')
    # Getting the type of 'None' (line 554)
    None_249024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 25), 'None')
    
    (may_be_249025, more_types_in_union_249026) = may_be_none(cost_reduction_249023, None_249024)

    if may_be_249025:

        if more_types_in_union_249026:
            # Runtime conditional SSA (line 554)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 555):
        
        # Assigning a BinOp to a Name (line 555):
        str_249027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 25), 'str', ' ')
        int_249028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 31), 'int')
        # Applying the binary operator '*' (line 555)
        result_mul_249029 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 25), '*', str_249027, int_249028)
        
        # Assigning a type to the variable 'cost_reduction' (line 555)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 8), 'cost_reduction', result_mul_249029)

        if more_types_in_union_249026:
            # Runtime conditional SSA for else branch (line 554)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_249025) or more_types_in_union_249026):
        
        # Assigning a Call to a Name (line 557):
        
        # Assigning a Call to a Name (line 557):
        
        # Call to format(...): (line 557)
        # Processing the call arguments (line 557)
        # Getting the type of 'cost_reduction' (line 557)
        cost_reduction_249032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 45), 'cost_reduction', False)
        # Processing the call keyword arguments (line 557)
        kwargs_249033 = {}
        str_249030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 25), 'str', '{0:^15.2e}')
        # Obtaining the member 'format' of a type (line 557)
        format_249031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 25), str_249030, 'format')
        # Calling format(args, kwargs) (line 557)
        format_call_result_249034 = invoke(stypy.reporting.localization.Localization(__file__, 557, 25), format_249031, *[cost_reduction_249032], **kwargs_249033)
        
        # Assigning a type to the variable 'cost_reduction' (line 557)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 8), 'cost_reduction', format_call_result_249034)

        if (may_be_249025 and more_types_in_union_249026):
            # SSA join for if statement (line 554)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 559)
    # Getting the type of 'step_norm' (line 559)
    step_norm_249035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 7), 'step_norm')
    # Getting the type of 'None' (line 559)
    None_249036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 20), 'None')
    
    (may_be_249037, more_types_in_union_249038) = may_be_none(step_norm_249035, None_249036)

    if may_be_249037:

        if more_types_in_union_249038:
            # Runtime conditional SSA (line 559)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 560):
        
        # Assigning a BinOp to a Name (line 560):
        str_249039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 20), 'str', ' ')
        int_249040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 26), 'int')
        # Applying the binary operator '*' (line 560)
        result_mul_249041 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 20), '*', str_249039, int_249040)
        
        # Assigning a type to the variable 'step_norm' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'step_norm', result_mul_249041)

        if more_types_in_union_249038:
            # Runtime conditional SSA for else branch (line 559)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_249037) or more_types_in_union_249038):
        
        # Assigning a Call to a Name (line 562):
        
        # Assigning a Call to a Name (line 562):
        
        # Call to format(...): (line 562)
        # Processing the call arguments (line 562)
        # Getting the type of 'step_norm' (line 562)
        step_norm_249044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 40), 'step_norm', False)
        # Processing the call keyword arguments (line 562)
        kwargs_249045 = {}
        str_249042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 20), 'str', '{0:^15.2e}')
        # Obtaining the member 'format' of a type (line 562)
        format_249043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 20), str_249042, 'format')
        # Calling format(args, kwargs) (line 562)
        format_call_result_249046 = invoke(stypy.reporting.localization.Localization(__file__, 562, 20), format_249043, *[step_norm_249044], **kwargs_249045)
        
        # Assigning a type to the variable 'step_norm' (line 562)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'step_norm', format_call_result_249046)

        if (may_be_249037 and more_types_in_union_249038):
            # SSA join for if statement (line 559)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to print(...): (line 564)
    # Processing the call arguments (line 564)
    
    # Call to format(...): (line 564)
    # Processing the call arguments (line 564)
    # Getting the type of 'iteration' (line 565)
    iteration_249050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 18), 'iteration', False)
    # Getting the type of 'nfev' (line 565)
    nfev_249051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 29), 'nfev', False)
    # Getting the type of 'cost' (line 565)
    cost_249052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 35), 'cost', False)
    # Getting the type of 'cost_reduction' (line 565)
    cost_reduction_249053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 41), 'cost_reduction', False)
    # Getting the type of 'step_norm' (line 566)
    step_norm_249054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 18), 'step_norm', False)
    # Getting the type of 'optimality' (line 566)
    optimality_249055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 29), 'optimality', False)
    # Processing the call keyword arguments (line 564)
    kwargs_249056 = {}
    str_249048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 10), 'str', '{0:^15}{1:^15}{2:^15.4e}{3}{4}{5:^15.2e}')
    # Obtaining the member 'format' of a type (line 564)
    format_249049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 10), str_249048, 'format')
    # Calling format(args, kwargs) (line 564)
    format_call_result_249057 = invoke(stypy.reporting.localization.Localization(__file__, 564, 10), format_249049, *[iteration_249050, nfev_249051, cost_249052, cost_reduction_249053, step_norm_249054, optimality_249055], **kwargs_249056)
    
    # Processing the call keyword arguments (line 564)
    kwargs_249058 = {}
    # Getting the type of 'print' (line 564)
    print_249047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 4), 'print', False)
    # Calling print(args, kwargs) (line 564)
    print_call_result_249059 = invoke(stypy.reporting.localization.Localization(__file__, 564, 4), print_249047, *[format_call_result_249057], **kwargs_249058)
    
    
    # ################# End of 'print_iteration_nonlinear(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'print_iteration_nonlinear' in the type store
    # Getting the type of 'stypy_return_type' (line 552)
    stypy_return_type_249060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_249060)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'print_iteration_nonlinear'
    return stypy_return_type_249060

# Assigning a type to the variable 'print_iteration_nonlinear' (line 552)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 0), 'print_iteration_nonlinear', print_iteration_nonlinear)

@norecursion
def print_header_linear(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'print_header_linear'
    module_type_store = module_type_store.open_function_context('print_header_linear', 569, 0, False)
    
    # Passed parameters checking function
    print_header_linear.stypy_localization = localization
    print_header_linear.stypy_type_of_self = None
    print_header_linear.stypy_type_store = module_type_store
    print_header_linear.stypy_function_name = 'print_header_linear'
    print_header_linear.stypy_param_names_list = []
    print_header_linear.stypy_varargs_param_name = None
    print_header_linear.stypy_kwargs_param_name = None
    print_header_linear.stypy_call_defaults = defaults
    print_header_linear.stypy_call_varargs = varargs
    print_header_linear.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'print_header_linear', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'print_header_linear', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'print_header_linear(...)' code ##################

    
    # Call to print(...): (line 570)
    # Processing the call arguments (line 570)
    
    # Call to format(...): (line 570)
    # Processing the call arguments (line 570)
    str_249064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 18), 'str', 'Iteration')
    str_249065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 31), 'str', 'Cost')
    str_249066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 39), 'str', 'Cost reduction')
    str_249067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 57), 'str', 'Step norm')
    str_249068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 18), 'str', 'Optimality')
    # Processing the call keyword arguments (line 570)
    kwargs_249069 = {}
    str_249062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 10), 'str', '{0:^15}{1:^15}{2:^15}{3:^15}{4:^15}')
    # Obtaining the member 'format' of a type (line 570)
    format_249063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 10), str_249062, 'format')
    # Calling format(args, kwargs) (line 570)
    format_call_result_249070 = invoke(stypy.reporting.localization.Localization(__file__, 570, 10), format_249063, *[str_249064, str_249065, str_249066, str_249067, str_249068], **kwargs_249069)
    
    # Processing the call keyword arguments (line 570)
    kwargs_249071 = {}
    # Getting the type of 'print' (line 570)
    print_249061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 4), 'print', False)
    # Calling print(args, kwargs) (line 570)
    print_call_result_249072 = invoke(stypy.reporting.localization.Localization(__file__, 570, 4), print_249061, *[format_call_result_249070], **kwargs_249071)
    
    
    # ################# End of 'print_header_linear(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'print_header_linear' in the type store
    # Getting the type of 'stypy_return_type' (line 569)
    stypy_return_type_249073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_249073)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'print_header_linear'
    return stypy_return_type_249073

# Assigning a type to the variable 'print_header_linear' (line 569)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 0), 'print_header_linear', print_header_linear)

@norecursion
def print_iteration_linear(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'print_iteration_linear'
    module_type_store = module_type_store.open_function_context('print_iteration_linear', 575, 0, False)
    
    # Passed parameters checking function
    print_iteration_linear.stypy_localization = localization
    print_iteration_linear.stypy_type_of_self = None
    print_iteration_linear.stypy_type_store = module_type_store
    print_iteration_linear.stypy_function_name = 'print_iteration_linear'
    print_iteration_linear.stypy_param_names_list = ['iteration', 'cost', 'cost_reduction', 'step_norm', 'optimality']
    print_iteration_linear.stypy_varargs_param_name = None
    print_iteration_linear.stypy_kwargs_param_name = None
    print_iteration_linear.stypy_call_defaults = defaults
    print_iteration_linear.stypy_call_varargs = varargs
    print_iteration_linear.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'print_iteration_linear', ['iteration', 'cost', 'cost_reduction', 'step_norm', 'optimality'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'print_iteration_linear', localization, ['iteration', 'cost', 'cost_reduction', 'step_norm', 'optimality'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'print_iteration_linear(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 577)
    # Getting the type of 'cost_reduction' (line 577)
    cost_reduction_249074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 7), 'cost_reduction')
    # Getting the type of 'None' (line 577)
    None_249075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 25), 'None')
    
    (may_be_249076, more_types_in_union_249077) = may_be_none(cost_reduction_249074, None_249075)

    if may_be_249076:

        if more_types_in_union_249077:
            # Runtime conditional SSA (line 577)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 578):
        
        # Assigning a BinOp to a Name (line 578):
        str_249078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 25), 'str', ' ')
        int_249079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 31), 'int')
        # Applying the binary operator '*' (line 578)
        result_mul_249080 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 25), '*', str_249078, int_249079)
        
        # Assigning a type to the variable 'cost_reduction' (line 578)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 8), 'cost_reduction', result_mul_249080)

        if more_types_in_union_249077:
            # Runtime conditional SSA for else branch (line 577)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_249076) or more_types_in_union_249077):
        
        # Assigning a Call to a Name (line 580):
        
        # Assigning a Call to a Name (line 580):
        
        # Call to format(...): (line 580)
        # Processing the call arguments (line 580)
        # Getting the type of 'cost_reduction' (line 580)
        cost_reduction_249083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 45), 'cost_reduction', False)
        # Processing the call keyword arguments (line 580)
        kwargs_249084 = {}
        str_249081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 25), 'str', '{0:^15.2e}')
        # Obtaining the member 'format' of a type (line 580)
        format_249082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 25), str_249081, 'format')
        # Calling format(args, kwargs) (line 580)
        format_call_result_249085 = invoke(stypy.reporting.localization.Localization(__file__, 580, 25), format_249082, *[cost_reduction_249083], **kwargs_249084)
        
        # Assigning a type to the variable 'cost_reduction' (line 580)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 8), 'cost_reduction', format_call_result_249085)

        if (may_be_249076 and more_types_in_union_249077):
            # SSA join for if statement (line 577)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 582)
    # Getting the type of 'step_norm' (line 582)
    step_norm_249086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 7), 'step_norm')
    # Getting the type of 'None' (line 582)
    None_249087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 20), 'None')
    
    (may_be_249088, more_types_in_union_249089) = may_be_none(step_norm_249086, None_249087)

    if may_be_249088:

        if more_types_in_union_249089:
            # Runtime conditional SSA (line 582)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 583):
        
        # Assigning a BinOp to a Name (line 583):
        str_249090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 20), 'str', ' ')
        int_249091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 26), 'int')
        # Applying the binary operator '*' (line 583)
        result_mul_249092 = python_operator(stypy.reporting.localization.Localization(__file__, 583, 20), '*', str_249090, int_249091)
        
        # Assigning a type to the variable 'step_norm' (line 583)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 8), 'step_norm', result_mul_249092)

        if more_types_in_union_249089:
            # Runtime conditional SSA for else branch (line 582)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_249088) or more_types_in_union_249089):
        
        # Assigning a Call to a Name (line 585):
        
        # Assigning a Call to a Name (line 585):
        
        # Call to format(...): (line 585)
        # Processing the call arguments (line 585)
        # Getting the type of 'step_norm' (line 585)
        step_norm_249095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 40), 'step_norm', False)
        # Processing the call keyword arguments (line 585)
        kwargs_249096 = {}
        str_249093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 20), 'str', '{0:^15.2e}')
        # Obtaining the member 'format' of a type (line 585)
        format_249094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 20), str_249093, 'format')
        # Calling format(args, kwargs) (line 585)
        format_call_result_249097 = invoke(stypy.reporting.localization.Localization(__file__, 585, 20), format_249094, *[step_norm_249095], **kwargs_249096)
        
        # Assigning a type to the variable 'step_norm' (line 585)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'step_norm', format_call_result_249097)

        if (may_be_249088 and more_types_in_union_249089):
            # SSA join for if statement (line 582)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to print(...): (line 587)
    # Processing the call arguments (line 587)
    
    # Call to format(...): (line 587)
    # Processing the call arguments (line 587)
    # Getting the type of 'iteration' (line 588)
    iteration_249101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'iteration', False)
    # Getting the type of 'cost' (line 588)
    cost_249102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 19), 'cost', False)
    # Getting the type of 'cost_reduction' (line 588)
    cost_reduction_249103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 25), 'cost_reduction', False)
    # Getting the type of 'step_norm' (line 588)
    step_norm_249104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 41), 'step_norm', False)
    # Getting the type of 'optimality' (line 588)
    optimality_249105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 52), 'optimality', False)
    # Processing the call keyword arguments (line 587)
    kwargs_249106 = {}
    str_249099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 10), 'str', '{0:^15}{1:^15.4e}{2}{3}{4:^15.2e}')
    # Obtaining the member 'format' of a type (line 587)
    format_249100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 10), str_249099, 'format')
    # Calling format(args, kwargs) (line 587)
    format_call_result_249107 = invoke(stypy.reporting.localization.Localization(__file__, 587, 10), format_249100, *[iteration_249101, cost_249102, cost_reduction_249103, step_norm_249104, optimality_249105], **kwargs_249106)
    
    # Processing the call keyword arguments (line 587)
    kwargs_249108 = {}
    # Getting the type of 'print' (line 587)
    print_249098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 4), 'print', False)
    # Calling print(args, kwargs) (line 587)
    print_call_result_249109 = invoke(stypy.reporting.localization.Localization(__file__, 587, 4), print_249098, *[format_call_result_249107], **kwargs_249108)
    
    
    # ################# End of 'print_iteration_linear(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'print_iteration_linear' in the type store
    # Getting the type of 'stypy_return_type' (line 575)
    stypy_return_type_249110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_249110)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'print_iteration_linear'
    return stypy_return_type_249110

# Assigning a type to the variable 'print_iteration_linear' (line 575)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 0), 'print_iteration_linear', print_iteration_linear)

@norecursion
def compute_grad(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'compute_grad'
    module_type_store = module_type_store.open_function_context('compute_grad', 594, 0, False)
    
    # Passed parameters checking function
    compute_grad.stypy_localization = localization
    compute_grad.stypy_type_of_self = None
    compute_grad.stypy_type_store = module_type_store
    compute_grad.stypy_function_name = 'compute_grad'
    compute_grad.stypy_param_names_list = ['J', 'f']
    compute_grad.stypy_varargs_param_name = None
    compute_grad.stypy_kwargs_param_name = None
    compute_grad.stypy_call_defaults = defaults
    compute_grad.stypy_call_varargs = varargs
    compute_grad.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'compute_grad', ['J', 'f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'compute_grad', localization, ['J', 'f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'compute_grad(...)' code ##################

    str_249111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 4), 'str', 'Compute gradient of the least-squares cost function.')
    
    
    # Call to isinstance(...): (line 596)
    # Processing the call arguments (line 596)
    # Getting the type of 'J' (line 596)
    J_249113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 18), 'J', False)
    # Getting the type of 'LinearOperator' (line 596)
    LinearOperator_249114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 21), 'LinearOperator', False)
    # Processing the call keyword arguments (line 596)
    kwargs_249115 = {}
    # Getting the type of 'isinstance' (line 596)
    isinstance_249112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 596)
    isinstance_call_result_249116 = invoke(stypy.reporting.localization.Localization(__file__, 596, 7), isinstance_249112, *[J_249113, LinearOperator_249114], **kwargs_249115)
    
    # Testing the type of an if condition (line 596)
    if_condition_249117 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 596, 4), isinstance_call_result_249116)
    # Assigning a type to the variable 'if_condition_249117' (line 596)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'if_condition_249117', if_condition_249117)
    # SSA begins for if statement (line 596)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to rmatvec(...): (line 597)
    # Processing the call arguments (line 597)
    # Getting the type of 'f' (line 597)
    f_249120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 25), 'f', False)
    # Processing the call keyword arguments (line 597)
    kwargs_249121 = {}
    # Getting the type of 'J' (line 597)
    J_249118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 15), 'J', False)
    # Obtaining the member 'rmatvec' of a type (line 597)
    rmatvec_249119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 15), J_249118, 'rmatvec')
    # Calling rmatvec(args, kwargs) (line 597)
    rmatvec_call_result_249122 = invoke(stypy.reporting.localization.Localization(__file__, 597, 15), rmatvec_249119, *[f_249120], **kwargs_249121)
    
    # Assigning a type to the variable 'stypy_return_type' (line 597)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'stypy_return_type', rmatvec_call_result_249122)
    # SSA branch for the else part of an if statement (line 596)
    module_type_store.open_ssa_branch('else')
    
    # Call to dot(...): (line 599)
    # Processing the call arguments (line 599)
    # Getting the type of 'f' (line 599)
    f_249126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 23), 'f', False)
    # Processing the call keyword arguments (line 599)
    kwargs_249127 = {}
    # Getting the type of 'J' (line 599)
    J_249123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 15), 'J', False)
    # Obtaining the member 'T' of a type (line 599)
    T_249124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 15), J_249123, 'T')
    # Obtaining the member 'dot' of a type (line 599)
    dot_249125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 15), T_249124, 'dot')
    # Calling dot(args, kwargs) (line 599)
    dot_call_result_249128 = invoke(stypy.reporting.localization.Localization(__file__, 599, 15), dot_249125, *[f_249126], **kwargs_249127)
    
    # Assigning a type to the variable 'stypy_return_type' (line 599)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'stypy_return_type', dot_call_result_249128)
    # SSA join for if statement (line 596)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'compute_grad(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'compute_grad' in the type store
    # Getting the type of 'stypy_return_type' (line 594)
    stypy_return_type_249129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_249129)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'compute_grad'
    return stypy_return_type_249129

# Assigning a type to the variable 'compute_grad' (line 594)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 0), 'compute_grad', compute_grad)

@norecursion
def compute_jac_scale(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 602)
    None_249130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 39), 'None')
    defaults = [None_249130]
    # Create a new context for function 'compute_jac_scale'
    module_type_store = module_type_store.open_function_context('compute_jac_scale', 602, 0, False)
    
    # Passed parameters checking function
    compute_jac_scale.stypy_localization = localization
    compute_jac_scale.stypy_type_of_self = None
    compute_jac_scale.stypy_type_store = module_type_store
    compute_jac_scale.stypy_function_name = 'compute_jac_scale'
    compute_jac_scale.stypy_param_names_list = ['J', 'scale_inv_old']
    compute_jac_scale.stypy_varargs_param_name = None
    compute_jac_scale.stypy_kwargs_param_name = None
    compute_jac_scale.stypy_call_defaults = defaults
    compute_jac_scale.stypy_call_varargs = varargs
    compute_jac_scale.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'compute_jac_scale', ['J', 'scale_inv_old'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'compute_jac_scale', localization, ['J', 'scale_inv_old'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'compute_jac_scale(...)' code ##################

    str_249131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 4), 'str', 'Compute variables scale based on the Jacobian matrix.')
    
    
    # Call to issparse(...): (line 604)
    # Processing the call arguments (line 604)
    # Getting the type of 'J' (line 604)
    J_249133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 16), 'J', False)
    # Processing the call keyword arguments (line 604)
    kwargs_249134 = {}
    # Getting the type of 'issparse' (line 604)
    issparse_249132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 7), 'issparse', False)
    # Calling issparse(args, kwargs) (line 604)
    issparse_call_result_249135 = invoke(stypy.reporting.localization.Localization(__file__, 604, 7), issparse_249132, *[J_249133], **kwargs_249134)
    
    # Testing the type of an if condition (line 604)
    if_condition_249136 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 604, 4), issparse_call_result_249135)
    # Assigning a type to the variable 'if_condition_249136' (line 604)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 4), 'if_condition_249136', if_condition_249136)
    # SSA begins for if statement (line 604)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 605):
    
    # Assigning a BinOp to a Name (line 605):
    
    # Call to ravel(...): (line 605)
    # Processing the call keyword arguments (line 605)
    kwargs_249152 = {}
    
    # Call to asarray(...): (line 605)
    # Processing the call arguments (line 605)
    
    # Call to sum(...): (line 605)
    # Processing the call keyword arguments (line 605)
    int_249145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 51), 'int')
    keyword_249146 = int_249145
    kwargs_249147 = {'axis': keyword_249146}
    
    # Call to power(...): (line 605)
    # Processing the call arguments (line 605)
    int_249141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 39), 'int')
    # Processing the call keyword arguments (line 605)
    kwargs_249142 = {}
    # Getting the type of 'J' (line 605)
    J_249139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 31), 'J', False)
    # Obtaining the member 'power' of a type (line 605)
    power_249140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 31), J_249139, 'power')
    # Calling power(args, kwargs) (line 605)
    power_call_result_249143 = invoke(stypy.reporting.localization.Localization(__file__, 605, 31), power_249140, *[int_249141], **kwargs_249142)
    
    # Obtaining the member 'sum' of a type (line 605)
    sum_249144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 31), power_call_result_249143, 'sum')
    # Calling sum(args, kwargs) (line 605)
    sum_call_result_249148 = invoke(stypy.reporting.localization.Localization(__file__, 605, 31), sum_249144, *[], **kwargs_249147)
    
    # Processing the call keyword arguments (line 605)
    kwargs_249149 = {}
    # Getting the type of 'np' (line 605)
    np_249137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 20), 'np', False)
    # Obtaining the member 'asarray' of a type (line 605)
    asarray_249138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 20), np_249137, 'asarray')
    # Calling asarray(args, kwargs) (line 605)
    asarray_call_result_249150 = invoke(stypy.reporting.localization.Localization(__file__, 605, 20), asarray_249138, *[sum_call_result_249148], **kwargs_249149)
    
    # Obtaining the member 'ravel' of a type (line 605)
    ravel_249151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 20), asarray_call_result_249150, 'ravel')
    # Calling ravel(args, kwargs) (line 605)
    ravel_call_result_249153 = invoke(stypy.reporting.localization.Localization(__file__, 605, 20), ravel_249151, *[], **kwargs_249152)
    
    float_249154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 64), 'float')
    # Applying the binary operator '**' (line 605)
    result_pow_249155 = python_operator(stypy.reporting.localization.Localization(__file__, 605, 20), '**', ravel_call_result_249153, float_249154)
    
    # Assigning a type to the variable 'scale_inv' (line 605)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 8), 'scale_inv', result_pow_249155)
    # SSA branch for the else part of an if statement (line 604)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 607):
    
    # Assigning a BinOp to a Name (line 607):
    
    # Call to sum(...): (line 607)
    # Processing the call arguments (line 607)
    # Getting the type of 'J' (line 607)
    J_249158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 27), 'J', False)
    int_249159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 30), 'int')
    # Applying the binary operator '**' (line 607)
    result_pow_249160 = python_operator(stypy.reporting.localization.Localization(__file__, 607, 27), '**', J_249158, int_249159)
    
    # Processing the call keyword arguments (line 607)
    int_249161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 38), 'int')
    keyword_249162 = int_249161
    kwargs_249163 = {'axis': keyword_249162}
    # Getting the type of 'np' (line 607)
    np_249156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 20), 'np', False)
    # Obtaining the member 'sum' of a type (line 607)
    sum_249157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 20), np_249156, 'sum')
    # Calling sum(args, kwargs) (line 607)
    sum_call_result_249164 = invoke(stypy.reporting.localization.Localization(__file__, 607, 20), sum_249157, *[result_pow_249160], **kwargs_249163)
    
    float_249165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 42), 'float')
    # Applying the binary operator '**' (line 607)
    result_pow_249166 = python_operator(stypy.reporting.localization.Localization(__file__, 607, 20), '**', sum_call_result_249164, float_249165)
    
    # Assigning a type to the variable 'scale_inv' (line 607)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 8), 'scale_inv', result_pow_249166)
    # SSA join for if statement (line 604)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 609)
    # Getting the type of 'scale_inv_old' (line 609)
    scale_inv_old_249167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 7), 'scale_inv_old')
    # Getting the type of 'None' (line 609)
    None_249168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 24), 'None')
    
    (may_be_249169, more_types_in_union_249170) = may_be_none(scale_inv_old_249167, None_249168)

    if may_be_249169:

        if more_types_in_union_249170:
            # Runtime conditional SSA (line 609)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Num to a Subscript (line 610):
        
        # Assigning a Num to a Subscript (line 610):
        int_249171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 36), 'int')
        # Getting the type of 'scale_inv' (line 610)
        scale_inv_249172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 8), 'scale_inv')
        
        # Getting the type of 'scale_inv' (line 610)
        scale_inv_249173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 18), 'scale_inv')
        int_249174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 31), 'int')
        # Applying the binary operator '==' (line 610)
        result_eq_249175 = python_operator(stypy.reporting.localization.Localization(__file__, 610, 18), '==', scale_inv_249173, int_249174)
        
        # Storing an element on a container (line 610)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 8), scale_inv_249172, (result_eq_249175, int_249171))

        if more_types_in_union_249170:
            # Runtime conditional SSA for else branch (line 609)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_249169) or more_types_in_union_249170):
        
        # Assigning a Call to a Name (line 612):
        
        # Assigning a Call to a Name (line 612):
        
        # Call to maximum(...): (line 612)
        # Processing the call arguments (line 612)
        # Getting the type of 'scale_inv' (line 612)
        scale_inv_249178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 31), 'scale_inv', False)
        # Getting the type of 'scale_inv_old' (line 612)
        scale_inv_old_249179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 42), 'scale_inv_old', False)
        # Processing the call keyword arguments (line 612)
        kwargs_249180 = {}
        # Getting the type of 'np' (line 612)
        np_249176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 20), 'np', False)
        # Obtaining the member 'maximum' of a type (line 612)
        maximum_249177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 20), np_249176, 'maximum')
        # Calling maximum(args, kwargs) (line 612)
        maximum_call_result_249181 = invoke(stypy.reporting.localization.Localization(__file__, 612, 20), maximum_249177, *[scale_inv_249178, scale_inv_old_249179], **kwargs_249180)
        
        # Assigning a type to the variable 'scale_inv' (line 612)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 8), 'scale_inv', maximum_call_result_249181)

        if (may_be_249169 and more_types_in_union_249170):
            # SSA join for if statement (line 609)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Obtaining an instance of the builtin type 'tuple' (line 614)
    tuple_249182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 614)
    # Adding element type (line 614)
    int_249183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 11), 'int')
    # Getting the type of 'scale_inv' (line 614)
    scale_inv_249184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 15), 'scale_inv')
    # Applying the binary operator 'div' (line 614)
    result_div_249185 = python_operator(stypy.reporting.localization.Localization(__file__, 614, 11), 'div', int_249183, scale_inv_249184)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 11), tuple_249182, result_div_249185)
    # Adding element type (line 614)
    # Getting the type of 'scale_inv' (line 614)
    scale_inv_249186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 26), 'scale_inv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 11), tuple_249182, scale_inv_249186)
    
    # Assigning a type to the variable 'stypy_return_type' (line 614)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 4), 'stypy_return_type', tuple_249182)
    
    # ################# End of 'compute_jac_scale(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'compute_jac_scale' in the type store
    # Getting the type of 'stypy_return_type' (line 602)
    stypy_return_type_249187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_249187)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'compute_jac_scale'
    return stypy_return_type_249187

# Assigning a type to the variable 'compute_jac_scale' (line 602)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 0), 'compute_jac_scale', compute_jac_scale)

@norecursion
def left_multiplied_operator(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'left_multiplied_operator'
    module_type_store = module_type_store.open_function_context('left_multiplied_operator', 617, 0, False)
    
    # Passed parameters checking function
    left_multiplied_operator.stypy_localization = localization
    left_multiplied_operator.stypy_type_of_self = None
    left_multiplied_operator.stypy_type_store = module_type_store
    left_multiplied_operator.stypy_function_name = 'left_multiplied_operator'
    left_multiplied_operator.stypy_param_names_list = ['J', 'd']
    left_multiplied_operator.stypy_varargs_param_name = None
    left_multiplied_operator.stypy_kwargs_param_name = None
    left_multiplied_operator.stypy_call_defaults = defaults
    left_multiplied_operator.stypy_call_varargs = varargs
    left_multiplied_operator.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'left_multiplied_operator', ['J', 'd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'left_multiplied_operator', localization, ['J', 'd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'left_multiplied_operator(...)' code ##################

    str_249188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 4), 'str', 'Return diag(d) J as LinearOperator.')
    
    # Assigning a Call to a Name (line 619):
    
    # Assigning a Call to a Name (line 619):
    
    # Call to aslinearoperator(...): (line 619)
    # Processing the call arguments (line 619)
    # Getting the type of 'J' (line 619)
    J_249190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 25), 'J', False)
    # Processing the call keyword arguments (line 619)
    kwargs_249191 = {}
    # Getting the type of 'aslinearoperator' (line 619)
    aslinearoperator_249189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 8), 'aslinearoperator', False)
    # Calling aslinearoperator(args, kwargs) (line 619)
    aslinearoperator_call_result_249192 = invoke(stypy.reporting.localization.Localization(__file__, 619, 8), aslinearoperator_249189, *[J_249190], **kwargs_249191)
    
    # Assigning a type to the variable 'J' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 4), 'J', aslinearoperator_call_result_249192)

    @norecursion
    def matvec(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'matvec'
        module_type_store = module_type_store.open_function_context('matvec', 621, 4, False)
        
        # Passed parameters checking function
        matvec.stypy_localization = localization
        matvec.stypy_type_of_self = None
        matvec.stypy_type_store = module_type_store
        matvec.stypy_function_name = 'matvec'
        matvec.stypy_param_names_list = ['x']
        matvec.stypy_varargs_param_name = None
        matvec.stypy_kwargs_param_name = None
        matvec.stypy_call_defaults = defaults
        matvec.stypy_call_varargs = varargs
        matvec.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'matvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'matvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'matvec(...)' code ##################

        # Getting the type of 'd' (line 622)
        d_249193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 15), 'd')
        
        # Call to matvec(...): (line 622)
        # Processing the call arguments (line 622)
        # Getting the type of 'x' (line 622)
        x_249196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 28), 'x', False)
        # Processing the call keyword arguments (line 622)
        kwargs_249197 = {}
        # Getting the type of 'J' (line 622)
        J_249194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 19), 'J', False)
        # Obtaining the member 'matvec' of a type (line 622)
        matvec_249195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 19), J_249194, 'matvec')
        # Calling matvec(args, kwargs) (line 622)
        matvec_call_result_249198 = invoke(stypy.reporting.localization.Localization(__file__, 622, 19), matvec_249195, *[x_249196], **kwargs_249197)
        
        # Applying the binary operator '*' (line 622)
        result_mul_249199 = python_operator(stypy.reporting.localization.Localization(__file__, 622, 15), '*', d_249193, matvec_call_result_249198)
        
        # Assigning a type to the variable 'stypy_return_type' (line 622)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 8), 'stypy_return_type', result_mul_249199)
        
        # ################# End of 'matvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'matvec' in the type store
        # Getting the type of 'stypy_return_type' (line 621)
        stypy_return_type_249200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_249200)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'matvec'
        return stypy_return_type_249200

    # Assigning a type to the variable 'matvec' (line 621)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 4), 'matvec', matvec)

    @norecursion
    def matmat(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'matmat'
        module_type_store = module_type_store.open_function_context('matmat', 624, 4, False)
        
        # Passed parameters checking function
        matmat.stypy_localization = localization
        matmat.stypy_type_of_self = None
        matmat.stypy_type_store = module_type_store
        matmat.stypy_function_name = 'matmat'
        matmat.stypy_param_names_list = ['X']
        matmat.stypy_varargs_param_name = None
        matmat.stypy_kwargs_param_name = None
        matmat.stypy_call_defaults = defaults
        matmat.stypy_call_varargs = varargs
        matmat.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'matmat', ['X'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'matmat', localization, ['X'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'matmat(...)' code ##################

        # Getting the type of 'd' (line 625)
        d_249201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 15), 'd')
        
        # Call to matmat(...): (line 625)
        # Processing the call arguments (line 625)
        # Getting the type of 'X' (line 625)
        X_249204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 28), 'X', False)
        # Processing the call keyword arguments (line 625)
        kwargs_249205 = {}
        # Getting the type of 'J' (line 625)
        J_249202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 19), 'J', False)
        # Obtaining the member 'matmat' of a type (line 625)
        matmat_249203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 19), J_249202, 'matmat')
        # Calling matmat(args, kwargs) (line 625)
        matmat_call_result_249206 = invoke(stypy.reporting.localization.Localization(__file__, 625, 19), matmat_249203, *[X_249204], **kwargs_249205)
        
        # Applying the binary operator '*' (line 625)
        result_mul_249207 = python_operator(stypy.reporting.localization.Localization(__file__, 625, 15), '*', d_249201, matmat_call_result_249206)
        
        # Assigning a type to the variable 'stypy_return_type' (line 625)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 8), 'stypy_return_type', result_mul_249207)
        
        # ################# End of 'matmat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'matmat' in the type store
        # Getting the type of 'stypy_return_type' (line 624)
        stypy_return_type_249208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_249208)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'matmat'
        return stypy_return_type_249208

    # Assigning a type to the variable 'matmat' (line 624)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 4), 'matmat', matmat)

    @norecursion
    def rmatvec(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'rmatvec'
        module_type_store = module_type_store.open_function_context('rmatvec', 627, 4, False)
        
        # Passed parameters checking function
        rmatvec.stypy_localization = localization
        rmatvec.stypy_type_of_self = None
        rmatvec.stypy_type_store = module_type_store
        rmatvec.stypy_function_name = 'rmatvec'
        rmatvec.stypy_param_names_list = ['x']
        rmatvec.stypy_varargs_param_name = None
        rmatvec.stypy_kwargs_param_name = None
        rmatvec.stypy_call_defaults = defaults
        rmatvec.stypy_call_varargs = varargs
        rmatvec.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'rmatvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'rmatvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'rmatvec(...)' code ##################

        
        # Call to rmatvec(...): (line 628)
        # Processing the call arguments (line 628)
        
        # Call to ravel(...): (line 628)
        # Processing the call keyword arguments (line 628)
        kwargs_249213 = {}
        # Getting the type of 'x' (line 628)
        x_249211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 25), 'x', False)
        # Obtaining the member 'ravel' of a type (line 628)
        ravel_249212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 25), x_249211, 'ravel')
        # Calling ravel(args, kwargs) (line 628)
        ravel_call_result_249214 = invoke(stypy.reporting.localization.Localization(__file__, 628, 25), ravel_249212, *[], **kwargs_249213)
        
        # Getting the type of 'd' (line 628)
        d_249215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 37), 'd', False)
        # Applying the binary operator '*' (line 628)
        result_mul_249216 = python_operator(stypy.reporting.localization.Localization(__file__, 628, 25), '*', ravel_call_result_249214, d_249215)
        
        # Processing the call keyword arguments (line 628)
        kwargs_249217 = {}
        # Getting the type of 'J' (line 628)
        J_249209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 15), 'J', False)
        # Obtaining the member 'rmatvec' of a type (line 628)
        rmatvec_249210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 15), J_249209, 'rmatvec')
        # Calling rmatvec(args, kwargs) (line 628)
        rmatvec_call_result_249218 = invoke(stypy.reporting.localization.Localization(__file__, 628, 15), rmatvec_249210, *[result_mul_249216], **kwargs_249217)
        
        # Assigning a type to the variable 'stypy_return_type' (line 628)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 8), 'stypy_return_type', rmatvec_call_result_249218)
        
        # ################# End of 'rmatvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'rmatvec' in the type store
        # Getting the type of 'stypy_return_type' (line 627)
        stypy_return_type_249219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_249219)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'rmatvec'
        return stypy_return_type_249219

    # Assigning a type to the variable 'rmatvec' (line 627)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 4), 'rmatvec', rmatvec)
    
    # Call to LinearOperator(...): (line 630)
    # Processing the call arguments (line 630)
    # Getting the type of 'J' (line 630)
    J_249221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 26), 'J', False)
    # Obtaining the member 'shape' of a type (line 630)
    shape_249222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 26), J_249221, 'shape')
    # Processing the call keyword arguments (line 630)
    # Getting the type of 'matvec' (line 630)
    matvec_249223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 42), 'matvec', False)
    keyword_249224 = matvec_249223
    # Getting the type of 'matmat' (line 630)
    matmat_249225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 57), 'matmat', False)
    keyword_249226 = matmat_249225
    # Getting the type of 'rmatvec' (line 631)
    rmatvec_249227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 34), 'rmatvec', False)
    keyword_249228 = rmatvec_249227
    kwargs_249229 = {'matmat': keyword_249226, 'matvec': keyword_249224, 'rmatvec': keyword_249228}
    # Getting the type of 'LinearOperator' (line 630)
    LinearOperator_249220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 11), 'LinearOperator', False)
    # Calling LinearOperator(args, kwargs) (line 630)
    LinearOperator_call_result_249230 = invoke(stypy.reporting.localization.Localization(__file__, 630, 11), LinearOperator_249220, *[shape_249222], **kwargs_249229)
    
    # Assigning a type to the variable 'stypy_return_type' (line 630)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 4), 'stypy_return_type', LinearOperator_call_result_249230)
    
    # ################# End of 'left_multiplied_operator(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'left_multiplied_operator' in the type store
    # Getting the type of 'stypy_return_type' (line 617)
    stypy_return_type_249231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_249231)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'left_multiplied_operator'
    return stypy_return_type_249231

# Assigning a type to the variable 'left_multiplied_operator' (line 617)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 0), 'left_multiplied_operator', left_multiplied_operator)

@norecursion
def right_multiplied_operator(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'right_multiplied_operator'
    module_type_store = module_type_store.open_function_context('right_multiplied_operator', 634, 0, False)
    
    # Passed parameters checking function
    right_multiplied_operator.stypy_localization = localization
    right_multiplied_operator.stypy_type_of_self = None
    right_multiplied_operator.stypy_type_store = module_type_store
    right_multiplied_operator.stypy_function_name = 'right_multiplied_operator'
    right_multiplied_operator.stypy_param_names_list = ['J', 'd']
    right_multiplied_operator.stypy_varargs_param_name = None
    right_multiplied_operator.stypy_kwargs_param_name = None
    right_multiplied_operator.stypy_call_defaults = defaults
    right_multiplied_operator.stypy_call_varargs = varargs
    right_multiplied_operator.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'right_multiplied_operator', ['J', 'd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'right_multiplied_operator', localization, ['J', 'd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'right_multiplied_operator(...)' code ##################

    str_249232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 4), 'str', 'Return J diag(d) as LinearOperator.')
    
    # Assigning a Call to a Name (line 636):
    
    # Assigning a Call to a Name (line 636):
    
    # Call to aslinearoperator(...): (line 636)
    # Processing the call arguments (line 636)
    # Getting the type of 'J' (line 636)
    J_249234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 25), 'J', False)
    # Processing the call keyword arguments (line 636)
    kwargs_249235 = {}
    # Getting the type of 'aslinearoperator' (line 636)
    aslinearoperator_249233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 8), 'aslinearoperator', False)
    # Calling aslinearoperator(args, kwargs) (line 636)
    aslinearoperator_call_result_249236 = invoke(stypy.reporting.localization.Localization(__file__, 636, 8), aslinearoperator_249233, *[J_249234], **kwargs_249235)
    
    # Assigning a type to the variable 'J' (line 636)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 4), 'J', aslinearoperator_call_result_249236)

    @norecursion
    def matvec(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'matvec'
        module_type_store = module_type_store.open_function_context('matvec', 638, 4, False)
        
        # Passed parameters checking function
        matvec.stypy_localization = localization
        matvec.stypy_type_of_self = None
        matvec.stypy_type_store = module_type_store
        matvec.stypy_function_name = 'matvec'
        matvec.stypy_param_names_list = ['x']
        matvec.stypy_varargs_param_name = None
        matvec.stypy_kwargs_param_name = None
        matvec.stypy_call_defaults = defaults
        matvec.stypy_call_varargs = varargs
        matvec.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'matvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'matvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'matvec(...)' code ##################

        
        # Call to matvec(...): (line 639)
        # Processing the call arguments (line 639)
        
        # Call to ravel(...): (line 639)
        # Processing the call arguments (line 639)
        # Getting the type of 'x' (line 639)
        x_249241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 33), 'x', False)
        # Processing the call keyword arguments (line 639)
        kwargs_249242 = {}
        # Getting the type of 'np' (line 639)
        np_249239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 24), 'np', False)
        # Obtaining the member 'ravel' of a type (line 639)
        ravel_249240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 24), np_249239, 'ravel')
        # Calling ravel(args, kwargs) (line 639)
        ravel_call_result_249243 = invoke(stypy.reporting.localization.Localization(__file__, 639, 24), ravel_249240, *[x_249241], **kwargs_249242)
        
        # Getting the type of 'd' (line 639)
        d_249244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 38), 'd', False)
        # Applying the binary operator '*' (line 639)
        result_mul_249245 = python_operator(stypy.reporting.localization.Localization(__file__, 639, 24), '*', ravel_call_result_249243, d_249244)
        
        # Processing the call keyword arguments (line 639)
        kwargs_249246 = {}
        # Getting the type of 'J' (line 639)
        J_249237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 15), 'J', False)
        # Obtaining the member 'matvec' of a type (line 639)
        matvec_249238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 15), J_249237, 'matvec')
        # Calling matvec(args, kwargs) (line 639)
        matvec_call_result_249247 = invoke(stypy.reporting.localization.Localization(__file__, 639, 15), matvec_249238, *[result_mul_249245], **kwargs_249246)
        
        # Assigning a type to the variable 'stypy_return_type' (line 639)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 8), 'stypy_return_type', matvec_call_result_249247)
        
        # ################# End of 'matvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'matvec' in the type store
        # Getting the type of 'stypy_return_type' (line 638)
        stypy_return_type_249248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_249248)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'matvec'
        return stypy_return_type_249248

    # Assigning a type to the variable 'matvec' (line 638)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'matvec', matvec)

    @norecursion
    def matmat(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'matmat'
        module_type_store = module_type_store.open_function_context('matmat', 641, 4, False)
        
        # Passed parameters checking function
        matmat.stypy_localization = localization
        matmat.stypy_type_of_self = None
        matmat.stypy_type_store = module_type_store
        matmat.stypy_function_name = 'matmat'
        matmat.stypy_param_names_list = ['X']
        matmat.stypy_varargs_param_name = None
        matmat.stypy_kwargs_param_name = None
        matmat.stypy_call_defaults = defaults
        matmat.stypy_call_varargs = varargs
        matmat.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'matmat', ['X'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'matmat', localization, ['X'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'matmat(...)' code ##################

        
        # Call to matmat(...): (line 642)
        # Processing the call arguments (line 642)
        # Getting the type of 'X' (line 642)
        X_249251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 24), 'X', False)
        
        # Obtaining the type of the subscript
        slice_249252 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 642, 28), None, None, None)
        # Getting the type of 'np' (line 642)
        np_249253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 33), 'np', False)
        # Obtaining the member 'newaxis' of a type (line 642)
        newaxis_249254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 33), np_249253, 'newaxis')
        # Getting the type of 'd' (line 642)
        d_249255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 28), 'd', False)
        # Obtaining the member '__getitem__' of a type (line 642)
        getitem___249256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 28), d_249255, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 642)
        subscript_call_result_249257 = invoke(stypy.reporting.localization.Localization(__file__, 642, 28), getitem___249256, (slice_249252, newaxis_249254))
        
        # Applying the binary operator '*' (line 642)
        result_mul_249258 = python_operator(stypy.reporting.localization.Localization(__file__, 642, 24), '*', X_249251, subscript_call_result_249257)
        
        # Processing the call keyword arguments (line 642)
        kwargs_249259 = {}
        # Getting the type of 'J' (line 642)
        J_249249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 15), 'J', False)
        # Obtaining the member 'matmat' of a type (line 642)
        matmat_249250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 15), J_249249, 'matmat')
        # Calling matmat(args, kwargs) (line 642)
        matmat_call_result_249260 = invoke(stypy.reporting.localization.Localization(__file__, 642, 15), matmat_249250, *[result_mul_249258], **kwargs_249259)
        
        # Assigning a type to the variable 'stypy_return_type' (line 642)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 8), 'stypy_return_type', matmat_call_result_249260)
        
        # ################# End of 'matmat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'matmat' in the type store
        # Getting the type of 'stypy_return_type' (line 641)
        stypy_return_type_249261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_249261)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'matmat'
        return stypy_return_type_249261

    # Assigning a type to the variable 'matmat' (line 641)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 4), 'matmat', matmat)

    @norecursion
    def rmatvec(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'rmatvec'
        module_type_store = module_type_store.open_function_context('rmatvec', 644, 4, False)
        
        # Passed parameters checking function
        rmatvec.stypy_localization = localization
        rmatvec.stypy_type_of_self = None
        rmatvec.stypy_type_store = module_type_store
        rmatvec.stypy_function_name = 'rmatvec'
        rmatvec.stypy_param_names_list = ['x']
        rmatvec.stypy_varargs_param_name = None
        rmatvec.stypy_kwargs_param_name = None
        rmatvec.stypy_call_defaults = defaults
        rmatvec.stypy_call_varargs = varargs
        rmatvec.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'rmatvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'rmatvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'rmatvec(...)' code ##################

        # Getting the type of 'd' (line 645)
        d_249262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 15), 'd')
        
        # Call to rmatvec(...): (line 645)
        # Processing the call arguments (line 645)
        # Getting the type of 'x' (line 645)
        x_249265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 29), 'x', False)
        # Processing the call keyword arguments (line 645)
        kwargs_249266 = {}
        # Getting the type of 'J' (line 645)
        J_249263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 19), 'J', False)
        # Obtaining the member 'rmatvec' of a type (line 645)
        rmatvec_249264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 19), J_249263, 'rmatvec')
        # Calling rmatvec(args, kwargs) (line 645)
        rmatvec_call_result_249267 = invoke(stypy.reporting.localization.Localization(__file__, 645, 19), rmatvec_249264, *[x_249265], **kwargs_249266)
        
        # Applying the binary operator '*' (line 645)
        result_mul_249268 = python_operator(stypy.reporting.localization.Localization(__file__, 645, 15), '*', d_249262, rmatvec_call_result_249267)
        
        # Assigning a type to the variable 'stypy_return_type' (line 645)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 8), 'stypy_return_type', result_mul_249268)
        
        # ################# End of 'rmatvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'rmatvec' in the type store
        # Getting the type of 'stypy_return_type' (line 644)
        stypy_return_type_249269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_249269)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'rmatvec'
        return stypy_return_type_249269

    # Assigning a type to the variable 'rmatvec' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 4), 'rmatvec', rmatvec)
    
    # Call to LinearOperator(...): (line 647)
    # Processing the call arguments (line 647)
    # Getting the type of 'J' (line 647)
    J_249271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 26), 'J', False)
    # Obtaining the member 'shape' of a type (line 647)
    shape_249272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 26), J_249271, 'shape')
    # Processing the call keyword arguments (line 647)
    # Getting the type of 'matvec' (line 647)
    matvec_249273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 42), 'matvec', False)
    keyword_249274 = matvec_249273
    # Getting the type of 'matmat' (line 647)
    matmat_249275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 57), 'matmat', False)
    keyword_249276 = matmat_249275
    # Getting the type of 'rmatvec' (line 648)
    rmatvec_249277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 34), 'rmatvec', False)
    keyword_249278 = rmatvec_249277
    kwargs_249279 = {'matmat': keyword_249276, 'matvec': keyword_249274, 'rmatvec': keyword_249278}
    # Getting the type of 'LinearOperator' (line 647)
    LinearOperator_249270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 11), 'LinearOperator', False)
    # Calling LinearOperator(args, kwargs) (line 647)
    LinearOperator_call_result_249280 = invoke(stypy.reporting.localization.Localization(__file__, 647, 11), LinearOperator_249270, *[shape_249272], **kwargs_249279)
    
    # Assigning a type to the variable 'stypy_return_type' (line 647)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 4), 'stypy_return_type', LinearOperator_call_result_249280)
    
    # ################# End of 'right_multiplied_operator(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'right_multiplied_operator' in the type store
    # Getting the type of 'stypy_return_type' (line 634)
    stypy_return_type_249281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_249281)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'right_multiplied_operator'
    return stypy_return_type_249281

# Assigning a type to the variable 'right_multiplied_operator' (line 634)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 0), 'right_multiplied_operator', right_multiplied_operator)

@norecursion
def regularized_lsq_operator(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'regularized_lsq_operator'
    module_type_store = module_type_store.open_function_context('regularized_lsq_operator', 651, 0, False)
    
    # Passed parameters checking function
    regularized_lsq_operator.stypy_localization = localization
    regularized_lsq_operator.stypy_type_of_self = None
    regularized_lsq_operator.stypy_type_store = module_type_store
    regularized_lsq_operator.stypy_function_name = 'regularized_lsq_operator'
    regularized_lsq_operator.stypy_param_names_list = ['J', 'diag']
    regularized_lsq_operator.stypy_varargs_param_name = None
    regularized_lsq_operator.stypy_kwargs_param_name = None
    regularized_lsq_operator.stypy_call_defaults = defaults
    regularized_lsq_operator.stypy_call_varargs = varargs
    regularized_lsq_operator.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'regularized_lsq_operator', ['J', 'diag'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'regularized_lsq_operator', localization, ['J', 'diag'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'regularized_lsq_operator(...)' code ##################

    str_249282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, (-1)), 'str', 'Return a matrix arising in regularized least squares as LinearOperator.\n    \n    The matrix is\n        [ J ]\n        [ D ]\n    where D is diagonal matrix with elements from `diag`.\n    ')
    
    # Assigning a Call to a Name (line 659):
    
    # Assigning a Call to a Name (line 659):
    
    # Call to aslinearoperator(...): (line 659)
    # Processing the call arguments (line 659)
    # Getting the type of 'J' (line 659)
    J_249284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 25), 'J', False)
    # Processing the call keyword arguments (line 659)
    kwargs_249285 = {}
    # Getting the type of 'aslinearoperator' (line 659)
    aslinearoperator_249283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 8), 'aslinearoperator', False)
    # Calling aslinearoperator(args, kwargs) (line 659)
    aslinearoperator_call_result_249286 = invoke(stypy.reporting.localization.Localization(__file__, 659, 8), aslinearoperator_249283, *[J_249284], **kwargs_249285)
    
    # Assigning a type to the variable 'J' (line 659)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 4), 'J', aslinearoperator_call_result_249286)
    
    # Assigning a Attribute to a Tuple (line 660):
    
    # Assigning a Subscript to a Name (line 660):
    
    # Obtaining the type of the subscript
    int_249287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 4), 'int')
    # Getting the type of 'J' (line 660)
    J_249288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 11), 'J')
    # Obtaining the member 'shape' of a type (line 660)
    shape_249289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 11), J_249288, 'shape')
    # Obtaining the member '__getitem__' of a type (line 660)
    getitem___249290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 4), shape_249289, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 660)
    subscript_call_result_249291 = invoke(stypy.reporting.localization.Localization(__file__, 660, 4), getitem___249290, int_249287)
    
    # Assigning a type to the variable 'tuple_var_assignment_247675' (line 660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 4), 'tuple_var_assignment_247675', subscript_call_result_249291)
    
    # Assigning a Subscript to a Name (line 660):
    
    # Obtaining the type of the subscript
    int_249292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 4), 'int')
    # Getting the type of 'J' (line 660)
    J_249293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 11), 'J')
    # Obtaining the member 'shape' of a type (line 660)
    shape_249294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 11), J_249293, 'shape')
    # Obtaining the member '__getitem__' of a type (line 660)
    getitem___249295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 4), shape_249294, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 660)
    subscript_call_result_249296 = invoke(stypy.reporting.localization.Localization(__file__, 660, 4), getitem___249295, int_249292)
    
    # Assigning a type to the variable 'tuple_var_assignment_247676' (line 660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 4), 'tuple_var_assignment_247676', subscript_call_result_249296)
    
    # Assigning a Name to a Name (line 660):
    # Getting the type of 'tuple_var_assignment_247675' (line 660)
    tuple_var_assignment_247675_249297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 4), 'tuple_var_assignment_247675')
    # Assigning a type to the variable 'm' (line 660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 4), 'm', tuple_var_assignment_247675_249297)
    
    # Assigning a Name to a Name (line 660):
    # Getting the type of 'tuple_var_assignment_247676' (line 660)
    tuple_var_assignment_247676_249298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 4), 'tuple_var_assignment_247676')
    # Assigning a type to the variable 'n' (line 660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 7), 'n', tuple_var_assignment_247676_249298)

    @norecursion
    def matvec(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'matvec'
        module_type_store = module_type_store.open_function_context('matvec', 662, 4, False)
        
        # Passed parameters checking function
        matvec.stypy_localization = localization
        matvec.stypy_type_of_self = None
        matvec.stypy_type_store = module_type_store
        matvec.stypy_function_name = 'matvec'
        matvec.stypy_param_names_list = ['x']
        matvec.stypy_varargs_param_name = None
        matvec.stypy_kwargs_param_name = None
        matvec.stypy_call_defaults = defaults
        matvec.stypy_call_varargs = varargs
        matvec.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'matvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'matvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'matvec(...)' code ##################

        
        # Call to hstack(...): (line 663)
        # Processing the call arguments (line 663)
        
        # Obtaining an instance of the builtin type 'tuple' (line 663)
        tuple_249301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 663)
        # Adding element type (line 663)
        
        # Call to matvec(...): (line 663)
        # Processing the call arguments (line 663)
        # Getting the type of 'x' (line 663)
        x_249304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 35), 'x', False)
        # Processing the call keyword arguments (line 663)
        kwargs_249305 = {}
        # Getting the type of 'J' (line 663)
        J_249302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 26), 'J', False)
        # Obtaining the member 'matvec' of a type (line 663)
        matvec_249303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 26), J_249302, 'matvec')
        # Calling matvec(args, kwargs) (line 663)
        matvec_call_result_249306 = invoke(stypy.reporting.localization.Localization(__file__, 663, 26), matvec_249303, *[x_249304], **kwargs_249305)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 663, 26), tuple_249301, matvec_call_result_249306)
        # Adding element type (line 663)
        # Getting the type of 'diag' (line 663)
        diag_249307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 39), 'diag', False)
        # Getting the type of 'x' (line 663)
        x_249308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 46), 'x', False)
        # Applying the binary operator '*' (line 663)
        result_mul_249309 = python_operator(stypy.reporting.localization.Localization(__file__, 663, 39), '*', diag_249307, x_249308)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 663, 26), tuple_249301, result_mul_249309)
        
        # Processing the call keyword arguments (line 663)
        kwargs_249310 = {}
        # Getting the type of 'np' (line 663)
        np_249299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 15), 'np', False)
        # Obtaining the member 'hstack' of a type (line 663)
        hstack_249300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 15), np_249299, 'hstack')
        # Calling hstack(args, kwargs) (line 663)
        hstack_call_result_249311 = invoke(stypy.reporting.localization.Localization(__file__, 663, 15), hstack_249300, *[tuple_249301], **kwargs_249310)
        
        # Assigning a type to the variable 'stypy_return_type' (line 663)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'stypy_return_type', hstack_call_result_249311)
        
        # ################# End of 'matvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'matvec' in the type store
        # Getting the type of 'stypy_return_type' (line 662)
        stypy_return_type_249312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_249312)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'matvec'
        return stypy_return_type_249312

    # Assigning a type to the variable 'matvec' (line 662)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 4), 'matvec', matvec)

    @norecursion
    def rmatvec(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'rmatvec'
        module_type_store = module_type_store.open_function_context('rmatvec', 665, 4, False)
        
        # Passed parameters checking function
        rmatvec.stypy_localization = localization
        rmatvec.stypy_type_of_self = None
        rmatvec.stypy_type_store = module_type_store
        rmatvec.stypy_function_name = 'rmatvec'
        rmatvec.stypy_param_names_list = ['x']
        rmatvec.stypy_varargs_param_name = None
        rmatvec.stypy_kwargs_param_name = None
        rmatvec.stypy_call_defaults = defaults
        rmatvec.stypy_call_varargs = varargs
        rmatvec.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'rmatvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'rmatvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'rmatvec(...)' code ##################

        
        # Assigning a Subscript to a Name (line 666):
        
        # Assigning a Subscript to a Name (line 666):
        
        # Obtaining the type of the subscript
        # Getting the type of 'm' (line 666)
        m_249313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 16), 'm')
        slice_249314 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 666, 13), None, m_249313, None)
        # Getting the type of 'x' (line 666)
        x_249315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 13), 'x')
        # Obtaining the member '__getitem__' of a type (line 666)
        getitem___249316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 13), x_249315, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 666)
        subscript_call_result_249317 = invoke(stypy.reporting.localization.Localization(__file__, 666, 13), getitem___249316, slice_249314)
        
        # Assigning a type to the variable 'x1' (line 666)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 8), 'x1', subscript_call_result_249317)
        
        # Assigning a Subscript to a Name (line 667):
        
        # Assigning a Subscript to a Name (line 667):
        
        # Obtaining the type of the subscript
        # Getting the type of 'm' (line 667)
        m_249318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 15), 'm')
        slice_249319 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 667, 13), m_249318, None, None)
        # Getting the type of 'x' (line 667)
        x_249320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 13), 'x')
        # Obtaining the member '__getitem__' of a type (line 667)
        getitem___249321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 13), x_249320, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 667)
        subscript_call_result_249322 = invoke(stypy.reporting.localization.Localization(__file__, 667, 13), getitem___249321, slice_249319)
        
        # Assigning a type to the variable 'x2' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 8), 'x2', subscript_call_result_249322)
        
        # Call to rmatvec(...): (line 668)
        # Processing the call arguments (line 668)
        # Getting the type of 'x1' (line 668)
        x1_249325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 25), 'x1', False)
        # Processing the call keyword arguments (line 668)
        kwargs_249326 = {}
        # Getting the type of 'J' (line 668)
        J_249323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 15), 'J', False)
        # Obtaining the member 'rmatvec' of a type (line 668)
        rmatvec_249324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 15), J_249323, 'rmatvec')
        # Calling rmatvec(args, kwargs) (line 668)
        rmatvec_call_result_249327 = invoke(stypy.reporting.localization.Localization(__file__, 668, 15), rmatvec_249324, *[x1_249325], **kwargs_249326)
        
        # Getting the type of 'diag' (line 668)
        diag_249328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 31), 'diag')
        # Getting the type of 'x2' (line 668)
        x2_249329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 38), 'x2')
        # Applying the binary operator '*' (line 668)
        result_mul_249330 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 31), '*', diag_249328, x2_249329)
        
        # Applying the binary operator '+' (line 668)
        result_add_249331 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 15), '+', rmatvec_call_result_249327, result_mul_249330)
        
        # Assigning a type to the variable 'stypy_return_type' (line 668)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 8), 'stypy_return_type', result_add_249331)
        
        # ################# End of 'rmatvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'rmatvec' in the type store
        # Getting the type of 'stypy_return_type' (line 665)
        stypy_return_type_249332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_249332)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'rmatvec'
        return stypy_return_type_249332

    # Assigning a type to the variable 'rmatvec' (line 665)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 4), 'rmatvec', rmatvec)
    
    # Call to LinearOperator(...): (line 670)
    # Processing the call arguments (line 670)
    
    # Obtaining an instance of the builtin type 'tuple' (line 670)
    tuple_249334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 670)
    # Adding element type (line 670)
    # Getting the type of 'm' (line 670)
    m_249335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 27), 'm', False)
    # Getting the type of 'n' (line 670)
    n_249336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 31), 'n', False)
    # Applying the binary operator '+' (line 670)
    result_add_249337 = python_operator(stypy.reporting.localization.Localization(__file__, 670, 27), '+', m_249335, n_249336)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 670, 27), tuple_249334, result_add_249337)
    # Adding element type (line 670)
    # Getting the type of 'n' (line 670)
    n_249338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 34), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 670, 27), tuple_249334, n_249338)
    
    # Processing the call keyword arguments (line 670)
    # Getting the type of 'matvec' (line 670)
    matvec_249339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 45), 'matvec', False)
    keyword_249340 = matvec_249339
    # Getting the type of 'rmatvec' (line 670)
    rmatvec_249341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 61), 'rmatvec', False)
    keyword_249342 = rmatvec_249341
    kwargs_249343 = {'rmatvec': keyword_249342, 'matvec': keyword_249340}
    # Getting the type of 'LinearOperator' (line 670)
    LinearOperator_249333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 11), 'LinearOperator', False)
    # Calling LinearOperator(args, kwargs) (line 670)
    LinearOperator_call_result_249344 = invoke(stypy.reporting.localization.Localization(__file__, 670, 11), LinearOperator_249333, *[tuple_249334], **kwargs_249343)
    
    # Assigning a type to the variable 'stypy_return_type' (line 670)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 4), 'stypy_return_type', LinearOperator_call_result_249344)
    
    # ################# End of 'regularized_lsq_operator(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'regularized_lsq_operator' in the type store
    # Getting the type of 'stypy_return_type' (line 651)
    stypy_return_type_249345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_249345)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'regularized_lsq_operator'
    return stypy_return_type_249345

# Assigning a type to the variable 'regularized_lsq_operator' (line 651)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 0), 'regularized_lsq_operator', regularized_lsq_operator)

@norecursion
def right_multiply(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 673)
    True_249346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 30), 'True')
    defaults = [True_249346]
    # Create a new context for function 'right_multiply'
    module_type_store = module_type_store.open_function_context('right_multiply', 673, 0, False)
    
    # Passed parameters checking function
    right_multiply.stypy_localization = localization
    right_multiply.stypy_type_of_self = None
    right_multiply.stypy_type_store = module_type_store
    right_multiply.stypy_function_name = 'right_multiply'
    right_multiply.stypy_param_names_list = ['J', 'd', 'copy']
    right_multiply.stypy_varargs_param_name = None
    right_multiply.stypy_kwargs_param_name = None
    right_multiply.stypy_call_defaults = defaults
    right_multiply.stypy_call_varargs = varargs
    right_multiply.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'right_multiply', ['J', 'd', 'copy'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'right_multiply', localization, ['J', 'd', 'copy'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'right_multiply(...)' code ##################

    str_249347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, (-1)), 'str', 'Compute J diag(d).\n    \n    If `copy` is False, `J` is modified in place (unless being LinearOperator).\n    ')
    
    
    # Evaluating a boolean operation
    # Getting the type of 'copy' (line 678)
    copy_249348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 7), 'copy')
    
    
    # Call to isinstance(...): (line 678)
    # Processing the call arguments (line 678)
    # Getting the type of 'J' (line 678)
    J_249350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 31), 'J', False)
    # Getting the type of 'LinearOperator' (line 678)
    LinearOperator_249351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 34), 'LinearOperator', False)
    # Processing the call keyword arguments (line 678)
    kwargs_249352 = {}
    # Getting the type of 'isinstance' (line 678)
    isinstance_249349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 20), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 678)
    isinstance_call_result_249353 = invoke(stypy.reporting.localization.Localization(__file__, 678, 20), isinstance_249349, *[J_249350, LinearOperator_249351], **kwargs_249352)
    
    # Applying the 'not' unary operator (line 678)
    result_not__249354 = python_operator(stypy.reporting.localization.Localization(__file__, 678, 16), 'not', isinstance_call_result_249353)
    
    # Applying the binary operator 'and' (line 678)
    result_and_keyword_249355 = python_operator(stypy.reporting.localization.Localization(__file__, 678, 7), 'and', copy_249348, result_not__249354)
    
    # Testing the type of an if condition (line 678)
    if_condition_249356 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 678, 4), result_and_keyword_249355)
    # Assigning a type to the variable 'if_condition_249356' (line 678)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 4), 'if_condition_249356', if_condition_249356)
    # SSA begins for if statement (line 678)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 679):
    
    # Assigning a Call to a Name (line 679):
    
    # Call to copy(...): (line 679)
    # Processing the call keyword arguments (line 679)
    kwargs_249359 = {}
    # Getting the type of 'J' (line 679)
    J_249357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 12), 'J', False)
    # Obtaining the member 'copy' of a type (line 679)
    copy_249358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 12), J_249357, 'copy')
    # Calling copy(args, kwargs) (line 679)
    copy_call_result_249360 = invoke(stypy.reporting.localization.Localization(__file__, 679, 12), copy_249358, *[], **kwargs_249359)
    
    # Assigning a type to the variable 'J' (line 679)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 8), 'J', copy_call_result_249360)
    # SSA join for if statement (line 678)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to issparse(...): (line 681)
    # Processing the call arguments (line 681)
    # Getting the type of 'J' (line 681)
    J_249362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 16), 'J', False)
    # Processing the call keyword arguments (line 681)
    kwargs_249363 = {}
    # Getting the type of 'issparse' (line 681)
    issparse_249361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 7), 'issparse', False)
    # Calling issparse(args, kwargs) (line 681)
    issparse_call_result_249364 = invoke(stypy.reporting.localization.Localization(__file__, 681, 7), issparse_249361, *[J_249362], **kwargs_249363)
    
    # Testing the type of an if condition (line 681)
    if_condition_249365 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 681, 4), issparse_call_result_249364)
    # Assigning a type to the variable 'if_condition_249365' (line 681)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 4), 'if_condition_249365', if_condition_249365)
    # SSA begins for if statement (line 681)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'J' (line 682)
    J_249366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 8), 'J')
    # Obtaining the member 'data' of a type (line 682)
    data_249367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 8), J_249366, 'data')
    
    # Call to take(...): (line 682)
    # Processing the call arguments (line 682)
    # Getting the type of 'J' (line 682)
    J_249370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 25), 'J', False)
    # Obtaining the member 'indices' of a type (line 682)
    indices_249371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 25), J_249370, 'indices')
    # Processing the call keyword arguments (line 682)
    str_249372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 41), 'str', 'clip')
    keyword_249373 = str_249372
    kwargs_249374 = {'mode': keyword_249373}
    # Getting the type of 'd' (line 682)
    d_249368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 18), 'd', False)
    # Obtaining the member 'take' of a type (line 682)
    take_249369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 18), d_249368, 'take')
    # Calling take(args, kwargs) (line 682)
    take_call_result_249375 = invoke(stypy.reporting.localization.Localization(__file__, 682, 18), take_249369, *[indices_249371], **kwargs_249374)
    
    # Applying the binary operator '*=' (line 682)
    result_imul_249376 = python_operator(stypy.reporting.localization.Localization(__file__, 682, 8), '*=', data_249367, take_call_result_249375)
    # Getting the type of 'J' (line 682)
    J_249377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 8), 'J')
    # Setting the type of the member 'data' of a type (line 682)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 8), J_249377, 'data', result_imul_249376)
    
    # SSA branch for the else part of an if statement (line 681)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isinstance(...): (line 683)
    # Processing the call arguments (line 683)
    # Getting the type of 'J' (line 683)
    J_249379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 20), 'J', False)
    # Getting the type of 'LinearOperator' (line 683)
    LinearOperator_249380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 23), 'LinearOperator', False)
    # Processing the call keyword arguments (line 683)
    kwargs_249381 = {}
    # Getting the type of 'isinstance' (line 683)
    isinstance_249378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 9), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 683)
    isinstance_call_result_249382 = invoke(stypy.reporting.localization.Localization(__file__, 683, 9), isinstance_249378, *[J_249379, LinearOperator_249380], **kwargs_249381)
    
    # Testing the type of an if condition (line 683)
    if_condition_249383 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 683, 9), isinstance_call_result_249382)
    # Assigning a type to the variable 'if_condition_249383' (line 683)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 9), 'if_condition_249383', if_condition_249383)
    # SSA begins for if statement (line 683)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 684):
    
    # Assigning a Call to a Name (line 684):
    
    # Call to right_multiplied_operator(...): (line 684)
    # Processing the call arguments (line 684)
    # Getting the type of 'J' (line 684)
    J_249385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 38), 'J', False)
    # Getting the type of 'd' (line 684)
    d_249386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 41), 'd', False)
    # Processing the call keyword arguments (line 684)
    kwargs_249387 = {}
    # Getting the type of 'right_multiplied_operator' (line 684)
    right_multiplied_operator_249384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 12), 'right_multiplied_operator', False)
    # Calling right_multiplied_operator(args, kwargs) (line 684)
    right_multiplied_operator_call_result_249388 = invoke(stypy.reporting.localization.Localization(__file__, 684, 12), right_multiplied_operator_249384, *[J_249385, d_249386], **kwargs_249387)
    
    # Assigning a type to the variable 'J' (line 684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 8), 'J', right_multiplied_operator_call_result_249388)
    # SSA branch for the else part of an if statement (line 683)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'J' (line 686)
    J_249389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 8), 'J')
    # Getting the type of 'd' (line 686)
    d_249390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 13), 'd')
    # Applying the binary operator '*=' (line 686)
    result_imul_249391 = python_operator(stypy.reporting.localization.Localization(__file__, 686, 8), '*=', J_249389, d_249390)
    # Assigning a type to the variable 'J' (line 686)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 8), 'J', result_imul_249391)
    
    # SSA join for if statement (line 683)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 681)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'J' (line 688)
    J_249392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 11), 'J')
    # Assigning a type to the variable 'stypy_return_type' (line 688)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 688, 4), 'stypy_return_type', J_249392)
    
    # ################# End of 'right_multiply(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'right_multiply' in the type store
    # Getting the type of 'stypy_return_type' (line 673)
    stypy_return_type_249393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_249393)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'right_multiply'
    return stypy_return_type_249393

# Assigning a type to the variable 'right_multiply' (line 673)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 0), 'right_multiply', right_multiply)

@norecursion
def left_multiply(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 691)
    True_249394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 29), 'True')
    defaults = [True_249394]
    # Create a new context for function 'left_multiply'
    module_type_store = module_type_store.open_function_context('left_multiply', 691, 0, False)
    
    # Passed parameters checking function
    left_multiply.stypy_localization = localization
    left_multiply.stypy_type_of_self = None
    left_multiply.stypy_type_store = module_type_store
    left_multiply.stypy_function_name = 'left_multiply'
    left_multiply.stypy_param_names_list = ['J', 'd', 'copy']
    left_multiply.stypy_varargs_param_name = None
    left_multiply.stypy_kwargs_param_name = None
    left_multiply.stypy_call_defaults = defaults
    left_multiply.stypy_call_varargs = varargs
    left_multiply.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'left_multiply', ['J', 'd', 'copy'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'left_multiply', localization, ['J', 'd', 'copy'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'left_multiply(...)' code ##################

    str_249395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, (-1)), 'str', 'Compute diag(d) J.\n    \n    If `copy` is False, `J` is modified in place (unless being LinearOperator).\n    ')
    
    
    # Evaluating a boolean operation
    # Getting the type of 'copy' (line 696)
    copy_249396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 7), 'copy')
    
    
    # Call to isinstance(...): (line 696)
    # Processing the call arguments (line 696)
    # Getting the type of 'J' (line 696)
    J_249398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 31), 'J', False)
    # Getting the type of 'LinearOperator' (line 696)
    LinearOperator_249399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 34), 'LinearOperator', False)
    # Processing the call keyword arguments (line 696)
    kwargs_249400 = {}
    # Getting the type of 'isinstance' (line 696)
    isinstance_249397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 20), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 696)
    isinstance_call_result_249401 = invoke(stypy.reporting.localization.Localization(__file__, 696, 20), isinstance_249397, *[J_249398, LinearOperator_249399], **kwargs_249400)
    
    # Applying the 'not' unary operator (line 696)
    result_not__249402 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 16), 'not', isinstance_call_result_249401)
    
    # Applying the binary operator 'and' (line 696)
    result_and_keyword_249403 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 7), 'and', copy_249396, result_not__249402)
    
    # Testing the type of an if condition (line 696)
    if_condition_249404 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 696, 4), result_and_keyword_249403)
    # Assigning a type to the variable 'if_condition_249404' (line 696)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 4), 'if_condition_249404', if_condition_249404)
    # SSA begins for if statement (line 696)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 697):
    
    # Assigning a Call to a Name (line 697):
    
    # Call to copy(...): (line 697)
    # Processing the call keyword arguments (line 697)
    kwargs_249407 = {}
    # Getting the type of 'J' (line 697)
    J_249405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 12), 'J', False)
    # Obtaining the member 'copy' of a type (line 697)
    copy_249406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 12), J_249405, 'copy')
    # Calling copy(args, kwargs) (line 697)
    copy_call_result_249408 = invoke(stypy.reporting.localization.Localization(__file__, 697, 12), copy_249406, *[], **kwargs_249407)
    
    # Assigning a type to the variable 'J' (line 697)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 8), 'J', copy_call_result_249408)
    # SSA join for if statement (line 696)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to issparse(...): (line 699)
    # Processing the call arguments (line 699)
    # Getting the type of 'J' (line 699)
    J_249410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 16), 'J', False)
    # Processing the call keyword arguments (line 699)
    kwargs_249411 = {}
    # Getting the type of 'issparse' (line 699)
    issparse_249409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 7), 'issparse', False)
    # Calling issparse(args, kwargs) (line 699)
    issparse_call_result_249412 = invoke(stypy.reporting.localization.Localization(__file__, 699, 7), issparse_249409, *[J_249410], **kwargs_249411)
    
    # Testing the type of an if condition (line 699)
    if_condition_249413 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 699, 4), issparse_call_result_249412)
    # Assigning a type to the variable 'if_condition_249413' (line 699)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 4), 'if_condition_249413', if_condition_249413)
    # SSA begins for if statement (line 699)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'J' (line 700)
    J_249414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 8), 'J')
    # Obtaining the member 'data' of a type (line 700)
    data_249415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 8), J_249414, 'data')
    
    # Call to repeat(...): (line 700)
    # Processing the call arguments (line 700)
    # Getting the type of 'd' (line 700)
    d_249418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 28), 'd', False)
    
    # Call to diff(...): (line 700)
    # Processing the call arguments (line 700)
    # Getting the type of 'J' (line 700)
    J_249421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 39), 'J', False)
    # Obtaining the member 'indptr' of a type (line 700)
    indptr_249422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 39), J_249421, 'indptr')
    # Processing the call keyword arguments (line 700)
    kwargs_249423 = {}
    # Getting the type of 'np' (line 700)
    np_249419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 31), 'np', False)
    # Obtaining the member 'diff' of a type (line 700)
    diff_249420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 31), np_249419, 'diff')
    # Calling diff(args, kwargs) (line 700)
    diff_call_result_249424 = invoke(stypy.reporting.localization.Localization(__file__, 700, 31), diff_249420, *[indptr_249422], **kwargs_249423)
    
    # Processing the call keyword arguments (line 700)
    kwargs_249425 = {}
    # Getting the type of 'np' (line 700)
    np_249416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 18), 'np', False)
    # Obtaining the member 'repeat' of a type (line 700)
    repeat_249417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 18), np_249416, 'repeat')
    # Calling repeat(args, kwargs) (line 700)
    repeat_call_result_249426 = invoke(stypy.reporting.localization.Localization(__file__, 700, 18), repeat_249417, *[d_249418, diff_call_result_249424], **kwargs_249425)
    
    # Applying the binary operator '*=' (line 700)
    result_imul_249427 = python_operator(stypy.reporting.localization.Localization(__file__, 700, 8), '*=', data_249415, repeat_call_result_249426)
    # Getting the type of 'J' (line 700)
    J_249428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 8), 'J')
    # Setting the type of the member 'data' of a type (line 700)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 8), J_249428, 'data', result_imul_249427)
    
    # SSA branch for the else part of an if statement (line 699)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isinstance(...): (line 701)
    # Processing the call arguments (line 701)
    # Getting the type of 'J' (line 701)
    J_249430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 20), 'J', False)
    # Getting the type of 'LinearOperator' (line 701)
    LinearOperator_249431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 23), 'LinearOperator', False)
    # Processing the call keyword arguments (line 701)
    kwargs_249432 = {}
    # Getting the type of 'isinstance' (line 701)
    isinstance_249429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 9), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 701)
    isinstance_call_result_249433 = invoke(stypy.reporting.localization.Localization(__file__, 701, 9), isinstance_249429, *[J_249430, LinearOperator_249431], **kwargs_249432)
    
    # Testing the type of an if condition (line 701)
    if_condition_249434 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 701, 9), isinstance_call_result_249433)
    # Assigning a type to the variable 'if_condition_249434' (line 701)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 9), 'if_condition_249434', if_condition_249434)
    # SSA begins for if statement (line 701)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 702):
    
    # Assigning a Call to a Name (line 702):
    
    # Call to left_multiplied_operator(...): (line 702)
    # Processing the call arguments (line 702)
    # Getting the type of 'J' (line 702)
    J_249436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 37), 'J', False)
    # Getting the type of 'd' (line 702)
    d_249437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 40), 'd', False)
    # Processing the call keyword arguments (line 702)
    kwargs_249438 = {}
    # Getting the type of 'left_multiplied_operator' (line 702)
    left_multiplied_operator_249435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 12), 'left_multiplied_operator', False)
    # Calling left_multiplied_operator(args, kwargs) (line 702)
    left_multiplied_operator_call_result_249439 = invoke(stypy.reporting.localization.Localization(__file__, 702, 12), left_multiplied_operator_249435, *[J_249436, d_249437], **kwargs_249438)
    
    # Assigning a type to the variable 'J' (line 702)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 8), 'J', left_multiplied_operator_call_result_249439)
    # SSA branch for the else part of an if statement (line 701)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'J' (line 704)
    J_249440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 8), 'J')
    
    # Obtaining the type of the subscript
    slice_249441 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 704, 13), None, None, None)
    # Getting the type of 'np' (line 704)
    np_249442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 18), 'np')
    # Obtaining the member 'newaxis' of a type (line 704)
    newaxis_249443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 18), np_249442, 'newaxis')
    # Getting the type of 'd' (line 704)
    d_249444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 13), 'd')
    # Obtaining the member '__getitem__' of a type (line 704)
    getitem___249445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 13), d_249444, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 704)
    subscript_call_result_249446 = invoke(stypy.reporting.localization.Localization(__file__, 704, 13), getitem___249445, (slice_249441, newaxis_249443))
    
    # Applying the binary operator '*=' (line 704)
    result_imul_249447 = python_operator(stypy.reporting.localization.Localization(__file__, 704, 8), '*=', J_249440, subscript_call_result_249446)
    # Assigning a type to the variable 'J' (line 704)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 8), 'J', result_imul_249447)
    
    # SSA join for if statement (line 701)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 699)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'J' (line 706)
    J_249448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 11), 'J')
    # Assigning a type to the variable 'stypy_return_type' (line 706)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 4), 'stypy_return_type', J_249448)
    
    # ################# End of 'left_multiply(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'left_multiply' in the type store
    # Getting the type of 'stypy_return_type' (line 691)
    stypy_return_type_249449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_249449)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'left_multiply'
    return stypy_return_type_249449

# Assigning a type to the variable 'left_multiply' (line 691)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 0), 'left_multiply', left_multiply)

@norecursion
def check_termination(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_termination'
    module_type_store = module_type_store.open_function_context('check_termination', 709, 0, False)
    
    # Passed parameters checking function
    check_termination.stypy_localization = localization
    check_termination.stypy_type_of_self = None
    check_termination.stypy_type_store = module_type_store
    check_termination.stypy_function_name = 'check_termination'
    check_termination.stypy_param_names_list = ['dF', 'F', 'dx_norm', 'x_norm', 'ratio', 'ftol', 'xtol']
    check_termination.stypy_varargs_param_name = None
    check_termination.stypy_kwargs_param_name = None
    check_termination.stypy_call_defaults = defaults
    check_termination.stypy_call_varargs = varargs
    check_termination.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_termination', ['dF', 'F', 'dx_norm', 'x_norm', 'ratio', 'ftol', 'xtol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_termination', localization, ['dF', 'F', 'dx_norm', 'x_norm', 'ratio', 'ftol', 'xtol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_termination(...)' code ##################

    str_249450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 4), 'str', 'Check termination condition for nonlinear least squares.')
    
    # Assigning a BoolOp to a Name (line 711):
    
    # Assigning a BoolOp to a Name (line 711):
    
    # Evaluating a boolean operation
    
    # Getting the type of 'dF' (line 711)
    dF_249451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 21), 'dF')
    # Getting the type of 'ftol' (line 711)
    ftol_249452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 26), 'ftol')
    # Getting the type of 'F' (line 711)
    F_249453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 33), 'F')
    # Applying the binary operator '*' (line 711)
    result_mul_249454 = python_operator(stypy.reporting.localization.Localization(__file__, 711, 26), '*', ftol_249452, F_249453)
    
    # Applying the binary operator '<' (line 711)
    result_lt_249455 = python_operator(stypy.reporting.localization.Localization(__file__, 711, 21), '<', dF_249451, result_mul_249454)
    
    
    # Getting the type of 'ratio' (line 711)
    ratio_249456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 39), 'ratio')
    float_249457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 47), 'float')
    # Applying the binary operator '>' (line 711)
    result_gt_249458 = python_operator(stypy.reporting.localization.Localization(__file__, 711, 39), '>', ratio_249456, float_249457)
    
    # Applying the binary operator 'and' (line 711)
    result_and_keyword_249459 = python_operator(stypy.reporting.localization.Localization(__file__, 711, 21), 'and', result_lt_249455, result_gt_249458)
    
    # Assigning a type to the variable 'ftol_satisfied' (line 711)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 4), 'ftol_satisfied', result_and_keyword_249459)
    
    # Assigning a Compare to a Name (line 712):
    
    # Assigning a Compare to a Name (line 712):
    
    # Getting the type of 'dx_norm' (line 712)
    dx_norm_249460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 21), 'dx_norm')
    # Getting the type of 'xtol' (line 712)
    xtol_249461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 31), 'xtol')
    # Getting the type of 'xtol' (line 712)
    xtol_249462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 39), 'xtol')
    # Getting the type of 'x_norm' (line 712)
    x_norm_249463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 46), 'x_norm')
    # Applying the binary operator '+' (line 712)
    result_add_249464 = python_operator(stypy.reporting.localization.Localization(__file__, 712, 39), '+', xtol_249462, x_norm_249463)
    
    # Applying the binary operator '*' (line 712)
    result_mul_249465 = python_operator(stypy.reporting.localization.Localization(__file__, 712, 31), '*', xtol_249461, result_add_249464)
    
    # Applying the binary operator '<' (line 712)
    result_lt_249466 = python_operator(stypy.reporting.localization.Localization(__file__, 712, 21), '<', dx_norm_249460, result_mul_249465)
    
    # Assigning a type to the variable 'xtol_satisfied' (line 712)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 4), 'xtol_satisfied', result_lt_249466)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'ftol_satisfied' (line 714)
    ftol_satisfied_249467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 7), 'ftol_satisfied')
    # Getting the type of 'xtol_satisfied' (line 714)
    xtol_satisfied_249468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 26), 'xtol_satisfied')
    # Applying the binary operator 'and' (line 714)
    result_and_keyword_249469 = python_operator(stypy.reporting.localization.Localization(__file__, 714, 7), 'and', ftol_satisfied_249467, xtol_satisfied_249468)
    
    # Testing the type of an if condition (line 714)
    if_condition_249470 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 714, 4), result_and_keyword_249469)
    # Assigning a type to the variable 'if_condition_249470' (line 714)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 4), 'if_condition_249470', if_condition_249470)
    # SSA begins for if statement (line 714)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_249471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 715)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 8), 'stypy_return_type', int_249471)
    # SSA branch for the else part of an if statement (line 714)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'ftol_satisfied' (line 716)
    ftol_satisfied_249472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 9), 'ftol_satisfied')
    # Testing the type of an if condition (line 716)
    if_condition_249473 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 716, 9), ftol_satisfied_249472)
    # Assigning a type to the variable 'if_condition_249473' (line 716)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 9), 'if_condition_249473', if_condition_249473)
    # SSA begins for if statement (line 716)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_249474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 717)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 8), 'stypy_return_type', int_249474)
    # SSA branch for the else part of an if statement (line 716)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'xtol_satisfied' (line 718)
    xtol_satisfied_249475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 9), 'xtol_satisfied')
    # Testing the type of an if condition (line 718)
    if_condition_249476 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 718, 9), xtol_satisfied_249475)
    # Assigning a type to the variable 'if_condition_249476' (line 718)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 9), 'if_condition_249476', if_condition_249476)
    # SSA begins for if statement (line 718)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_249477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 719)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 8), 'stypy_return_type', int_249477)
    # SSA branch for the else part of an if statement (line 718)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'None' (line 721)
    None_249478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 721)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 8), 'stypy_return_type', None_249478)
    # SSA join for if statement (line 718)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 716)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 714)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'check_termination(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_termination' in the type store
    # Getting the type of 'stypy_return_type' (line 709)
    stypy_return_type_249479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_249479)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_termination'
    return stypy_return_type_249479

# Assigning a type to the variable 'check_termination' (line 709)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 0), 'check_termination', check_termination)

@norecursion
def scale_for_robust_loss_function(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'scale_for_robust_loss_function'
    module_type_store = module_type_store.open_function_context('scale_for_robust_loss_function', 724, 0, False)
    
    # Passed parameters checking function
    scale_for_robust_loss_function.stypy_localization = localization
    scale_for_robust_loss_function.stypy_type_of_self = None
    scale_for_robust_loss_function.stypy_type_store = module_type_store
    scale_for_robust_loss_function.stypy_function_name = 'scale_for_robust_loss_function'
    scale_for_robust_loss_function.stypy_param_names_list = ['J', 'f', 'rho']
    scale_for_robust_loss_function.stypy_varargs_param_name = None
    scale_for_robust_loss_function.stypy_kwargs_param_name = None
    scale_for_robust_loss_function.stypy_call_defaults = defaults
    scale_for_robust_loss_function.stypy_call_varargs = varargs
    scale_for_robust_loss_function.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'scale_for_robust_loss_function', ['J', 'f', 'rho'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'scale_for_robust_loss_function', localization, ['J', 'f', 'rho'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'scale_for_robust_loss_function(...)' code ##################

    str_249480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, (-1)), 'str', 'Scale Jacobian and residuals for a robust loss function.\n    \n    Arrays are modified in place.\n    ')
    
    # Assigning a BinOp to a Name (line 729):
    
    # Assigning a BinOp to a Name (line 729):
    
    # Obtaining the type of the subscript
    int_249481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 18), 'int')
    # Getting the type of 'rho' (line 729)
    rho_249482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 14), 'rho')
    # Obtaining the member '__getitem__' of a type (line 729)
    getitem___249483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 14), rho_249482, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 729)
    subscript_call_result_249484 = invoke(stypy.reporting.localization.Localization(__file__, 729, 14), getitem___249483, int_249481)
    
    int_249485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 23), 'int')
    
    # Obtaining the type of the subscript
    int_249486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 31), 'int')
    # Getting the type of 'rho' (line 729)
    rho_249487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 27), 'rho')
    # Obtaining the member '__getitem__' of a type (line 729)
    getitem___249488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 27), rho_249487, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 729)
    subscript_call_result_249489 = invoke(stypy.reporting.localization.Localization(__file__, 729, 27), getitem___249488, int_249486)
    
    # Applying the binary operator '*' (line 729)
    result_mul_249490 = python_operator(stypy.reporting.localization.Localization(__file__, 729, 23), '*', int_249485, subscript_call_result_249489)
    
    # Getting the type of 'f' (line 729)
    f_249491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 36), 'f')
    int_249492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 39), 'int')
    # Applying the binary operator '**' (line 729)
    result_pow_249493 = python_operator(stypy.reporting.localization.Localization(__file__, 729, 36), '**', f_249491, int_249492)
    
    # Applying the binary operator '*' (line 729)
    result_mul_249494 = python_operator(stypy.reporting.localization.Localization(__file__, 729, 34), '*', result_mul_249490, result_pow_249493)
    
    # Applying the binary operator '+' (line 729)
    result_add_249495 = python_operator(stypy.reporting.localization.Localization(__file__, 729, 14), '+', subscript_call_result_249484, result_mul_249494)
    
    # Assigning a type to the variable 'J_scale' (line 729)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 4), 'J_scale', result_add_249495)
    
    # Assigning a Name to a Subscript (line 730):
    
    # Assigning a Name to a Subscript (line 730):
    # Getting the type of 'EPS' (line 730)
    EPS_249496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 29), 'EPS')
    # Getting the type of 'J_scale' (line 730)
    J_scale_249497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 4), 'J_scale')
    
    # Getting the type of 'J_scale' (line 730)
    J_scale_249498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 12), 'J_scale')
    # Getting the type of 'EPS' (line 730)
    EPS_249499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 22), 'EPS')
    # Applying the binary operator '<' (line 730)
    result_lt_249500 = python_operator(stypy.reporting.localization.Localization(__file__, 730, 12), '<', J_scale_249498, EPS_249499)
    
    # Storing an element on a container (line 730)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 730, 4), J_scale_249497, (result_lt_249500, EPS_249496))
    
    # Getting the type of 'J_scale' (line 731)
    J_scale_249501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 4), 'J_scale')
    float_249502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, 16), 'float')
    # Applying the binary operator '**=' (line 731)
    result_ipow_249503 = python_operator(stypy.reporting.localization.Localization(__file__, 731, 4), '**=', J_scale_249501, float_249502)
    # Assigning a type to the variable 'J_scale' (line 731)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 4), 'J_scale', result_ipow_249503)
    
    
    # Getting the type of 'f' (line 733)
    f_249504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 4), 'f')
    
    # Obtaining the type of the subscript
    int_249505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 13), 'int')
    # Getting the type of 'rho' (line 733)
    rho_249506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 9), 'rho')
    # Obtaining the member '__getitem__' of a type (line 733)
    getitem___249507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 733, 9), rho_249506, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 733)
    subscript_call_result_249508 = invoke(stypy.reporting.localization.Localization(__file__, 733, 9), getitem___249507, int_249505)
    
    # Getting the type of 'J_scale' (line 733)
    J_scale_249509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 18), 'J_scale')
    # Applying the binary operator 'div' (line 733)
    result_div_249510 = python_operator(stypy.reporting.localization.Localization(__file__, 733, 9), 'div', subscript_call_result_249508, J_scale_249509)
    
    # Applying the binary operator '*=' (line 733)
    result_imul_249511 = python_operator(stypy.reporting.localization.Localization(__file__, 733, 4), '*=', f_249504, result_div_249510)
    # Assigning a type to the variable 'f' (line 733)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 4), 'f', result_imul_249511)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 735)
    tuple_249512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 735)
    # Adding element type (line 735)
    
    # Call to left_multiply(...): (line 735)
    # Processing the call arguments (line 735)
    # Getting the type of 'J' (line 735)
    J_249514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 25), 'J', False)
    # Getting the type of 'J_scale' (line 735)
    J_scale_249515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 28), 'J_scale', False)
    # Processing the call keyword arguments (line 735)
    # Getting the type of 'False' (line 735)
    False_249516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 42), 'False', False)
    keyword_249517 = False_249516
    kwargs_249518 = {'copy': keyword_249517}
    # Getting the type of 'left_multiply' (line 735)
    left_multiply_249513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 11), 'left_multiply', False)
    # Calling left_multiply(args, kwargs) (line 735)
    left_multiply_call_result_249519 = invoke(stypy.reporting.localization.Localization(__file__, 735, 11), left_multiply_249513, *[J_249514, J_scale_249515], **kwargs_249518)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 735, 11), tuple_249512, left_multiply_call_result_249519)
    # Adding element type (line 735)
    # Getting the type of 'f' (line 735)
    f_249520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 50), 'f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 735, 11), tuple_249512, f_249520)
    
    # Assigning a type to the variable 'stypy_return_type' (line 735)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'stypy_return_type', tuple_249512)
    
    # ################# End of 'scale_for_robust_loss_function(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'scale_for_robust_loss_function' in the type store
    # Getting the type of 'stypy_return_type' (line 724)
    stypy_return_type_249521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_249521)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'scale_for_robust_loss_function'
    return stypy_return_type_249521

# Assigning a type to the variable 'scale_for_robust_loss_function' (line 724)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 0), 'scale_for_robust_loss_function', scale_for_robust_loss_function)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
