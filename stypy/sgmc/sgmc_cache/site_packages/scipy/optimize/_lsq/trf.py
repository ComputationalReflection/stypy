
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Trust Region Reflective algorithm for least-squares optimization.
2: 
3: The algorithm is based on ideas from paper [STIR]_. The main idea is to
4: account for presence of the bounds by appropriate scaling of the variables (or
5: equivalently changing a trust-region shape). Let's introduce a vector v:
6: 
7:            | ub[i] - x[i], if g[i] < 0 and ub[i] < np.inf
8:     v[i] = | x[i] - lb[i], if g[i] > 0 and lb[i] > -np.inf
9:            | 1,           otherwise
10: 
11: where g is the gradient of a cost function and lb, ub are the bounds. Its
12: components are distances to the bounds at which the anti-gradient points (if
13: this distance is finite). Define a scaling matrix D = diag(v**0.5).
14: First-order optimality conditions can be stated as
15: 
16:     D^2 g(x) = 0.
17: 
18: Meaning that components of the gradient should be zero for strictly interior
19: variables, and components must point inside the feasible region for variables
20: on the bound.
21: 
22: Now consider this system of equations as a new optimization problem. If the
23: point x is strictly interior (not on the bound) then the left-hand side is
24: differentiable and the Newton step for it satisfies
25: 
26:     (D^2 H + diag(g) Jv) p = -D^2 g
27: 
28: where H is the Hessian matrix (or its J^T J approximation in least squares),
29: Jv is the Jacobian matrix of v with components -1, 1 or 0, such that all
30: elements of matrix C = diag(g) Jv are non-negative. Introduce the change
31: of the variables x = D x_h (_h would be "hat" in LaTeX). In the new variables
32: we have a Newton step satisfying
33: 
34:     B_h p_h = -g_h,
35: 
36: where B_h = D H D + C, g_h = D g. In least squares B_h = J_h^T J_h, where
37: J_h = J D. Note that J_h and g_h are proper Jacobian and gradient with respect
38: to "hat" variables. To guarantee global convergence we formulate a
39: trust-region problem based on the Newton step in the new variables:
40: 
41:     0.5 * p_h^T B_h p + g_h^T p_h -> min, ||p_h|| <= Delta
42: 
43: In the original space B = H + D^{-1} C D^{-1}, and the equivalent trust-region
44: problem is
45: 
46:     0.5 * p^T B p + g^T p -> min, ||D^{-1} p|| <= Delta
47: 
48: Here the meaning of the matrix D becomes more clear: it alters the shape
49: of a trust-region, such that large steps towards the bounds are not allowed.
50: In the implementation the trust-region problem is solved in "hat" space,
51: but handling of the bounds is done in the original space (see below and read
52: the code).
53: 
54: The introduction of the matrix D doesn't allow to ignore bounds, the algorithm
55: must keep iterates strictly feasible (to satisfy aforementioned
56: differentiability), the parameter theta controls step back from the boundary
57: (see the code for details).
58: 
59: The algorithm does another important trick. If the trust-region solution
60: doesn't fit into the bounds, then a reflected (from a firstly encountered
61: bound) search direction is considered. For motivation and analysis refer to
62: [STIR]_ paper (and other papers of the authors). In practice it doesn't need
63: a lot of justifications, the algorithm simply chooses the best step among
64: three: a constrained trust-region step, a reflected step and a constrained
65: Cauchy step (a minimizer along -g_h in "hat" space, or -D^2 g in the original
66: space).
67: 
68: Another feature is that a trust-region radius control strategy is modified to
69: account for appearance of the diagonal C matrix (called diag_h in the code).
70: 
71: Note, that all described peculiarities are completely gone as we consider
72: problems without bounds (the algorithm becomes a standard trust-region type
73: algorithm very similar to ones implemented in MINPACK).
74: 
75: The implementation supports two methods of solving the trust-region problem.
76: The first, called 'exact', applies SVD on Jacobian and then solves the problem
77: very accurately using the algorithm described in [JJMore]_. It is not
78: applicable to large problem. The second, called 'lsmr', uses the 2-D subspace
79: approach (sometimes called "indefinite dogleg"), where the problem is solved
80: in a subspace spanned by the gradient and the approximate Gauss-Newton step
81: found by ``scipy.sparse.linalg.lsmr``. A 2-D trust-region problem is
82: reformulated as a 4-th order algebraic equation and solved very accurately by
83: ``numpy.roots``. The subspace approach allows to solve very large problems
84: (up to couple of millions of residuals on a regular PC), provided the Jacobian
85: matrix is sufficiently sparse.
86: 
87: References
88: ----------
89: .. [STIR] Branch, M.A., T.F. Coleman, and Y. Li, "A Subspace, Interior,
90:       and Conjugate Gradient Method for Large-Scale Bound-Constrained
91:       Minimization Problems," SIAM Journal on Scientific Computing,
92:       Vol. 21, Number 1, pp 1-23, 1999.
93: .. [JJMore] More, J. J., "The Levenberg-Marquardt Algorithm: Implementation
94:     and Theory," Numerical Analysis, ed. G. A. Watson, Lecture
95: '''
96: from __future__ import division, print_function, absolute_import
97: 
98: import numpy as np
99: from numpy.linalg import norm
100: from scipy.linalg import svd, qr
101: from scipy.sparse.linalg import LinearOperator, lsmr
102: from scipy.optimize import OptimizeResult
103: from scipy._lib.six import string_types
104: 
105: from .common import (
106:     step_size_to_bound, find_active_constraints, in_bounds,
107:     make_strictly_feasible, intersect_trust_region, solve_lsq_trust_region,
108:     solve_trust_region_2d, minimize_quadratic_1d, build_quadratic_1d,
109:     evaluate_quadratic, right_multiplied_operator, regularized_lsq_operator,
110:     CL_scaling_vector, compute_grad, compute_jac_scale, check_termination,
111:     update_tr_radius, scale_for_robust_loss_function, print_header_nonlinear,
112:     print_iteration_nonlinear)
113: 
114: 
115: def trf(fun, jac, x0, f0, J0, lb, ub, ftol, xtol, gtol, max_nfev, x_scale,
116:         loss_function, tr_solver, tr_options, verbose):
117:     # For efficiency it makes sense to run the simplified version of the
118:     # algorithm when no bounds are imposed. We decided to write the two
119:     # separate functions. It violates DRY principle, but the individual
120:     # functions are kept the most readable.
121:     if np.all(lb == -np.inf) and np.all(ub == np.inf):
122:         return trf_no_bounds(
123:             fun, jac, x0, f0, J0, ftol, xtol, gtol, max_nfev, x_scale,
124:             loss_function, tr_solver, tr_options, verbose)
125:     else:
126:         return trf_bounds(
127:             fun, jac, x0, f0, J0, lb, ub, ftol, xtol, gtol, max_nfev, x_scale,
128:             loss_function, tr_solver, tr_options, verbose)
129: 
130: 
131: def select_step(x, J_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta):
132:     '''Select the best step according to Trust Region Reflective algorithm.'''
133:     if in_bounds(x + p, lb, ub):
134:         p_value = evaluate_quadratic(J_h, g_h, p_h, diag=diag_h)
135:         return p, p_h, -p_value
136: 
137:     p_stride, hits = step_size_to_bound(x, p, lb, ub)
138: 
139:     # Compute the reflected direction.
140:     r_h = np.copy(p_h)
141:     r_h[hits.astype(bool)] *= -1
142:     r = d * r_h
143: 
144:     # Restrict trust-region step, such that it hits the bound.
145:     p *= p_stride
146:     p_h *= p_stride
147:     x_on_bound = x + p
148: 
149:     # Reflected direction will cross first either feasible region or trust
150:     # region boundary.
151:     _, to_tr = intersect_trust_region(p_h, r_h, Delta)
152:     to_bound, _ = step_size_to_bound(x_on_bound, r, lb, ub)
153: 
154:     # Find lower and upper bounds on a step size along the reflected
155:     # direction, considering the strict feasibility requirement. There is no
156:     # single correct way to do that, the chosen approach seems to work best
157:     # on test problems.
158:     r_stride = min(to_bound, to_tr)
159:     if r_stride > 0:
160:         r_stride_l = (1 - theta) * p_stride / r_stride
161:         if r_stride == to_bound:
162:             r_stride_u = theta * to_bound
163:         else:
164:             r_stride_u = to_tr
165:     else:
166:         r_stride_l = 0
167:         r_stride_u = -1
168: 
169:     # Check if reflection step is available.
170:     if r_stride_l <= r_stride_u:
171:         a, b, c = build_quadratic_1d(J_h, g_h, r_h, s0=p_h, diag=diag_h)
172:         r_stride, r_value = minimize_quadratic_1d(
173:             a, b, r_stride_l, r_stride_u, c=c)
174:         r_h *= r_stride
175:         r_h += p_h
176:         r = r_h * d
177:     else:
178:         r_value = np.inf
179: 
180:     # Now correct p_h to make it strictly interior.
181:     p *= theta
182:     p_h *= theta
183:     p_value = evaluate_quadratic(J_h, g_h, p_h, diag=diag_h)
184: 
185:     ag_h = -g_h
186:     ag = d * ag_h
187: 
188:     to_tr = Delta / norm(ag_h)
189:     to_bound, _ = step_size_to_bound(x, ag, lb, ub)
190:     if to_bound < to_tr:
191:         ag_stride = theta * to_bound
192:     else:
193:         ag_stride = to_tr
194: 
195:     a, b = build_quadratic_1d(J_h, g_h, ag_h, diag=diag_h)
196:     ag_stride, ag_value = minimize_quadratic_1d(a, b, 0, ag_stride)
197:     ag_h *= ag_stride
198:     ag *= ag_stride
199: 
200:     if p_value < r_value and p_value < ag_value:
201:         return p, p_h, -p_value
202:     elif r_value < p_value and r_value < ag_value:
203:         return r, r_h, -r_value
204:     else:
205:         return ag, ag_h, -ag_value
206: 
207: 
208: def trf_bounds(fun, jac, x0, f0, J0, lb, ub, ftol, xtol, gtol, max_nfev,
209:                x_scale, loss_function, tr_solver, tr_options, verbose):
210:     x = x0.copy()
211: 
212:     f = f0
213:     f_true = f.copy()
214:     nfev = 1
215: 
216:     J = J0
217:     njev = 1
218:     m, n = J.shape
219: 
220:     if loss_function is not None:
221:         rho = loss_function(f)
222:         cost = 0.5 * np.sum(rho[0])
223:         J, f = scale_for_robust_loss_function(J, f, rho)
224:     else:
225:         cost = 0.5 * np.dot(f, f)
226: 
227:     g = compute_grad(J, f)
228: 
229:     jac_scale = isinstance(x_scale, string_types) and x_scale == 'jac'
230:     if jac_scale:
231:         scale, scale_inv = compute_jac_scale(J)
232:     else:
233:         scale, scale_inv = x_scale, 1 / x_scale
234: 
235:     v, dv = CL_scaling_vector(x, g, lb, ub)
236:     v[dv != 0] *= scale_inv[dv != 0]
237:     Delta = norm(x0 * scale_inv / v**0.5)
238:     if Delta == 0:
239:         Delta = 1.0
240: 
241:     g_norm = norm(g * v, ord=np.inf)
242: 
243:     f_augmented = np.zeros((m + n))
244:     if tr_solver == 'exact':
245:         J_augmented = np.empty((m + n, n))
246:     elif tr_solver == 'lsmr':
247:         reg_term = 0.0
248:         regularize = tr_options.pop('regularize', True)
249: 
250:     if max_nfev is None:
251:         max_nfev = x0.size * 100
252: 
253:     alpha = 0.0  # "Levenberg-Marquardt" parameter
254: 
255:     termination_status = None
256:     iteration = 0
257:     step_norm = None
258:     actual_reduction = None
259: 
260:     if verbose == 2:
261:         print_header_nonlinear()
262: 
263:     while True:
264:         v, dv = CL_scaling_vector(x, g, lb, ub)
265: 
266:         g_norm = norm(g * v, ord=np.inf)
267:         if g_norm < gtol:
268:             termination_status = 1
269: 
270:         if verbose == 2:
271:             print_iteration_nonlinear(iteration, nfev, cost, actual_reduction,
272:                                       step_norm, g_norm)
273: 
274:         if termination_status is not None or nfev == max_nfev:
275:             break
276: 
277:         # Now compute variables in "hat" space. Here we also account for
278:         # scaling introduced by `x_scale` parameter. This part is a bit tricky,
279:         # you have to write down the formulas and see how the trust-region
280:         # problem is formulated when the two types of scaling are applied.
281:         # The idea is that first we apply `x_scale` and then apply Coleman-Li
282:         # approach in the new variables.
283: 
284:         # v is recomputed in the variables after applying `x_scale`, note that
285:         # components which were identically 1 not affected.
286:         v[dv != 0] *= scale_inv[dv != 0]
287: 
288:         # Here we apply two types of scaling.
289:         d = v**0.5 * scale
290: 
291:         # C = diag(g * scale) Jv
292:         diag_h = g * dv * scale
293: 
294:         # After all this were done, we continue normally.
295: 
296:         # "hat" gradient.
297:         g_h = d * g
298: 
299:         f_augmented[:m] = f
300:         if tr_solver == 'exact':
301:             J_augmented[:m] = J * d
302:             J_h = J_augmented[:m]  # Memory view.
303:             J_augmented[m:] = np.diag(diag_h**0.5)
304:             U, s, V = svd(J_augmented, full_matrices=False)
305:             V = V.T
306:             uf = U.T.dot(f_augmented)
307:         elif tr_solver == 'lsmr':
308:             J_h = right_multiplied_operator(J, d)
309: 
310:             if regularize:
311:                 a, b = build_quadratic_1d(J_h, g_h, -g_h, diag=diag_h)
312:                 to_tr = Delta / norm(g_h)
313:                 ag_value = minimize_quadratic_1d(a, b, 0, to_tr)[1]
314:                 reg_term = -ag_value / Delta**2
315: 
316:             lsmr_op = regularized_lsq_operator(J_h, (diag_h + reg_term)**0.5)
317:             gn_h = lsmr(lsmr_op, f_augmented, **tr_options)[0]
318:             S = np.vstack((g_h, gn_h)).T
319:             S, _ = qr(S, mode='economic')
320:             JS = J_h.dot(S)  # LinearOperator does dot too.
321:             B_S = np.dot(JS.T, JS) + np.dot(S.T * diag_h, S)
322:             g_S = S.T.dot(g_h)
323: 
324:         # theta controls step back step ratio from the bounds.
325:         theta = max(0.995, 1 - g_norm)
326: 
327:         actual_reduction = -1
328:         while actual_reduction <= 0 and nfev < max_nfev:
329:             if tr_solver == 'exact':
330:                 p_h, alpha, n_iter = solve_lsq_trust_region(
331:                     n, m, uf, s, V, Delta, initial_alpha=alpha)
332:             elif tr_solver == 'lsmr':
333:                 p_S, _ = solve_trust_region_2d(B_S, g_S, Delta)
334:                 p_h = S.dot(p_S)
335: 
336:             p = d * p_h  # Trust-region solution in the original space.
337:             step, step_h, predicted_reduction = select_step(
338:                 x, J_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta)
339: 
340:             x_new = make_strictly_feasible(x + step, lb, ub, rstep=0)
341:             f_new = fun(x_new)
342:             nfev += 1
343: 
344:             step_h_norm = norm(step_h)
345: 
346:             if not np.all(np.isfinite(f_new)):
347:                 Delta = 0.25 * step_h_norm
348:                 continue
349: 
350:             # Usual trust-region step quality estimation.
351:             if loss_function is not None:
352:                 cost_new = loss_function(f_new, cost_only=True)
353:             else:
354:                 cost_new = 0.5 * np.dot(f_new, f_new)
355:             actual_reduction = cost - cost_new
356:             # Correction term is specific to the algorithm,
357:             # vanishes in unbounded case.
358:             correction = 0.5 * np.dot(step_h * diag_h, step_h)
359: 
360:             Delta_new, ratio = update_tr_radius(
361:                 Delta, actual_reduction - correction, predicted_reduction,
362:                 step_h_norm, step_h_norm > 0.95 * Delta
363:             )
364:             alpha *= Delta / Delta_new
365:             Delta = Delta_new
366: 
367:             step_norm = norm(step)
368:             termination_status = check_termination(
369:                 actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol)
370: 
371:             if termination_status is not None:
372:                 break
373: 
374:         if actual_reduction > 0:
375:             x = x_new
376: 
377:             f = f_new
378:             f_true = f.copy()
379: 
380:             cost = cost_new
381: 
382:             J = jac(x, f)
383:             njev += 1
384: 
385:             if loss_function is not None:
386:                 rho = loss_function(f)
387:                 J, f = scale_for_robust_loss_function(J, f, rho)
388: 
389:             g = compute_grad(J, f)
390: 
391:             if jac_scale:
392:                 scale, scale_inv = compute_jac_scale(J, scale_inv)
393:         else:
394:             step_norm = 0
395:             actual_reduction = 0
396: 
397:         iteration += 1
398: 
399:     if termination_status is None:
400:         termination_status = 0
401: 
402:     active_mask = find_active_constraints(x, lb, ub, rtol=xtol)
403:     return OptimizeResult(
404:         x=x, cost=cost, fun=f_true, jac=J, grad=g, optimality=g_norm,
405:         active_mask=active_mask, nfev=nfev, njev=njev,
406:         status=termination_status)
407: 
408: 
409: def trf_no_bounds(fun, jac, x0, f0, J0, ftol, xtol, gtol, max_nfev,
410:                   x_scale, loss_function, tr_solver, tr_options, verbose):
411:     x = x0.copy()
412: 
413:     f = f0
414:     f_true = f.copy()
415:     nfev = 1
416: 
417:     J = J0
418:     njev = 1
419:     m, n = J.shape
420: 
421:     if loss_function is not None:
422:         rho = loss_function(f)
423:         cost = 0.5 * np.sum(rho[0])
424:         J, f = scale_for_robust_loss_function(J, f, rho)
425:     else:
426:         cost = 0.5 * np.dot(f, f)
427: 
428:     g = compute_grad(J, f)
429: 
430:     jac_scale = isinstance(x_scale, string_types) and x_scale == 'jac'
431:     if jac_scale:
432:         scale, scale_inv = compute_jac_scale(J)
433:     else:
434:         scale, scale_inv = x_scale, 1 / x_scale
435: 
436:     Delta = norm(x0 * scale_inv)
437:     if Delta == 0:
438:         Delta = 1.0
439: 
440:     if tr_solver == 'lsmr':
441:         reg_term = 0
442:         damp = tr_options.pop('damp', 0.0)
443:         regularize = tr_options.pop('regularize', True)
444: 
445:     if max_nfev is None:
446:         max_nfev = x0.size * 100
447: 
448:     alpha = 0.0  # "Levenberg-Marquardt" parameter
449: 
450:     termination_status = None
451:     iteration = 0
452:     step_norm = None
453:     actual_reduction = None
454: 
455:     if verbose == 2:
456:         print_header_nonlinear()
457: 
458:     while True:
459:         g_norm = norm(g, ord=np.inf)
460:         if g_norm < gtol:
461:             termination_status = 1
462: 
463:         if verbose == 2:
464:             print_iteration_nonlinear(iteration, nfev, cost, actual_reduction,
465:                                       step_norm, g_norm)
466: 
467:         if termination_status is not None or nfev == max_nfev:
468:             break
469: 
470:         d = scale
471:         g_h = d * g
472: 
473:         if tr_solver == 'exact':
474:             J_h = J * d
475:             U, s, V = svd(J_h, full_matrices=False)
476:             V = V.T
477:             uf = U.T.dot(f)
478:         elif tr_solver == 'lsmr':
479:             J_h = right_multiplied_operator(J, d)
480: 
481:             if regularize:
482:                 a, b = build_quadratic_1d(J_h, g_h, -g_h)
483:                 to_tr = Delta / norm(g_h)
484:                 ag_value = minimize_quadratic_1d(a, b, 0, to_tr)[1]
485:                 reg_term = -ag_value / Delta**2
486: 
487:             damp_full = (damp**2 + reg_term)**0.5
488:             gn_h = lsmr(J_h, f, damp=damp_full, **tr_options)[0]
489:             S = np.vstack((g_h, gn_h)).T
490:             S, _ = qr(S, mode='economic')
491:             JS = J_h.dot(S)
492:             B_S = np.dot(JS.T, JS)
493:             g_S = S.T.dot(g_h)
494: 
495:         actual_reduction = -1
496:         while actual_reduction <= 0 and nfev < max_nfev:
497:             if tr_solver == 'exact':
498:                 step_h, alpha, n_iter = solve_lsq_trust_region(
499:                     n, m, uf, s, V, Delta, initial_alpha=alpha)
500:             elif tr_solver == 'lsmr':
501:                 p_S, _ = solve_trust_region_2d(B_S, g_S, Delta)
502:                 step_h = S.dot(p_S)
503: 
504:             predicted_reduction = -evaluate_quadratic(J_h, g_h, step_h)
505:             step = d * step_h
506:             x_new = x + step
507:             f_new = fun(x_new)
508:             nfev += 1
509: 
510:             step_h_norm = norm(step_h)
511: 
512:             if not np.all(np.isfinite(f_new)):
513:                 Delta = 0.25 * step_h_norm
514:                 continue
515: 
516:             # Usual trust-region step quality estimation.
517:             if loss_function is not None:
518:                 cost_new = loss_function(f_new, cost_only=True)
519:             else:
520:                 cost_new = 0.5 * np.dot(f_new, f_new)
521:             actual_reduction = cost - cost_new
522: 
523:             Delta_new, ratio = update_tr_radius(
524:                 Delta, actual_reduction, predicted_reduction,
525:                 step_h_norm, step_h_norm > 0.95 * Delta)
526:             alpha *= Delta / Delta_new
527:             Delta = Delta_new
528: 
529:             step_norm = norm(step)
530:             termination_status = check_termination(
531:                 actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol)
532: 
533:             if termination_status is not None:
534:                 break
535: 
536:         if actual_reduction > 0:
537:             x = x_new
538: 
539:             f = f_new
540:             f_true = f.copy()
541: 
542:             cost = cost_new
543: 
544:             J = jac(x, f)
545:             njev += 1
546: 
547:             if loss_function is not None:
548:                 rho = loss_function(f)
549:                 J, f = scale_for_robust_loss_function(J, f, rho)
550: 
551:             g = compute_grad(J, f)
552: 
553:             if jac_scale:
554:                 scale, scale_inv = compute_jac_scale(J, scale_inv)
555:         else:
556:             step_norm = 0
557:             actual_reduction = 0
558: 
559:         iteration += 1
560: 
561:     if termination_status is None:
562:         termination_status = 0
563: 
564:     active_mask = np.zeros_like(x)
565:     return OptimizeResult(
566:         x=x, cost=cost, fun=f_true, jac=J, grad=g, optimality=g_norm,
567:         active_mask=active_mask, nfev=nfev, njev=njev,
568:         status=termination_status)
569: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_252661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, (-1)), 'str', 'Trust Region Reflective algorithm for least-squares optimization.\n\nThe algorithm is based on ideas from paper [STIR]_. The main idea is to\naccount for presence of the bounds by appropriate scaling of the variables (or\nequivalently changing a trust-region shape). Let\'s introduce a vector v:\n\n           | ub[i] - x[i], if g[i] < 0 and ub[i] < np.inf\n    v[i] = | x[i] - lb[i], if g[i] > 0 and lb[i] > -np.inf\n           | 1,           otherwise\n\nwhere g is the gradient of a cost function and lb, ub are the bounds. Its\ncomponents are distances to the bounds at which the anti-gradient points (if\nthis distance is finite). Define a scaling matrix D = diag(v**0.5).\nFirst-order optimality conditions can be stated as\n\n    D^2 g(x) = 0.\n\nMeaning that components of the gradient should be zero for strictly interior\nvariables, and components must point inside the feasible region for variables\non the bound.\n\nNow consider this system of equations as a new optimization problem. If the\npoint x is strictly interior (not on the bound) then the left-hand side is\ndifferentiable and the Newton step for it satisfies\n\n    (D^2 H + diag(g) Jv) p = -D^2 g\n\nwhere H is the Hessian matrix (or its J^T J approximation in least squares),\nJv is the Jacobian matrix of v with components -1, 1 or 0, such that all\nelements of matrix C = diag(g) Jv are non-negative. Introduce the change\nof the variables x = D x_h (_h would be "hat" in LaTeX). In the new variables\nwe have a Newton step satisfying\n\n    B_h p_h = -g_h,\n\nwhere B_h = D H D + C, g_h = D g. In least squares B_h = J_h^T J_h, where\nJ_h = J D. Note that J_h and g_h are proper Jacobian and gradient with respect\nto "hat" variables. To guarantee global convergence we formulate a\ntrust-region problem based on the Newton step in the new variables:\n\n    0.5 * p_h^T B_h p + g_h^T p_h -> min, ||p_h|| <= Delta\n\nIn the original space B = H + D^{-1} C D^{-1}, and the equivalent trust-region\nproblem is\n\n    0.5 * p^T B p + g^T p -> min, ||D^{-1} p|| <= Delta\n\nHere the meaning of the matrix D becomes more clear: it alters the shape\nof a trust-region, such that large steps towards the bounds are not allowed.\nIn the implementation the trust-region problem is solved in "hat" space,\nbut handling of the bounds is done in the original space (see below and read\nthe code).\n\nThe introduction of the matrix D doesn\'t allow to ignore bounds, the algorithm\nmust keep iterates strictly feasible (to satisfy aforementioned\ndifferentiability), the parameter theta controls step back from the boundary\n(see the code for details).\n\nThe algorithm does another important trick. If the trust-region solution\ndoesn\'t fit into the bounds, then a reflected (from a firstly encountered\nbound) search direction is considered. For motivation and analysis refer to\n[STIR]_ paper (and other papers of the authors). In practice it doesn\'t need\na lot of justifications, the algorithm simply chooses the best step among\nthree: a constrained trust-region step, a reflected step and a constrained\nCauchy step (a minimizer along -g_h in "hat" space, or -D^2 g in the original\nspace).\n\nAnother feature is that a trust-region radius control strategy is modified to\naccount for appearance of the diagonal C matrix (called diag_h in the code).\n\nNote, that all described peculiarities are completely gone as we consider\nproblems without bounds (the algorithm becomes a standard trust-region type\nalgorithm very similar to ones implemented in MINPACK).\n\nThe implementation supports two methods of solving the trust-region problem.\nThe first, called \'exact\', applies SVD on Jacobian and then solves the problem\nvery accurately using the algorithm described in [JJMore]_. It is not\napplicable to large problem. The second, called \'lsmr\', uses the 2-D subspace\napproach (sometimes called "indefinite dogleg"), where the problem is solved\nin a subspace spanned by the gradient and the approximate Gauss-Newton step\nfound by ``scipy.sparse.linalg.lsmr``. A 2-D trust-region problem is\nreformulated as a 4-th order algebraic equation and solved very accurately by\n``numpy.roots``. The subspace approach allows to solve very large problems\n(up to couple of millions of residuals on a regular PC), provided the Jacobian\nmatrix is sufficiently sparse.\n\nReferences\n----------\n.. [STIR] Branch, M.A., T.F. Coleman, and Y. Li, "A Subspace, Interior,\n      and Conjugate Gradient Method for Large-Scale Bound-Constrained\n      Minimization Problems," SIAM Journal on Scientific Computing,\n      Vol. 21, Number 1, pp 1-23, 1999.\n.. [JJMore] More, J. J., "The Levenberg-Marquardt Algorithm: Implementation\n    and Theory," Numerical Analysis, ed. G. A. Watson, Lecture\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 98, 0))

# 'import numpy' statement (line 98)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_252662 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 98, 0), 'numpy')

if (type(import_252662) is not StypyTypeError):

    if (import_252662 != 'pyd_module'):
        __import__(import_252662)
        sys_modules_252663 = sys.modules[import_252662]
        import_module(stypy.reporting.localization.Localization(__file__, 98, 0), 'np', sys_modules_252663.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 98, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'numpy', import_252662)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 99, 0))

# 'from numpy.linalg import norm' statement (line 99)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_252664 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 99, 0), 'numpy.linalg')

if (type(import_252664) is not StypyTypeError):

    if (import_252664 != 'pyd_module'):
        __import__(import_252664)
        sys_modules_252665 = sys.modules[import_252664]
        import_from_module(stypy.reporting.localization.Localization(__file__, 99, 0), 'numpy.linalg', sys_modules_252665.module_type_store, module_type_store, ['norm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 99, 0), __file__, sys_modules_252665, sys_modules_252665.module_type_store, module_type_store)
    else:
        from numpy.linalg import norm

        import_from_module(stypy.reporting.localization.Localization(__file__, 99, 0), 'numpy.linalg', None, module_type_store, ['norm'], [norm])

else:
    # Assigning a type to the variable 'numpy.linalg' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 0), 'numpy.linalg', import_252664)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 100, 0))

# 'from scipy.linalg import svd, qr' statement (line 100)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_252666 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 100, 0), 'scipy.linalg')

if (type(import_252666) is not StypyTypeError):

    if (import_252666 != 'pyd_module'):
        __import__(import_252666)
        sys_modules_252667 = sys.modules[import_252666]
        import_from_module(stypy.reporting.localization.Localization(__file__, 100, 0), 'scipy.linalg', sys_modules_252667.module_type_store, module_type_store, ['svd', 'qr'])
        nest_module(stypy.reporting.localization.Localization(__file__, 100, 0), __file__, sys_modules_252667, sys_modules_252667.module_type_store, module_type_store)
    else:
        from scipy.linalg import svd, qr

        import_from_module(stypy.reporting.localization.Localization(__file__, 100, 0), 'scipy.linalg', None, module_type_store, ['svd', 'qr'], [svd, qr])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'scipy.linalg', import_252666)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 101, 0))

# 'from scipy.sparse.linalg import LinearOperator, lsmr' statement (line 101)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_252668 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 101, 0), 'scipy.sparse.linalg')

if (type(import_252668) is not StypyTypeError):

    if (import_252668 != 'pyd_module'):
        __import__(import_252668)
        sys_modules_252669 = sys.modules[import_252668]
        import_from_module(stypy.reporting.localization.Localization(__file__, 101, 0), 'scipy.sparse.linalg', sys_modules_252669.module_type_store, module_type_store, ['LinearOperator', 'lsmr'])
        nest_module(stypy.reporting.localization.Localization(__file__, 101, 0), __file__, sys_modules_252669, sys_modules_252669.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import LinearOperator, lsmr

        import_from_module(stypy.reporting.localization.Localization(__file__, 101, 0), 'scipy.sparse.linalg', None, module_type_store, ['LinearOperator', 'lsmr'], [LinearOperator, lsmr])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), 'scipy.sparse.linalg', import_252668)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 102, 0))

# 'from scipy.optimize import OptimizeResult' statement (line 102)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_252670 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 102, 0), 'scipy.optimize')

if (type(import_252670) is not StypyTypeError):

    if (import_252670 != 'pyd_module'):
        __import__(import_252670)
        sys_modules_252671 = sys.modules[import_252670]
        import_from_module(stypy.reporting.localization.Localization(__file__, 102, 0), 'scipy.optimize', sys_modules_252671.module_type_store, module_type_store, ['OptimizeResult'])
        nest_module(stypy.reporting.localization.Localization(__file__, 102, 0), __file__, sys_modules_252671, sys_modules_252671.module_type_store, module_type_store)
    else:
        from scipy.optimize import OptimizeResult

        import_from_module(stypy.reporting.localization.Localization(__file__, 102, 0), 'scipy.optimize', None, module_type_store, ['OptimizeResult'], [OptimizeResult])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'scipy.optimize', import_252670)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 103, 0))

# 'from scipy._lib.six import string_types' statement (line 103)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_252672 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 103, 0), 'scipy._lib.six')

if (type(import_252672) is not StypyTypeError):

    if (import_252672 != 'pyd_module'):
        __import__(import_252672)
        sys_modules_252673 = sys.modules[import_252672]
        import_from_module(stypy.reporting.localization.Localization(__file__, 103, 0), 'scipy._lib.six', sys_modules_252673.module_type_store, module_type_store, ['string_types'])
        nest_module(stypy.reporting.localization.Localization(__file__, 103, 0), __file__, sys_modules_252673, sys_modules_252673.module_type_store, module_type_store)
    else:
        from scipy._lib.six import string_types

        import_from_module(stypy.reporting.localization.Localization(__file__, 103, 0), 'scipy._lib.six', None, module_type_store, ['string_types'], [string_types])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'scipy._lib.six', import_252672)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 105, 0))

# 'from scipy.optimize._lsq.common import step_size_to_bound, find_active_constraints, in_bounds, make_strictly_feasible, intersect_trust_region, solve_lsq_trust_region, solve_trust_region_2d, minimize_quadratic_1d, build_quadratic_1d, evaluate_quadratic, right_multiplied_operator, regularized_lsq_operator, CL_scaling_vector, compute_grad, compute_jac_scale, check_termination, update_tr_radius, scale_for_robust_loss_function, print_header_nonlinear, print_iteration_nonlinear' statement (line 105)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_252674 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 105, 0), 'scipy.optimize._lsq.common')

if (type(import_252674) is not StypyTypeError):

    if (import_252674 != 'pyd_module'):
        __import__(import_252674)
        sys_modules_252675 = sys.modules[import_252674]
        import_from_module(stypy.reporting.localization.Localization(__file__, 105, 0), 'scipy.optimize._lsq.common', sys_modules_252675.module_type_store, module_type_store, ['step_size_to_bound', 'find_active_constraints', 'in_bounds', 'make_strictly_feasible', 'intersect_trust_region', 'solve_lsq_trust_region', 'solve_trust_region_2d', 'minimize_quadratic_1d', 'build_quadratic_1d', 'evaluate_quadratic', 'right_multiplied_operator', 'regularized_lsq_operator', 'CL_scaling_vector', 'compute_grad', 'compute_jac_scale', 'check_termination', 'update_tr_radius', 'scale_for_robust_loss_function', 'print_header_nonlinear', 'print_iteration_nonlinear'])
        nest_module(stypy.reporting.localization.Localization(__file__, 105, 0), __file__, sys_modules_252675, sys_modules_252675.module_type_store, module_type_store)
    else:
        from scipy.optimize._lsq.common import step_size_to_bound, find_active_constraints, in_bounds, make_strictly_feasible, intersect_trust_region, solve_lsq_trust_region, solve_trust_region_2d, minimize_quadratic_1d, build_quadratic_1d, evaluate_quadratic, right_multiplied_operator, regularized_lsq_operator, CL_scaling_vector, compute_grad, compute_jac_scale, check_termination, update_tr_radius, scale_for_robust_loss_function, print_header_nonlinear, print_iteration_nonlinear

        import_from_module(stypy.reporting.localization.Localization(__file__, 105, 0), 'scipy.optimize._lsq.common', None, module_type_store, ['step_size_to_bound', 'find_active_constraints', 'in_bounds', 'make_strictly_feasible', 'intersect_trust_region', 'solve_lsq_trust_region', 'solve_trust_region_2d', 'minimize_quadratic_1d', 'build_quadratic_1d', 'evaluate_quadratic', 'right_multiplied_operator', 'regularized_lsq_operator', 'CL_scaling_vector', 'compute_grad', 'compute_jac_scale', 'check_termination', 'update_tr_radius', 'scale_for_robust_loss_function', 'print_header_nonlinear', 'print_iteration_nonlinear'], [step_size_to_bound, find_active_constraints, in_bounds, make_strictly_feasible, intersect_trust_region, solve_lsq_trust_region, solve_trust_region_2d, minimize_quadratic_1d, build_quadratic_1d, evaluate_quadratic, right_multiplied_operator, regularized_lsq_operator, CL_scaling_vector, compute_grad, compute_jac_scale, check_termination, update_tr_radius, scale_for_robust_loss_function, print_header_nonlinear, print_iteration_nonlinear])

else:
    # Assigning a type to the variable 'scipy.optimize._lsq.common' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'scipy.optimize._lsq.common', import_252674)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')


@norecursion
def trf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'trf'
    module_type_store = module_type_store.open_function_context('trf', 115, 0, False)
    
    # Passed parameters checking function
    trf.stypy_localization = localization
    trf.stypy_type_of_self = None
    trf.stypy_type_store = module_type_store
    trf.stypy_function_name = 'trf'
    trf.stypy_param_names_list = ['fun', 'jac', 'x0', 'f0', 'J0', 'lb', 'ub', 'ftol', 'xtol', 'gtol', 'max_nfev', 'x_scale', 'loss_function', 'tr_solver', 'tr_options', 'verbose']
    trf.stypy_varargs_param_name = None
    trf.stypy_kwargs_param_name = None
    trf.stypy_call_defaults = defaults
    trf.stypy_call_varargs = varargs
    trf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'trf', ['fun', 'jac', 'x0', 'f0', 'J0', 'lb', 'ub', 'ftol', 'xtol', 'gtol', 'max_nfev', 'x_scale', 'loss_function', 'tr_solver', 'tr_options', 'verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'trf', localization, ['fun', 'jac', 'x0', 'f0', 'J0', 'lb', 'ub', 'ftol', 'xtol', 'gtol', 'max_nfev', 'x_scale', 'loss_function', 'tr_solver', 'tr_options', 'verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'trf(...)' code ##################

    
    
    # Evaluating a boolean operation
    
    # Call to all(...): (line 121)
    # Processing the call arguments (line 121)
    
    # Getting the type of 'lb' (line 121)
    lb_252678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 14), 'lb', False)
    
    # Getting the type of 'np' (line 121)
    np_252679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 21), 'np', False)
    # Obtaining the member 'inf' of a type (line 121)
    inf_252680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 21), np_252679, 'inf')
    # Applying the 'usub' unary operator (line 121)
    result___neg___252681 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 20), 'usub', inf_252680)
    
    # Applying the binary operator '==' (line 121)
    result_eq_252682 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 14), '==', lb_252678, result___neg___252681)
    
    # Processing the call keyword arguments (line 121)
    kwargs_252683 = {}
    # Getting the type of 'np' (line 121)
    np_252676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 7), 'np', False)
    # Obtaining the member 'all' of a type (line 121)
    all_252677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 7), np_252676, 'all')
    # Calling all(args, kwargs) (line 121)
    all_call_result_252684 = invoke(stypy.reporting.localization.Localization(__file__, 121, 7), all_252677, *[result_eq_252682], **kwargs_252683)
    
    
    # Call to all(...): (line 121)
    # Processing the call arguments (line 121)
    
    # Getting the type of 'ub' (line 121)
    ub_252687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 40), 'ub', False)
    # Getting the type of 'np' (line 121)
    np_252688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 46), 'np', False)
    # Obtaining the member 'inf' of a type (line 121)
    inf_252689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 46), np_252688, 'inf')
    # Applying the binary operator '==' (line 121)
    result_eq_252690 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 40), '==', ub_252687, inf_252689)
    
    # Processing the call keyword arguments (line 121)
    kwargs_252691 = {}
    # Getting the type of 'np' (line 121)
    np_252685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 33), 'np', False)
    # Obtaining the member 'all' of a type (line 121)
    all_252686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 33), np_252685, 'all')
    # Calling all(args, kwargs) (line 121)
    all_call_result_252692 = invoke(stypy.reporting.localization.Localization(__file__, 121, 33), all_252686, *[result_eq_252690], **kwargs_252691)
    
    # Applying the binary operator 'and' (line 121)
    result_and_keyword_252693 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 7), 'and', all_call_result_252684, all_call_result_252692)
    
    # Testing the type of an if condition (line 121)
    if_condition_252694 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 4), result_and_keyword_252693)
    # Assigning a type to the variable 'if_condition_252694' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'if_condition_252694', if_condition_252694)
    # SSA begins for if statement (line 121)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to trf_no_bounds(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'fun' (line 123)
    fun_252696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'fun', False)
    # Getting the type of 'jac' (line 123)
    jac_252697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 17), 'jac', False)
    # Getting the type of 'x0' (line 123)
    x0_252698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 22), 'x0', False)
    # Getting the type of 'f0' (line 123)
    f0_252699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 26), 'f0', False)
    # Getting the type of 'J0' (line 123)
    J0_252700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 30), 'J0', False)
    # Getting the type of 'ftol' (line 123)
    ftol_252701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 34), 'ftol', False)
    # Getting the type of 'xtol' (line 123)
    xtol_252702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 40), 'xtol', False)
    # Getting the type of 'gtol' (line 123)
    gtol_252703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 46), 'gtol', False)
    # Getting the type of 'max_nfev' (line 123)
    max_nfev_252704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 52), 'max_nfev', False)
    # Getting the type of 'x_scale' (line 123)
    x_scale_252705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 62), 'x_scale', False)
    # Getting the type of 'loss_function' (line 124)
    loss_function_252706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'loss_function', False)
    # Getting the type of 'tr_solver' (line 124)
    tr_solver_252707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 27), 'tr_solver', False)
    # Getting the type of 'tr_options' (line 124)
    tr_options_252708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 38), 'tr_options', False)
    # Getting the type of 'verbose' (line 124)
    verbose_252709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 50), 'verbose', False)
    # Processing the call keyword arguments (line 122)
    kwargs_252710 = {}
    # Getting the type of 'trf_no_bounds' (line 122)
    trf_no_bounds_252695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 15), 'trf_no_bounds', False)
    # Calling trf_no_bounds(args, kwargs) (line 122)
    trf_no_bounds_call_result_252711 = invoke(stypy.reporting.localization.Localization(__file__, 122, 15), trf_no_bounds_252695, *[fun_252696, jac_252697, x0_252698, f0_252699, J0_252700, ftol_252701, xtol_252702, gtol_252703, max_nfev_252704, x_scale_252705, loss_function_252706, tr_solver_252707, tr_options_252708, verbose_252709], **kwargs_252710)
    
    # Assigning a type to the variable 'stypy_return_type' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'stypy_return_type', trf_no_bounds_call_result_252711)
    # SSA branch for the else part of an if statement (line 121)
    module_type_store.open_ssa_branch('else')
    
    # Call to trf_bounds(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'fun' (line 127)
    fun_252713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'fun', False)
    # Getting the type of 'jac' (line 127)
    jac_252714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 17), 'jac', False)
    # Getting the type of 'x0' (line 127)
    x0_252715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 22), 'x0', False)
    # Getting the type of 'f0' (line 127)
    f0_252716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 26), 'f0', False)
    # Getting the type of 'J0' (line 127)
    J0_252717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 30), 'J0', False)
    # Getting the type of 'lb' (line 127)
    lb_252718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 34), 'lb', False)
    # Getting the type of 'ub' (line 127)
    ub_252719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 38), 'ub', False)
    # Getting the type of 'ftol' (line 127)
    ftol_252720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 42), 'ftol', False)
    # Getting the type of 'xtol' (line 127)
    xtol_252721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 48), 'xtol', False)
    # Getting the type of 'gtol' (line 127)
    gtol_252722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 54), 'gtol', False)
    # Getting the type of 'max_nfev' (line 127)
    max_nfev_252723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 60), 'max_nfev', False)
    # Getting the type of 'x_scale' (line 127)
    x_scale_252724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 70), 'x_scale', False)
    # Getting the type of 'loss_function' (line 128)
    loss_function_252725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'loss_function', False)
    # Getting the type of 'tr_solver' (line 128)
    tr_solver_252726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 27), 'tr_solver', False)
    # Getting the type of 'tr_options' (line 128)
    tr_options_252727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 38), 'tr_options', False)
    # Getting the type of 'verbose' (line 128)
    verbose_252728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 50), 'verbose', False)
    # Processing the call keyword arguments (line 126)
    kwargs_252729 = {}
    # Getting the type of 'trf_bounds' (line 126)
    trf_bounds_252712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 15), 'trf_bounds', False)
    # Calling trf_bounds(args, kwargs) (line 126)
    trf_bounds_call_result_252730 = invoke(stypy.reporting.localization.Localization(__file__, 126, 15), trf_bounds_252712, *[fun_252713, jac_252714, x0_252715, f0_252716, J0_252717, lb_252718, ub_252719, ftol_252720, xtol_252721, gtol_252722, max_nfev_252723, x_scale_252724, loss_function_252725, tr_solver_252726, tr_options_252727, verbose_252728], **kwargs_252729)
    
    # Assigning a type to the variable 'stypy_return_type' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'stypy_return_type', trf_bounds_call_result_252730)
    # SSA join for if statement (line 121)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'trf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'trf' in the type store
    # Getting the type of 'stypy_return_type' (line 115)
    stypy_return_type_252731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_252731)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'trf'
    return stypy_return_type_252731

# Assigning a type to the variable 'trf' (line 115)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 0), 'trf', trf)

@norecursion
def select_step(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'select_step'
    module_type_store = module_type_store.open_function_context('select_step', 131, 0, False)
    
    # Passed parameters checking function
    select_step.stypy_localization = localization
    select_step.stypy_type_of_self = None
    select_step.stypy_type_store = module_type_store
    select_step.stypy_function_name = 'select_step'
    select_step.stypy_param_names_list = ['x', 'J_h', 'diag_h', 'g_h', 'p', 'p_h', 'd', 'Delta', 'lb', 'ub', 'theta']
    select_step.stypy_varargs_param_name = None
    select_step.stypy_kwargs_param_name = None
    select_step.stypy_call_defaults = defaults
    select_step.stypy_call_varargs = varargs
    select_step.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'select_step', ['x', 'J_h', 'diag_h', 'g_h', 'p', 'p_h', 'd', 'Delta', 'lb', 'ub', 'theta'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'select_step', localization, ['x', 'J_h', 'diag_h', 'g_h', 'p', 'p_h', 'd', 'Delta', 'lb', 'ub', 'theta'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'select_step(...)' code ##################

    str_252732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 4), 'str', 'Select the best step according to Trust Region Reflective algorithm.')
    
    
    # Call to in_bounds(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'x' (line 133)
    x_252734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 17), 'x', False)
    # Getting the type of 'p' (line 133)
    p_252735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 21), 'p', False)
    # Applying the binary operator '+' (line 133)
    result_add_252736 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 17), '+', x_252734, p_252735)
    
    # Getting the type of 'lb' (line 133)
    lb_252737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 24), 'lb', False)
    # Getting the type of 'ub' (line 133)
    ub_252738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 28), 'ub', False)
    # Processing the call keyword arguments (line 133)
    kwargs_252739 = {}
    # Getting the type of 'in_bounds' (line 133)
    in_bounds_252733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 7), 'in_bounds', False)
    # Calling in_bounds(args, kwargs) (line 133)
    in_bounds_call_result_252740 = invoke(stypy.reporting.localization.Localization(__file__, 133, 7), in_bounds_252733, *[result_add_252736, lb_252737, ub_252738], **kwargs_252739)
    
    # Testing the type of an if condition (line 133)
    if_condition_252741 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 4), in_bounds_call_result_252740)
    # Assigning a type to the variable 'if_condition_252741' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'if_condition_252741', if_condition_252741)
    # SSA begins for if statement (line 133)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 134):
    
    # Assigning a Call to a Name (line 134):
    
    # Call to evaluate_quadratic(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'J_h' (line 134)
    J_h_252743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 37), 'J_h', False)
    # Getting the type of 'g_h' (line 134)
    g_h_252744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 42), 'g_h', False)
    # Getting the type of 'p_h' (line 134)
    p_h_252745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 47), 'p_h', False)
    # Processing the call keyword arguments (line 134)
    # Getting the type of 'diag_h' (line 134)
    diag_h_252746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 57), 'diag_h', False)
    keyword_252747 = diag_h_252746
    kwargs_252748 = {'diag': keyword_252747}
    # Getting the type of 'evaluate_quadratic' (line 134)
    evaluate_quadratic_252742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 18), 'evaluate_quadratic', False)
    # Calling evaluate_quadratic(args, kwargs) (line 134)
    evaluate_quadratic_call_result_252749 = invoke(stypy.reporting.localization.Localization(__file__, 134, 18), evaluate_quadratic_252742, *[J_h_252743, g_h_252744, p_h_252745], **kwargs_252748)
    
    # Assigning a type to the variable 'p_value' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'p_value', evaluate_quadratic_call_result_252749)
    
    # Obtaining an instance of the builtin type 'tuple' (line 135)
    tuple_252750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 135)
    # Adding element type (line 135)
    # Getting the type of 'p' (line 135)
    p_252751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 15), 'p')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 15), tuple_252750, p_252751)
    # Adding element type (line 135)
    # Getting the type of 'p_h' (line 135)
    p_h_252752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 18), 'p_h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 15), tuple_252750, p_h_252752)
    # Adding element type (line 135)
    
    # Getting the type of 'p_value' (line 135)
    p_value_252753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 24), 'p_value')
    # Applying the 'usub' unary operator (line 135)
    result___neg___252754 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 23), 'usub', p_value_252753)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 15), tuple_252750, result___neg___252754)
    
    # Assigning a type to the variable 'stypy_return_type' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'stypy_return_type', tuple_252750)
    # SSA join for if statement (line 133)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 137):
    
    # Assigning a Subscript to a Name (line 137):
    
    # Obtaining the type of the subscript
    int_252755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 4), 'int')
    
    # Call to step_size_to_bound(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'x' (line 137)
    x_252757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 40), 'x', False)
    # Getting the type of 'p' (line 137)
    p_252758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 43), 'p', False)
    # Getting the type of 'lb' (line 137)
    lb_252759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 46), 'lb', False)
    # Getting the type of 'ub' (line 137)
    ub_252760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 50), 'ub', False)
    # Processing the call keyword arguments (line 137)
    kwargs_252761 = {}
    # Getting the type of 'step_size_to_bound' (line 137)
    step_size_to_bound_252756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 21), 'step_size_to_bound', False)
    # Calling step_size_to_bound(args, kwargs) (line 137)
    step_size_to_bound_call_result_252762 = invoke(stypy.reporting.localization.Localization(__file__, 137, 21), step_size_to_bound_252756, *[x_252757, p_252758, lb_252759, ub_252760], **kwargs_252761)
    
    # Obtaining the member '__getitem__' of a type (line 137)
    getitem___252763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 4), step_size_to_bound_call_result_252762, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 137)
    subscript_call_result_252764 = invoke(stypy.reporting.localization.Localization(__file__, 137, 4), getitem___252763, int_252755)
    
    # Assigning a type to the variable 'tuple_var_assignment_252585' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'tuple_var_assignment_252585', subscript_call_result_252764)
    
    # Assigning a Subscript to a Name (line 137):
    
    # Obtaining the type of the subscript
    int_252765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 4), 'int')
    
    # Call to step_size_to_bound(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'x' (line 137)
    x_252767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 40), 'x', False)
    # Getting the type of 'p' (line 137)
    p_252768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 43), 'p', False)
    # Getting the type of 'lb' (line 137)
    lb_252769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 46), 'lb', False)
    # Getting the type of 'ub' (line 137)
    ub_252770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 50), 'ub', False)
    # Processing the call keyword arguments (line 137)
    kwargs_252771 = {}
    # Getting the type of 'step_size_to_bound' (line 137)
    step_size_to_bound_252766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 21), 'step_size_to_bound', False)
    # Calling step_size_to_bound(args, kwargs) (line 137)
    step_size_to_bound_call_result_252772 = invoke(stypy.reporting.localization.Localization(__file__, 137, 21), step_size_to_bound_252766, *[x_252767, p_252768, lb_252769, ub_252770], **kwargs_252771)
    
    # Obtaining the member '__getitem__' of a type (line 137)
    getitem___252773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 4), step_size_to_bound_call_result_252772, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 137)
    subscript_call_result_252774 = invoke(stypy.reporting.localization.Localization(__file__, 137, 4), getitem___252773, int_252765)
    
    # Assigning a type to the variable 'tuple_var_assignment_252586' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'tuple_var_assignment_252586', subscript_call_result_252774)
    
    # Assigning a Name to a Name (line 137):
    # Getting the type of 'tuple_var_assignment_252585' (line 137)
    tuple_var_assignment_252585_252775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'tuple_var_assignment_252585')
    # Assigning a type to the variable 'p_stride' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'p_stride', tuple_var_assignment_252585_252775)
    
    # Assigning a Name to a Name (line 137):
    # Getting the type of 'tuple_var_assignment_252586' (line 137)
    tuple_var_assignment_252586_252776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'tuple_var_assignment_252586')
    # Assigning a type to the variable 'hits' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 14), 'hits', tuple_var_assignment_252586_252776)
    
    # Assigning a Call to a Name (line 140):
    
    # Assigning a Call to a Name (line 140):
    
    # Call to copy(...): (line 140)
    # Processing the call arguments (line 140)
    # Getting the type of 'p_h' (line 140)
    p_h_252779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 18), 'p_h', False)
    # Processing the call keyword arguments (line 140)
    kwargs_252780 = {}
    # Getting the type of 'np' (line 140)
    np_252777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 10), 'np', False)
    # Obtaining the member 'copy' of a type (line 140)
    copy_252778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 10), np_252777, 'copy')
    # Calling copy(args, kwargs) (line 140)
    copy_call_result_252781 = invoke(stypy.reporting.localization.Localization(__file__, 140, 10), copy_252778, *[p_h_252779], **kwargs_252780)
    
    # Assigning a type to the variable 'r_h' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'r_h', copy_call_result_252781)
    
    # Getting the type of 'r_h' (line 141)
    r_h_252782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'r_h')
    
    # Obtaining the type of the subscript
    
    # Call to astype(...): (line 141)
    # Processing the call arguments (line 141)
    # Getting the type of 'bool' (line 141)
    bool_252785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 20), 'bool', False)
    # Processing the call keyword arguments (line 141)
    kwargs_252786 = {}
    # Getting the type of 'hits' (line 141)
    hits_252783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'hits', False)
    # Obtaining the member 'astype' of a type (line 141)
    astype_252784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), hits_252783, 'astype')
    # Calling astype(args, kwargs) (line 141)
    astype_call_result_252787 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), astype_252784, *[bool_252785], **kwargs_252786)
    
    # Getting the type of 'r_h' (line 141)
    r_h_252788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'r_h')
    # Obtaining the member '__getitem__' of a type (line 141)
    getitem___252789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 4), r_h_252788, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 141)
    subscript_call_result_252790 = invoke(stypy.reporting.localization.Localization(__file__, 141, 4), getitem___252789, astype_call_result_252787)
    
    int_252791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 30), 'int')
    # Applying the binary operator '*=' (line 141)
    result_imul_252792 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 4), '*=', subscript_call_result_252790, int_252791)
    # Getting the type of 'r_h' (line 141)
    r_h_252793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'r_h')
    
    # Call to astype(...): (line 141)
    # Processing the call arguments (line 141)
    # Getting the type of 'bool' (line 141)
    bool_252796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 20), 'bool', False)
    # Processing the call keyword arguments (line 141)
    kwargs_252797 = {}
    # Getting the type of 'hits' (line 141)
    hits_252794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'hits', False)
    # Obtaining the member 'astype' of a type (line 141)
    astype_252795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), hits_252794, 'astype')
    # Calling astype(args, kwargs) (line 141)
    astype_call_result_252798 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), astype_252795, *[bool_252796], **kwargs_252797)
    
    # Storing an element on a container (line 141)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 4), r_h_252793, (astype_call_result_252798, result_imul_252792))
    
    
    # Assigning a BinOp to a Name (line 142):
    
    # Assigning a BinOp to a Name (line 142):
    # Getting the type of 'd' (line 142)
    d_252799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'd')
    # Getting the type of 'r_h' (line 142)
    r_h_252800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'r_h')
    # Applying the binary operator '*' (line 142)
    result_mul_252801 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 8), '*', d_252799, r_h_252800)
    
    # Assigning a type to the variable 'r' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'r', result_mul_252801)
    
    # Getting the type of 'p' (line 145)
    p_252802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'p')
    # Getting the type of 'p_stride' (line 145)
    p_stride_252803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 9), 'p_stride')
    # Applying the binary operator '*=' (line 145)
    result_imul_252804 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 4), '*=', p_252802, p_stride_252803)
    # Assigning a type to the variable 'p' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'p', result_imul_252804)
    
    
    # Getting the type of 'p_h' (line 146)
    p_h_252805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'p_h')
    # Getting the type of 'p_stride' (line 146)
    p_stride_252806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 11), 'p_stride')
    # Applying the binary operator '*=' (line 146)
    result_imul_252807 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 4), '*=', p_h_252805, p_stride_252806)
    # Assigning a type to the variable 'p_h' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'p_h', result_imul_252807)
    
    
    # Assigning a BinOp to a Name (line 147):
    
    # Assigning a BinOp to a Name (line 147):
    # Getting the type of 'x' (line 147)
    x_252808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 17), 'x')
    # Getting the type of 'p' (line 147)
    p_252809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 21), 'p')
    # Applying the binary operator '+' (line 147)
    result_add_252810 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 17), '+', x_252808, p_252809)
    
    # Assigning a type to the variable 'x_on_bound' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'x_on_bound', result_add_252810)
    
    # Assigning a Call to a Tuple (line 151):
    
    # Assigning a Subscript to a Name (line 151):
    
    # Obtaining the type of the subscript
    int_252811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 4), 'int')
    
    # Call to intersect_trust_region(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'p_h' (line 151)
    p_h_252813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 38), 'p_h', False)
    # Getting the type of 'r_h' (line 151)
    r_h_252814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 43), 'r_h', False)
    # Getting the type of 'Delta' (line 151)
    Delta_252815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 48), 'Delta', False)
    # Processing the call keyword arguments (line 151)
    kwargs_252816 = {}
    # Getting the type of 'intersect_trust_region' (line 151)
    intersect_trust_region_252812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 15), 'intersect_trust_region', False)
    # Calling intersect_trust_region(args, kwargs) (line 151)
    intersect_trust_region_call_result_252817 = invoke(stypy.reporting.localization.Localization(__file__, 151, 15), intersect_trust_region_252812, *[p_h_252813, r_h_252814, Delta_252815], **kwargs_252816)
    
    # Obtaining the member '__getitem__' of a type (line 151)
    getitem___252818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 4), intersect_trust_region_call_result_252817, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 151)
    subscript_call_result_252819 = invoke(stypy.reporting.localization.Localization(__file__, 151, 4), getitem___252818, int_252811)
    
    # Assigning a type to the variable 'tuple_var_assignment_252587' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'tuple_var_assignment_252587', subscript_call_result_252819)
    
    # Assigning a Subscript to a Name (line 151):
    
    # Obtaining the type of the subscript
    int_252820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 4), 'int')
    
    # Call to intersect_trust_region(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'p_h' (line 151)
    p_h_252822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 38), 'p_h', False)
    # Getting the type of 'r_h' (line 151)
    r_h_252823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 43), 'r_h', False)
    # Getting the type of 'Delta' (line 151)
    Delta_252824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 48), 'Delta', False)
    # Processing the call keyword arguments (line 151)
    kwargs_252825 = {}
    # Getting the type of 'intersect_trust_region' (line 151)
    intersect_trust_region_252821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 15), 'intersect_trust_region', False)
    # Calling intersect_trust_region(args, kwargs) (line 151)
    intersect_trust_region_call_result_252826 = invoke(stypy.reporting.localization.Localization(__file__, 151, 15), intersect_trust_region_252821, *[p_h_252822, r_h_252823, Delta_252824], **kwargs_252825)
    
    # Obtaining the member '__getitem__' of a type (line 151)
    getitem___252827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 4), intersect_trust_region_call_result_252826, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 151)
    subscript_call_result_252828 = invoke(stypy.reporting.localization.Localization(__file__, 151, 4), getitem___252827, int_252820)
    
    # Assigning a type to the variable 'tuple_var_assignment_252588' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'tuple_var_assignment_252588', subscript_call_result_252828)
    
    # Assigning a Name to a Name (line 151):
    # Getting the type of 'tuple_var_assignment_252587' (line 151)
    tuple_var_assignment_252587_252829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'tuple_var_assignment_252587')
    # Assigning a type to the variable '_' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), '_', tuple_var_assignment_252587_252829)
    
    # Assigning a Name to a Name (line 151):
    # Getting the type of 'tuple_var_assignment_252588' (line 151)
    tuple_var_assignment_252588_252830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'tuple_var_assignment_252588')
    # Assigning a type to the variable 'to_tr' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 7), 'to_tr', tuple_var_assignment_252588_252830)
    
    # Assigning a Call to a Tuple (line 152):
    
    # Assigning a Subscript to a Name (line 152):
    
    # Obtaining the type of the subscript
    int_252831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 4), 'int')
    
    # Call to step_size_to_bound(...): (line 152)
    # Processing the call arguments (line 152)
    # Getting the type of 'x_on_bound' (line 152)
    x_on_bound_252833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 37), 'x_on_bound', False)
    # Getting the type of 'r' (line 152)
    r_252834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 49), 'r', False)
    # Getting the type of 'lb' (line 152)
    lb_252835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 52), 'lb', False)
    # Getting the type of 'ub' (line 152)
    ub_252836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 56), 'ub', False)
    # Processing the call keyword arguments (line 152)
    kwargs_252837 = {}
    # Getting the type of 'step_size_to_bound' (line 152)
    step_size_to_bound_252832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 18), 'step_size_to_bound', False)
    # Calling step_size_to_bound(args, kwargs) (line 152)
    step_size_to_bound_call_result_252838 = invoke(stypy.reporting.localization.Localization(__file__, 152, 18), step_size_to_bound_252832, *[x_on_bound_252833, r_252834, lb_252835, ub_252836], **kwargs_252837)
    
    # Obtaining the member '__getitem__' of a type (line 152)
    getitem___252839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 4), step_size_to_bound_call_result_252838, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 152)
    subscript_call_result_252840 = invoke(stypy.reporting.localization.Localization(__file__, 152, 4), getitem___252839, int_252831)
    
    # Assigning a type to the variable 'tuple_var_assignment_252589' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'tuple_var_assignment_252589', subscript_call_result_252840)
    
    # Assigning a Subscript to a Name (line 152):
    
    # Obtaining the type of the subscript
    int_252841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 4), 'int')
    
    # Call to step_size_to_bound(...): (line 152)
    # Processing the call arguments (line 152)
    # Getting the type of 'x_on_bound' (line 152)
    x_on_bound_252843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 37), 'x_on_bound', False)
    # Getting the type of 'r' (line 152)
    r_252844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 49), 'r', False)
    # Getting the type of 'lb' (line 152)
    lb_252845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 52), 'lb', False)
    # Getting the type of 'ub' (line 152)
    ub_252846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 56), 'ub', False)
    # Processing the call keyword arguments (line 152)
    kwargs_252847 = {}
    # Getting the type of 'step_size_to_bound' (line 152)
    step_size_to_bound_252842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 18), 'step_size_to_bound', False)
    # Calling step_size_to_bound(args, kwargs) (line 152)
    step_size_to_bound_call_result_252848 = invoke(stypy.reporting.localization.Localization(__file__, 152, 18), step_size_to_bound_252842, *[x_on_bound_252843, r_252844, lb_252845, ub_252846], **kwargs_252847)
    
    # Obtaining the member '__getitem__' of a type (line 152)
    getitem___252849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 4), step_size_to_bound_call_result_252848, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 152)
    subscript_call_result_252850 = invoke(stypy.reporting.localization.Localization(__file__, 152, 4), getitem___252849, int_252841)
    
    # Assigning a type to the variable 'tuple_var_assignment_252590' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'tuple_var_assignment_252590', subscript_call_result_252850)
    
    # Assigning a Name to a Name (line 152):
    # Getting the type of 'tuple_var_assignment_252589' (line 152)
    tuple_var_assignment_252589_252851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'tuple_var_assignment_252589')
    # Assigning a type to the variable 'to_bound' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'to_bound', tuple_var_assignment_252589_252851)
    
    # Assigning a Name to a Name (line 152):
    # Getting the type of 'tuple_var_assignment_252590' (line 152)
    tuple_var_assignment_252590_252852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'tuple_var_assignment_252590')
    # Assigning a type to the variable '_' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 14), '_', tuple_var_assignment_252590_252852)
    
    # Assigning a Call to a Name (line 158):
    
    # Assigning a Call to a Name (line 158):
    
    # Call to min(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'to_bound' (line 158)
    to_bound_252854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 19), 'to_bound', False)
    # Getting the type of 'to_tr' (line 158)
    to_tr_252855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 29), 'to_tr', False)
    # Processing the call keyword arguments (line 158)
    kwargs_252856 = {}
    # Getting the type of 'min' (line 158)
    min_252853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 15), 'min', False)
    # Calling min(args, kwargs) (line 158)
    min_call_result_252857 = invoke(stypy.reporting.localization.Localization(__file__, 158, 15), min_252853, *[to_bound_252854, to_tr_252855], **kwargs_252856)
    
    # Assigning a type to the variable 'r_stride' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'r_stride', min_call_result_252857)
    
    
    # Getting the type of 'r_stride' (line 159)
    r_stride_252858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 7), 'r_stride')
    int_252859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 18), 'int')
    # Applying the binary operator '>' (line 159)
    result_gt_252860 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 7), '>', r_stride_252858, int_252859)
    
    # Testing the type of an if condition (line 159)
    if_condition_252861 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 4), result_gt_252860)
    # Assigning a type to the variable 'if_condition_252861' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'if_condition_252861', if_condition_252861)
    # SSA begins for if statement (line 159)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 160):
    
    # Assigning a BinOp to a Name (line 160):
    int_252862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 22), 'int')
    # Getting the type of 'theta' (line 160)
    theta_252863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 26), 'theta')
    # Applying the binary operator '-' (line 160)
    result_sub_252864 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 22), '-', int_252862, theta_252863)
    
    # Getting the type of 'p_stride' (line 160)
    p_stride_252865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 35), 'p_stride')
    # Applying the binary operator '*' (line 160)
    result_mul_252866 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 21), '*', result_sub_252864, p_stride_252865)
    
    # Getting the type of 'r_stride' (line 160)
    r_stride_252867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 46), 'r_stride')
    # Applying the binary operator 'div' (line 160)
    result_div_252868 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 44), 'div', result_mul_252866, r_stride_252867)
    
    # Assigning a type to the variable 'r_stride_l' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'r_stride_l', result_div_252868)
    
    
    # Getting the type of 'r_stride' (line 161)
    r_stride_252869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 11), 'r_stride')
    # Getting the type of 'to_bound' (line 161)
    to_bound_252870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 23), 'to_bound')
    # Applying the binary operator '==' (line 161)
    result_eq_252871 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 11), '==', r_stride_252869, to_bound_252870)
    
    # Testing the type of an if condition (line 161)
    if_condition_252872 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 8), result_eq_252871)
    # Assigning a type to the variable 'if_condition_252872' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'if_condition_252872', if_condition_252872)
    # SSA begins for if statement (line 161)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 162):
    
    # Assigning a BinOp to a Name (line 162):
    # Getting the type of 'theta' (line 162)
    theta_252873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 25), 'theta')
    # Getting the type of 'to_bound' (line 162)
    to_bound_252874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 33), 'to_bound')
    # Applying the binary operator '*' (line 162)
    result_mul_252875 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 25), '*', theta_252873, to_bound_252874)
    
    # Assigning a type to the variable 'r_stride_u' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'r_stride_u', result_mul_252875)
    # SSA branch for the else part of an if statement (line 161)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 164):
    
    # Assigning a Name to a Name (line 164):
    # Getting the type of 'to_tr' (line 164)
    to_tr_252876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 25), 'to_tr')
    # Assigning a type to the variable 'r_stride_u' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'r_stride_u', to_tr_252876)
    # SSA join for if statement (line 161)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 159)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 166):
    
    # Assigning a Num to a Name (line 166):
    int_252877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 21), 'int')
    # Assigning a type to the variable 'r_stride_l' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'r_stride_l', int_252877)
    
    # Assigning a Num to a Name (line 167):
    
    # Assigning a Num to a Name (line 167):
    int_252878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 21), 'int')
    # Assigning a type to the variable 'r_stride_u' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'r_stride_u', int_252878)
    # SSA join for if statement (line 159)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'r_stride_l' (line 170)
    r_stride_l_252879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 7), 'r_stride_l')
    # Getting the type of 'r_stride_u' (line 170)
    r_stride_u_252880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 21), 'r_stride_u')
    # Applying the binary operator '<=' (line 170)
    result_le_252881 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 7), '<=', r_stride_l_252879, r_stride_u_252880)
    
    # Testing the type of an if condition (line 170)
    if_condition_252882 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 4), result_le_252881)
    # Assigning a type to the variable 'if_condition_252882' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'if_condition_252882', if_condition_252882)
    # SSA begins for if statement (line 170)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 171):
    
    # Assigning a Subscript to a Name (line 171):
    
    # Obtaining the type of the subscript
    int_252883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 8), 'int')
    
    # Call to build_quadratic_1d(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'J_h' (line 171)
    J_h_252885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 37), 'J_h', False)
    # Getting the type of 'g_h' (line 171)
    g_h_252886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 42), 'g_h', False)
    # Getting the type of 'r_h' (line 171)
    r_h_252887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 47), 'r_h', False)
    # Processing the call keyword arguments (line 171)
    # Getting the type of 'p_h' (line 171)
    p_h_252888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 55), 'p_h', False)
    keyword_252889 = p_h_252888
    # Getting the type of 'diag_h' (line 171)
    diag_h_252890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 65), 'diag_h', False)
    keyword_252891 = diag_h_252890
    kwargs_252892 = {'diag': keyword_252891, 's0': keyword_252889}
    # Getting the type of 'build_quadratic_1d' (line 171)
    build_quadratic_1d_252884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 18), 'build_quadratic_1d', False)
    # Calling build_quadratic_1d(args, kwargs) (line 171)
    build_quadratic_1d_call_result_252893 = invoke(stypy.reporting.localization.Localization(__file__, 171, 18), build_quadratic_1d_252884, *[J_h_252885, g_h_252886, r_h_252887], **kwargs_252892)
    
    # Obtaining the member '__getitem__' of a type (line 171)
    getitem___252894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), build_quadratic_1d_call_result_252893, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 171)
    subscript_call_result_252895 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), getitem___252894, int_252883)
    
    # Assigning a type to the variable 'tuple_var_assignment_252591' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'tuple_var_assignment_252591', subscript_call_result_252895)
    
    # Assigning a Subscript to a Name (line 171):
    
    # Obtaining the type of the subscript
    int_252896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 8), 'int')
    
    # Call to build_quadratic_1d(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'J_h' (line 171)
    J_h_252898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 37), 'J_h', False)
    # Getting the type of 'g_h' (line 171)
    g_h_252899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 42), 'g_h', False)
    # Getting the type of 'r_h' (line 171)
    r_h_252900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 47), 'r_h', False)
    # Processing the call keyword arguments (line 171)
    # Getting the type of 'p_h' (line 171)
    p_h_252901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 55), 'p_h', False)
    keyword_252902 = p_h_252901
    # Getting the type of 'diag_h' (line 171)
    diag_h_252903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 65), 'diag_h', False)
    keyword_252904 = diag_h_252903
    kwargs_252905 = {'diag': keyword_252904, 's0': keyword_252902}
    # Getting the type of 'build_quadratic_1d' (line 171)
    build_quadratic_1d_252897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 18), 'build_quadratic_1d', False)
    # Calling build_quadratic_1d(args, kwargs) (line 171)
    build_quadratic_1d_call_result_252906 = invoke(stypy.reporting.localization.Localization(__file__, 171, 18), build_quadratic_1d_252897, *[J_h_252898, g_h_252899, r_h_252900], **kwargs_252905)
    
    # Obtaining the member '__getitem__' of a type (line 171)
    getitem___252907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), build_quadratic_1d_call_result_252906, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 171)
    subscript_call_result_252908 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), getitem___252907, int_252896)
    
    # Assigning a type to the variable 'tuple_var_assignment_252592' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'tuple_var_assignment_252592', subscript_call_result_252908)
    
    # Assigning a Subscript to a Name (line 171):
    
    # Obtaining the type of the subscript
    int_252909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 8), 'int')
    
    # Call to build_quadratic_1d(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'J_h' (line 171)
    J_h_252911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 37), 'J_h', False)
    # Getting the type of 'g_h' (line 171)
    g_h_252912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 42), 'g_h', False)
    # Getting the type of 'r_h' (line 171)
    r_h_252913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 47), 'r_h', False)
    # Processing the call keyword arguments (line 171)
    # Getting the type of 'p_h' (line 171)
    p_h_252914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 55), 'p_h', False)
    keyword_252915 = p_h_252914
    # Getting the type of 'diag_h' (line 171)
    diag_h_252916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 65), 'diag_h', False)
    keyword_252917 = diag_h_252916
    kwargs_252918 = {'diag': keyword_252917, 's0': keyword_252915}
    # Getting the type of 'build_quadratic_1d' (line 171)
    build_quadratic_1d_252910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 18), 'build_quadratic_1d', False)
    # Calling build_quadratic_1d(args, kwargs) (line 171)
    build_quadratic_1d_call_result_252919 = invoke(stypy.reporting.localization.Localization(__file__, 171, 18), build_quadratic_1d_252910, *[J_h_252911, g_h_252912, r_h_252913], **kwargs_252918)
    
    # Obtaining the member '__getitem__' of a type (line 171)
    getitem___252920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), build_quadratic_1d_call_result_252919, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 171)
    subscript_call_result_252921 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), getitem___252920, int_252909)
    
    # Assigning a type to the variable 'tuple_var_assignment_252593' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'tuple_var_assignment_252593', subscript_call_result_252921)
    
    # Assigning a Name to a Name (line 171):
    # Getting the type of 'tuple_var_assignment_252591' (line 171)
    tuple_var_assignment_252591_252922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'tuple_var_assignment_252591')
    # Assigning a type to the variable 'a' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'a', tuple_var_assignment_252591_252922)
    
    # Assigning a Name to a Name (line 171):
    # Getting the type of 'tuple_var_assignment_252592' (line 171)
    tuple_var_assignment_252592_252923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'tuple_var_assignment_252592')
    # Assigning a type to the variable 'b' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 11), 'b', tuple_var_assignment_252592_252923)
    
    # Assigning a Name to a Name (line 171):
    # Getting the type of 'tuple_var_assignment_252593' (line 171)
    tuple_var_assignment_252593_252924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'tuple_var_assignment_252593')
    # Assigning a type to the variable 'c' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 14), 'c', tuple_var_assignment_252593_252924)
    
    # Assigning a Call to a Tuple (line 172):
    
    # Assigning a Subscript to a Name (line 172):
    
    # Obtaining the type of the subscript
    int_252925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 8), 'int')
    
    # Call to minimize_quadratic_1d(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'a' (line 173)
    a_252927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'a', False)
    # Getting the type of 'b' (line 173)
    b_252928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 'b', False)
    # Getting the type of 'r_stride_l' (line 173)
    r_stride_l_252929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 18), 'r_stride_l', False)
    # Getting the type of 'r_stride_u' (line 173)
    r_stride_u_252930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 30), 'r_stride_u', False)
    # Processing the call keyword arguments (line 172)
    # Getting the type of 'c' (line 173)
    c_252931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 44), 'c', False)
    keyword_252932 = c_252931
    kwargs_252933 = {'c': keyword_252932}
    # Getting the type of 'minimize_quadratic_1d' (line 172)
    minimize_quadratic_1d_252926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 28), 'minimize_quadratic_1d', False)
    # Calling minimize_quadratic_1d(args, kwargs) (line 172)
    minimize_quadratic_1d_call_result_252934 = invoke(stypy.reporting.localization.Localization(__file__, 172, 28), minimize_quadratic_1d_252926, *[a_252927, b_252928, r_stride_l_252929, r_stride_u_252930], **kwargs_252933)
    
    # Obtaining the member '__getitem__' of a type (line 172)
    getitem___252935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), minimize_quadratic_1d_call_result_252934, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 172)
    subscript_call_result_252936 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), getitem___252935, int_252925)
    
    # Assigning a type to the variable 'tuple_var_assignment_252594' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'tuple_var_assignment_252594', subscript_call_result_252936)
    
    # Assigning a Subscript to a Name (line 172):
    
    # Obtaining the type of the subscript
    int_252937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 8), 'int')
    
    # Call to minimize_quadratic_1d(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'a' (line 173)
    a_252939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'a', False)
    # Getting the type of 'b' (line 173)
    b_252940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 'b', False)
    # Getting the type of 'r_stride_l' (line 173)
    r_stride_l_252941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 18), 'r_stride_l', False)
    # Getting the type of 'r_stride_u' (line 173)
    r_stride_u_252942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 30), 'r_stride_u', False)
    # Processing the call keyword arguments (line 172)
    # Getting the type of 'c' (line 173)
    c_252943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 44), 'c', False)
    keyword_252944 = c_252943
    kwargs_252945 = {'c': keyword_252944}
    # Getting the type of 'minimize_quadratic_1d' (line 172)
    minimize_quadratic_1d_252938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 28), 'minimize_quadratic_1d', False)
    # Calling minimize_quadratic_1d(args, kwargs) (line 172)
    minimize_quadratic_1d_call_result_252946 = invoke(stypy.reporting.localization.Localization(__file__, 172, 28), minimize_quadratic_1d_252938, *[a_252939, b_252940, r_stride_l_252941, r_stride_u_252942], **kwargs_252945)
    
    # Obtaining the member '__getitem__' of a type (line 172)
    getitem___252947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), minimize_quadratic_1d_call_result_252946, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 172)
    subscript_call_result_252948 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), getitem___252947, int_252937)
    
    # Assigning a type to the variable 'tuple_var_assignment_252595' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'tuple_var_assignment_252595', subscript_call_result_252948)
    
    # Assigning a Name to a Name (line 172):
    # Getting the type of 'tuple_var_assignment_252594' (line 172)
    tuple_var_assignment_252594_252949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'tuple_var_assignment_252594')
    # Assigning a type to the variable 'r_stride' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'r_stride', tuple_var_assignment_252594_252949)
    
    # Assigning a Name to a Name (line 172):
    # Getting the type of 'tuple_var_assignment_252595' (line 172)
    tuple_var_assignment_252595_252950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'tuple_var_assignment_252595')
    # Assigning a type to the variable 'r_value' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 18), 'r_value', tuple_var_assignment_252595_252950)
    
    # Getting the type of 'r_h' (line 174)
    r_h_252951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'r_h')
    # Getting the type of 'r_stride' (line 174)
    r_stride_252952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 15), 'r_stride')
    # Applying the binary operator '*=' (line 174)
    result_imul_252953 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 8), '*=', r_h_252951, r_stride_252952)
    # Assigning a type to the variable 'r_h' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'r_h', result_imul_252953)
    
    
    # Getting the type of 'r_h' (line 175)
    r_h_252954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'r_h')
    # Getting the type of 'p_h' (line 175)
    p_h_252955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 15), 'p_h')
    # Applying the binary operator '+=' (line 175)
    result_iadd_252956 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 8), '+=', r_h_252954, p_h_252955)
    # Assigning a type to the variable 'r_h' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'r_h', result_iadd_252956)
    
    
    # Assigning a BinOp to a Name (line 176):
    
    # Assigning a BinOp to a Name (line 176):
    # Getting the type of 'r_h' (line 176)
    r_h_252957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'r_h')
    # Getting the type of 'd' (line 176)
    d_252958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 18), 'd')
    # Applying the binary operator '*' (line 176)
    result_mul_252959 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 12), '*', r_h_252957, d_252958)
    
    # Assigning a type to the variable 'r' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'r', result_mul_252959)
    # SSA branch for the else part of an if statement (line 170)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Attribute to a Name (line 178):
    
    # Assigning a Attribute to a Name (line 178):
    # Getting the type of 'np' (line 178)
    np_252960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 18), 'np')
    # Obtaining the member 'inf' of a type (line 178)
    inf_252961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 18), np_252960, 'inf')
    # Assigning a type to the variable 'r_value' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'r_value', inf_252961)
    # SSA join for if statement (line 170)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'p' (line 181)
    p_252962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'p')
    # Getting the type of 'theta' (line 181)
    theta_252963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 9), 'theta')
    # Applying the binary operator '*=' (line 181)
    result_imul_252964 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 4), '*=', p_252962, theta_252963)
    # Assigning a type to the variable 'p' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'p', result_imul_252964)
    
    
    # Getting the type of 'p_h' (line 182)
    p_h_252965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'p_h')
    # Getting the type of 'theta' (line 182)
    theta_252966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 11), 'theta')
    # Applying the binary operator '*=' (line 182)
    result_imul_252967 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 4), '*=', p_h_252965, theta_252966)
    # Assigning a type to the variable 'p_h' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'p_h', result_imul_252967)
    
    
    # Assigning a Call to a Name (line 183):
    
    # Assigning a Call to a Name (line 183):
    
    # Call to evaluate_quadratic(...): (line 183)
    # Processing the call arguments (line 183)
    # Getting the type of 'J_h' (line 183)
    J_h_252969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 33), 'J_h', False)
    # Getting the type of 'g_h' (line 183)
    g_h_252970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 38), 'g_h', False)
    # Getting the type of 'p_h' (line 183)
    p_h_252971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 43), 'p_h', False)
    # Processing the call keyword arguments (line 183)
    # Getting the type of 'diag_h' (line 183)
    diag_h_252972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 53), 'diag_h', False)
    keyword_252973 = diag_h_252972
    kwargs_252974 = {'diag': keyword_252973}
    # Getting the type of 'evaluate_quadratic' (line 183)
    evaluate_quadratic_252968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 14), 'evaluate_quadratic', False)
    # Calling evaluate_quadratic(args, kwargs) (line 183)
    evaluate_quadratic_call_result_252975 = invoke(stypy.reporting.localization.Localization(__file__, 183, 14), evaluate_quadratic_252968, *[J_h_252969, g_h_252970, p_h_252971], **kwargs_252974)
    
    # Assigning a type to the variable 'p_value' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'p_value', evaluate_quadratic_call_result_252975)
    
    # Assigning a UnaryOp to a Name (line 185):
    
    # Assigning a UnaryOp to a Name (line 185):
    
    # Getting the type of 'g_h' (line 185)
    g_h_252976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'g_h')
    # Applying the 'usub' unary operator (line 185)
    result___neg___252977 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 11), 'usub', g_h_252976)
    
    # Assigning a type to the variable 'ag_h' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'ag_h', result___neg___252977)
    
    # Assigning a BinOp to a Name (line 186):
    
    # Assigning a BinOp to a Name (line 186):
    # Getting the type of 'd' (line 186)
    d_252978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 9), 'd')
    # Getting the type of 'ag_h' (line 186)
    ag_h_252979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 13), 'ag_h')
    # Applying the binary operator '*' (line 186)
    result_mul_252980 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 9), '*', d_252978, ag_h_252979)
    
    # Assigning a type to the variable 'ag' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'ag', result_mul_252980)
    
    # Assigning a BinOp to a Name (line 188):
    
    # Assigning a BinOp to a Name (line 188):
    # Getting the type of 'Delta' (line 188)
    Delta_252981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'Delta')
    
    # Call to norm(...): (line 188)
    # Processing the call arguments (line 188)
    # Getting the type of 'ag_h' (line 188)
    ag_h_252983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 25), 'ag_h', False)
    # Processing the call keyword arguments (line 188)
    kwargs_252984 = {}
    # Getting the type of 'norm' (line 188)
    norm_252982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 20), 'norm', False)
    # Calling norm(args, kwargs) (line 188)
    norm_call_result_252985 = invoke(stypy.reporting.localization.Localization(__file__, 188, 20), norm_252982, *[ag_h_252983], **kwargs_252984)
    
    # Applying the binary operator 'div' (line 188)
    result_div_252986 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 12), 'div', Delta_252981, norm_call_result_252985)
    
    # Assigning a type to the variable 'to_tr' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'to_tr', result_div_252986)
    
    # Assigning a Call to a Tuple (line 189):
    
    # Assigning a Subscript to a Name (line 189):
    
    # Obtaining the type of the subscript
    int_252987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 4), 'int')
    
    # Call to step_size_to_bound(...): (line 189)
    # Processing the call arguments (line 189)
    # Getting the type of 'x' (line 189)
    x_252989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 37), 'x', False)
    # Getting the type of 'ag' (line 189)
    ag_252990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 40), 'ag', False)
    # Getting the type of 'lb' (line 189)
    lb_252991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 44), 'lb', False)
    # Getting the type of 'ub' (line 189)
    ub_252992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 48), 'ub', False)
    # Processing the call keyword arguments (line 189)
    kwargs_252993 = {}
    # Getting the type of 'step_size_to_bound' (line 189)
    step_size_to_bound_252988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 18), 'step_size_to_bound', False)
    # Calling step_size_to_bound(args, kwargs) (line 189)
    step_size_to_bound_call_result_252994 = invoke(stypy.reporting.localization.Localization(__file__, 189, 18), step_size_to_bound_252988, *[x_252989, ag_252990, lb_252991, ub_252992], **kwargs_252993)
    
    # Obtaining the member '__getitem__' of a type (line 189)
    getitem___252995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 4), step_size_to_bound_call_result_252994, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 189)
    subscript_call_result_252996 = invoke(stypy.reporting.localization.Localization(__file__, 189, 4), getitem___252995, int_252987)
    
    # Assigning a type to the variable 'tuple_var_assignment_252596' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'tuple_var_assignment_252596', subscript_call_result_252996)
    
    # Assigning a Subscript to a Name (line 189):
    
    # Obtaining the type of the subscript
    int_252997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 4), 'int')
    
    # Call to step_size_to_bound(...): (line 189)
    # Processing the call arguments (line 189)
    # Getting the type of 'x' (line 189)
    x_252999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 37), 'x', False)
    # Getting the type of 'ag' (line 189)
    ag_253000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 40), 'ag', False)
    # Getting the type of 'lb' (line 189)
    lb_253001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 44), 'lb', False)
    # Getting the type of 'ub' (line 189)
    ub_253002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 48), 'ub', False)
    # Processing the call keyword arguments (line 189)
    kwargs_253003 = {}
    # Getting the type of 'step_size_to_bound' (line 189)
    step_size_to_bound_252998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 18), 'step_size_to_bound', False)
    # Calling step_size_to_bound(args, kwargs) (line 189)
    step_size_to_bound_call_result_253004 = invoke(stypy.reporting.localization.Localization(__file__, 189, 18), step_size_to_bound_252998, *[x_252999, ag_253000, lb_253001, ub_253002], **kwargs_253003)
    
    # Obtaining the member '__getitem__' of a type (line 189)
    getitem___253005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 4), step_size_to_bound_call_result_253004, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 189)
    subscript_call_result_253006 = invoke(stypy.reporting.localization.Localization(__file__, 189, 4), getitem___253005, int_252997)
    
    # Assigning a type to the variable 'tuple_var_assignment_252597' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'tuple_var_assignment_252597', subscript_call_result_253006)
    
    # Assigning a Name to a Name (line 189):
    # Getting the type of 'tuple_var_assignment_252596' (line 189)
    tuple_var_assignment_252596_253007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'tuple_var_assignment_252596')
    # Assigning a type to the variable 'to_bound' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'to_bound', tuple_var_assignment_252596_253007)
    
    # Assigning a Name to a Name (line 189):
    # Getting the type of 'tuple_var_assignment_252597' (line 189)
    tuple_var_assignment_252597_253008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'tuple_var_assignment_252597')
    # Assigning a type to the variable '_' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 14), '_', tuple_var_assignment_252597_253008)
    
    
    # Getting the type of 'to_bound' (line 190)
    to_bound_253009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 7), 'to_bound')
    # Getting the type of 'to_tr' (line 190)
    to_tr_253010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 18), 'to_tr')
    # Applying the binary operator '<' (line 190)
    result_lt_253011 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 7), '<', to_bound_253009, to_tr_253010)
    
    # Testing the type of an if condition (line 190)
    if_condition_253012 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 190, 4), result_lt_253011)
    # Assigning a type to the variable 'if_condition_253012' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'if_condition_253012', if_condition_253012)
    # SSA begins for if statement (line 190)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 191):
    
    # Assigning a BinOp to a Name (line 191):
    # Getting the type of 'theta' (line 191)
    theta_253013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 20), 'theta')
    # Getting the type of 'to_bound' (line 191)
    to_bound_253014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 28), 'to_bound')
    # Applying the binary operator '*' (line 191)
    result_mul_253015 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 20), '*', theta_253013, to_bound_253014)
    
    # Assigning a type to the variable 'ag_stride' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'ag_stride', result_mul_253015)
    # SSA branch for the else part of an if statement (line 190)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 193):
    
    # Assigning a Name to a Name (line 193):
    # Getting the type of 'to_tr' (line 193)
    to_tr_253016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 20), 'to_tr')
    # Assigning a type to the variable 'ag_stride' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'ag_stride', to_tr_253016)
    # SSA join for if statement (line 190)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 195):
    
    # Assigning a Subscript to a Name (line 195):
    
    # Obtaining the type of the subscript
    int_253017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 4), 'int')
    
    # Call to build_quadratic_1d(...): (line 195)
    # Processing the call arguments (line 195)
    # Getting the type of 'J_h' (line 195)
    J_h_253019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 30), 'J_h', False)
    # Getting the type of 'g_h' (line 195)
    g_h_253020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 35), 'g_h', False)
    # Getting the type of 'ag_h' (line 195)
    ag_h_253021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 40), 'ag_h', False)
    # Processing the call keyword arguments (line 195)
    # Getting the type of 'diag_h' (line 195)
    diag_h_253022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 51), 'diag_h', False)
    keyword_253023 = diag_h_253022
    kwargs_253024 = {'diag': keyword_253023}
    # Getting the type of 'build_quadratic_1d' (line 195)
    build_quadratic_1d_253018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 11), 'build_quadratic_1d', False)
    # Calling build_quadratic_1d(args, kwargs) (line 195)
    build_quadratic_1d_call_result_253025 = invoke(stypy.reporting.localization.Localization(__file__, 195, 11), build_quadratic_1d_253018, *[J_h_253019, g_h_253020, ag_h_253021], **kwargs_253024)
    
    # Obtaining the member '__getitem__' of a type (line 195)
    getitem___253026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 4), build_quadratic_1d_call_result_253025, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 195)
    subscript_call_result_253027 = invoke(stypy.reporting.localization.Localization(__file__, 195, 4), getitem___253026, int_253017)
    
    # Assigning a type to the variable 'tuple_var_assignment_252598' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'tuple_var_assignment_252598', subscript_call_result_253027)
    
    # Assigning a Subscript to a Name (line 195):
    
    # Obtaining the type of the subscript
    int_253028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 4), 'int')
    
    # Call to build_quadratic_1d(...): (line 195)
    # Processing the call arguments (line 195)
    # Getting the type of 'J_h' (line 195)
    J_h_253030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 30), 'J_h', False)
    # Getting the type of 'g_h' (line 195)
    g_h_253031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 35), 'g_h', False)
    # Getting the type of 'ag_h' (line 195)
    ag_h_253032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 40), 'ag_h', False)
    # Processing the call keyword arguments (line 195)
    # Getting the type of 'diag_h' (line 195)
    diag_h_253033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 51), 'diag_h', False)
    keyword_253034 = diag_h_253033
    kwargs_253035 = {'diag': keyword_253034}
    # Getting the type of 'build_quadratic_1d' (line 195)
    build_quadratic_1d_253029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 11), 'build_quadratic_1d', False)
    # Calling build_quadratic_1d(args, kwargs) (line 195)
    build_quadratic_1d_call_result_253036 = invoke(stypy.reporting.localization.Localization(__file__, 195, 11), build_quadratic_1d_253029, *[J_h_253030, g_h_253031, ag_h_253032], **kwargs_253035)
    
    # Obtaining the member '__getitem__' of a type (line 195)
    getitem___253037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 4), build_quadratic_1d_call_result_253036, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 195)
    subscript_call_result_253038 = invoke(stypy.reporting.localization.Localization(__file__, 195, 4), getitem___253037, int_253028)
    
    # Assigning a type to the variable 'tuple_var_assignment_252599' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'tuple_var_assignment_252599', subscript_call_result_253038)
    
    # Assigning a Name to a Name (line 195):
    # Getting the type of 'tuple_var_assignment_252598' (line 195)
    tuple_var_assignment_252598_253039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'tuple_var_assignment_252598')
    # Assigning a type to the variable 'a' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'a', tuple_var_assignment_252598_253039)
    
    # Assigning a Name to a Name (line 195):
    # Getting the type of 'tuple_var_assignment_252599' (line 195)
    tuple_var_assignment_252599_253040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'tuple_var_assignment_252599')
    # Assigning a type to the variable 'b' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 7), 'b', tuple_var_assignment_252599_253040)
    
    # Assigning a Call to a Tuple (line 196):
    
    # Assigning a Subscript to a Name (line 196):
    
    # Obtaining the type of the subscript
    int_253041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 4), 'int')
    
    # Call to minimize_quadratic_1d(...): (line 196)
    # Processing the call arguments (line 196)
    # Getting the type of 'a' (line 196)
    a_253043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 48), 'a', False)
    # Getting the type of 'b' (line 196)
    b_253044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 51), 'b', False)
    int_253045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 54), 'int')
    # Getting the type of 'ag_stride' (line 196)
    ag_stride_253046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 57), 'ag_stride', False)
    # Processing the call keyword arguments (line 196)
    kwargs_253047 = {}
    # Getting the type of 'minimize_quadratic_1d' (line 196)
    minimize_quadratic_1d_253042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 26), 'minimize_quadratic_1d', False)
    # Calling minimize_quadratic_1d(args, kwargs) (line 196)
    minimize_quadratic_1d_call_result_253048 = invoke(stypy.reporting.localization.Localization(__file__, 196, 26), minimize_quadratic_1d_253042, *[a_253043, b_253044, int_253045, ag_stride_253046], **kwargs_253047)
    
    # Obtaining the member '__getitem__' of a type (line 196)
    getitem___253049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 4), minimize_quadratic_1d_call_result_253048, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 196)
    subscript_call_result_253050 = invoke(stypy.reporting.localization.Localization(__file__, 196, 4), getitem___253049, int_253041)
    
    # Assigning a type to the variable 'tuple_var_assignment_252600' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'tuple_var_assignment_252600', subscript_call_result_253050)
    
    # Assigning a Subscript to a Name (line 196):
    
    # Obtaining the type of the subscript
    int_253051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 4), 'int')
    
    # Call to minimize_quadratic_1d(...): (line 196)
    # Processing the call arguments (line 196)
    # Getting the type of 'a' (line 196)
    a_253053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 48), 'a', False)
    # Getting the type of 'b' (line 196)
    b_253054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 51), 'b', False)
    int_253055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 54), 'int')
    # Getting the type of 'ag_stride' (line 196)
    ag_stride_253056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 57), 'ag_stride', False)
    # Processing the call keyword arguments (line 196)
    kwargs_253057 = {}
    # Getting the type of 'minimize_quadratic_1d' (line 196)
    minimize_quadratic_1d_253052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 26), 'minimize_quadratic_1d', False)
    # Calling minimize_quadratic_1d(args, kwargs) (line 196)
    minimize_quadratic_1d_call_result_253058 = invoke(stypy.reporting.localization.Localization(__file__, 196, 26), minimize_quadratic_1d_253052, *[a_253053, b_253054, int_253055, ag_stride_253056], **kwargs_253057)
    
    # Obtaining the member '__getitem__' of a type (line 196)
    getitem___253059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 4), minimize_quadratic_1d_call_result_253058, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 196)
    subscript_call_result_253060 = invoke(stypy.reporting.localization.Localization(__file__, 196, 4), getitem___253059, int_253051)
    
    # Assigning a type to the variable 'tuple_var_assignment_252601' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'tuple_var_assignment_252601', subscript_call_result_253060)
    
    # Assigning a Name to a Name (line 196):
    # Getting the type of 'tuple_var_assignment_252600' (line 196)
    tuple_var_assignment_252600_253061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'tuple_var_assignment_252600')
    # Assigning a type to the variable 'ag_stride' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'ag_stride', tuple_var_assignment_252600_253061)
    
    # Assigning a Name to a Name (line 196):
    # Getting the type of 'tuple_var_assignment_252601' (line 196)
    tuple_var_assignment_252601_253062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'tuple_var_assignment_252601')
    # Assigning a type to the variable 'ag_value' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 15), 'ag_value', tuple_var_assignment_252601_253062)
    
    # Getting the type of 'ag_h' (line 197)
    ag_h_253063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'ag_h')
    # Getting the type of 'ag_stride' (line 197)
    ag_stride_253064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'ag_stride')
    # Applying the binary operator '*=' (line 197)
    result_imul_253065 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 4), '*=', ag_h_253063, ag_stride_253064)
    # Assigning a type to the variable 'ag_h' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'ag_h', result_imul_253065)
    
    
    # Getting the type of 'ag' (line 198)
    ag_253066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'ag')
    # Getting the type of 'ag_stride' (line 198)
    ag_stride_253067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 10), 'ag_stride')
    # Applying the binary operator '*=' (line 198)
    result_imul_253068 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 4), '*=', ag_253066, ag_stride_253067)
    # Assigning a type to the variable 'ag' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'ag', result_imul_253068)
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'p_value' (line 200)
    p_value_253069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 7), 'p_value')
    # Getting the type of 'r_value' (line 200)
    r_value_253070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 17), 'r_value')
    # Applying the binary operator '<' (line 200)
    result_lt_253071 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 7), '<', p_value_253069, r_value_253070)
    
    
    # Getting the type of 'p_value' (line 200)
    p_value_253072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 29), 'p_value')
    # Getting the type of 'ag_value' (line 200)
    ag_value_253073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 39), 'ag_value')
    # Applying the binary operator '<' (line 200)
    result_lt_253074 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 29), '<', p_value_253072, ag_value_253073)
    
    # Applying the binary operator 'and' (line 200)
    result_and_keyword_253075 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 7), 'and', result_lt_253071, result_lt_253074)
    
    # Testing the type of an if condition (line 200)
    if_condition_253076 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 4), result_and_keyword_253075)
    # Assigning a type to the variable 'if_condition_253076' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'if_condition_253076', if_condition_253076)
    # SSA begins for if statement (line 200)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 201)
    tuple_253077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 201)
    # Adding element type (line 201)
    # Getting the type of 'p' (line 201)
    p_253078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 15), 'p')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 15), tuple_253077, p_253078)
    # Adding element type (line 201)
    # Getting the type of 'p_h' (line 201)
    p_h_253079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 18), 'p_h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 15), tuple_253077, p_h_253079)
    # Adding element type (line 201)
    
    # Getting the type of 'p_value' (line 201)
    p_value_253080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 24), 'p_value')
    # Applying the 'usub' unary operator (line 201)
    result___neg___253081 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 23), 'usub', p_value_253080)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 15), tuple_253077, result___neg___253081)
    
    # Assigning a type to the variable 'stypy_return_type' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'stypy_return_type', tuple_253077)
    # SSA branch for the else part of an if statement (line 200)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'r_value' (line 202)
    r_value_253082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 9), 'r_value')
    # Getting the type of 'p_value' (line 202)
    p_value_253083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 19), 'p_value')
    # Applying the binary operator '<' (line 202)
    result_lt_253084 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 9), '<', r_value_253082, p_value_253083)
    
    
    # Getting the type of 'r_value' (line 202)
    r_value_253085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 31), 'r_value')
    # Getting the type of 'ag_value' (line 202)
    ag_value_253086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 41), 'ag_value')
    # Applying the binary operator '<' (line 202)
    result_lt_253087 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 31), '<', r_value_253085, ag_value_253086)
    
    # Applying the binary operator 'and' (line 202)
    result_and_keyword_253088 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 9), 'and', result_lt_253084, result_lt_253087)
    
    # Testing the type of an if condition (line 202)
    if_condition_253089 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 9), result_and_keyword_253088)
    # Assigning a type to the variable 'if_condition_253089' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 9), 'if_condition_253089', if_condition_253089)
    # SSA begins for if statement (line 202)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 203)
    tuple_253090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 203)
    # Adding element type (line 203)
    # Getting the type of 'r' (line 203)
    r_253091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 15), 'r')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 15), tuple_253090, r_253091)
    # Adding element type (line 203)
    # Getting the type of 'r_h' (line 203)
    r_h_253092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 18), 'r_h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 15), tuple_253090, r_h_253092)
    # Adding element type (line 203)
    
    # Getting the type of 'r_value' (line 203)
    r_value_253093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 24), 'r_value')
    # Applying the 'usub' unary operator (line 203)
    result___neg___253094 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 23), 'usub', r_value_253093)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 15), tuple_253090, result___neg___253094)
    
    # Assigning a type to the variable 'stypy_return_type' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'stypy_return_type', tuple_253090)
    # SSA branch for the else part of an if statement (line 202)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 205)
    tuple_253095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 205)
    # Adding element type (line 205)
    # Getting the type of 'ag' (line 205)
    ag_253096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 15), 'ag')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 15), tuple_253095, ag_253096)
    # Adding element type (line 205)
    # Getting the type of 'ag_h' (line 205)
    ag_h_253097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 19), 'ag_h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 15), tuple_253095, ag_h_253097)
    # Adding element type (line 205)
    
    # Getting the type of 'ag_value' (line 205)
    ag_value_253098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 26), 'ag_value')
    # Applying the 'usub' unary operator (line 205)
    result___neg___253099 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 25), 'usub', ag_value_253098)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 15), tuple_253095, result___neg___253099)
    
    # Assigning a type to the variable 'stypy_return_type' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'stypy_return_type', tuple_253095)
    # SSA join for if statement (line 202)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 200)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'select_step(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'select_step' in the type store
    # Getting the type of 'stypy_return_type' (line 131)
    stypy_return_type_253100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_253100)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'select_step'
    return stypy_return_type_253100

# Assigning a type to the variable 'select_step' (line 131)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 0), 'select_step', select_step)

@norecursion
def trf_bounds(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'trf_bounds'
    module_type_store = module_type_store.open_function_context('trf_bounds', 208, 0, False)
    
    # Passed parameters checking function
    trf_bounds.stypy_localization = localization
    trf_bounds.stypy_type_of_self = None
    trf_bounds.stypy_type_store = module_type_store
    trf_bounds.stypy_function_name = 'trf_bounds'
    trf_bounds.stypy_param_names_list = ['fun', 'jac', 'x0', 'f0', 'J0', 'lb', 'ub', 'ftol', 'xtol', 'gtol', 'max_nfev', 'x_scale', 'loss_function', 'tr_solver', 'tr_options', 'verbose']
    trf_bounds.stypy_varargs_param_name = None
    trf_bounds.stypy_kwargs_param_name = None
    trf_bounds.stypy_call_defaults = defaults
    trf_bounds.stypy_call_varargs = varargs
    trf_bounds.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'trf_bounds', ['fun', 'jac', 'x0', 'f0', 'J0', 'lb', 'ub', 'ftol', 'xtol', 'gtol', 'max_nfev', 'x_scale', 'loss_function', 'tr_solver', 'tr_options', 'verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'trf_bounds', localization, ['fun', 'jac', 'x0', 'f0', 'J0', 'lb', 'ub', 'ftol', 'xtol', 'gtol', 'max_nfev', 'x_scale', 'loss_function', 'tr_solver', 'tr_options', 'verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'trf_bounds(...)' code ##################

    
    # Assigning a Call to a Name (line 210):
    
    # Assigning a Call to a Name (line 210):
    
    # Call to copy(...): (line 210)
    # Processing the call keyword arguments (line 210)
    kwargs_253103 = {}
    # Getting the type of 'x0' (line 210)
    x0_253101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'x0', False)
    # Obtaining the member 'copy' of a type (line 210)
    copy_253102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 8), x0_253101, 'copy')
    # Calling copy(args, kwargs) (line 210)
    copy_call_result_253104 = invoke(stypy.reporting.localization.Localization(__file__, 210, 8), copy_253102, *[], **kwargs_253103)
    
    # Assigning a type to the variable 'x' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'x', copy_call_result_253104)
    
    # Assigning a Name to a Name (line 212):
    
    # Assigning a Name to a Name (line 212):
    # Getting the type of 'f0' (line 212)
    f0_253105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'f0')
    # Assigning a type to the variable 'f' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'f', f0_253105)
    
    # Assigning a Call to a Name (line 213):
    
    # Assigning a Call to a Name (line 213):
    
    # Call to copy(...): (line 213)
    # Processing the call keyword arguments (line 213)
    kwargs_253108 = {}
    # Getting the type of 'f' (line 213)
    f_253106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 13), 'f', False)
    # Obtaining the member 'copy' of a type (line 213)
    copy_253107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 13), f_253106, 'copy')
    # Calling copy(args, kwargs) (line 213)
    copy_call_result_253109 = invoke(stypy.reporting.localization.Localization(__file__, 213, 13), copy_253107, *[], **kwargs_253108)
    
    # Assigning a type to the variable 'f_true' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'f_true', copy_call_result_253109)
    
    # Assigning a Num to a Name (line 214):
    
    # Assigning a Num to a Name (line 214):
    int_253110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 11), 'int')
    # Assigning a type to the variable 'nfev' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'nfev', int_253110)
    
    # Assigning a Name to a Name (line 216):
    
    # Assigning a Name to a Name (line 216):
    # Getting the type of 'J0' (line 216)
    J0_253111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'J0')
    # Assigning a type to the variable 'J' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'J', J0_253111)
    
    # Assigning a Num to a Name (line 217):
    
    # Assigning a Num to a Name (line 217):
    int_253112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 11), 'int')
    # Assigning a type to the variable 'njev' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'njev', int_253112)
    
    # Assigning a Attribute to a Tuple (line 218):
    
    # Assigning a Subscript to a Name (line 218):
    
    # Obtaining the type of the subscript
    int_253113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 4), 'int')
    # Getting the type of 'J' (line 218)
    J_253114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 11), 'J')
    # Obtaining the member 'shape' of a type (line 218)
    shape_253115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 11), J_253114, 'shape')
    # Obtaining the member '__getitem__' of a type (line 218)
    getitem___253116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 4), shape_253115, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 218)
    subscript_call_result_253117 = invoke(stypy.reporting.localization.Localization(__file__, 218, 4), getitem___253116, int_253113)
    
    # Assigning a type to the variable 'tuple_var_assignment_252602' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'tuple_var_assignment_252602', subscript_call_result_253117)
    
    # Assigning a Subscript to a Name (line 218):
    
    # Obtaining the type of the subscript
    int_253118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 4), 'int')
    # Getting the type of 'J' (line 218)
    J_253119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 11), 'J')
    # Obtaining the member 'shape' of a type (line 218)
    shape_253120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 11), J_253119, 'shape')
    # Obtaining the member '__getitem__' of a type (line 218)
    getitem___253121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 4), shape_253120, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 218)
    subscript_call_result_253122 = invoke(stypy.reporting.localization.Localization(__file__, 218, 4), getitem___253121, int_253118)
    
    # Assigning a type to the variable 'tuple_var_assignment_252603' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'tuple_var_assignment_252603', subscript_call_result_253122)
    
    # Assigning a Name to a Name (line 218):
    # Getting the type of 'tuple_var_assignment_252602' (line 218)
    tuple_var_assignment_252602_253123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'tuple_var_assignment_252602')
    # Assigning a type to the variable 'm' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'm', tuple_var_assignment_252602_253123)
    
    # Assigning a Name to a Name (line 218):
    # Getting the type of 'tuple_var_assignment_252603' (line 218)
    tuple_var_assignment_252603_253124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'tuple_var_assignment_252603')
    # Assigning a type to the variable 'n' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 7), 'n', tuple_var_assignment_252603_253124)
    
    # Type idiom detected: calculating its left and rigth part (line 220)
    # Getting the type of 'loss_function' (line 220)
    loss_function_253125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'loss_function')
    # Getting the type of 'None' (line 220)
    None_253126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 28), 'None')
    
    (may_be_253127, more_types_in_union_253128) = may_not_be_none(loss_function_253125, None_253126)

    if may_be_253127:

        if more_types_in_union_253128:
            # Runtime conditional SSA (line 220)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 221):
        
        # Assigning a Call to a Name (line 221):
        
        # Call to loss_function(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'f' (line 221)
        f_253130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 28), 'f', False)
        # Processing the call keyword arguments (line 221)
        kwargs_253131 = {}
        # Getting the type of 'loss_function' (line 221)
        loss_function_253129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 14), 'loss_function', False)
        # Calling loss_function(args, kwargs) (line 221)
        loss_function_call_result_253132 = invoke(stypy.reporting.localization.Localization(__file__, 221, 14), loss_function_253129, *[f_253130], **kwargs_253131)
        
        # Assigning a type to the variable 'rho' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'rho', loss_function_call_result_253132)
        
        # Assigning a BinOp to a Name (line 222):
        
        # Assigning a BinOp to a Name (line 222):
        float_253133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 15), 'float')
        
        # Call to sum(...): (line 222)
        # Processing the call arguments (line 222)
        
        # Obtaining the type of the subscript
        int_253136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 32), 'int')
        # Getting the type of 'rho' (line 222)
        rho_253137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 28), 'rho', False)
        # Obtaining the member '__getitem__' of a type (line 222)
        getitem___253138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 28), rho_253137, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 222)
        subscript_call_result_253139 = invoke(stypy.reporting.localization.Localization(__file__, 222, 28), getitem___253138, int_253136)
        
        # Processing the call keyword arguments (line 222)
        kwargs_253140 = {}
        # Getting the type of 'np' (line 222)
        np_253134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 21), 'np', False)
        # Obtaining the member 'sum' of a type (line 222)
        sum_253135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 21), np_253134, 'sum')
        # Calling sum(args, kwargs) (line 222)
        sum_call_result_253141 = invoke(stypy.reporting.localization.Localization(__file__, 222, 21), sum_253135, *[subscript_call_result_253139], **kwargs_253140)
        
        # Applying the binary operator '*' (line 222)
        result_mul_253142 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 15), '*', float_253133, sum_call_result_253141)
        
        # Assigning a type to the variable 'cost' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'cost', result_mul_253142)
        
        # Assigning a Call to a Tuple (line 223):
        
        # Assigning a Subscript to a Name (line 223):
        
        # Obtaining the type of the subscript
        int_253143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 8), 'int')
        
        # Call to scale_for_robust_loss_function(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'J' (line 223)
        J_253145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 46), 'J', False)
        # Getting the type of 'f' (line 223)
        f_253146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 49), 'f', False)
        # Getting the type of 'rho' (line 223)
        rho_253147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 52), 'rho', False)
        # Processing the call keyword arguments (line 223)
        kwargs_253148 = {}
        # Getting the type of 'scale_for_robust_loss_function' (line 223)
        scale_for_robust_loss_function_253144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 15), 'scale_for_robust_loss_function', False)
        # Calling scale_for_robust_loss_function(args, kwargs) (line 223)
        scale_for_robust_loss_function_call_result_253149 = invoke(stypy.reporting.localization.Localization(__file__, 223, 15), scale_for_robust_loss_function_253144, *[J_253145, f_253146, rho_253147], **kwargs_253148)
        
        # Obtaining the member '__getitem__' of a type (line 223)
        getitem___253150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), scale_for_robust_loss_function_call_result_253149, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 223)
        subscript_call_result_253151 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), getitem___253150, int_253143)
        
        # Assigning a type to the variable 'tuple_var_assignment_252604' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'tuple_var_assignment_252604', subscript_call_result_253151)
        
        # Assigning a Subscript to a Name (line 223):
        
        # Obtaining the type of the subscript
        int_253152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 8), 'int')
        
        # Call to scale_for_robust_loss_function(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'J' (line 223)
        J_253154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 46), 'J', False)
        # Getting the type of 'f' (line 223)
        f_253155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 49), 'f', False)
        # Getting the type of 'rho' (line 223)
        rho_253156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 52), 'rho', False)
        # Processing the call keyword arguments (line 223)
        kwargs_253157 = {}
        # Getting the type of 'scale_for_robust_loss_function' (line 223)
        scale_for_robust_loss_function_253153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 15), 'scale_for_robust_loss_function', False)
        # Calling scale_for_robust_loss_function(args, kwargs) (line 223)
        scale_for_robust_loss_function_call_result_253158 = invoke(stypy.reporting.localization.Localization(__file__, 223, 15), scale_for_robust_loss_function_253153, *[J_253154, f_253155, rho_253156], **kwargs_253157)
        
        # Obtaining the member '__getitem__' of a type (line 223)
        getitem___253159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), scale_for_robust_loss_function_call_result_253158, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 223)
        subscript_call_result_253160 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), getitem___253159, int_253152)
        
        # Assigning a type to the variable 'tuple_var_assignment_252605' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'tuple_var_assignment_252605', subscript_call_result_253160)
        
        # Assigning a Name to a Name (line 223):
        # Getting the type of 'tuple_var_assignment_252604' (line 223)
        tuple_var_assignment_252604_253161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'tuple_var_assignment_252604')
        # Assigning a type to the variable 'J' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'J', tuple_var_assignment_252604_253161)
        
        # Assigning a Name to a Name (line 223):
        # Getting the type of 'tuple_var_assignment_252605' (line 223)
        tuple_var_assignment_252605_253162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'tuple_var_assignment_252605')
        # Assigning a type to the variable 'f' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 11), 'f', tuple_var_assignment_252605_253162)

        if more_types_in_union_253128:
            # Runtime conditional SSA for else branch (line 220)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_253127) or more_types_in_union_253128):
        
        # Assigning a BinOp to a Name (line 225):
        
        # Assigning a BinOp to a Name (line 225):
        float_253163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 15), 'float')
        
        # Call to dot(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'f' (line 225)
        f_253166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 28), 'f', False)
        # Getting the type of 'f' (line 225)
        f_253167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 31), 'f', False)
        # Processing the call keyword arguments (line 225)
        kwargs_253168 = {}
        # Getting the type of 'np' (line 225)
        np_253164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 21), 'np', False)
        # Obtaining the member 'dot' of a type (line 225)
        dot_253165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 21), np_253164, 'dot')
        # Calling dot(args, kwargs) (line 225)
        dot_call_result_253169 = invoke(stypy.reporting.localization.Localization(__file__, 225, 21), dot_253165, *[f_253166, f_253167], **kwargs_253168)
        
        # Applying the binary operator '*' (line 225)
        result_mul_253170 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 15), '*', float_253163, dot_call_result_253169)
        
        # Assigning a type to the variable 'cost' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'cost', result_mul_253170)

        if (may_be_253127 and more_types_in_union_253128):
            # SSA join for if statement (line 220)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 227):
    
    # Assigning a Call to a Name (line 227):
    
    # Call to compute_grad(...): (line 227)
    # Processing the call arguments (line 227)
    # Getting the type of 'J' (line 227)
    J_253172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 21), 'J', False)
    # Getting the type of 'f' (line 227)
    f_253173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 24), 'f', False)
    # Processing the call keyword arguments (line 227)
    kwargs_253174 = {}
    # Getting the type of 'compute_grad' (line 227)
    compute_grad_253171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'compute_grad', False)
    # Calling compute_grad(args, kwargs) (line 227)
    compute_grad_call_result_253175 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), compute_grad_253171, *[J_253172, f_253173], **kwargs_253174)
    
    # Assigning a type to the variable 'g' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'g', compute_grad_call_result_253175)
    
    # Assigning a BoolOp to a Name (line 229):
    
    # Assigning a BoolOp to a Name (line 229):
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 229)
    # Processing the call arguments (line 229)
    # Getting the type of 'x_scale' (line 229)
    x_scale_253177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 27), 'x_scale', False)
    # Getting the type of 'string_types' (line 229)
    string_types_253178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 36), 'string_types', False)
    # Processing the call keyword arguments (line 229)
    kwargs_253179 = {}
    # Getting the type of 'isinstance' (line 229)
    isinstance_253176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 16), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 229)
    isinstance_call_result_253180 = invoke(stypy.reporting.localization.Localization(__file__, 229, 16), isinstance_253176, *[x_scale_253177, string_types_253178], **kwargs_253179)
    
    
    # Getting the type of 'x_scale' (line 229)
    x_scale_253181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 54), 'x_scale')
    str_253182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 65), 'str', 'jac')
    # Applying the binary operator '==' (line 229)
    result_eq_253183 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 54), '==', x_scale_253181, str_253182)
    
    # Applying the binary operator 'and' (line 229)
    result_and_keyword_253184 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 16), 'and', isinstance_call_result_253180, result_eq_253183)
    
    # Assigning a type to the variable 'jac_scale' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'jac_scale', result_and_keyword_253184)
    
    # Getting the type of 'jac_scale' (line 230)
    jac_scale_253185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 7), 'jac_scale')
    # Testing the type of an if condition (line 230)
    if_condition_253186 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 4), jac_scale_253185)
    # Assigning a type to the variable 'if_condition_253186' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'if_condition_253186', if_condition_253186)
    # SSA begins for if statement (line 230)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 231):
    
    # Assigning a Subscript to a Name (line 231):
    
    # Obtaining the type of the subscript
    int_253187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 8), 'int')
    
    # Call to compute_jac_scale(...): (line 231)
    # Processing the call arguments (line 231)
    # Getting the type of 'J' (line 231)
    J_253189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 45), 'J', False)
    # Processing the call keyword arguments (line 231)
    kwargs_253190 = {}
    # Getting the type of 'compute_jac_scale' (line 231)
    compute_jac_scale_253188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 27), 'compute_jac_scale', False)
    # Calling compute_jac_scale(args, kwargs) (line 231)
    compute_jac_scale_call_result_253191 = invoke(stypy.reporting.localization.Localization(__file__, 231, 27), compute_jac_scale_253188, *[J_253189], **kwargs_253190)
    
    # Obtaining the member '__getitem__' of a type (line 231)
    getitem___253192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 8), compute_jac_scale_call_result_253191, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 231)
    subscript_call_result_253193 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), getitem___253192, int_253187)
    
    # Assigning a type to the variable 'tuple_var_assignment_252606' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'tuple_var_assignment_252606', subscript_call_result_253193)
    
    # Assigning a Subscript to a Name (line 231):
    
    # Obtaining the type of the subscript
    int_253194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 8), 'int')
    
    # Call to compute_jac_scale(...): (line 231)
    # Processing the call arguments (line 231)
    # Getting the type of 'J' (line 231)
    J_253196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 45), 'J', False)
    # Processing the call keyword arguments (line 231)
    kwargs_253197 = {}
    # Getting the type of 'compute_jac_scale' (line 231)
    compute_jac_scale_253195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 27), 'compute_jac_scale', False)
    # Calling compute_jac_scale(args, kwargs) (line 231)
    compute_jac_scale_call_result_253198 = invoke(stypy.reporting.localization.Localization(__file__, 231, 27), compute_jac_scale_253195, *[J_253196], **kwargs_253197)
    
    # Obtaining the member '__getitem__' of a type (line 231)
    getitem___253199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 8), compute_jac_scale_call_result_253198, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 231)
    subscript_call_result_253200 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), getitem___253199, int_253194)
    
    # Assigning a type to the variable 'tuple_var_assignment_252607' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'tuple_var_assignment_252607', subscript_call_result_253200)
    
    # Assigning a Name to a Name (line 231):
    # Getting the type of 'tuple_var_assignment_252606' (line 231)
    tuple_var_assignment_252606_253201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'tuple_var_assignment_252606')
    # Assigning a type to the variable 'scale' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'scale', tuple_var_assignment_252606_253201)
    
    # Assigning a Name to a Name (line 231):
    # Getting the type of 'tuple_var_assignment_252607' (line 231)
    tuple_var_assignment_252607_253202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'tuple_var_assignment_252607')
    # Assigning a type to the variable 'scale_inv' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 15), 'scale_inv', tuple_var_assignment_252607_253202)
    # SSA branch for the else part of an if statement (line 230)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Tuple to a Tuple (line 233):
    
    # Assigning a Name to a Name (line 233):
    # Getting the type of 'x_scale' (line 233)
    x_scale_253203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 27), 'x_scale')
    # Assigning a type to the variable 'tuple_assignment_252608' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'tuple_assignment_252608', x_scale_253203)
    
    # Assigning a BinOp to a Name (line 233):
    int_253204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 36), 'int')
    # Getting the type of 'x_scale' (line 233)
    x_scale_253205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 40), 'x_scale')
    # Applying the binary operator 'div' (line 233)
    result_div_253206 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 36), 'div', int_253204, x_scale_253205)
    
    # Assigning a type to the variable 'tuple_assignment_252609' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'tuple_assignment_252609', result_div_253206)
    
    # Assigning a Name to a Name (line 233):
    # Getting the type of 'tuple_assignment_252608' (line 233)
    tuple_assignment_252608_253207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'tuple_assignment_252608')
    # Assigning a type to the variable 'scale' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'scale', tuple_assignment_252608_253207)
    
    # Assigning a Name to a Name (line 233):
    # Getting the type of 'tuple_assignment_252609' (line 233)
    tuple_assignment_252609_253208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'tuple_assignment_252609')
    # Assigning a type to the variable 'scale_inv' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 15), 'scale_inv', tuple_assignment_252609_253208)
    # SSA join for if statement (line 230)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 235):
    
    # Assigning a Subscript to a Name (line 235):
    
    # Obtaining the type of the subscript
    int_253209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 4), 'int')
    
    # Call to CL_scaling_vector(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'x' (line 235)
    x_253211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 30), 'x', False)
    # Getting the type of 'g' (line 235)
    g_253212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 33), 'g', False)
    # Getting the type of 'lb' (line 235)
    lb_253213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 36), 'lb', False)
    # Getting the type of 'ub' (line 235)
    ub_253214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 40), 'ub', False)
    # Processing the call keyword arguments (line 235)
    kwargs_253215 = {}
    # Getting the type of 'CL_scaling_vector' (line 235)
    CL_scaling_vector_253210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'CL_scaling_vector', False)
    # Calling CL_scaling_vector(args, kwargs) (line 235)
    CL_scaling_vector_call_result_253216 = invoke(stypy.reporting.localization.Localization(__file__, 235, 12), CL_scaling_vector_253210, *[x_253211, g_253212, lb_253213, ub_253214], **kwargs_253215)
    
    # Obtaining the member '__getitem__' of a type (line 235)
    getitem___253217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 4), CL_scaling_vector_call_result_253216, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 235)
    subscript_call_result_253218 = invoke(stypy.reporting.localization.Localization(__file__, 235, 4), getitem___253217, int_253209)
    
    # Assigning a type to the variable 'tuple_var_assignment_252610' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'tuple_var_assignment_252610', subscript_call_result_253218)
    
    # Assigning a Subscript to a Name (line 235):
    
    # Obtaining the type of the subscript
    int_253219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 4), 'int')
    
    # Call to CL_scaling_vector(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'x' (line 235)
    x_253221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 30), 'x', False)
    # Getting the type of 'g' (line 235)
    g_253222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 33), 'g', False)
    # Getting the type of 'lb' (line 235)
    lb_253223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 36), 'lb', False)
    # Getting the type of 'ub' (line 235)
    ub_253224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 40), 'ub', False)
    # Processing the call keyword arguments (line 235)
    kwargs_253225 = {}
    # Getting the type of 'CL_scaling_vector' (line 235)
    CL_scaling_vector_253220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'CL_scaling_vector', False)
    # Calling CL_scaling_vector(args, kwargs) (line 235)
    CL_scaling_vector_call_result_253226 = invoke(stypy.reporting.localization.Localization(__file__, 235, 12), CL_scaling_vector_253220, *[x_253221, g_253222, lb_253223, ub_253224], **kwargs_253225)
    
    # Obtaining the member '__getitem__' of a type (line 235)
    getitem___253227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 4), CL_scaling_vector_call_result_253226, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 235)
    subscript_call_result_253228 = invoke(stypy.reporting.localization.Localization(__file__, 235, 4), getitem___253227, int_253219)
    
    # Assigning a type to the variable 'tuple_var_assignment_252611' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'tuple_var_assignment_252611', subscript_call_result_253228)
    
    # Assigning a Name to a Name (line 235):
    # Getting the type of 'tuple_var_assignment_252610' (line 235)
    tuple_var_assignment_252610_253229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'tuple_var_assignment_252610')
    # Assigning a type to the variable 'v' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'v', tuple_var_assignment_252610_253229)
    
    # Assigning a Name to a Name (line 235):
    # Getting the type of 'tuple_var_assignment_252611' (line 235)
    tuple_var_assignment_252611_253230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'tuple_var_assignment_252611')
    # Assigning a type to the variable 'dv' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 7), 'dv', tuple_var_assignment_252611_253230)
    
    # Getting the type of 'v' (line 236)
    v_253231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'v')
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'dv' (line 236)
    dv_253232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 6), 'dv')
    int_253233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 12), 'int')
    # Applying the binary operator '!=' (line 236)
    result_ne_253234 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 6), '!=', dv_253232, int_253233)
    
    # Getting the type of 'v' (line 236)
    v_253235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'v')
    # Obtaining the member '__getitem__' of a type (line 236)
    getitem___253236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 4), v_253235, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 236)
    subscript_call_result_253237 = invoke(stypy.reporting.localization.Localization(__file__, 236, 4), getitem___253236, result_ne_253234)
    
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'dv' (line 236)
    dv_253238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 28), 'dv')
    int_253239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 34), 'int')
    # Applying the binary operator '!=' (line 236)
    result_ne_253240 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 28), '!=', dv_253238, int_253239)
    
    # Getting the type of 'scale_inv' (line 236)
    scale_inv_253241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 18), 'scale_inv')
    # Obtaining the member '__getitem__' of a type (line 236)
    getitem___253242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 18), scale_inv_253241, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 236)
    subscript_call_result_253243 = invoke(stypy.reporting.localization.Localization(__file__, 236, 18), getitem___253242, result_ne_253240)
    
    # Applying the binary operator '*=' (line 236)
    result_imul_253244 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 4), '*=', subscript_call_result_253237, subscript_call_result_253243)
    # Getting the type of 'v' (line 236)
    v_253245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'v')
    
    # Getting the type of 'dv' (line 236)
    dv_253246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 6), 'dv')
    int_253247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 12), 'int')
    # Applying the binary operator '!=' (line 236)
    result_ne_253248 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 6), '!=', dv_253246, int_253247)
    
    # Storing an element on a container (line 236)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 4), v_253245, (result_ne_253248, result_imul_253244))
    
    
    # Assigning a Call to a Name (line 237):
    
    # Assigning a Call to a Name (line 237):
    
    # Call to norm(...): (line 237)
    # Processing the call arguments (line 237)
    # Getting the type of 'x0' (line 237)
    x0_253250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 17), 'x0', False)
    # Getting the type of 'scale_inv' (line 237)
    scale_inv_253251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 22), 'scale_inv', False)
    # Applying the binary operator '*' (line 237)
    result_mul_253252 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 17), '*', x0_253250, scale_inv_253251)
    
    # Getting the type of 'v' (line 237)
    v_253253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 34), 'v', False)
    float_253254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 37), 'float')
    # Applying the binary operator '**' (line 237)
    result_pow_253255 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 34), '**', v_253253, float_253254)
    
    # Applying the binary operator 'div' (line 237)
    result_div_253256 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 32), 'div', result_mul_253252, result_pow_253255)
    
    # Processing the call keyword arguments (line 237)
    kwargs_253257 = {}
    # Getting the type of 'norm' (line 237)
    norm_253249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'norm', False)
    # Calling norm(args, kwargs) (line 237)
    norm_call_result_253258 = invoke(stypy.reporting.localization.Localization(__file__, 237, 12), norm_253249, *[result_div_253256], **kwargs_253257)
    
    # Assigning a type to the variable 'Delta' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'Delta', norm_call_result_253258)
    
    
    # Getting the type of 'Delta' (line 238)
    Delta_253259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 7), 'Delta')
    int_253260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 16), 'int')
    # Applying the binary operator '==' (line 238)
    result_eq_253261 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 7), '==', Delta_253259, int_253260)
    
    # Testing the type of an if condition (line 238)
    if_condition_253262 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 238, 4), result_eq_253261)
    # Assigning a type to the variable 'if_condition_253262' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'if_condition_253262', if_condition_253262)
    # SSA begins for if statement (line 238)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 239):
    
    # Assigning a Num to a Name (line 239):
    float_253263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 16), 'float')
    # Assigning a type to the variable 'Delta' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'Delta', float_253263)
    # SSA join for if statement (line 238)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 241):
    
    # Assigning a Call to a Name (line 241):
    
    # Call to norm(...): (line 241)
    # Processing the call arguments (line 241)
    # Getting the type of 'g' (line 241)
    g_253265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 18), 'g', False)
    # Getting the type of 'v' (line 241)
    v_253266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 22), 'v', False)
    # Applying the binary operator '*' (line 241)
    result_mul_253267 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 18), '*', g_253265, v_253266)
    
    # Processing the call keyword arguments (line 241)
    # Getting the type of 'np' (line 241)
    np_253268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 29), 'np', False)
    # Obtaining the member 'inf' of a type (line 241)
    inf_253269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 29), np_253268, 'inf')
    keyword_253270 = inf_253269
    kwargs_253271 = {'ord': keyword_253270}
    # Getting the type of 'norm' (line 241)
    norm_253264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 13), 'norm', False)
    # Calling norm(args, kwargs) (line 241)
    norm_call_result_253272 = invoke(stypy.reporting.localization.Localization(__file__, 241, 13), norm_253264, *[result_mul_253267], **kwargs_253271)
    
    # Assigning a type to the variable 'g_norm' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'g_norm', norm_call_result_253272)
    
    # Assigning a Call to a Name (line 243):
    
    # Assigning a Call to a Name (line 243):
    
    # Call to zeros(...): (line 243)
    # Processing the call arguments (line 243)
    # Getting the type of 'm' (line 243)
    m_253275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 28), 'm', False)
    # Getting the type of 'n' (line 243)
    n_253276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 32), 'n', False)
    # Applying the binary operator '+' (line 243)
    result_add_253277 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 28), '+', m_253275, n_253276)
    
    # Processing the call keyword arguments (line 243)
    kwargs_253278 = {}
    # Getting the type of 'np' (line 243)
    np_253273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 18), 'np', False)
    # Obtaining the member 'zeros' of a type (line 243)
    zeros_253274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 18), np_253273, 'zeros')
    # Calling zeros(args, kwargs) (line 243)
    zeros_call_result_253279 = invoke(stypy.reporting.localization.Localization(__file__, 243, 18), zeros_253274, *[result_add_253277], **kwargs_253278)
    
    # Assigning a type to the variable 'f_augmented' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'f_augmented', zeros_call_result_253279)
    
    
    # Getting the type of 'tr_solver' (line 244)
    tr_solver_253280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 7), 'tr_solver')
    str_253281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 20), 'str', 'exact')
    # Applying the binary operator '==' (line 244)
    result_eq_253282 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 7), '==', tr_solver_253280, str_253281)
    
    # Testing the type of an if condition (line 244)
    if_condition_253283 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 244, 4), result_eq_253282)
    # Assigning a type to the variable 'if_condition_253283' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'if_condition_253283', if_condition_253283)
    # SSA begins for if statement (line 244)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 245):
    
    # Assigning a Call to a Name (line 245):
    
    # Call to empty(...): (line 245)
    # Processing the call arguments (line 245)
    
    # Obtaining an instance of the builtin type 'tuple' (line 245)
    tuple_253286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 245)
    # Adding element type (line 245)
    # Getting the type of 'm' (line 245)
    m_253287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 32), 'm', False)
    # Getting the type of 'n' (line 245)
    n_253288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 36), 'n', False)
    # Applying the binary operator '+' (line 245)
    result_add_253289 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 32), '+', m_253287, n_253288)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 32), tuple_253286, result_add_253289)
    # Adding element type (line 245)
    # Getting the type of 'n' (line 245)
    n_253290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 39), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 32), tuple_253286, n_253290)
    
    # Processing the call keyword arguments (line 245)
    kwargs_253291 = {}
    # Getting the type of 'np' (line 245)
    np_253284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 22), 'np', False)
    # Obtaining the member 'empty' of a type (line 245)
    empty_253285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 22), np_253284, 'empty')
    # Calling empty(args, kwargs) (line 245)
    empty_call_result_253292 = invoke(stypy.reporting.localization.Localization(__file__, 245, 22), empty_253285, *[tuple_253286], **kwargs_253291)
    
    # Assigning a type to the variable 'J_augmented' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'J_augmented', empty_call_result_253292)
    # SSA branch for the else part of an if statement (line 244)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'tr_solver' (line 246)
    tr_solver_253293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 9), 'tr_solver')
    str_253294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 22), 'str', 'lsmr')
    # Applying the binary operator '==' (line 246)
    result_eq_253295 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 9), '==', tr_solver_253293, str_253294)
    
    # Testing the type of an if condition (line 246)
    if_condition_253296 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 9), result_eq_253295)
    # Assigning a type to the variable 'if_condition_253296' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 9), 'if_condition_253296', if_condition_253296)
    # SSA begins for if statement (line 246)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 247):
    
    # Assigning a Num to a Name (line 247):
    float_253297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 19), 'float')
    # Assigning a type to the variable 'reg_term' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'reg_term', float_253297)
    
    # Assigning a Call to a Name (line 248):
    
    # Assigning a Call to a Name (line 248):
    
    # Call to pop(...): (line 248)
    # Processing the call arguments (line 248)
    str_253300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 36), 'str', 'regularize')
    # Getting the type of 'True' (line 248)
    True_253301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 50), 'True', False)
    # Processing the call keyword arguments (line 248)
    kwargs_253302 = {}
    # Getting the type of 'tr_options' (line 248)
    tr_options_253298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 21), 'tr_options', False)
    # Obtaining the member 'pop' of a type (line 248)
    pop_253299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 21), tr_options_253298, 'pop')
    # Calling pop(args, kwargs) (line 248)
    pop_call_result_253303 = invoke(stypy.reporting.localization.Localization(__file__, 248, 21), pop_253299, *[str_253300, True_253301], **kwargs_253302)
    
    # Assigning a type to the variable 'regularize' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'regularize', pop_call_result_253303)
    # SSA join for if statement (line 246)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 244)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 250)
    # Getting the type of 'max_nfev' (line 250)
    max_nfev_253304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 7), 'max_nfev')
    # Getting the type of 'None' (line 250)
    None_253305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 19), 'None')
    
    (may_be_253306, more_types_in_union_253307) = may_be_none(max_nfev_253304, None_253305)

    if may_be_253306:

        if more_types_in_union_253307:
            # Runtime conditional SSA (line 250)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 251):
        
        # Assigning a BinOp to a Name (line 251):
        # Getting the type of 'x0' (line 251)
        x0_253308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 19), 'x0')
        # Obtaining the member 'size' of a type (line 251)
        size_253309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 19), x0_253308, 'size')
        int_253310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 29), 'int')
        # Applying the binary operator '*' (line 251)
        result_mul_253311 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 19), '*', size_253309, int_253310)
        
        # Assigning a type to the variable 'max_nfev' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'max_nfev', result_mul_253311)

        if more_types_in_union_253307:
            # SSA join for if statement (line 250)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Num to a Name (line 253):
    
    # Assigning a Num to a Name (line 253):
    float_253312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 12), 'float')
    # Assigning a type to the variable 'alpha' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'alpha', float_253312)
    
    # Assigning a Name to a Name (line 255):
    
    # Assigning a Name to a Name (line 255):
    # Getting the type of 'None' (line 255)
    None_253313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 25), 'None')
    # Assigning a type to the variable 'termination_status' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'termination_status', None_253313)
    
    # Assigning a Num to a Name (line 256):
    
    # Assigning a Num to a Name (line 256):
    int_253314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 16), 'int')
    # Assigning a type to the variable 'iteration' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'iteration', int_253314)
    
    # Assigning a Name to a Name (line 257):
    
    # Assigning a Name to a Name (line 257):
    # Getting the type of 'None' (line 257)
    None_253315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'None')
    # Assigning a type to the variable 'step_norm' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'step_norm', None_253315)
    
    # Assigning a Name to a Name (line 258):
    
    # Assigning a Name to a Name (line 258):
    # Getting the type of 'None' (line 258)
    None_253316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 23), 'None')
    # Assigning a type to the variable 'actual_reduction' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'actual_reduction', None_253316)
    
    
    # Getting the type of 'verbose' (line 260)
    verbose_253317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 7), 'verbose')
    int_253318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 18), 'int')
    # Applying the binary operator '==' (line 260)
    result_eq_253319 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 7), '==', verbose_253317, int_253318)
    
    # Testing the type of an if condition (line 260)
    if_condition_253320 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 260, 4), result_eq_253319)
    # Assigning a type to the variable 'if_condition_253320' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'if_condition_253320', if_condition_253320)
    # SSA begins for if statement (line 260)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print_header_nonlinear(...): (line 261)
    # Processing the call keyword arguments (line 261)
    kwargs_253322 = {}
    # Getting the type of 'print_header_nonlinear' (line 261)
    print_header_nonlinear_253321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'print_header_nonlinear', False)
    # Calling print_header_nonlinear(args, kwargs) (line 261)
    print_header_nonlinear_call_result_253323 = invoke(stypy.reporting.localization.Localization(__file__, 261, 8), print_header_nonlinear_253321, *[], **kwargs_253322)
    
    # SSA join for if statement (line 260)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'True' (line 263)
    True_253324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 10), 'True')
    # Testing the type of an if condition (line 263)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 263, 4), True_253324)
    # SSA begins for while statement (line 263)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Tuple (line 264):
    
    # Assigning a Subscript to a Name (line 264):
    
    # Obtaining the type of the subscript
    int_253325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 8), 'int')
    
    # Call to CL_scaling_vector(...): (line 264)
    # Processing the call arguments (line 264)
    # Getting the type of 'x' (line 264)
    x_253327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 34), 'x', False)
    # Getting the type of 'g' (line 264)
    g_253328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 37), 'g', False)
    # Getting the type of 'lb' (line 264)
    lb_253329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 40), 'lb', False)
    # Getting the type of 'ub' (line 264)
    ub_253330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 44), 'ub', False)
    # Processing the call keyword arguments (line 264)
    kwargs_253331 = {}
    # Getting the type of 'CL_scaling_vector' (line 264)
    CL_scaling_vector_253326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 16), 'CL_scaling_vector', False)
    # Calling CL_scaling_vector(args, kwargs) (line 264)
    CL_scaling_vector_call_result_253332 = invoke(stypy.reporting.localization.Localization(__file__, 264, 16), CL_scaling_vector_253326, *[x_253327, g_253328, lb_253329, ub_253330], **kwargs_253331)
    
    # Obtaining the member '__getitem__' of a type (line 264)
    getitem___253333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 8), CL_scaling_vector_call_result_253332, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 264)
    subscript_call_result_253334 = invoke(stypy.reporting.localization.Localization(__file__, 264, 8), getitem___253333, int_253325)
    
    # Assigning a type to the variable 'tuple_var_assignment_252612' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'tuple_var_assignment_252612', subscript_call_result_253334)
    
    # Assigning a Subscript to a Name (line 264):
    
    # Obtaining the type of the subscript
    int_253335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 8), 'int')
    
    # Call to CL_scaling_vector(...): (line 264)
    # Processing the call arguments (line 264)
    # Getting the type of 'x' (line 264)
    x_253337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 34), 'x', False)
    # Getting the type of 'g' (line 264)
    g_253338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 37), 'g', False)
    # Getting the type of 'lb' (line 264)
    lb_253339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 40), 'lb', False)
    # Getting the type of 'ub' (line 264)
    ub_253340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 44), 'ub', False)
    # Processing the call keyword arguments (line 264)
    kwargs_253341 = {}
    # Getting the type of 'CL_scaling_vector' (line 264)
    CL_scaling_vector_253336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 16), 'CL_scaling_vector', False)
    # Calling CL_scaling_vector(args, kwargs) (line 264)
    CL_scaling_vector_call_result_253342 = invoke(stypy.reporting.localization.Localization(__file__, 264, 16), CL_scaling_vector_253336, *[x_253337, g_253338, lb_253339, ub_253340], **kwargs_253341)
    
    # Obtaining the member '__getitem__' of a type (line 264)
    getitem___253343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 8), CL_scaling_vector_call_result_253342, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 264)
    subscript_call_result_253344 = invoke(stypy.reporting.localization.Localization(__file__, 264, 8), getitem___253343, int_253335)
    
    # Assigning a type to the variable 'tuple_var_assignment_252613' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'tuple_var_assignment_252613', subscript_call_result_253344)
    
    # Assigning a Name to a Name (line 264):
    # Getting the type of 'tuple_var_assignment_252612' (line 264)
    tuple_var_assignment_252612_253345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'tuple_var_assignment_252612')
    # Assigning a type to the variable 'v' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'v', tuple_var_assignment_252612_253345)
    
    # Assigning a Name to a Name (line 264):
    # Getting the type of 'tuple_var_assignment_252613' (line 264)
    tuple_var_assignment_252613_253346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'tuple_var_assignment_252613')
    # Assigning a type to the variable 'dv' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 11), 'dv', tuple_var_assignment_252613_253346)
    
    # Assigning a Call to a Name (line 266):
    
    # Assigning a Call to a Name (line 266):
    
    # Call to norm(...): (line 266)
    # Processing the call arguments (line 266)
    # Getting the type of 'g' (line 266)
    g_253348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 22), 'g', False)
    # Getting the type of 'v' (line 266)
    v_253349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 26), 'v', False)
    # Applying the binary operator '*' (line 266)
    result_mul_253350 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 22), '*', g_253348, v_253349)
    
    # Processing the call keyword arguments (line 266)
    # Getting the type of 'np' (line 266)
    np_253351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 33), 'np', False)
    # Obtaining the member 'inf' of a type (line 266)
    inf_253352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 33), np_253351, 'inf')
    keyword_253353 = inf_253352
    kwargs_253354 = {'ord': keyword_253353}
    # Getting the type of 'norm' (line 266)
    norm_253347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 17), 'norm', False)
    # Calling norm(args, kwargs) (line 266)
    norm_call_result_253355 = invoke(stypy.reporting.localization.Localization(__file__, 266, 17), norm_253347, *[result_mul_253350], **kwargs_253354)
    
    # Assigning a type to the variable 'g_norm' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'g_norm', norm_call_result_253355)
    
    
    # Getting the type of 'g_norm' (line 267)
    g_norm_253356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 11), 'g_norm')
    # Getting the type of 'gtol' (line 267)
    gtol_253357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 20), 'gtol')
    # Applying the binary operator '<' (line 267)
    result_lt_253358 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 11), '<', g_norm_253356, gtol_253357)
    
    # Testing the type of an if condition (line 267)
    if_condition_253359 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 267, 8), result_lt_253358)
    # Assigning a type to the variable 'if_condition_253359' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'if_condition_253359', if_condition_253359)
    # SSA begins for if statement (line 267)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 268):
    
    # Assigning a Num to a Name (line 268):
    int_253360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 33), 'int')
    # Assigning a type to the variable 'termination_status' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'termination_status', int_253360)
    # SSA join for if statement (line 267)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'verbose' (line 270)
    verbose_253361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 11), 'verbose')
    int_253362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 22), 'int')
    # Applying the binary operator '==' (line 270)
    result_eq_253363 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 11), '==', verbose_253361, int_253362)
    
    # Testing the type of an if condition (line 270)
    if_condition_253364 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 270, 8), result_eq_253363)
    # Assigning a type to the variable 'if_condition_253364' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'if_condition_253364', if_condition_253364)
    # SSA begins for if statement (line 270)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print_iteration_nonlinear(...): (line 271)
    # Processing the call arguments (line 271)
    # Getting the type of 'iteration' (line 271)
    iteration_253366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 38), 'iteration', False)
    # Getting the type of 'nfev' (line 271)
    nfev_253367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 49), 'nfev', False)
    # Getting the type of 'cost' (line 271)
    cost_253368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 55), 'cost', False)
    # Getting the type of 'actual_reduction' (line 271)
    actual_reduction_253369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 61), 'actual_reduction', False)
    # Getting the type of 'step_norm' (line 272)
    step_norm_253370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 38), 'step_norm', False)
    # Getting the type of 'g_norm' (line 272)
    g_norm_253371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 49), 'g_norm', False)
    # Processing the call keyword arguments (line 271)
    kwargs_253372 = {}
    # Getting the type of 'print_iteration_nonlinear' (line 271)
    print_iteration_nonlinear_253365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'print_iteration_nonlinear', False)
    # Calling print_iteration_nonlinear(args, kwargs) (line 271)
    print_iteration_nonlinear_call_result_253373 = invoke(stypy.reporting.localization.Localization(__file__, 271, 12), print_iteration_nonlinear_253365, *[iteration_253366, nfev_253367, cost_253368, actual_reduction_253369, step_norm_253370, g_norm_253371], **kwargs_253372)
    
    # SSA join for if statement (line 270)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'termination_status' (line 274)
    termination_status_253374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 11), 'termination_status')
    # Getting the type of 'None' (line 274)
    None_253375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 37), 'None')
    # Applying the binary operator 'isnot' (line 274)
    result_is_not_253376 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 11), 'isnot', termination_status_253374, None_253375)
    
    
    # Getting the type of 'nfev' (line 274)
    nfev_253377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 45), 'nfev')
    # Getting the type of 'max_nfev' (line 274)
    max_nfev_253378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 53), 'max_nfev')
    # Applying the binary operator '==' (line 274)
    result_eq_253379 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 45), '==', nfev_253377, max_nfev_253378)
    
    # Applying the binary operator 'or' (line 274)
    result_or_keyword_253380 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 11), 'or', result_is_not_253376, result_eq_253379)
    
    # Testing the type of an if condition (line 274)
    if_condition_253381 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 274, 8), result_or_keyword_253380)
    # Assigning a type to the variable 'if_condition_253381' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'if_condition_253381', if_condition_253381)
    # SSA begins for if statement (line 274)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 274)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'v' (line 286)
    v_253382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'v')
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'dv' (line 286)
    dv_253383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 10), 'dv')
    int_253384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 16), 'int')
    # Applying the binary operator '!=' (line 286)
    result_ne_253385 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 10), '!=', dv_253383, int_253384)
    
    # Getting the type of 'v' (line 286)
    v_253386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'v')
    # Obtaining the member '__getitem__' of a type (line 286)
    getitem___253387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 8), v_253386, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 286)
    subscript_call_result_253388 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), getitem___253387, result_ne_253385)
    
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'dv' (line 286)
    dv_253389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 32), 'dv')
    int_253390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 38), 'int')
    # Applying the binary operator '!=' (line 286)
    result_ne_253391 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 32), '!=', dv_253389, int_253390)
    
    # Getting the type of 'scale_inv' (line 286)
    scale_inv_253392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 22), 'scale_inv')
    # Obtaining the member '__getitem__' of a type (line 286)
    getitem___253393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 22), scale_inv_253392, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 286)
    subscript_call_result_253394 = invoke(stypy.reporting.localization.Localization(__file__, 286, 22), getitem___253393, result_ne_253391)
    
    # Applying the binary operator '*=' (line 286)
    result_imul_253395 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 8), '*=', subscript_call_result_253388, subscript_call_result_253394)
    # Getting the type of 'v' (line 286)
    v_253396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'v')
    
    # Getting the type of 'dv' (line 286)
    dv_253397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 10), 'dv')
    int_253398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 16), 'int')
    # Applying the binary operator '!=' (line 286)
    result_ne_253399 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 10), '!=', dv_253397, int_253398)
    
    # Storing an element on a container (line 286)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 8), v_253396, (result_ne_253399, result_imul_253395))
    
    
    # Assigning a BinOp to a Name (line 289):
    
    # Assigning a BinOp to a Name (line 289):
    # Getting the type of 'v' (line 289)
    v_253400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'v')
    float_253401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 15), 'float')
    # Applying the binary operator '**' (line 289)
    result_pow_253402 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 12), '**', v_253400, float_253401)
    
    # Getting the type of 'scale' (line 289)
    scale_253403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 21), 'scale')
    # Applying the binary operator '*' (line 289)
    result_mul_253404 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 12), '*', result_pow_253402, scale_253403)
    
    # Assigning a type to the variable 'd' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'd', result_mul_253404)
    
    # Assigning a BinOp to a Name (line 292):
    
    # Assigning a BinOp to a Name (line 292):
    # Getting the type of 'g' (line 292)
    g_253405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 17), 'g')
    # Getting the type of 'dv' (line 292)
    dv_253406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 21), 'dv')
    # Applying the binary operator '*' (line 292)
    result_mul_253407 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 17), '*', g_253405, dv_253406)
    
    # Getting the type of 'scale' (line 292)
    scale_253408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 26), 'scale')
    # Applying the binary operator '*' (line 292)
    result_mul_253409 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 24), '*', result_mul_253407, scale_253408)
    
    # Assigning a type to the variable 'diag_h' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'diag_h', result_mul_253409)
    
    # Assigning a BinOp to a Name (line 297):
    
    # Assigning a BinOp to a Name (line 297):
    # Getting the type of 'd' (line 297)
    d_253410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 14), 'd')
    # Getting the type of 'g' (line 297)
    g_253411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 18), 'g')
    # Applying the binary operator '*' (line 297)
    result_mul_253412 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 14), '*', d_253410, g_253411)
    
    # Assigning a type to the variable 'g_h' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'g_h', result_mul_253412)
    
    # Assigning a Name to a Subscript (line 299):
    
    # Assigning a Name to a Subscript (line 299):
    # Getting the type of 'f' (line 299)
    f_253413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 26), 'f')
    # Getting the type of 'f_augmented' (line 299)
    f_augmented_253414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'f_augmented')
    # Getting the type of 'm' (line 299)
    m_253415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 21), 'm')
    slice_253416 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 299, 8), None, m_253415, None)
    # Storing an element on a container (line 299)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 8), f_augmented_253414, (slice_253416, f_253413))
    
    
    # Getting the type of 'tr_solver' (line 300)
    tr_solver_253417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 11), 'tr_solver')
    str_253418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 24), 'str', 'exact')
    # Applying the binary operator '==' (line 300)
    result_eq_253419 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 11), '==', tr_solver_253417, str_253418)
    
    # Testing the type of an if condition (line 300)
    if_condition_253420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 300, 8), result_eq_253419)
    # Assigning a type to the variable 'if_condition_253420' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'if_condition_253420', if_condition_253420)
    # SSA begins for if statement (line 300)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 301):
    
    # Assigning a BinOp to a Subscript (line 301):
    # Getting the type of 'J' (line 301)
    J_253421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 30), 'J')
    # Getting the type of 'd' (line 301)
    d_253422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 34), 'd')
    # Applying the binary operator '*' (line 301)
    result_mul_253423 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 30), '*', J_253421, d_253422)
    
    # Getting the type of 'J_augmented' (line 301)
    J_augmented_253424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'J_augmented')
    # Getting the type of 'm' (line 301)
    m_253425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 25), 'm')
    slice_253426 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 301, 12), None, m_253425, None)
    # Storing an element on a container (line 301)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 12), J_augmented_253424, (slice_253426, result_mul_253423))
    
    # Assigning a Subscript to a Name (line 302):
    
    # Assigning a Subscript to a Name (line 302):
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 302)
    m_253427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 31), 'm')
    slice_253428 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 302, 18), None, m_253427, None)
    # Getting the type of 'J_augmented' (line 302)
    J_augmented_253429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 18), 'J_augmented')
    # Obtaining the member '__getitem__' of a type (line 302)
    getitem___253430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 18), J_augmented_253429, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 302)
    subscript_call_result_253431 = invoke(stypy.reporting.localization.Localization(__file__, 302, 18), getitem___253430, slice_253428)
    
    # Assigning a type to the variable 'J_h' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'J_h', subscript_call_result_253431)
    
    # Assigning a Call to a Subscript (line 303):
    
    # Assigning a Call to a Subscript (line 303):
    
    # Call to diag(...): (line 303)
    # Processing the call arguments (line 303)
    # Getting the type of 'diag_h' (line 303)
    diag_h_253434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 38), 'diag_h', False)
    float_253435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 46), 'float')
    # Applying the binary operator '**' (line 303)
    result_pow_253436 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 38), '**', diag_h_253434, float_253435)
    
    # Processing the call keyword arguments (line 303)
    kwargs_253437 = {}
    # Getting the type of 'np' (line 303)
    np_253432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 30), 'np', False)
    # Obtaining the member 'diag' of a type (line 303)
    diag_253433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 30), np_253432, 'diag')
    # Calling diag(args, kwargs) (line 303)
    diag_call_result_253438 = invoke(stypy.reporting.localization.Localization(__file__, 303, 30), diag_253433, *[result_pow_253436], **kwargs_253437)
    
    # Getting the type of 'J_augmented' (line 303)
    J_augmented_253439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'J_augmented')
    # Getting the type of 'm' (line 303)
    m_253440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 24), 'm')
    slice_253441 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 303, 12), m_253440, None, None)
    # Storing an element on a container (line 303)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 12), J_augmented_253439, (slice_253441, diag_call_result_253438))
    
    # Assigning a Call to a Tuple (line 304):
    
    # Assigning a Subscript to a Name (line 304):
    
    # Obtaining the type of the subscript
    int_253442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 12), 'int')
    
    # Call to svd(...): (line 304)
    # Processing the call arguments (line 304)
    # Getting the type of 'J_augmented' (line 304)
    J_augmented_253444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 26), 'J_augmented', False)
    # Processing the call keyword arguments (line 304)
    # Getting the type of 'False' (line 304)
    False_253445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 53), 'False', False)
    keyword_253446 = False_253445
    kwargs_253447 = {'full_matrices': keyword_253446}
    # Getting the type of 'svd' (line 304)
    svd_253443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 22), 'svd', False)
    # Calling svd(args, kwargs) (line 304)
    svd_call_result_253448 = invoke(stypy.reporting.localization.Localization(__file__, 304, 22), svd_253443, *[J_augmented_253444], **kwargs_253447)
    
    # Obtaining the member '__getitem__' of a type (line 304)
    getitem___253449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 12), svd_call_result_253448, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 304)
    subscript_call_result_253450 = invoke(stypy.reporting.localization.Localization(__file__, 304, 12), getitem___253449, int_253442)
    
    # Assigning a type to the variable 'tuple_var_assignment_252614' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'tuple_var_assignment_252614', subscript_call_result_253450)
    
    # Assigning a Subscript to a Name (line 304):
    
    # Obtaining the type of the subscript
    int_253451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 12), 'int')
    
    # Call to svd(...): (line 304)
    # Processing the call arguments (line 304)
    # Getting the type of 'J_augmented' (line 304)
    J_augmented_253453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 26), 'J_augmented', False)
    # Processing the call keyword arguments (line 304)
    # Getting the type of 'False' (line 304)
    False_253454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 53), 'False', False)
    keyword_253455 = False_253454
    kwargs_253456 = {'full_matrices': keyword_253455}
    # Getting the type of 'svd' (line 304)
    svd_253452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 22), 'svd', False)
    # Calling svd(args, kwargs) (line 304)
    svd_call_result_253457 = invoke(stypy.reporting.localization.Localization(__file__, 304, 22), svd_253452, *[J_augmented_253453], **kwargs_253456)
    
    # Obtaining the member '__getitem__' of a type (line 304)
    getitem___253458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 12), svd_call_result_253457, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 304)
    subscript_call_result_253459 = invoke(stypy.reporting.localization.Localization(__file__, 304, 12), getitem___253458, int_253451)
    
    # Assigning a type to the variable 'tuple_var_assignment_252615' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'tuple_var_assignment_252615', subscript_call_result_253459)
    
    # Assigning a Subscript to a Name (line 304):
    
    # Obtaining the type of the subscript
    int_253460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 12), 'int')
    
    # Call to svd(...): (line 304)
    # Processing the call arguments (line 304)
    # Getting the type of 'J_augmented' (line 304)
    J_augmented_253462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 26), 'J_augmented', False)
    # Processing the call keyword arguments (line 304)
    # Getting the type of 'False' (line 304)
    False_253463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 53), 'False', False)
    keyword_253464 = False_253463
    kwargs_253465 = {'full_matrices': keyword_253464}
    # Getting the type of 'svd' (line 304)
    svd_253461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 22), 'svd', False)
    # Calling svd(args, kwargs) (line 304)
    svd_call_result_253466 = invoke(stypy.reporting.localization.Localization(__file__, 304, 22), svd_253461, *[J_augmented_253462], **kwargs_253465)
    
    # Obtaining the member '__getitem__' of a type (line 304)
    getitem___253467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 12), svd_call_result_253466, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 304)
    subscript_call_result_253468 = invoke(stypy.reporting.localization.Localization(__file__, 304, 12), getitem___253467, int_253460)
    
    # Assigning a type to the variable 'tuple_var_assignment_252616' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'tuple_var_assignment_252616', subscript_call_result_253468)
    
    # Assigning a Name to a Name (line 304):
    # Getting the type of 'tuple_var_assignment_252614' (line 304)
    tuple_var_assignment_252614_253469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'tuple_var_assignment_252614')
    # Assigning a type to the variable 'U' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'U', tuple_var_assignment_252614_253469)
    
    # Assigning a Name to a Name (line 304):
    # Getting the type of 'tuple_var_assignment_252615' (line 304)
    tuple_var_assignment_252615_253470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'tuple_var_assignment_252615')
    # Assigning a type to the variable 's' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 15), 's', tuple_var_assignment_252615_253470)
    
    # Assigning a Name to a Name (line 304):
    # Getting the type of 'tuple_var_assignment_252616' (line 304)
    tuple_var_assignment_252616_253471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'tuple_var_assignment_252616')
    # Assigning a type to the variable 'V' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 18), 'V', tuple_var_assignment_252616_253471)
    
    # Assigning a Attribute to a Name (line 305):
    
    # Assigning a Attribute to a Name (line 305):
    # Getting the type of 'V' (line 305)
    V_253472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'V')
    # Obtaining the member 'T' of a type (line 305)
    T_253473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 16), V_253472, 'T')
    # Assigning a type to the variable 'V' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'V', T_253473)
    
    # Assigning a Call to a Name (line 306):
    
    # Assigning a Call to a Name (line 306):
    
    # Call to dot(...): (line 306)
    # Processing the call arguments (line 306)
    # Getting the type of 'f_augmented' (line 306)
    f_augmented_253477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 25), 'f_augmented', False)
    # Processing the call keyword arguments (line 306)
    kwargs_253478 = {}
    # Getting the type of 'U' (line 306)
    U_253474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 17), 'U', False)
    # Obtaining the member 'T' of a type (line 306)
    T_253475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 17), U_253474, 'T')
    # Obtaining the member 'dot' of a type (line 306)
    dot_253476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 17), T_253475, 'dot')
    # Calling dot(args, kwargs) (line 306)
    dot_call_result_253479 = invoke(stypy.reporting.localization.Localization(__file__, 306, 17), dot_253476, *[f_augmented_253477], **kwargs_253478)
    
    # Assigning a type to the variable 'uf' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'uf', dot_call_result_253479)
    # SSA branch for the else part of an if statement (line 300)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'tr_solver' (line 307)
    tr_solver_253480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 13), 'tr_solver')
    str_253481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 26), 'str', 'lsmr')
    # Applying the binary operator '==' (line 307)
    result_eq_253482 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 13), '==', tr_solver_253480, str_253481)
    
    # Testing the type of an if condition (line 307)
    if_condition_253483 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 307, 13), result_eq_253482)
    # Assigning a type to the variable 'if_condition_253483' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 13), 'if_condition_253483', if_condition_253483)
    # SSA begins for if statement (line 307)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 308):
    
    # Assigning a Call to a Name (line 308):
    
    # Call to right_multiplied_operator(...): (line 308)
    # Processing the call arguments (line 308)
    # Getting the type of 'J' (line 308)
    J_253485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 44), 'J', False)
    # Getting the type of 'd' (line 308)
    d_253486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 47), 'd', False)
    # Processing the call keyword arguments (line 308)
    kwargs_253487 = {}
    # Getting the type of 'right_multiplied_operator' (line 308)
    right_multiplied_operator_253484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 18), 'right_multiplied_operator', False)
    # Calling right_multiplied_operator(args, kwargs) (line 308)
    right_multiplied_operator_call_result_253488 = invoke(stypy.reporting.localization.Localization(__file__, 308, 18), right_multiplied_operator_253484, *[J_253485, d_253486], **kwargs_253487)
    
    # Assigning a type to the variable 'J_h' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'J_h', right_multiplied_operator_call_result_253488)
    
    # Getting the type of 'regularize' (line 310)
    regularize_253489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 15), 'regularize')
    # Testing the type of an if condition (line 310)
    if_condition_253490 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 310, 12), regularize_253489)
    # Assigning a type to the variable 'if_condition_253490' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'if_condition_253490', if_condition_253490)
    # SSA begins for if statement (line 310)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 311):
    
    # Assigning a Subscript to a Name (line 311):
    
    # Obtaining the type of the subscript
    int_253491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 16), 'int')
    
    # Call to build_quadratic_1d(...): (line 311)
    # Processing the call arguments (line 311)
    # Getting the type of 'J_h' (line 311)
    J_h_253493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 42), 'J_h', False)
    # Getting the type of 'g_h' (line 311)
    g_h_253494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 47), 'g_h', False)
    
    # Getting the type of 'g_h' (line 311)
    g_h_253495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 53), 'g_h', False)
    # Applying the 'usub' unary operator (line 311)
    result___neg___253496 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 52), 'usub', g_h_253495)
    
    # Processing the call keyword arguments (line 311)
    # Getting the type of 'diag_h' (line 311)
    diag_h_253497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 63), 'diag_h', False)
    keyword_253498 = diag_h_253497
    kwargs_253499 = {'diag': keyword_253498}
    # Getting the type of 'build_quadratic_1d' (line 311)
    build_quadratic_1d_253492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 23), 'build_quadratic_1d', False)
    # Calling build_quadratic_1d(args, kwargs) (line 311)
    build_quadratic_1d_call_result_253500 = invoke(stypy.reporting.localization.Localization(__file__, 311, 23), build_quadratic_1d_253492, *[J_h_253493, g_h_253494, result___neg___253496], **kwargs_253499)
    
    # Obtaining the member '__getitem__' of a type (line 311)
    getitem___253501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 16), build_quadratic_1d_call_result_253500, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 311)
    subscript_call_result_253502 = invoke(stypy.reporting.localization.Localization(__file__, 311, 16), getitem___253501, int_253491)
    
    # Assigning a type to the variable 'tuple_var_assignment_252617' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 16), 'tuple_var_assignment_252617', subscript_call_result_253502)
    
    # Assigning a Subscript to a Name (line 311):
    
    # Obtaining the type of the subscript
    int_253503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 16), 'int')
    
    # Call to build_quadratic_1d(...): (line 311)
    # Processing the call arguments (line 311)
    # Getting the type of 'J_h' (line 311)
    J_h_253505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 42), 'J_h', False)
    # Getting the type of 'g_h' (line 311)
    g_h_253506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 47), 'g_h', False)
    
    # Getting the type of 'g_h' (line 311)
    g_h_253507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 53), 'g_h', False)
    # Applying the 'usub' unary operator (line 311)
    result___neg___253508 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 52), 'usub', g_h_253507)
    
    # Processing the call keyword arguments (line 311)
    # Getting the type of 'diag_h' (line 311)
    diag_h_253509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 63), 'diag_h', False)
    keyword_253510 = diag_h_253509
    kwargs_253511 = {'diag': keyword_253510}
    # Getting the type of 'build_quadratic_1d' (line 311)
    build_quadratic_1d_253504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 23), 'build_quadratic_1d', False)
    # Calling build_quadratic_1d(args, kwargs) (line 311)
    build_quadratic_1d_call_result_253512 = invoke(stypy.reporting.localization.Localization(__file__, 311, 23), build_quadratic_1d_253504, *[J_h_253505, g_h_253506, result___neg___253508], **kwargs_253511)
    
    # Obtaining the member '__getitem__' of a type (line 311)
    getitem___253513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 16), build_quadratic_1d_call_result_253512, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 311)
    subscript_call_result_253514 = invoke(stypy.reporting.localization.Localization(__file__, 311, 16), getitem___253513, int_253503)
    
    # Assigning a type to the variable 'tuple_var_assignment_252618' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 16), 'tuple_var_assignment_252618', subscript_call_result_253514)
    
    # Assigning a Name to a Name (line 311):
    # Getting the type of 'tuple_var_assignment_252617' (line 311)
    tuple_var_assignment_252617_253515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 16), 'tuple_var_assignment_252617')
    # Assigning a type to the variable 'a' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 16), 'a', tuple_var_assignment_252617_253515)
    
    # Assigning a Name to a Name (line 311):
    # Getting the type of 'tuple_var_assignment_252618' (line 311)
    tuple_var_assignment_252618_253516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 16), 'tuple_var_assignment_252618')
    # Assigning a type to the variable 'b' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 19), 'b', tuple_var_assignment_252618_253516)
    
    # Assigning a BinOp to a Name (line 312):
    
    # Assigning a BinOp to a Name (line 312):
    # Getting the type of 'Delta' (line 312)
    Delta_253517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 24), 'Delta')
    
    # Call to norm(...): (line 312)
    # Processing the call arguments (line 312)
    # Getting the type of 'g_h' (line 312)
    g_h_253519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 37), 'g_h', False)
    # Processing the call keyword arguments (line 312)
    kwargs_253520 = {}
    # Getting the type of 'norm' (line 312)
    norm_253518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 32), 'norm', False)
    # Calling norm(args, kwargs) (line 312)
    norm_call_result_253521 = invoke(stypy.reporting.localization.Localization(__file__, 312, 32), norm_253518, *[g_h_253519], **kwargs_253520)
    
    # Applying the binary operator 'div' (line 312)
    result_div_253522 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 24), 'div', Delta_253517, norm_call_result_253521)
    
    # Assigning a type to the variable 'to_tr' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 16), 'to_tr', result_div_253522)
    
    # Assigning a Subscript to a Name (line 313):
    
    # Assigning a Subscript to a Name (line 313):
    
    # Obtaining the type of the subscript
    int_253523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 65), 'int')
    
    # Call to minimize_quadratic_1d(...): (line 313)
    # Processing the call arguments (line 313)
    # Getting the type of 'a' (line 313)
    a_253525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 49), 'a', False)
    # Getting the type of 'b' (line 313)
    b_253526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 52), 'b', False)
    int_253527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 55), 'int')
    # Getting the type of 'to_tr' (line 313)
    to_tr_253528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 58), 'to_tr', False)
    # Processing the call keyword arguments (line 313)
    kwargs_253529 = {}
    # Getting the type of 'minimize_quadratic_1d' (line 313)
    minimize_quadratic_1d_253524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 27), 'minimize_quadratic_1d', False)
    # Calling minimize_quadratic_1d(args, kwargs) (line 313)
    minimize_quadratic_1d_call_result_253530 = invoke(stypy.reporting.localization.Localization(__file__, 313, 27), minimize_quadratic_1d_253524, *[a_253525, b_253526, int_253527, to_tr_253528], **kwargs_253529)
    
    # Obtaining the member '__getitem__' of a type (line 313)
    getitem___253531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 27), minimize_quadratic_1d_call_result_253530, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 313)
    subscript_call_result_253532 = invoke(stypy.reporting.localization.Localization(__file__, 313, 27), getitem___253531, int_253523)
    
    # Assigning a type to the variable 'ag_value' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 16), 'ag_value', subscript_call_result_253532)
    
    # Assigning a BinOp to a Name (line 314):
    
    # Assigning a BinOp to a Name (line 314):
    
    # Getting the type of 'ag_value' (line 314)
    ag_value_253533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 28), 'ag_value')
    # Applying the 'usub' unary operator (line 314)
    result___neg___253534 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 27), 'usub', ag_value_253533)
    
    # Getting the type of 'Delta' (line 314)
    Delta_253535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 39), 'Delta')
    int_253536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 46), 'int')
    # Applying the binary operator '**' (line 314)
    result_pow_253537 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 39), '**', Delta_253535, int_253536)
    
    # Applying the binary operator 'div' (line 314)
    result_div_253538 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 27), 'div', result___neg___253534, result_pow_253537)
    
    # Assigning a type to the variable 'reg_term' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 16), 'reg_term', result_div_253538)
    # SSA join for if statement (line 310)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 316):
    
    # Assigning a Call to a Name (line 316):
    
    # Call to regularized_lsq_operator(...): (line 316)
    # Processing the call arguments (line 316)
    # Getting the type of 'J_h' (line 316)
    J_h_253540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 47), 'J_h', False)
    # Getting the type of 'diag_h' (line 316)
    diag_h_253541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 53), 'diag_h', False)
    # Getting the type of 'reg_term' (line 316)
    reg_term_253542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 62), 'reg_term', False)
    # Applying the binary operator '+' (line 316)
    result_add_253543 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 53), '+', diag_h_253541, reg_term_253542)
    
    float_253544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 73), 'float')
    # Applying the binary operator '**' (line 316)
    result_pow_253545 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 52), '**', result_add_253543, float_253544)
    
    # Processing the call keyword arguments (line 316)
    kwargs_253546 = {}
    # Getting the type of 'regularized_lsq_operator' (line 316)
    regularized_lsq_operator_253539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 22), 'regularized_lsq_operator', False)
    # Calling regularized_lsq_operator(args, kwargs) (line 316)
    regularized_lsq_operator_call_result_253547 = invoke(stypy.reporting.localization.Localization(__file__, 316, 22), regularized_lsq_operator_253539, *[J_h_253540, result_pow_253545], **kwargs_253546)
    
    # Assigning a type to the variable 'lsmr_op' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'lsmr_op', regularized_lsq_operator_call_result_253547)
    
    # Assigning a Subscript to a Name (line 317):
    
    # Assigning a Subscript to a Name (line 317):
    
    # Obtaining the type of the subscript
    int_253548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 60), 'int')
    
    # Call to lsmr(...): (line 317)
    # Processing the call arguments (line 317)
    # Getting the type of 'lsmr_op' (line 317)
    lsmr_op_253550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 24), 'lsmr_op', False)
    # Getting the type of 'f_augmented' (line 317)
    f_augmented_253551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 33), 'f_augmented', False)
    # Processing the call keyword arguments (line 317)
    # Getting the type of 'tr_options' (line 317)
    tr_options_253552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 48), 'tr_options', False)
    kwargs_253553 = {'tr_options_253552': tr_options_253552}
    # Getting the type of 'lsmr' (line 317)
    lsmr_253549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 19), 'lsmr', False)
    # Calling lsmr(args, kwargs) (line 317)
    lsmr_call_result_253554 = invoke(stypy.reporting.localization.Localization(__file__, 317, 19), lsmr_253549, *[lsmr_op_253550, f_augmented_253551], **kwargs_253553)
    
    # Obtaining the member '__getitem__' of a type (line 317)
    getitem___253555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 19), lsmr_call_result_253554, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 317)
    subscript_call_result_253556 = invoke(stypy.reporting.localization.Localization(__file__, 317, 19), getitem___253555, int_253548)
    
    # Assigning a type to the variable 'gn_h' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'gn_h', subscript_call_result_253556)
    
    # Assigning a Attribute to a Name (line 318):
    
    # Assigning a Attribute to a Name (line 318):
    
    # Call to vstack(...): (line 318)
    # Processing the call arguments (line 318)
    
    # Obtaining an instance of the builtin type 'tuple' (line 318)
    tuple_253559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 318)
    # Adding element type (line 318)
    # Getting the type of 'g_h' (line 318)
    g_h_253560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 27), 'g_h', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 27), tuple_253559, g_h_253560)
    # Adding element type (line 318)
    # Getting the type of 'gn_h' (line 318)
    gn_h_253561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 32), 'gn_h', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 27), tuple_253559, gn_h_253561)
    
    # Processing the call keyword arguments (line 318)
    kwargs_253562 = {}
    # Getting the type of 'np' (line 318)
    np_253557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 16), 'np', False)
    # Obtaining the member 'vstack' of a type (line 318)
    vstack_253558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 16), np_253557, 'vstack')
    # Calling vstack(args, kwargs) (line 318)
    vstack_call_result_253563 = invoke(stypy.reporting.localization.Localization(__file__, 318, 16), vstack_253558, *[tuple_253559], **kwargs_253562)
    
    # Obtaining the member 'T' of a type (line 318)
    T_253564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 16), vstack_call_result_253563, 'T')
    # Assigning a type to the variable 'S' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'S', T_253564)
    
    # Assigning a Call to a Tuple (line 319):
    
    # Assigning a Subscript to a Name (line 319):
    
    # Obtaining the type of the subscript
    int_253565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 12), 'int')
    
    # Call to qr(...): (line 319)
    # Processing the call arguments (line 319)
    # Getting the type of 'S' (line 319)
    S_253567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 22), 'S', False)
    # Processing the call keyword arguments (line 319)
    str_253568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 30), 'str', 'economic')
    keyword_253569 = str_253568
    kwargs_253570 = {'mode': keyword_253569}
    # Getting the type of 'qr' (line 319)
    qr_253566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 19), 'qr', False)
    # Calling qr(args, kwargs) (line 319)
    qr_call_result_253571 = invoke(stypy.reporting.localization.Localization(__file__, 319, 19), qr_253566, *[S_253567], **kwargs_253570)
    
    # Obtaining the member '__getitem__' of a type (line 319)
    getitem___253572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 12), qr_call_result_253571, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 319)
    subscript_call_result_253573 = invoke(stypy.reporting.localization.Localization(__file__, 319, 12), getitem___253572, int_253565)
    
    # Assigning a type to the variable 'tuple_var_assignment_252619' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'tuple_var_assignment_252619', subscript_call_result_253573)
    
    # Assigning a Subscript to a Name (line 319):
    
    # Obtaining the type of the subscript
    int_253574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 12), 'int')
    
    # Call to qr(...): (line 319)
    # Processing the call arguments (line 319)
    # Getting the type of 'S' (line 319)
    S_253576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 22), 'S', False)
    # Processing the call keyword arguments (line 319)
    str_253577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 30), 'str', 'economic')
    keyword_253578 = str_253577
    kwargs_253579 = {'mode': keyword_253578}
    # Getting the type of 'qr' (line 319)
    qr_253575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 19), 'qr', False)
    # Calling qr(args, kwargs) (line 319)
    qr_call_result_253580 = invoke(stypy.reporting.localization.Localization(__file__, 319, 19), qr_253575, *[S_253576], **kwargs_253579)
    
    # Obtaining the member '__getitem__' of a type (line 319)
    getitem___253581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 12), qr_call_result_253580, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 319)
    subscript_call_result_253582 = invoke(stypy.reporting.localization.Localization(__file__, 319, 12), getitem___253581, int_253574)
    
    # Assigning a type to the variable 'tuple_var_assignment_252620' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'tuple_var_assignment_252620', subscript_call_result_253582)
    
    # Assigning a Name to a Name (line 319):
    # Getting the type of 'tuple_var_assignment_252619' (line 319)
    tuple_var_assignment_252619_253583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'tuple_var_assignment_252619')
    # Assigning a type to the variable 'S' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'S', tuple_var_assignment_252619_253583)
    
    # Assigning a Name to a Name (line 319):
    # Getting the type of 'tuple_var_assignment_252620' (line 319)
    tuple_var_assignment_252620_253584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'tuple_var_assignment_252620')
    # Assigning a type to the variable '_' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 15), '_', tuple_var_assignment_252620_253584)
    
    # Assigning a Call to a Name (line 320):
    
    # Assigning a Call to a Name (line 320):
    
    # Call to dot(...): (line 320)
    # Processing the call arguments (line 320)
    # Getting the type of 'S' (line 320)
    S_253587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 25), 'S', False)
    # Processing the call keyword arguments (line 320)
    kwargs_253588 = {}
    # Getting the type of 'J_h' (line 320)
    J_h_253585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 17), 'J_h', False)
    # Obtaining the member 'dot' of a type (line 320)
    dot_253586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 17), J_h_253585, 'dot')
    # Calling dot(args, kwargs) (line 320)
    dot_call_result_253589 = invoke(stypy.reporting.localization.Localization(__file__, 320, 17), dot_253586, *[S_253587], **kwargs_253588)
    
    # Assigning a type to the variable 'JS' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'JS', dot_call_result_253589)
    
    # Assigning a BinOp to a Name (line 321):
    
    # Assigning a BinOp to a Name (line 321):
    
    # Call to dot(...): (line 321)
    # Processing the call arguments (line 321)
    # Getting the type of 'JS' (line 321)
    JS_253592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 25), 'JS', False)
    # Obtaining the member 'T' of a type (line 321)
    T_253593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 25), JS_253592, 'T')
    # Getting the type of 'JS' (line 321)
    JS_253594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 31), 'JS', False)
    # Processing the call keyword arguments (line 321)
    kwargs_253595 = {}
    # Getting the type of 'np' (line 321)
    np_253590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 18), 'np', False)
    # Obtaining the member 'dot' of a type (line 321)
    dot_253591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 18), np_253590, 'dot')
    # Calling dot(args, kwargs) (line 321)
    dot_call_result_253596 = invoke(stypy.reporting.localization.Localization(__file__, 321, 18), dot_253591, *[T_253593, JS_253594], **kwargs_253595)
    
    
    # Call to dot(...): (line 321)
    # Processing the call arguments (line 321)
    # Getting the type of 'S' (line 321)
    S_253599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 44), 'S', False)
    # Obtaining the member 'T' of a type (line 321)
    T_253600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 44), S_253599, 'T')
    # Getting the type of 'diag_h' (line 321)
    diag_h_253601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 50), 'diag_h', False)
    # Applying the binary operator '*' (line 321)
    result_mul_253602 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 44), '*', T_253600, diag_h_253601)
    
    # Getting the type of 'S' (line 321)
    S_253603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 58), 'S', False)
    # Processing the call keyword arguments (line 321)
    kwargs_253604 = {}
    # Getting the type of 'np' (line 321)
    np_253597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 37), 'np', False)
    # Obtaining the member 'dot' of a type (line 321)
    dot_253598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 37), np_253597, 'dot')
    # Calling dot(args, kwargs) (line 321)
    dot_call_result_253605 = invoke(stypy.reporting.localization.Localization(__file__, 321, 37), dot_253598, *[result_mul_253602, S_253603], **kwargs_253604)
    
    # Applying the binary operator '+' (line 321)
    result_add_253606 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 18), '+', dot_call_result_253596, dot_call_result_253605)
    
    # Assigning a type to the variable 'B_S' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'B_S', result_add_253606)
    
    # Assigning a Call to a Name (line 322):
    
    # Assigning a Call to a Name (line 322):
    
    # Call to dot(...): (line 322)
    # Processing the call arguments (line 322)
    # Getting the type of 'g_h' (line 322)
    g_h_253610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 26), 'g_h', False)
    # Processing the call keyword arguments (line 322)
    kwargs_253611 = {}
    # Getting the type of 'S' (line 322)
    S_253607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 18), 'S', False)
    # Obtaining the member 'T' of a type (line 322)
    T_253608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 18), S_253607, 'T')
    # Obtaining the member 'dot' of a type (line 322)
    dot_253609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 18), T_253608, 'dot')
    # Calling dot(args, kwargs) (line 322)
    dot_call_result_253612 = invoke(stypy.reporting.localization.Localization(__file__, 322, 18), dot_253609, *[g_h_253610], **kwargs_253611)
    
    # Assigning a type to the variable 'g_S' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'g_S', dot_call_result_253612)
    # SSA join for if statement (line 307)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 300)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 325):
    
    # Assigning a Call to a Name (line 325):
    
    # Call to max(...): (line 325)
    # Processing the call arguments (line 325)
    float_253614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 20), 'float')
    int_253615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 27), 'int')
    # Getting the type of 'g_norm' (line 325)
    g_norm_253616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 31), 'g_norm', False)
    # Applying the binary operator '-' (line 325)
    result_sub_253617 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 27), '-', int_253615, g_norm_253616)
    
    # Processing the call keyword arguments (line 325)
    kwargs_253618 = {}
    # Getting the type of 'max' (line 325)
    max_253613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 16), 'max', False)
    # Calling max(args, kwargs) (line 325)
    max_call_result_253619 = invoke(stypy.reporting.localization.Localization(__file__, 325, 16), max_253613, *[float_253614, result_sub_253617], **kwargs_253618)
    
    # Assigning a type to the variable 'theta' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'theta', max_call_result_253619)
    
    # Assigning a Num to a Name (line 327):
    
    # Assigning a Num to a Name (line 327):
    int_253620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 27), 'int')
    # Assigning a type to the variable 'actual_reduction' (line 327)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'actual_reduction', int_253620)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'actual_reduction' (line 328)
    actual_reduction_253621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 14), 'actual_reduction')
    int_253622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 34), 'int')
    # Applying the binary operator '<=' (line 328)
    result_le_253623 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 14), '<=', actual_reduction_253621, int_253622)
    
    
    # Getting the type of 'nfev' (line 328)
    nfev_253624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 40), 'nfev')
    # Getting the type of 'max_nfev' (line 328)
    max_nfev_253625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 47), 'max_nfev')
    # Applying the binary operator '<' (line 328)
    result_lt_253626 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 40), '<', nfev_253624, max_nfev_253625)
    
    # Applying the binary operator 'and' (line 328)
    result_and_keyword_253627 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 14), 'and', result_le_253623, result_lt_253626)
    
    # Testing the type of an if condition (line 328)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 328, 8), result_and_keyword_253627)
    # SSA begins for while statement (line 328)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    
    # Getting the type of 'tr_solver' (line 329)
    tr_solver_253628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 15), 'tr_solver')
    str_253629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 28), 'str', 'exact')
    # Applying the binary operator '==' (line 329)
    result_eq_253630 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 15), '==', tr_solver_253628, str_253629)
    
    # Testing the type of an if condition (line 329)
    if_condition_253631 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 329, 12), result_eq_253630)
    # Assigning a type to the variable 'if_condition_253631' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'if_condition_253631', if_condition_253631)
    # SSA begins for if statement (line 329)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 330):
    
    # Assigning a Subscript to a Name (line 330):
    
    # Obtaining the type of the subscript
    int_253632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 16), 'int')
    
    # Call to solve_lsq_trust_region(...): (line 330)
    # Processing the call arguments (line 330)
    # Getting the type of 'n' (line 331)
    n_253634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 20), 'n', False)
    # Getting the type of 'm' (line 331)
    m_253635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 23), 'm', False)
    # Getting the type of 'uf' (line 331)
    uf_253636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 26), 'uf', False)
    # Getting the type of 's' (line 331)
    s_253637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 30), 's', False)
    # Getting the type of 'V' (line 331)
    V_253638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 33), 'V', False)
    # Getting the type of 'Delta' (line 331)
    Delta_253639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 36), 'Delta', False)
    # Processing the call keyword arguments (line 330)
    # Getting the type of 'alpha' (line 331)
    alpha_253640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 57), 'alpha', False)
    keyword_253641 = alpha_253640
    kwargs_253642 = {'initial_alpha': keyword_253641}
    # Getting the type of 'solve_lsq_trust_region' (line 330)
    solve_lsq_trust_region_253633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 37), 'solve_lsq_trust_region', False)
    # Calling solve_lsq_trust_region(args, kwargs) (line 330)
    solve_lsq_trust_region_call_result_253643 = invoke(stypy.reporting.localization.Localization(__file__, 330, 37), solve_lsq_trust_region_253633, *[n_253634, m_253635, uf_253636, s_253637, V_253638, Delta_253639], **kwargs_253642)
    
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___253644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 16), solve_lsq_trust_region_call_result_253643, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 330)
    subscript_call_result_253645 = invoke(stypy.reporting.localization.Localization(__file__, 330, 16), getitem___253644, int_253632)
    
    # Assigning a type to the variable 'tuple_var_assignment_252621' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'tuple_var_assignment_252621', subscript_call_result_253645)
    
    # Assigning a Subscript to a Name (line 330):
    
    # Obtaining the type of the subscript
    int_253646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 16), 'int')
    
    # Call to solve_lsq_trust_region(...): (line 330)
    # Processing the call arguments (line 330)
    # Getting the type of 'n' (line 331)
    n_253648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 20), 'n', False)
    # Getting the type of 'm' (line 331)
    m_253649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 23), 'm', False)
    # Getting the type of 'uf' (line 331)
    uf_253650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 26), 'uf', False)
    # Getting the type of 's' (line 331)
    s_253651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 30), 's', False)
    # Getting the type of 'V' (line 331)
    V_253652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 33), 'V', False)
    # Getting the type of 'Delta' (line 331)
    Delta_253653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 36), 'Delta', False)
    # Processing the call keyword arguments (line 330)
    # Getting the type of 'alpha' (line 331)
    alpha_253654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 57), 'alpha', False)
    keyword_253655 = alpha_253654
    kwargs_253656 = {'initial_alpha': keyword_253655}
    # Getting the type of 'solve_lsq_trust_region' (line 330)
    solve_lsq_trust_region_253647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 37), 'solve_lsq_trust_region', False)
    # Calling solve_lsq_trust_region(args, kwargs) (line 330)
    solve_lsq_trust_region_call_result_253657 = invoke(stypy.reporting.localization.Localization(__file__, 330, 37), solve_lsq_trust_region_253647, *[n_253648, m_253649, uf_253650, s_253651, V_253652, Delta_253653], **kwargs_253656)
    
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___253658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 16), solve_lsq_trust_region_call_result_253657, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 330)
    subscript_call_result_253659 = invoke(stypy.reporting.localization.Localization(__file__, 330, 16), getitem___253658, int_253646)
    
    # Assigning a type to the variable 'tuple_var_assignment_252622' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'tuple_var_assignment_252622', subscript_call_result_253659)
    
    # Assigning a Subscript to a Name (line 330):
    
    # Obtaining the type of the subscript
    int_253660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 16), 'int')
    
    # Call to solve_lsq_trust_region(...): (line 330)
    # Processing the call arguments (line 330)
    # Getting the type of 'n' (line 331)
    n_253662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 20), 'n', False)
    # Getting the type of 'm' (line 331)
    m_253663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 23), 'm', False)
    # Getting the type of 'uf' (line 331)
    uf_253664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 26), 'uf', False)
    # Getting the type of 's' (line 331)
    s_253665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 30), 's', False)
    # Getting the type of 'V' (line 331)
    V_253666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 33), 'V', False)
    # Getting the type of 'Delta' (line 331)
    Delta_253667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 36), 'Delta', False)
    # Processing the call keyword arguments (line 330)
    # Getting the type of 'alpha' (line 331)
    alpha_253668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 57), 'alpha', False)
    keyword_253669 = alpha_253668
    kwargs_253670 = {'initial_alpha': keyword_253669}
    # Getting the type of 'solve_lsq_trust_region' (line 330)
    solve_lsq_trust_region_253661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 37), 'solve_lsq_trust_region', False)
    # Calling solve_lsq_trust_region(args, kwargs) (line 330)
    solve_lsq_trust_region_call_result_253671 = invoke(stypy.reporting.localization.Localization(__file__, 330, 37), solve_lsq_trust_region_253661, *[n_253662, m_253663, uf_253664, s_253665, V_253666, Delta_253667], **kwargs_253670)
    
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___253672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 16), solve_lsq_trust_region_call_result_253671, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 330)
    subscript_call_result_253673 = invoke(stypy.reporting.localization.Localization(__file__, 330, 16), getitem___253672, int_253660)
    
    # Assigning a type to the variable 'tuple_var_assignment_252623' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'tuple_var_assignment_252623', subscript_call_result_253673)
    
    # Assigning a Name to a Name (line 330):
    # Getting the type of 'tuple_var_assignment_252621' (line 330)
    tuple_var_assignment_252621_253674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'tuple_var_assignment_252621')
    # Assigning a type to the variable 'p_h' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'p_h', tuple_var_assignment_252621_253674)
    
    # Assigning a Name to a Name (line 330):
    # Getting the type of 'tuple_var_assignment_252622' (line 330)
    tuple_var_assignment_252622_253675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'tuple_var_assignment_252622')
    # Assigning a type to the variable 'alpha' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 21), 'alpha', tuple_var_assignment_252622_253675)
    
    # Assigning a Name to a Name (line 330):
    # Getting the type of 'tuple_var_assignment_252623' (line 330)
    tuple_var_assignment_252623_253676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'tuple_var_assignment_252623')
    # Assigning a type to the variable 'n_iter' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 28), 'n_iter', tuple_var_assignment_252623_253676)
    # SSA branch for the else part of an if statement (line 329)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'tr_solver' (line 332)
    tr_solver_253677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 17), 'tr_solver')
    str_253678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 30), 'str', 'lsmr')
    # Applying the binary operator '==' (line 332)
    result_eq_253679 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 17), '==', tr_solver_253677, str_253678)
    
    # Testing the type of an if condition (line 332)
    if_condition_253680 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 332, 17), result_eq_253679)
    # Assigning a type to the variable 'if_condition_253680' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 17), 'if_condition_253680', if_condition_253680)
    # SSA begins for if statement (line 332)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 333):
    
    # Assigning a Subscript to a Name (line 333):
    
    # Obtaining the type of the subscript
    int_253681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 16), 'int')
    
    # Call to solve_trust_region_2d(...): (line 333)
    # Processing the call arguments (line 333)
    # Getting the type of 'B_S' (line 333)
    B_S_253683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 47), 'B_S', False)
    # Getting the type of 'g_S' (line 333)
    g_S_253684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 52), 'g_S', False)
    # Getting the type of 'Delta' (line 333)
    Delta_253685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 57), 'Delta', False)
    # Processing the call keyword arguments (line 333)
    kwargs_253686 = {}
    # Getting the type of 'solve_trust_region_2d' (line 333)
    solve_trust_region_2d_253682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 25), 'solve_trust_region_2d', False)
    # Calling solve_trust_region_2d(args, kwargs) (line 333)
    solve_trust_region_2d_call_result_253687 = invoke(stypy.reporting.localization.Localization(__file__, 333, 25), solve_trust_region_2d_253682, *[B_S_253683, g_S_253684, Delta_253685], **kwargs_253686)
    
    # Obtaining the member '__getitem__' of a type (line 333)
    getitem___253688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 16), solve_trust_region_2d_call_result_253687, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 333)
    subscript_call_result_253689 = invoke(stypy.reporting.localization.Localization(__file__, 333, 16), getitem___253688, int_253681)
    
    # Assigning a type to the variable 'tuple_var_assignment_252624' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 16), 'tuple_var_assignment_252624', subscript_call_result_253689)
    
    # Assigning a Subscript to a Name (line 333):
    
    # Obtaining the type of the subscript
    int_253690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 16), 'int')
    
    # Call to solve_trust_region_2d(...): (line 333)
    # Processing the call arguments (line 333)
    # Getting the type of 'B_S' (line 333)
    B_S_253692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 47), 'B_S', False)
    # Getting the type of 'g_S' (line 333)
    g_S_253693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 52), 'g_S', False)
    # Getting the type of 'Delta' (line 333)
    Delta_253694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 57), 'Delta', False)
    # Processing the call keyword arguments (line 333)
    kwargs_253695 = {}
    # Getting the type of 'solve_trust_region_2d' (line 333)
    solve_trust_region_2d_253691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 25), 'solve_trust_region_2d', False)
    # Calling solve_trust_region_2d(args, kwargs) (line 333)
    solve_trust_region_2d_call_result_253696 = invoke(stypy.reporting.localization.Localization(__file__, 333, 25), solve_trust_region_2d_253691, *[B_S_253692, g_S_253693, Delta_253694], **kwargs_253695)
    
    # Obtaining the member '__getitem__' of a type (line 333)
    getitem___253697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 16), solve_trust_region_2d_call_result_253696, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 333)
    subscript_call_result_253698 = invoke(stypy.reporting.localization.Localization(__file__, 333, 16), getitem___253697, int_253690)
    
    # Assigning a type to the variable 'tuple_var_assignment_252625' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 16), 'tuple_var_assignment_252625', subscript_call_result_253698)
    
    # Assigning a Name to a Name (line 333):
    # Getting the type of 'tuple_var_assignment_252624' (line 333)
    tuple_var_assignment_252624_253699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 16), 'tuple_var_assignment_252624')
    # Assigning a type to the variable 'p_S' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 16), 'p_S', tuple_var_assignment_252624_253699)
    
    # Assigning a Name to a Name (line 333):
    # Getting the type of 'tuple_var_assignment_252625' (line 333)
    tuple_var_assignment_252625_253700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 16), 'tuple_var_assignment_252625')
    # Assigning a type to the variable '_' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 21), '_', tuple_var_assignment_252625_253700)
    
    # Assigning a Call to a Name (line 334):
    
    # Assigning a Call to a Name (line 334):
    
    # Call to dot(...): (line 334)
    # Processing the call arguments (line 334)
    # Getting the type of 'p_S' (line 334)
    p_S_253703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 28), 'p_S', False)
    # Processing the call keyword arguments (line 334)
    kwargs_253704 = {}
    # Getting the type of 'S' (line 334)
    S_253701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 22), 'S', False)
    # Obtaining the member 'dot' of a type (line 334)
    dot_253702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 22), S_253701, 'dot')
    # Calling dot(args, kwargs) (line 334)
    dot_call_result_253705 = invoke(stypy.reporting.localization.Localization(__file__, 334, 22), dot_253702, *[p_S_253703], **kwargs_253704)
    
    # Assigning a type to the variable 'p_h' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'p_h', dot_call_result_253705)
    # SSA join for if statement (line 332)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 329)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 336):
    
    # Assigning a BinOp to a Name (line 336):
    # Getting the type of 'd' (line 336)
    d_253706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 16), 'd')
    # Getting the type of 'p_h' (line 336)
    p_h_253707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 20), 'p_h')
    # Applying the binary operator '*' (line 336)
    result_mul_253708 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 16), '*', d_253706, p_h_253707)
    
    # Assigning a type to the variable 'p' (line 336)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'p', result_mul_253708)
    
    # Assigning a Call to a Tuple (line 337):
    
    # Assigning a Subscript to a Name (line 337):
    
    # Obtaining the type of the subscript
    int_253709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 12), 'int')
    
    # Call to select_step(...): (line 337)
    # Processing the call arguments (line 337)
    # Getting the type of 'x' (line 338)
    x_253711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 16), 'x', False)
    # Getting the type of 'J_h' (line 338)
    J_h_253712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 19), 'J_h', False)
    # Getting the type of 'diag_h' (line 338)
    diag_h_253713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 24), 'diag_h', False)
    # Getting the type of 'g_h' (line 338)
    g_h_253714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 32), 'g_h', False)
    # Getting the type of 'p' (line 338)
    p_253715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 37), 'p', False)
    # Getting the type of 'p_h' (line 338)
    p_h_253716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 40), 'p_h', False)
    # Getting the type of 'd' (line 338)
    d_253717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 45), 'd', False)
    # Getting the type of 'Delta' (line 338)
    Delta_253718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 48), 'Delta', False)
    # Getting the type of 'lb' (line 338)
    lb_253719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 55), 'lb', False)
    # Getting the type of 'ub' (line 338)
    ub_253720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 59), 'ub', False)
    # Getting the type of 'theta' (line 338)
    theta_253721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 63), 'theta', False)
    # Processing the call keyword arguments (line 337)
    kwargs_253722 = {}
    # Getting the type of 'select_step' (line 337)
    select_step_253710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 48), 'select_step', False)
    # Calling select_step(args, kwargs) (line 337)
    select_step_call_result_253723 = invoke(stypy.reporting.localization.Localization(__file__, 337, 48), select_step_253710, *[x_253711, J_h_253712, diag_h_253713, g_h_253714, p_253715, p_h_253716, d_253717, Delta_253718, lb_253719, ub_253720, theta_253721], **kwargs_253722)
    
    # Obtaining the member '__getitem__' of a type (line 337)
    getitem___253724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 12), select_step_call_result_253723, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 337)
    subscript_call_result_253725 = invoke(stypy.reporting.localization.Localization(__file__, 337, 12), getitem___253724, int_253709)
    
    # Assigning a type to the variable 'tuple_var_assignment_252626' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'tuple_var_assignment_252626', subscript_call_result_253725)
    
    # Assigning a Subscript to a Name (line 337):
    
    # Obtaining the type of the subscript
    int_253726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 12), 'int')
    
    # Call to select_step(...): (line 337)
    # Processing the call arguments (line 337)
    # Getting the type of 'x' (line 338)
    x_253728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 16), 'x', False)
    # Getting the type of 'J_h' (line 338)
    J_h_253729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 19), 'J_h', False)
    # Getting the type of 'diag_h' (line 338)
    diag_h_253730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 24), 'diag_h', False)
    # Getting the type of 'g_h' (line 338)
    g_h_253731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 32), 'g_h', False)
    # Getting the type of 'p' (line 338)
    p_253732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 37), 'p', False)
    # Getting the type of 'p_h' (line 338)
    p_h_253733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 40), 'p_h', False)
    # Getting the type of 'd' (line 338)
    d_253734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 45), 'd', False)
    # Getting the type of 'Delta' (line 338)
    Delta_253735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 48), 'Delta', False)
    # Getting the type of 'lb' (line 338)
    lb_253736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 55), 'lb', False)
    # Getting the type of 'ub' (line 338)
    ub_253737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 59), 'ub', False)
    # Getting the type of 'theta' (line 338)
    theta_253738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 63), 'theta', False)
    # Processing the call keyword arguments (line 337)
    kwargs_253739 = {}
    # Getting the type of 'select_step' (line 337)
    select_step_253727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 48), 'select_step', False)
    # Calling select_step(args, kwargs) (line 337)
    select_step_call_result_253740 = invoke(stypy.reporting.localization.Localization(__file__, 337, 48), select_step_253727, *[x_253728, J_h_253729, diag_h_253730, g_h_253731, p_253732, p_h_253733, d_253734, Delta_253735, lb_253736, ub_253737, theta_253738], **kwargs_253739)
    
    # Obtaining the member '__getitem__' of a type (line 337)
    getitem___253741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 12), select_step_call_result_253740, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 337)
    subscript_call_result_253742 = invoke(stypy.reporting.localization.Localization(__file__, 337, 12), getitem___253741, int_253726)
    
    # Assigning a type to the variable 'tuple_var_assignment_252627' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'tuple_var_assignment_252627', subscript_call_result_253742)
    
    # Assigning a Subscript to a Name (line 337):
    
    # Obtaining the type of the subscript
    int_253743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 12), 'int')
    
    # Call to select_step(...): (line 337)
    # Processing the call arguments (line 337)
    # Getting the type of 'x' (line 338)
    x_253745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 16), 'x', False)
    # Getting the type of 'J_h' (line 338)
    J_h_253746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 19), 'J_h', False)
    # Getting the type of 'diag_h' (line 338)
    diag_h_253747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 24), 'diag_h', False)
    # Getting the type of 'g_h' (line 338)
    g_h_253748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 32), 'g_h', False)
    # Getting the type of 'p' (line 338)
    p_253749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 37), 'p', False)
    # Getting the type of 'p_h' (line 338)
    p_h_253750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 40), 'p_h', False)
    # Getting the type of 'd' (line 338)
    d_253751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 45), 'd', False)
    # Getting the type of 'Delta' (line 338)
    Delta_253752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 48), 'Delta', False)
    # Getting the type of 'lb' (line 338)
    lb_253753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 55), 'lb', False)
    # Getting the type of 'ub' (line 338)
    ub_253754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 59), 'ub', False)
    # Getting the type of 'theta' (line 338)
    theta_253755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 63), 'theta', False)
    # Processing the call keyword arguments (line 337)
    kwargs_253756 = {}
    # Getting the type of 'select_step' (line 337)
    select_step_253744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 48), 'select_step', False)
    # Calling select_step(args, kwargs) (line 337)
    select_step_call_result_253757 = invoke(stypy.reporting.localization.Localization(__file__, 337, 48), select_step_253744, *[x_253745, J_h_253746, diag_h_253747, g_h_253748, p_253749, p_h_253750, d_253751, Delta_253752, lb_253753, ub_253754, theta_253755], **kwargs_253756)
    
    # Obtaining the member '__getitem__' of a type (line 337)
    getitem___253758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 12), select_step_call_result_253757, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 337)
    subscript_call_result_253759 = invoke(stypy.reporting.localization.Localization(__file__, 337, 12), getitem___253758, int_253743)
    
    # Assigning a type to the variable 'tuple_var_assignment_252628' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'tuple_var_assignment_252628', subscript_call_result_253759)
    
    # Assigning a Name to a Name (line 337):
    # Getting the type of 'tuple_var_assignment_252626' (line 337)
    tuple_var_assignment_252626_253760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'tuple_var_assignment_252626')
    # Assigning a type to the variable 'step' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'step', tuple_var_assignment_252626_253760)
    
    # Assigning a Name to a Name (line 337):
    # Getting the type of 'tuple_var_assignment_252627' (line 337)
    tuple_var_assignment_252627_253761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'tuple_var_assignment_252627')
    # Assigning a type to the variable 'step_h' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 18), 'step_h', tuple_var_assignment_252627_253761)
    
    # Assigning a Name to a Name (line 337):
    # Getting the type of 'tuple_var_assignment_252628' (line 337)
    tuple_var_assignment_252628_253762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'tuple_var_assignment_252628')
    # Assigning a type to the variable 'predicted_reduction' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 26), 'predicted_reduction', tuple_var_assignment_252628_253762)
    
    # Assigning a Call to a Name (line 340):
    
    # Assigning a Call to a Name (line 340):
    
    # Call to make_strictly_feasible(...): (line 340)
    # Processing the call arguments (line 340)
    # Getting the type of 'x' (line 340)
    x_253764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 43), 'x', False)
    # Getting the type of 'step' (line 340)
    step_253765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 47), 'step', False)
    # Applying the binary operator '+' (line 340)
    result_add_253766 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 43), '+', x_253764, step_253765)
    
    # Getting the type of 'lb' (line 340)
    lb_253767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 53), 'lb', False)
    # Getting the type of 'ub' (line 340)
    ub_253768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 57), 'ub', False)
    # Processing the call keyword arguments (line 340)
    int_253769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 67), 'int')
    keyword_253770 = int_253769
    kwargs_253771 = {'rstep': keyword_253770}
    # Getting the type of 'make_strictly_feasible' (line 340)
    make_strictly_feasible_253763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 20), 'make_strictly_feasible', False)
    # Calling make_strictly_feasible(args, kwargs) (line 340)
    make_strictly_feasible_call_result_253772 = invoke(stypy.reporting.localization.Localization(__file__, 340, 20), make_strictly_feasible_253763, *[result_add_253766, lb_253767, ub_253768], **kwargs_253771)
    
    # Assigning a type to the variable 'x_new' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 12), 'x_new', make_strictly_feasible_call_result_253772)
    
    # Assigning a Call to a Name (line 341):
    
    # Assigning a Call to a Name (line 341):
    
    # Call to fun(...): (line 341)
    # Processing the call arguments (line 341)
    # Getting the type of 'x_new' (line 341)
    x_new_253774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 24), 'x_new', False)
    # Processing the call keyword arguments (line 341)
    kwargs_253775 = {}
    # Getting the type of 'fun' (line 341)
    fun_253773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 20), 'fun', False)
    # Calling fun(args, kwargs) (line 341)
    fun_call_result_253776 = invoke(stypy.reporting.localization.Localization(__file__, 341, 20), fun_253773, *[x_new_253774], **kwargs_253775)
    
    # Assigning a type to the variable 'f_new' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'f_new', fun_call_result_253776)
    
    # Getting the type of 'nfev' (line 342)
    nfev_253777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'nfev')
    int_253778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 20), 'int')
    # Applying the binary operator '+=' (line 342)
    result_iadd_253779 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 12), '+=', nfev_253777, int_253778)
    # Assigning a type to the variable 'nfev' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'nfev', result_iadd_253779)
    
    
    # Assigning a Call to a Name (line 344):
    
    # Assigning a Call to a Name (line 344):
    
    # Call to norm(...): (line 344)
    # Processing the call arguments (line 344)
    # Getting the type of 'step_h' (line 344)
    step_h_253781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 31), 'step_h', False)
    # Processing the call keyword arguments (line 344)
    kwargs_253782 = {}
    # Getting the type of 'norm' (line 344)
    norm_253780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 26), 'norm', False)
    # Calling norm(args, kwargs) (line 344)
    norm_call_result_253783 = invoke(stypy.reporting.localization.Localization(__file__, 344, 26), norm_253780, *[step_h_253781], **kwargs_253782)
    
    # Assigning a type to the variable 'step_h_norm' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'step_h_norm', norm_call_result_253783)
    
    
    
    # Call to all(...): (line 346)
    # Processing the call arguments (line 346)
    
    # Call to isfinite(...): (line 346)
    # Processing the call arguments (line 346)
    # Getting the type of 'f_new' (line 346)
    f_new_253788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 38), 'f_new', False)
    # Processing the call keyword arguments (line 346)
    kwargs_253789 = {}
    # Getting the type of 'np' (line 346)
    np_253786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 26), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 346)
    isfinite_253787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 26), np_253786, 'isfinite')
    # Calling isfinite(args, kwargs) (line 346)
    isfinite_call_result_253790 = invoke(stypy.reporting.localization.Localization(__file__, 346, 26), isfinite_253787, *[f_new_253788], **kwargs_253789)
    
    # Processing the call keyword arguments (line 346)
    kwargs_253791 = {}
    # Getting the type of 'np' (line 346)
    np_253784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 19), 'np', False)
    # Obtaining the member 'all' of a type (line 346)
    all_253785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 19), np_253784, 'all')
    # Calling all(args, kwargs) (line 346)
    all_call_result_253792 = invoke(stypy.reporting.localization.Localization(__file__, 346, 19), all_253785, *[isfinite_call_result_253790], **kwargs_253791)
    
    # Applying the 'not' unary operator (line 346)
    result_not__253793 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 15), 'not', all_call_result_253792)
    
    # Testing the type of an if condition (line 346)
    if_condition_253794 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 346, 12), result_not__253793)
    # Assigning a type to the variable 'if_condition_253794' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 12), 'if_condition_253794', if_condition_253794)
    # SSA begins for if statement (line 346)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 347):
    
    # Assigning a BinOp to a Name (line 347):
    float_253795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 24), 'float')
    # Getting the type of 'step_h_norm' (line 347)
    step_h_norm_253796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 31), 'step_h_norm')
    # Applying the binary operator '*' (line 347)
    result_mul_253797 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 24), '*', float_253795, step_h_norm_253796)
    
    # Assigning a type to the variable 'Delta' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 16), 'Delta', result_mul_253797)
    # SSA join for if statement (line 346)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 351)
    # Getting the type of 'loss_function' (line 351)
    loss_function_253798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'loss_function')
    # Getting the type of 'None' (line 351)
    None_253799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 36), 'None')
    
    (may_be_253800, more_types_in_union_253801) = may_not_be_none(loss_function_253798, None_253799)

    if may_be_253800:

        if more_types_in_union_253801:
            # Runtime conditional SSA (line 351)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 352):
        
        # Assigning a Call to a Name (line 352):
        
        # Call to loss_function(...): (line 352)
        # Processing the call arguments (line 352)
        # Getting the type of 'f_new' (line 352)
        f_new_253803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 41), 'f_new', False)
        # Processing the call keyword arguments (line 352)
        # Getting the type of 'True' (line 352)
        True_253804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 58), 'True', False)
        keyword_253805 = True_253804
        kwargs_253806 = {'cost_only': keyword_253805}
        # Getting the type of 'loss_function' (line 352)
        loss_function_253802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 27), 'loss_function', False)
        # Calling loss_function(args, kwargs) (line 352)
        loss_function_call_result_253807 = invoke(stypy.reporting.localization.Localization(__file__, 352, 27), loss_function_253802, *[f_new_253803], **kwargs_253806)
        
        # Assigning a type to the variable 'cost_new' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 16), 'cost_new', loss_function_call_result_253807)

        if more_types_in_union_253801:
            # Runtime conditional SSA for else branch (line 351)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_253800) or more_types_in_union_253801):
        
        # Assigning a BinOp to a Name (line 354):
        
        # Assigning a BinOp to a Name (line 354):
        float_253808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 27), 'float')
        
        # Call to dot(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 'f_new' (line 354)
        f_new_253811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 40), 'f_new', False)
        # Getting the type of 'f_new' (line 354)
        f_new_253812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 47), 'f_new', False)
        # Processing the call keyword arguments (line 354)
        kwargs_253813 = {}
        # Getting the type of 'np' (line 354)
        np_253809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 33), 'np', False)
        # Obtaining the member 'dot' of a type (line 354)
        dot_253810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 33), np_253809, 'dot')
        # Calling dot(args, kwargs) (line 354)
        dot_call_result_253814 = invoke(stypy.reporting.localization.Localization(__file__, 354, 33), dot_253810, *[f_new_253811, f_new_253812], **kwargs_253813)
        
        # Applying the binary operator '*' (line 354)
        result_mul_253815 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 27), '*', float_253808, dot_call_result_253814)
        
        # Assigning a type to the variable 'cost_new' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 16), 'cost_new', result_mul_253815)

        if (may_be_253800 and more_types_in_union_253801):
            # SSA join for if statement (line 351)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 355):
    
    # Assigning a BinOp to a Name (line 355):
    # Getting the type of 'cost' (line 355)
    cost_253816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 31), 'cost')
    # Getting the type of 'cost_new' (line 355)
    cost_new_253817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 38), 'cost_new')
    # Applying the binary operator '-' (line 355)
    result_sub_253818 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 31), '-', cost_253816, cost_new_253817)
    
    # Assigning a type to the variable 'actual_reduction' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'actual_reduction', result_sub_253818)
    
    # Assigning a BinOp to a Name (line 358):
    
    # Assigning a BinOp to a Name (line 358):
    float_253819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 25), 'float')
    
    # Call to dot(...): (line 358)
    # Processing the call arguments (line 358)
    # Getting the type of 'step_h' (line 358)
    step_h_253822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 38), 'step_h', False)
    # Getting the type of 'diag_h' (line 358)
    diag_h_253823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 47), 'diag_h', False)
    # Applying the binary operator '*' (line 358)
    result_mul_253824 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 38), '*', step_h_253822, diag_h_253823)
    
    # Getting the type of 'step_h' (line 358)
    step_h_253825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 55), 'step_h', False)
    # Processing the call keyword arguments (line 358)
    kwargs_253826 = {}
    # Getting the type of 'np' (line 358)
    np_253820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 31), 'np', False)
    # Obtaining the member 'dot' of a type (line 358)
    dot_253821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 31), np_253820, 'dot')
    # Calling dot(args, kwargs) (line 358)
    dot_call_result_253827 = invoke(stypy.reporting.localization.Localization(__file__, 358, 31), dot_253821, *[result_mul_253824, step_h_253825], **kwargs_253826)
    
    # Applying the binary operator '*' (line 358)
    result_mul_253828 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 25), '*', float_253819, dot_call_result_253827)
    
    # Assigning a type to the variable 'correction' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'correction', result_mul_253828)
    
    # Assigning a Call to a Tuple (line 360):
    
    # Assigning a Subscript to a Name (line 360):
    
    # Obtaining the type of the subscript
    int_253829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 12), 'int')
    
    # Call to update_tr_radius(...): (line 360)
    # Processing the call arguments (line 360)
    # Getting the type of 'Delta' (line 361)
    Delta_253831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 16), 'Delta', False)
    # Getting the type of 'actual_reduction' (line 361)
    actual_reduction_253832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 23), 'actual_reduction', False)
    # Getting the type of 'correction' (line 361)
    correction_253833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 42), 'correction', False)
    # Applying the binary operator '-' (line 361)
    result_sub_253834 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 23), '-', actual_reduction_253832, correction_253833)
    
    # Getting the type of 'predicted_reduction' (line 361)
    predicted_reduction_253835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 54), 'predicted_reduction', False)
    # Getting the type of 'step_h_norm' (line 362)
    step_h_norm_253836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 16), 'step_h_norm', False)
    
    # Getting the type of 'step_h_norm' (line 362)
    step_h_norm_253837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 29), 'step_h_norm', False)
    float_253838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 43), 'float')
    # Getting the type of 'Delta' (line 362)
    Delta_253839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 50), 'Delta', False)
    # Applying the binary operator '*' (line 362)
    result_mul_253840 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 43), '*', float_253838, Delta_253839)
    
    # Applying the binary operator '>' (line 362)
    result_gt_253841 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 29), '>', step_h_norm_253837, result_mul_253840)
    
    # Processing the call keyword arguments (line 360)
    kwargs_253842 = {}
    # Getting the type of 'update_tr_radius' (line 360)
    update_tr_radius_253830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 31), 'update_tr_radius', False)
    # Calling update_tr_radius(args, kwargs) (line 360)
    update_tr_radius_call_result_253843 = invoke(stypy.reporting.localization.Localization(__file__, 360, 31), update_tr_radius_253830, *[Delta_253831, result_sub_253834, predicted_reduction_253835, step_h_norm_253836, result_gt_253841], **kwargs_253842)
    
    # Obtaining the member '__getitem__' of a type (line 360)
    getitem___253844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 12), update_tr_radius_call_result_253843, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 360)
    subscript_call_result_253845 = invoke(stypy.reporting.localization.Localization(__file__, 360, 12), getitem___253844, int_253829)
    
    # Assigning a type to the variable 'tuple_var_assignment_252629' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'tuple_var_assignment_252629', subscript_call_result_253845)
    
    # Assigning a Subscript to a Name (line 360):
    
    # Obtaining the type of the subscript
    int_253846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 12), 'int')
    
    # Call to update_tr_radius(...): (line 360)
    # Processing the call arguments (line 360)
    # Getting the type of 'Delta' (line 361)
    Delta_253848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 16), 'Delta', False)
    # Getting the type of 'actual_reduction' (line 361)
    actual_reduction_253849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 23), 'actual_reduction', False)
    # Getting the type of 'correction' (line 361)
    correction_253850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 42), 'correction', False)
    # Applying the binary operator '-' (line 361)
    result_sub_253851 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 23), '-', actual_reduction_253849, correction_253850)
    
    # Getting the type of 'predicted_reduction' (line 361)
    predicted_reduction_253852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 54), 'predicted_reduction', False)
    # Getting the type of 'step_h_norm' (line 362)
    step_h_norm_253853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 16), 'step_h_norm', False)
    
    # Getting the type of 'step_h_norm' (line 362)
    step_h_norm_253854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 29), 'step_h_norm', False)
    float_253855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 43), 'float')
    # Getting the type of 'Delta' (line 362)
    Delta_253856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 50), 'Delta', False)
    # Applying the binary operator '*' (line 362)
    result_mul_253857 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 43), '*', float_253855, Delta_253856)
    
    # Applying the binary operator '>' (line 362)
    result_gt_253858 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 29), '>', step_h_norm_253854, result_mul_253857)
    
    # Processing the call keyword arguments (line 360)
    kwargs_253859 = {}
    # Getting the type of 'update_tr_radius' (line 360)
    update_tr_radius_253847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 31), 'update_tr_radius', False)
    # Calling update_tr_radius(args, kwargs) (line 360)
    update_tr_radius_call_result_253860 = invoke(stypy.reporting.localization.Localization(__file__, 360, 31), update_tr_radius_253847, *[Delta_253848, result_sub_253851, predicted_reduction_253852, step_h_norm_253853, result_gt_253858], **kwargs_253859)
    
    # Obtaining the member '__getitem__' of a type (line 360)
    getitem___253861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 12), update_tr_radius_call_result_253860, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 360)
    subscript_call_result_253862 = invoke(stypy.reporting.localization.Localization(__file__, 360, 12), getitem___253861, int_253846)
    
    # Assigning a type to the variable 'tuple_var_assignment_252630' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'tuple_var_assignment_252630', subscript_call_result_253862)
    
    # Assigning a Name to a Name (line 360):
    # Getting the type of 'tuple_var_assignment_252629' (line 360)
    tuple_var_assignment_252629_253863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'tuple_var_assignment_252629')
    # Assigning a type to the variable 'Delta_new' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'Delta_new', tuple_var_assignment_252629_253863)
    
    # Assigning a Name to a Name (line 360):
    # Getting the type of 'tuple_var_assignment_252630' (line 360)
    tuple_var_assignment_252630_253864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'tuple_var_assignment_252630')
    # Assigning a type to the variable 'ratio' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 23), 'ratio', tuple_var_assignment_252630_253864)
    
    # Getting the type of 'alpha' (line 364)
    alpha_253865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 12), 'alpha')
    # Getting the type of 'Delta' (line 364)
    Delta_253866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 21), 'Delta')
    # Getting the type of 'Delta_new' (line 364)
    Delta_new_253867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 29), 'Delta_new')
    # Applying the binary operator 'div' (line 364)
    result_div_253868 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 21), 'div', Delta_253866, Delta_new_253867)
    
    # Applying the binary operator '*=' (line 364)
    result_imul_253869 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 12), '*=', alpha_253865, result_div_253868)
    # Assigning a type to the variable 'alpha' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 12), 'alpha', result_imul_253869)
    
    
    # Assigning a Name to a Name (line 365):
    
    # Assigning a Name to a Name (line 365):
    # Getting the type of 'Delta_new' (line 365)
    Delta_new_253870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 20), 'Delta_new')
    # Assigning a type to the variable 'Delta' (line 365)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'Delta', Delta_new_253870)
    
    # Assigning a Call to a Name (line 367):
    
    # Assigning a Call to a Name (line 367):
    
    # Call to norm(...): (line 367)
    # Processing the call arguments (line 367)
    # Getting the type of 'step' (line 367)
    step_253872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 29), 'step', False)
    # Processing the call keyword arguments (line 367)
    kwargs_253873 = {}
    # Getting the type of 'norm' (line 367)
    norm_253871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 24), 'norm', False)
    # Calling norm(args, kwargs) (line 367)
    norm_call_result_253874 = invoke(stypy.reporting.localization.Localization(__file__, 367, 24), norm_253871, *[step_253872], **kwargs_253873)
    
    # Assigning a type to the variable 'step_norm' (line 367)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'step_norm', norm_call_result_253874)
    
    # Assigning a Call to a Name (line 368):
    
    # Assigning a Call to a Name (line 368):
    
    # Call to check_termination(...): (line 368)
    # Processing the call arguments (line 368)
    # Getting the type of 'actual_reduction' (line 369)
    actual_reduction_253876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 16), 'actual_reduction', False)
    # Getting the type of 'cost' (line 369)
    cost_253877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 34), 'cost', False)
    # Getting the type of 'step_norm' (line 369)
    step_norm_253878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 40), 'step_norm', False)
    
    # Call to norm(...): (line 369)
    # Processing the call arguments (line 369)
    # Getting the type of 'x' (line 369)
    x_253880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 56), 'x', False)
    # Processing the call keyword arguments (line 369)
    kwargs_253881 = {}
    # Getting the type of 'norm' (line 369)
    norm_253879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 51), 'norm', False)
    # Calling norm(args, kwargs) (line 369)
    norm_call_result_253882 = invoke(stypy.reporting.localization.Localization(__file__, 369, 51), norm_253879, *[x_253880], **kwargs_253881)
    
    # Getting the type of 'ratio' (line 369)
    ratio_253883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 60), 'ratio', False)
    # Getting the type of 'ftol' (line 369)
    ftol_253884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 67), 'ftol', False)
    # Getting the type of 'xtol' (line 369)
    xtol_253885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 73), 'xtol', False)
    # Processing the call keyword arguments (line 368)
    kwargs_253886 = {}
    # Getting the type of 'check_termination' (line 368)
    check_termination_253875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 33), 'check_termination', False)
    # Calling check_termination(args, kwargs) (line 368)
    check_termination_call_result_253887 = invoke(stypy.reporting.localization.Localization(__file__, 368, 33), check_termination_253875, *[actual_reduction_253876, cost_253877, step_norm_253878, norm_call_result_253882, ratio_253883, ftol_253884, xtol_253885], **kwargs_253886)
    
    # Assigning a type to the variable 'termination_status' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'termination_status', check_termination_call_result_253887)
    
    # Type idiom detected: calculating its left and rigth part (line 371)
    # Getting the type of 'termination_status' (line 371)
    termination_status_253888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 12), 'termination_status')
    # Getting the type of 'None' (line 371)
    None_253889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 41), 'None')
    
    (may_be_253890, more_types_in_union_253891) = may_not_be_none(termination_status_253888, None_253889)

    if may_be_253890:

        if more_types_in_union_253891:
            # Runtime conditional SSA (line 371)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        if more_types_in_union_253891:
            # SSA join for if statement (line 371)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for while statement (line 328)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'actual_reduction' (line 374)
    actual_reduction_253892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 11), 'actual_reduction')
    int_253893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 30), 'int')
    # Applying the binary operator '>' (line 374)
    result_gt_253894 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 11), '>', actual_reduction_253892, int_253893)
    
    # Testing the type of an if condition (line 374)
    if_condition_253895 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 374, 8), result_gt_253894)
    # Assigning a type to the variable 'if_condition_253895' (line 374)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'if_condition_253895', if_condition_253895)
    # SSA begins for if statement (line 374)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 375):
    
    # Assigning a Name to a Name (line 375):
    # Getting the type of 'x_new' (line 375)
    x_new_253896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 16), 'x_new')
    # Assigning a type to the variable 'x' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 12), 'x', x_new_253896)
    
    # Assigning a Name to a Name (line 377):
    
    # Assigning a Name to a Name (line 377):
    # Getting the type of 'f_new' (line 377)
    f_new_253897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 16), 'f_new')
    # Assigning a type to the variable 'f' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 12), 'f', f_new_253897)
    
    # Assigning a Call to a Name (line 378):
    
    # Assigning a Call to a Name (line 378):
    
    # Call to copy(...): (line 378)
    # Processing the call keyword arguments (line 378)
    kwargs_253900 = {}
    # Getting the type of 'f' (line 378)
    f_253898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 21), 'f', False)
    # Obtaining the member 'copy' of a type (line 378)
    copy_253899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 21), f_253898, 'copy')
    # Calling copy(args, kwargs) (line 378)
    copy_call_result_253901 = invoke(stypy.reporting.localization.Localization(__file__, 378, 21), copy_253899, *[], **kwargs_253900)
    
    # Assigning a type to the variable 'f_true' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 12), 'f_true', copy_call_result_253901)
    
    # Assigning a Name to a Name (line 380):
    
    # Assigning a Name to a Name (line 380):
    # Getting the type of 'cost_new' (line 380)
    cost_new_253902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 19), 'cost_new')
    # Assigning a type to the variable 'cost' (line 380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 12), 'cost', cost_new_253902)
    
    # Assigning a Call to a Name (line 382):
    
    # Assigning a Call to a Name (line 382):
    
    # Call to jac(...): (line 382)
    # Processing the call arguments (line 382)
    # Getting the type of 'x' (line 382)
    x_253904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 20), 'x', False)
    # Getting the type of 'f' (line 382)
    f_253905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 23), 'f', False)
    # Processing the call keyword arguments (line 382)
    kwargs_253906 = {}
    # Getting the type of 'jac' (line 382)
    jac_253903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 16), 'jac', False)
    # Calling jac(args, kwargs) (line 382)
    jac_call_result_253907 = invoke(stypy.reporting.localization.Localization(__file__, 382, 16), jac_253903, *[x_253904, f_253905], **kwargs_253906)
    
    # Assigning a type to the variable 'J' (line 382)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'J', jac_call_result_253907)
    
    # Getting the type of 'njev' (line 383)
    njev_253908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'njev')
    int_253909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 20), 'int')
    # Applying the binary operator '+=' (line 383)
    result_iadd_253910 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 12), '+=', njev_253908, int_253909)
    # Assigning a type to the variable 'njev' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'njev', result_iadd_253910)
    
    
    # Type idiom detected: calculating its left and rigth part (line 385)
    # Getting the type of 'loss_function' (line 385)
    loss_function_253911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'loss_function')
    # Getting the type of 'None' (line 385)
    None_253912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 36), 'None')
    
    (may_be_253913, more_types_in_union_253914) = may_not_be_none(loss_function_253911, None_253912)

    if may_be_253913:

        if more_types_in_union_253914:
            # Runtime conditional SSA (line 385)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 386):
        
        # Assigning a Call to a Name (line 386):
        
        # Call to loss_function(...): (line 386)
        # Processing the call arguments (line 386)
        # Getting the type of 'f' (line 386)
        f_253916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 36), 'f', False)
        # Processing the call keyword arguments (line 386)
        kwargs_253917 = {}
        # Getting the type of 'loss_function' (line 386)
        loss_function_253915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 22), 'loss_function', False)
        # Calling loss_function(args, kwargs) (line 386)
        loss_function_call_result_253918 = invoke(stypy.reporting.localization.Localization(__file__, 386, 22), loss_function_253915, *[f_253916], **kwargs_253917)
        
        # Assigning a type to the variable 'rho' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 16), 'rho', loss_function_call_result_253918)
        
        # Assigning a Call to a Tuple (line 387):
        
        # Assigning a Subscript to a Name (line 387):
        
        # Obtaining the type of the subscript
        int_253919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 16), 'int')
        
        # Call to scale_for_robust_loss_function(...): (line 387)
        # Processing the call arguments (line 387)
        # Getting the type of 'J' (line 387)
        J_253921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 54), 'J', False)
        # Getting the type of 'f' (line 387)
        f_253922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 57), 'f', False)
        # Getting the type of 'rho' (line 387)
        rho_253923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 60), 'rho', False)
        # Processing the call keyword arguments (line 387)
        kwargs_253924 = {}
        # Getting the type of 'scale_for_robust_loss_function' (line 387)
        scale_for_robust_loss_function_253920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 23), 'scale_for_robust_loss_function', False)
        # Calling scale_for_robust_loss_function(args, kwargs) (line 387)
        scale_for_robust_loss_function_call_result_253925 = invoke(stypy.reporting.localization.Localization(__file__, 387, 23), scale_for_robust_loss_function_253920, *[J_253921, f_253922, rho_253923], **kwargs_253924)
        
        # Obtaining the member '__getitem__' of a type (line 387)
        getitem___253926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 16), scale_for_robust_loss_function_call_result_253925, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 387)
        subscript_call_result_253927 = invoke(stypy.reporting.localization.Localization(__file__, 387, 16), getitem___253926, int_253919)
        
        # Assigning a type to the variable 'tuple_var_assignment_252631' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 16), 'tuple_var_assignment_252631', subscript_call_result_253927)
        
        # Assigning a Subscript to a Name (line 387):
        
        # Obtaining the type of the subscript
        int_253928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 16), 'int')
        
        # Call to scale_for_robust_loss_function(...): (line 387)
        # Processing the call arguments (line 387)
        # Getting the type of 'J' (line 387)
        J_253930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 54), 'J', False)
        # Getting the type of 'f' (line 387)
        f_253931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 57), 'f', False)
        # Getting the type of 'rho' (line 387)
        rho_253932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 60), 'rho', False)
        # Processing the call keyword arguments (line 387)
        kwargs_253933 = {}
        # Getting the type of 'scale_for_robust_loss_function' (line 387)
        scale_for_robust_loss_function_253929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 23), 'scale_for_robust_loss_function', False)
        # Calling scale_for_robust_loss_function(args, kwargs) (line 387)
        scale_for_robust_loss_function_call_result_253934 = invoke(stypy.reporting.localization.Localization(__file__, 387, 23), scale_for_robust_loss_function_253929, *[J_253930, f_253931, rho_253932], **kwargs_253933)
        
        # Obtaining the member '__getitem__' of a type (line 387)
        getitem___253935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 16), scale_for_robust_loss_function_call_result_253934, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 387)
        subscript_call_result_253936 = invoke(stypy.reporting.localization.Localization(__file__, 387, 16), getitem___253935, int_253928)
        
        # Assigning a type to the variable 'tuple_var_assignment_252632' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 16), 'tuple_var_assignment_252632', subscript_call_result_253936)
        
        # Assigning a Name to a Name (line 387):
        # Getting the type of 'tuple_var_assignment_252631' (line 387)
        tuple_var_assignment_252631_253937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 16), 'tuple_var_assignment_252631')
        # Assigning a type to the variable 'J' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 16), 'J', tuple_var_assignment_252631_253937)
        
        # Assigning a Name to a Name (line 387):
        # Getting the type of 'tuple_var_assignment_252632' (line 387)
        tuple_var_assignment_252632_253938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 16), 'tuple_var_assignment_252632')
        # Assigning a type to the variable 'f' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 19), 'f', tuple_var_assignment_252632_253938)

        if more_types_in_union_253914:
            # SSA join for if statement (line 385)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 389):
    
    # Assigning a Call to a Name (line 389):
    
    # Call to compute_grad(...): (line 389)
    # Processing the call arguments (line 389)
    # Getting the type of 'J' (line 389)
    J_253940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 29), 'J', False)
    # Getting the type of 'f' (line 389)
    f_253941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 32), 'f', False)
    # Processing the call keyword arguments (line 389)
    kwargs_253942 = {}
    # Getting the type of 'compute_grad' (line 389)
    compute_grad_253939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 16), 'compute_grad', False)
    # Calling compute_grad(args, kwargs) (line 389)
    compute_grad_call_result_253943 = invoke(stypy.reporting.localization.Localization(__file__, 389, 16), compute_grad_253939, *[J_253940, f_253941], **kwargs_253942)
    
    # Assigning a type to the variable 'g' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'g', compute_grad_call_result_253943)
    
    # Getting the type of 'jac_scale' (line 391)
    jac_scale_253944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 15), 'jac_scale')
    # Testing the type of an if condition (line 391)
    if_condition_253945 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 391, 12), jac_scale_253944)
    # Assigning a type to the variable 'if_condition_253945' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'if_condition_253945', if_condition_253945)
    # SSA begins for if statement (line 391)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 392):
    
    # Assigning a Subscript to a Name (line 392):
    
    # Obtaining the type of the subscript
    int_253946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 16), 'int')
    
    # Call to compute_jac_scale(...): (line 392)
    # Processing the call arguments (line 392)
    # Getting the type of 'J' (line 392)
    J_253948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 53), 'J', False)
    # Getting the type of 'scale_inv' (line 392)
    scale_inv_253949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 56), 'scale_inv', False)
    # Processing the call keyword arguments (line 392)
    kwargs_253950 = {}
    # Getting the type of 'compute_jac_scale' (line 392)
    compute_jac_scale_253947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 35), 'compute_jac_scale', False)
    # Calling compute_jac_scale(args, kwargs) (line 392)
    compute_jac_scale_call_result_253951 = invoke(stypy.reporting.localization.Localization(__file__, 392, 35), compute_jac_scale_253947, *[J_253948, scale_inv_253949], **kwargs_253950)
    
    # Obtaining the member '__getitem__' of a type (line 392)
    getitem___253952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 16), compute_jac_scale_call_result_253951, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 392)
    subscript_call_result_253953 = invoke(stypy.reporting.localization.Localization(__file__, 392, 16), getitem___253952, int_253946)
    
    # Assigning a type to the variable 'tuple_var_assignment_252633' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 16), 'tuple_var_assignment_252633', subscript_call_result_253953)
    
    # Assigning a Subscript to a Name (line 392):
    
    # Obtaining the type of the subscript
    int_253954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 16), 'int')
    
    # Call to compute_jac_scale(...): (line 392)
    # Processing the call arguments (line 392)
    # Getting the type of 'J' (line 392)
    J_253956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 53), 'J', False)
    # Getting the type of 'scale_inv' (line 392)
    scale_inv_253957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 56), 'scale_inv', False)
    # Processing the call keyword arguments (line 392)
    kwargs_253958 = {}
    # Getting the type of 'compute_jac_scale' (line 392)
    compute_jac_scale_253955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 35), 'compute_jac_scale', False)
    # Calling compute_jac_scale(args, kwargs) (line 392)
    compute_jac_scale_call_result_253959 = invoke(stypy.reporting.localization.Localization(__file__, 392, 35), compute_jac_scale_253955, *[J_253956, scale_inv_253957], **kwargs_253958)
    
    # Obtaining the member '__getitem__' of a type (line 392)
    getitem___253960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 16), compute_jac_scale_call_result_253959, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 392)
    subscript_call_result_253961 = invoke(stypy.reporting.localization.Localization(__file__, 392, 16), getitem___253960, int_253954)
    
    # Assigning a type to the variable 'tuple_var_assignment_252634' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 16), 'tuple_var_assignment_252634', subscript_call_result_253961)
    
    # Assigning a Name to a Name (line 392):
    # Getting the type of 'tuple_var_assignment_252633' (line 392)
    tuple_var_assignment_252633_253962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 16), 'tuple_var_assignment_252633')
    # Assigning a type to the variable 'scale' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 16), 'scale', tuple_var_assignment_252633_253962)
    
    # Assigning a Name to a Name (line 392):
    # Getting the type of 'tuple_var_assignment_252634' (line 392)
    tuple_var_assignment_252634_253963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 16), 'tuple_var_assignment_252634')
    # Assigning a type to the variable 'scale_inv' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 23), 'scale_inv', tuple_var_assignment_252634_253963)
    # SSA join for if statement (line 391)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 374)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 394):
    
    # Assigning a Num to a Name (line 394):
    int_253964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 24), 'int')
    # Assigning a type to the variable 'step_norm' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'step_norm', int_253964)
    
    # Assigning a Num to a Name (line 395):
    
    # Assigning a Num to a Name (line 395):
    int_253965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 31), 'int')
    # Assigning a type to the variable 'actual_reduction' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'actual_reduction', int_253965)
    # SSA join for if statement (line 374)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'iteration' (line 397)
    iteration_253966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'iteration')
    int_253967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 21), 'int')
    # Applying the binary operator '+=' (line 397)
    result_iadd_253968 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 8), '+=', iteration_253966, int_253967)
    # Assigning a type to the variable 'iteration' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'iteration', result_iadd_253968)
    
    # SSA join for while statement (line 263)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 399)
    # Getting the type of 'termination_status' (line 399)
    termination_status_253969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 7), 'termination_status')
    # Getting the type of 'None' (line 399)
    None_253970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 29), 'None')
    
    (may_be_253971, more_types_in_union_253972) = may_be_none(termination_status_253969, None_253970)

    if may_be_253971:

        if more_types_in_union_253972:
            # Runtime conditional SSA (line 399)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Num to a Name (line 400):
        
        # Assigning a Num to a Name (line 400):
        int_253973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 29), 'int')
        # Assigning a type to the variable 'termination_status' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'termination_status', int_253973)

        if more_types_in_union_253972:
            # SSA join for if statement (line 399)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 402):
    
    # Assigning a Call to a Name (line 402):
    
    # Call to find_active_constraints(...): (line 402)
    # Processing the call arguments (line 402)
    # Getting the type of 'x' (line 402)
    x_253975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 42), 'x', False)
    # Getting the type of 'lb' (line 402)
    lb_253976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 45), 'lb', False)
    # Getting the type of 'ub' (line 402)
    ub_253977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 49), 'ub', False)
    # Processing the call keyword arguments (line 402)
    # Getting the type of 'xtol' (line 402)
    xtol_253978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 58), 'xtol', False)
    keyword_253979 = xtol_253978
    kwargs_253980 = {'rtol': keyword_253979}
    # Getting the type of 'find_active_constraints' (line 402)
    find_active_constraints_253974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 18), 'find_active_constraints', False)
    # Calling find_active_constraints(args, kwargs) (line 402)
    find_active_constraints_call_result_253981 = invoke(stypy.reporting.localization.Localization(__file__, 402, 18), find_active_constraints_253974, *[x_253975, lb_253976, ub_253977], **kwargs_253980)
    
    # Assigning a type to the variable 'active_mask' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'active_mask', find_active_constraints_call_result_253981)
    
    # Call to OptimizeResult(...): (line 403)
    # Processing the call keyword arguments (line 403)
    # Getting the type of 'x' (line 404)
    x_253983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 10), 'x', False)
    keyword_253984 = x_253983
    # Getting the type of 'cost' (line 404)
    cost_253985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 18), 'cost', False)
    keyword_253986 = cost_253985
    # Getting the type of 'f_true' (line 404)
    f_true_253987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 28), 'f_true', False)
    keyword_253988 = f_true_253987
    # Getting the type of 'J' (line 404)
    J_253989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 40), 'J', False)
    keyword_253990 = J_253989
    # Getting the type of 'g' (line 404)
    g_253991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 48), 'g', False)
    keyword_253992 = g_253991
    # Getting the type of 'g_norm' (line 404)
    g_norm_253993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 62), 'g_norm', False)
    keyword_253994 = g_norm_253993
    # Getting the type of 'active_mask' (line 405)
    active_mask_253995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 20), 'active_mask', False)
    keyword_253996 = active_mask_253995
    # Getting the type of 'nfev' (line 405)
    nfev_253997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 38), 'nfev', False)
    keyword_253998 = nfev_253997
    # Getting the type of 'njev' (line 405)
    njev_253999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 49), 'njev', False)
    keyword_254000 = njev_253999
    # Getting the type of 'termination_status' (line 406)
    termination_status_254001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 15), 'termination_status', False)
    keyword_254002 = termination_status_254001
    kwargs_254003 = {'status': keyword_254002, 'njev': keyword_254000, 'nfev': keyword_253998, 'active_mask': keyword_253996, 'cost': keyword_253986, 'optimality': keyword_253994, 'fun': keyword_253988, 'x': keyword_253984, 'grad': keyword_253992, 'jac': keyword_253990}
    # Getting the type of 'OptimizeResult' (line 403)
    OptimizeResult_253982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 11), 'OptimizeResult', False)
    # Calling OptimizeResult(args, kwargs) (line 403)
    OptimizeResult_call_result_254004 = invoke(stypy.reporting.localization.Localization(__file__, 403, 11), OptimizeResult_253982, *[], **kwargs_254003)
    
    # Assigning a type to the variable 'stypy_return_type' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'stypy_return_type', OptimizeResult_call_result_254004)
    
    # ################# End of 'trf_bounds(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'trf_bounds' in the type store
    # Getting the type of 'stypy_return_type' (line 208)
    stypy_return_type_254005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_254005)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'trf_bounds'
    return stypy_return_type_254005

# Assigning a type to the variable 'trf_bounds' (line 208)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 0), 'trf_bounds', trf_bounds)

@norecursion
def trf_no_bounds(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'trf_no_bounds'
    module_type_store = module_type_store.open_function_context('trf_no_bounds', 409, 0, False)
    
    # Passed parameters checking function
    trf_no_bounds.stypy_localization = localization
    trf_no_bounds.stypy_type_of_self = None
    trf_no_bounds.stypy_type_store = module_type_store
    trf_no_bounds.stypy_function_name = 'trf_no_bounds'
    trf_no_bounds.stypy_param_names_list = ['fun', 'jac', 'x0', 'f0', 'J0', 'ftol', 'xtol', 'gtol', 'max_nfev', 'x_scale', 'loss_function', 'tr_solver', 'tr_options', 'verbose']
    trf_no_bounds.stypy_varargs_param_name = None
    trf_no_bounds.stypy_kwargs_param_name = None
    trf_no_bounds.stypy_call_defaults = defaults
    trf_no_bounds.stypy_call_varargs = varargs
    trf_no_bounds.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'trf_no_bounds', ['fun', 'jac', 'x0', 'f0', 'J0', 'ftol', 'xtol', 'gtol', 'max_nfev', 'x_scale', 'loss_function', 'tr_solver', 'tr_options', 'verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'trf_no_bounds', localization, ['fun', 'jac', 'x0', 'f0', 'J0', 'ftol', 'xtol', 'gtol', 'max_nfev', 'x_scale', 'loss_function', 'tr_solver', 'tr_options', 'verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'trf_no_bounds(...)' code ##################

    
    # Assigning a Call to a Name (line 411):
    
    # Assigning a Call to a Name (line 411):
    
    # Call to copy(...): (line 411)
    # Processing the call keyword arguments (line 411)
    kwargs_254008 = {}
    # Getting the type of 'x0' (line 411)
    x0_254006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'x0', False)
    # Obtaining the member 'copy' of a type (line 411)
    copy_254007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 8), x0_254006, 'copy')
    # Calling copy(args, kwargs) (line 411)
    copy_call_result_254009 = invoke(stypy.reporting.localization.Localization(__file__, 411, 8), copy_254007, *[], **kwargs_254008)
    
    # Assigning a type to the variable 'x' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'x', copy_call_result_254009)
    
    # Assigning a Name to a Name (line 413):
    
    # Assigning a Name to a Name (line 413):
    # Getting the type of 'f0' (line 413)
    f0_254010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'f0')
    # Assigning a type to the variable 'f' (line 413)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'f', f0_254010)
    
    # Assigning a Call to a Name (line 414):
    
    # Assigning a Call to a Name (line 414):
    
    # Call to copy(...): (line 414)
    # Processing the call keyword arguments (line 414)
    kwargs_254013 = {}
    # Getting the type of 'f' (line 414)
    f_254011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 13), 'f', False)
    # Obtaining the member 'copy' of a type (line 414)
    copy_254012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 13), f_254011, 'copy')
    # Calling copy(args, kwargs) (line 414)
    copy_call_result_254014 = invoke(stypy.reporting.localization.Localization(__file__, 414, 13), copy_254012, *[], **kwargs_254013)
    
    # Assigning a type to the variable 'f_true' (line 414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'f_true', copy_call_result_254014)
    
    # Assigning a Num to a Name (line 415):
    
    # Assigning a Num to a Name (line 415):
    int_254015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 11), 'int')
    # Assigning a type to the variable 'nfev' (line 415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'nfev', int_254015)
    
    # Assigning a Name to a Name (line 417):
    
    # Assigning a Name to a Name (line 417):
    # Getting the type of 'J0' (line 417)
    J0_254016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'J0')
    # Assigning a type to the variable 'J' (line 417)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'J', J0_254016)
    
    # Assigning a Num to a Name (line 418):
    
    # Assigning a Num to a Name (line 418):
    int_254017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 11), 'int')
    # Assigning a type to the variable 'njev' (line 418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'njev', int_254017)
    
    # Assigning a Attribute to a Tuple (line 419):
    
    # Assigning a Subscript to a Name (line 419):
    
    # Obtaining the type of the subscript
    int_254018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 4), 'int')
    # Getting the type of 'J' (line 419)
    J_254019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 11), 'J')
    # Obtaining the member 'shape' of a type (line 419)
    shape_254020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 11), J_254019, 'shape')
    # Obtaining the member '__getitem__' of a type (line 419)
    getitem___254021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 4), shape_254020, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 419)
    subscript_call_result_254022 = invoke(stypy.reporting.localization.Localization(__file__, 419, 4), getitem___254021, int_254018)
    
    # Assigning a type to the variable 'tuple_var_assignment_252635' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'tuple_var_assignment_252635', subscript_call_result_254022)
    
    # Assigning a Subscript to a Name (line 419):
    
    # Obtaining the type of the subscript
    int_254023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 4), 'int')
    # Getting the type of 'J' (line 419)
    J_254024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 11), 'J')
    # Obtaining the member 'shape' of a type (line 419)
    shape_254025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 11), J_254024, 'shape')
    # Obtaining the member '__getitem__' of a type (line 419)
    getitem___254026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 4), shape_254025, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 419)
    subscript_call_result_254027 = invoke(stypy.reporting.localization.Localization(__file__, 419, 4), getitem___254026, int_254023)
    
    # Assigning a type to the variable 'tuple_var_assignment_252636' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'tuple_var_assignment_252636', subscript_call_result_254027)
    
    # Assigning a Name to a Name (line 419):
    # Getting the type of 'tuple_var_assignment_252635' (line 419)
    tuple_var_assignment_252635_254028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'tuple_var_assignment_252635')
    # Assigning a type to the variable 'm' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'm', tuple_var_assignment_252635_254028)
    
    # Assigning a Name to a Name (line 419):
    # Getting the type of 'tuple_var_assignment_252636' (line 419)
    tuple_var_assignment_252636_254029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'tuple_var_assignment_252636')
    # Assigning a type to the variable 'n' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 7), 'n', tuple_var_assignment_252636_254029)
    
    # Type idiom detected: calculating its left and rigth part (line 421)
    # Getting the type of 'loss_function' (line 421)
    loss_function_254030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 4), 'loss_function')
    # Getting the type of 'None' (line 421)
    None_254031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 28), 'None')
    
    (may_be_254032, more_types_in_union_254033) = may_not_be_none(loss_function_254030, None_254031)

    if may_be_254032:

        if more_types_in_union_254033:
            # Runtime conditional SSA (line 421)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 422):
        
        # Assigning a Call to a Name (line 422):
        
        # Call to loss_function(...): (line 422)
        # Processing the call arguments (line 422)
        # Getting the type of 'f' (line 422)
        f_254035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 28), 'f', False)
        # Processing the call keyword arguments (line 422)
        kwargs_254036 = {}
        # Getting the type of 'loss_function' (line 422)
        loss_function_254034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 14), 'loss_function', False)
        # Calling loss_function(args, kwargs) (line 422)
        loss_function_call_result_254037 = invoke(stypy.reporting.localization.Localization(__file__, 422, 14), loss_function_254034, *[f_254035], **kwargs_254036)
        
        # Assigning a type to the variable 'rho' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'rho', loss_function_call_result_254037)
        
        # Assigning a BinOp to a Name (line 423):
        
        # Assigning a BinOp to a Name (line 423):
        float_254038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 15), 'float')
        
        # Call to sum(...): (line 423)
        # Processing the call arguments (line 423)
        
        # Obtaining the type of the subscript
        int_254041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 32), 'int')
        # Getting the type of 'rho' (line 423)
        rho_254042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 28), 'rho', False)
        # Obtaining the member '__getitem__' of a type (line 423)
        getitem___254043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 28), rho_254042, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 423)
        subscript_call_result_254044 = invoke(stypy.reporting.localization.Localization(__file__, 423, 28), getitem___254043, int_254041)
        
        # Processing the call keyword arguments (line 423)
        kwargs_254045 = {}
        # Getting the type of 'np' (line 423)
        np_254039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 21), 'np', False)
        # Obtaining the member 'sum' of a type (line 423)
        sum_254040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 21), np_254039, 'sum')
        # Calling sum(args, kwargs) (line 423)
        sum_call_result_254046 = invoke(stypy.reporting.localization.Localization(__file__, 423, 21), sum_254040, *[subscript_call_result_254044], **kwargs_254045)
        
        # Applying the binary operator '*' (line 423)
        result_mul_254047 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 15), '*', float_254038, sum_call_result_254046)
        
        # Assigning a type to the variable 'cost' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'cost', result_mul_254047)
        
        # Assigning a Call to a Tuple (line 424):
        
        # Assigning a Subscript to a Name (line 424):
        
        # Obtaining the type of the subscript
        int_254048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 8), 'int')
        
        # Call to scale_for_robust_loss_function(...): (line 424)
        # Processing the call arguments (line 424)
        # Getting the type of 'J' (line 424)
        J_254050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 46), 'J', False)
        # Getting the type of 'f' (line 424)
        f_254051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 49), 'f', False)
        # Getting the type of 'rho' (line 424)
        rho_254052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 52), 'rho', False)
        # Processing the call keyword arguments (line 424)
        kwargs_254053 = {}
        # Getting the type of 'scale_for_robust_loss_function' (line 424)
        scale_for_robust_loss_function_254049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 15), 'scale_for_robust_loss_function', False)
        # Calling scale_for_robust_loss_function(args, kwargs) (line 424)
        scale_for_robust_loss_function_call_result_254054 = invoke(stypy.reporting.localization.Localization(__file__, 424, 15), scale_for_robust_loss_function_254049, *[J_254050, f_254051, rho_254052], **kwargs_254053)
        
        # Obtaining the member '__getitem__' of a type (line 424)
        getitem___254055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 8), scale_for_robust_loss_function_call_result_254054, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 424)
        subscript_call_result_254056 = invoke(stypy.reporting.localization.Localization(__file__, 424, 8), getitem___254055, int_254048)
        
        # Assigning a type to the variable 'tuple_var_assignment_252637' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'tuple_var_assignment_252637', subscript_call_result_254056)
        
        # Assigning a Subscript to a Name (line 424):
        
        # Obtaining the type of the subscript
        int_254057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 8), 'int')
        
        # Call to scale_for_robust_loss_function(...): (line 424)
        # Processing the call arguments (line 424)
        # Getting the type of 'J' (line 424)
        J_254059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 46), 'J', False)
        # Getting the type of 'f' (line 424)
        f_254060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 49), 'f', False)
        # Getting the type of 'rho' (line 424)
        rho_254061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 52), 'rho', False)
        # Processing the call keyword arguments (line 424)
        kwargs_254062 = {}
        # Getting the type of 'scale_for_robust_loss_function' (line 424)
        scale_for_robust_loss_function_254058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 15), 'scale_for_robust_loss_function', False)
        # Calling scale_for_robust_loss_function(args, kwargs) (line 424)
        scale_for_robust_loss_function_call_result_254063 = invoke(stypy.reporting.localization.Localization(__file__, 424, 15), scale_for_robust_loss_function_254058, *[J_254059, f_254060, rho_254061], **kwargs_254062)
        
        # Obtaining the member '__getitem__' of a type (line 424)
        getitem___254064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 8), scale_for_robust_loss_function_call_result_254063, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 424)
        subscript_call_result_254065 = invoke(stypy.reporting.localization.Localization(__file__, 424, 8), getitem___254064, int_254057)
        
        # Assigning a type to the variable 'tuple_var_assignment_252638' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'tuple_var_assignment_252638', subscript_call_result_254065)
        
        # Assigning a Name to a Name (line 424):
        # Getting the type of 'tuple_var_assignment_252637' (line 424)
        tuple_var_assignment_252637_254066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'tuple_var_assignment_252637')
        # Assigning a type to the variable 'J' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'J', tuple_var_assignment_252637_254066)
        
        # Assigning a Name to a Name (line 424):
        # Getting the type of 'tuple_var_assignment_252638' (line 424)
        tuple_var_assignment_252638_254067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'tuple_var_assignment_252638')
        # Assigning a type to the variable 'f' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 11), 'f', tuple_var_assignment_252638_254067)

        if more_types_in_union_254033:
            # Runtime conditional SSA for else branch (line 421)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_254032) or more_types_in_union_254033):
        
        # Assigning a BinOp to a Name (line 426):
        
        # Assigning a BinOp to a Name (line 426):
        float_254068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 15), 'float')
        
        # Call to dot(...): (line 426)
        # Processing the call arguments (line 426)
        # Getting the type of 'f' (line 426)
        f_254071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 28), 'f', False)
        # Getting the type of 'f' (line 426)
        f_254072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 31), 'f', False)
        # Processing the call keyword arguments (line 426)
        kwargs_254073 = {}
        # Getting the type of 'np' (line 426)
        np_254069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 21), 'np', False)
        # Obtaining the member 'dot' of a type (line 426)
        dot_254070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 21), np_254069, 'dot')
        # Calling dot(args, kwargs) (line 426)
        dot_call_result_254074 = invoke(stypy.reporting.localization.Localization(__file__, 426, 21), dot_254070, *[f_254071, f_254072], **kwargs_254073)
        
        # Applying the binary operator '*' (line 426)
        result_mul_254075 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 15), '*', float_254068, dot_call_result_254074)
        
        # Assigning a type to the variable 'cost' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'cost', result_mul_254075)

        if (may_be_254032 and more_types_in_union_254033):
            # SSA join for if statement (line 421)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 428):
    
    # Assigning a Call to a Name (line 428):
    
    # Call to compute_grad(...): (line 428)
    # Processing the call arguments (line 428)
    # Getting the type of 'J' (line 428)
    J_254077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 21), 'J', False)
    # Getting the type of 'f' (line 428)
    f_254078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 24), 'f', False)
    # Processing the call keyword arguments (line 428)
    kwargs_254079 = {}
    # Getting the type of 'compute_grad' (line 428)
    compute_grad_254076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'compute_grad', False)
    # Calling compute_grad(args, kwargs) (line 428)
    compute_grad_call_result_254080 = invoke(stypy.reporting.localization.Localization(__file__, 428, 8), compute_grad_254076, *[J_254077, f_254078], **kwargs_254079)
    
    # Assigning a type to the variable 'g' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'g', compute_grad_call_result_254080)
    
    # Assigning a BoolOp to a Name (line 430):
    
    # Assigning a BoolOp to a Name (line 430):
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 430)
    # Processing the call arguments (line 430)
    # Getting the type of 'x_scale' (line 430)
    x_scale_254082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 27), 'x_scale', False)
    # Getting the type of 'string_types' (line 430)
    string_types_254083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 36), 'string_types', False)
    # Processing the call keyword arguments (line 430)
    kwargs_254084 = {}
    # Getting the type of 'isinstance' (line 430)
    isinstance_254081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 16), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 430)
    isinstance_call_result_254085 = invoke(stypy.reporting.localization.Localization(__file__, 430, 16), isinstance_254081, *[x_scale_254082, string_types_254083], **kwargs_254084)
    
    
    # Getting the type of 'x_scale' (line 430)
    x_scale_254086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 54), 'x_scale')
    str_254087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 65), 'str', 'jac')
    # Applying the binary operator '==' (line 430)
    result_eq_254088 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 54), '==', x_scale_254086, str_254087)
    
    # Applying the binary operator 'and' (line 430)
    result_and_keyword_254089 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 16), 'and', isinstance_call_result_254085, result_eq_254088)
    
    # Assigning a type to the variable 'jac_scale' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'jac_scale', result_and_keyword_254089)
    
    # Getting the type of 'jac_scale' (line 431)
    jac_scale_254090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 7), 'jac_scale')
    # Testing the type of an if condition (line 431)
    if_condition_254091 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 431, 4), jac_scale_254090)
    # Assigning a type to the variable 'if_condition_254091' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'if_condition_254091', if_condition_254091)
    # SSA begins for if statement (line 431)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 432):
    
    # Assigning a Subscript to a Name (line 432):
    
    # Obtaining the type of the subscript
    int_254092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 8), 'int')
    
    # Call to compute_jac_scale(...): (line 432)
    # Processing the call arguments (line 432)
    # Getting the type of 'J' (line 432)
    J_254094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 45), 'J', False)
    # Processing the call keyword arguments (line 432)
    kwargs_254095 = {}
    # Getting the type of 'compute_jac_scale' (line 432)
    compute_jac_scale_254093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 27), 'compute_jac_scale', False)
    # Calling compute_jac_scale(args, kwargs) (line 432)
    compute_jac_scale_call_result_254096 = invoke(stypy.reporting.localization.Localization(__file__, 432, 27), compute_jac_scale_254093, *[J_254094], **kwargs_254095)
    
    # Obtaining the member '__getitem__' of a type (line 432)
    getitem___254097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 8), compute_jac_scale_call_result_254096, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 432)
    subscript_call_result_254098 = invoke(stypy.reporting.localization.Localization(__file__, 432, 8), getitem___254097, int_254092)
    
    # Assigning a type to the variable 'tuple_var_assignment_252639' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'tuple_var_assignment_252639', subscript_call_result_254098)
    
    # Assigning a Subscript to a Name (line 432):
    
    # Obtaining the type of the subscript
    int_254099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 8), 'int')
    
    # Call to compute_jac_scale(...): (line 432)
    # Processing the call arguments (line 432)
    # Getting the type of 'J' (line 432)
    J_254101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 45), 'J', False)
    # Processing the call keyword arguments (line 432)
    kwargs_254102 = {}
    # Getting the type of 'compute_jac_scale' (line 432)
    compute_jac_scale_254100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 27), 'compute_jac_scale', False)
    # Calling compute_jac_scale(args, kwargs) (line 432)
    compute_jac_scale_call_result_254103 = invoke(stypy.reporting.localization.Localization(__file__, 432, 27), compute_jac_scale_254100, *[J_254101], **kwargs_254102)
    
    # Obtaining the member '__getitem__' of a type (line 432)
    getitem___254104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 8), compute_jac_scale_call_result_254103, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 432)
    subscript_call_result_254105 = invoke(stypy.reporting.localization.Localization(__file__, 432, 8), getitem___254104, int_254099)
    
    # Assigning a type to the variable 'tuple_var_assignment_252640' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'tuple_var_assignment_252640', subscript_call_result_254105)
    
    # Assigning a Name to a Name (line 432):
    # Getting the type of 'tuple_var_assignment_252639' (line 432)
    tuple_var_assignment_252639_254106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'tuple_var_assignment_252639')
    # Assigning a type to the variable 'scale' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'scale', tuple_var_assignment_252639_254106)
    
    # Assigning a Name to a Name (line 432):
    # Getting the type of 'tuple_var_assignment_252640' (line 432)
    tuple_var_assignment_252640_254107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'tuple_var_assignment_252640')
    # Assigning a type to the variable 'scale_inv' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 15), 'scale_inv', tuple_var_assignment_252640_254107)
    # SSA branch for the else part of an if statement (line 431)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Tuple to a Tuple (line 434):
    
    # Assigning a Name to a Name (line 434):
    # Getting the type of 'x_scale' (line 434)
    x_scale_254108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 27), 'x_scale')
    # Assigning a type to the variable 'tuple_assignment_252641' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'tuple_assignment_252641', x_scale_254108)
    
    # Assigning a BinOp to a Name (line 434):
    int_254109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 36), 'int')
    # Getting the type of 'x_scale' (line 434)
    x_scale_254110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 40), 'x_scale')
    # Applying the binary operator 'div' (line 434)
    result_div_254111 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 36), 'div', int_254109, x_scale_254110)
    
    # Assigning a type to the variable 'tuple_assignment_252642' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'tuple_assignment_252642', result_div_254111)
    
    # Assigning a Name to a Name (line 434):
    # Getting the type of 'tuple_assignment_252641' (line 434)
    tuple_assignment_252641_254112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'tuple_assignment_252641')
    # Assigning a type to the variable 'scale' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'scale', tuple_assignment_252641_254112)
    
    # Assigning a Name to a Name (line 434):
    # Getting the type of 'tuple_assignment_252642' (line 434)
    tuple_assignment_252642_254113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'tuple_assignment_252642')
    # Assigning a type to the variable 'scale_inv' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 15), 'scale_inv', tuple_assignment_252642_254113)
    # SSA join for if statement (line 431)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 436):
    
    # Assigning a Call to a Name (line 436):
    
    # Call to norm(...): (line 436)
    # Processing the call arguments (line 436)
    # Getting the type of 'x0' (line 436)
    x0_254115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 17), 'x0', False)
    # Getting the type of 'scale_inv' (line 436)
    scale_inv_254116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 22), 'scale_inv', False)
    # Applying the binary operator '*' (line 436)
    result_mul_254117 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 17), '*', x0_254115, scale_inv_254116)
    
    # Processing the call keyword arguments (line 436)
    kwargs_254118 = {}
    # Getting the type of 'norm' (line 436)
    norm_254114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'norm', False)
    # Calling norm(args, kwargs) (line 436)
    norm_call_result_254119 = invoke(stypy.reporting.localization.Localization(__file__, 436, 12), norm_254114, *[result_mul_254117], **kwargs_254118)
    
    # Assigning a type to the variable 'Delta' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'Delta', norm_call_result_254119)
    
    
    # Getting the type of 'Delta' (line 437)
    Delta_254120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 7), 'Delta')
    int_254121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 16), 'int')
    # Applying the binary operator '==' (line 437)
    result_eq_254122 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 7), '==', Delta_254120, int_254121)
    
    # Testing the type of an if condition (line 437)
    if_condition_254123 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 437, 4), result_eq_254122)
    # Assigning a type to the variable 'if_condition_254123' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), 'if_condition_254123', if_condition_254123)
    # SSA begins for if statement (line 437)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 438):
    
    # Assigning a Num to a Name (line 438):
    float_254124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 16), 'float')
    # Assigning a type to the variable 'Delta' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'Delta', float_254124)
    # SSA join for if statement (line 437)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'tr_solver' (line 440)
    tr_solver_254125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 7), 'tr_solver')
    str_254126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 20), 'str', 'lsmr')
    # Applying the binary operator '==' (line 440)
    result_eq_254127 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 7), '==', tr_solver_254125, str_254126)
    
    # Testing the type of an if condition (line 440)
    if_condition_254128 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 440, 4), result_eq_254127)
    # Assigning a type to the variable 'if_condition_254128' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'if_condition_254128', if_condition_254128)
    # SSA begins for if statement (line 440)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 441):
    
    # Assigning a Num to a Name (line 441):
    int_254129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 19), 'int')
    # Assigning a type to the variable 'reg_term' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'reg_term', int_254129)
    
    # Assigning a Call to a Name (line 442):
    
    # Assigning a Call to a Name (line 442):
    
    # Call to pop(...): (line 442)
    # Processing the call arguments (line 442)
    str_254132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 30), 'str', 'damp')
    float_254133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 38), 'float')
    # Processing the call keyword arguments (line 442)
    kwargs_254134 = {}
    # Getting the type of 'tr_options' (line 442)
    tr_options_254130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 15), 'tr_options', False)
    # Obtaining the member 'pop' of a type (line 442)
    pop_254131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 15), tr_options_254130, 'pop')
    # Calling pop(args, kwargs) (line 442)
    pop_call_result_254135 = invoke(stypy.reporting.localization.Localization(__file__, 442, 15), pop_254131, *[str_254132, float_254133], **kwargs_254134)
    
    # Assigning a type to the variable 'damp' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'damp', pop_call_result_254135)
    
    # Assigning a Call to a Name (line 443):
    
    # Assigning a Call to a Name (line 443):
    
    # Call to pop(...): (line 443)
    # Processing the call arguments (line 443)
    str_254138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 36), 'str', 'regularize')
    # Getting the type of 'True' (line 443)
    True_254139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 50), 'True', False)
    # Processing the call keyword arguments (line 443)
    kwargs_254140 = {}
    # Getting the type of 'tr_options' (line 443)
    tr_options_254136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 21), 'tr_options', False)
    # Obtaining the member 'pop' of a type (line 443)
    pop_254137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 21), tr_options_254136, 'pop')
    # Calling pop(args, kwargs) (line 443)
    pop_call_result_254141 = invoke(stypy.reporting.localization.Localization(__file__, 443, 21), pop_254137, *[str_254138, True_254139], **kwargs_254140)
    
    # Assigning a type to the variable 'regularize' (line 443)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'regularize', pop_call_result_254141)
    # SSA join for if statement (line 440)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 445)
    # Getting the type of 'max_nfev' (line 445)
    max_nfev_254142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 7), 'max_nfev')
    # Getting the type of 'None' (line 445)
    None_254143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 19), 'None')
    
    (may_be_254144, more_types_in_union_254145) = may_be_none(max_nfev_254142, None_254143)

    if may_be_254144:

        if more_types_in_union_254145:
            # Runtime conditional SSA (line 445)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 446):
        
        # Assigning a BinOp to a Name (line 446):
        # Getting the type of 'x0' (line 446)
        x0_254146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 19), 'x0')
        # Obtaining the member 'size' of a type (line 446)
        size_254147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 19), x0_254146, 'size')
        int_254148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 29), 'int')
        # Applying the binary operator '*' (line 446)
        result_mul_254149 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 19), '*', size_254147, int_254148)
        
        # Assigning a type to the variable 'max_nfev' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'max_nfev', result_mul_254149)

        if more_types_in_union_254145:
            # SSA join for if statement (line 445)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Num to a Name (line 448):
    
    # Assigning a Num to a Name (line 448):
    float_254150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 12), 'float')
    # Assigning a type to the variable 'alpha' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'alpha', float_254150)
    
    # Assigning a Name to a Name (line 450):
    
    # Assigning a Name to a Name (line 450):
    # Getting the type of 'None' (line 450)
    None_254151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 25), 'None')
    # Assigning a type to the variable 'termination_status' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'termination_status', None_254151)
    
    # Assigning a Num to a Name (line 451):
    
    # Assigning a Num to a Name (line 451):
    int_254152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 16), 'int')
    # Assigning a type to the variable 'iteration' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'iteration', int_254152)
    
    # Assigning a Name to a Name (line 452):
    
    # Assigning a Name to a Name (line 452):
    # Getting the type of 'None' (line 452)
    None_254153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 16), 'None')
    # Assigning a type to the variable 'step_norm' (line 452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'step_norm', None_254153)
    
    # Assigning a Name to a Name (line 453):
    
    # Assigning a Name to a Name (line 453):
    # Getting the type of 'None' (line 453)
    None_254154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 23), 'None')
    # Assigning a type to the variable 'actual_reduction' (line 453)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'actual_reduction', None_254154)
    
    
    # Getting the type of 'verbose' (line 455)
    verbose_254155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 7), 'verbose')
    int_254156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 18), 'int')
    # Applying the binary operator '==' (line 455)
    result_eq_254157 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 7), '==', verbose_254155, int_254156)
    
    # Testing the type of an if condition (line 455)
    if_condition_254158 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 455, 4), result_eq_254157)
    # Assigning a type to the variable 'if_condition_254158' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'if_condition_254158', if_condition_254158)
    # SSA begins for if statement (line 455)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print_header_nonlinear(...): (line 456)
    # Processing the call keyword arguments (line 456)
    kwargs_254160 = {}
    # Getting the type of 'print_header_nonlinear' (line 456)
    print_header_nonlinear_254159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'print_header_nonlinear', False)
    # Calling print_header_nonlinear(args, kwargs) (line 456)
    print_header_nonlinear_call_result_254161 = invoke(stypy.reporting.localization.Localization(__file__, 456, 8), print_header_nonlinear_254159, *[], **kwargs_254160)
    
    # SSA join for if statement (line 455)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'True' (line 458)
    True_254162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 10), 'True')
    # Testing the type of an if condition (line 458)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 458, 4), True_254162)
    # SSA begins for while statement (line 458)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 459):
    
    # Assigning a Call to a Name (line 459):
    
    # Call to norm(...): (line 459)
    # Processing the call arguments (line 459)
    # Getting the type of 'g' (line 459)
    g_254164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 22), 'g', False)
    # Processing the call keyword arguments (line 459)
    # Getting the type of 'np' (line 459)
    np_254165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 29), 'np', False)
    # Obtaining the member 'inf' of a type (line 459)
    inf_254166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 29), np_254165, 'inf')
    keyword_254167 = inf_254166
    kwargs_254168 = {'ord': keyword_254167}
    # Getting the type of 'norm' (line 459)
    norm_254163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 17), 'norm', False)
    # Calling norm(args, kwargs) (line 459)
    norm_call_result_254169 = invoke(stypy.reporting.localization.Localization(__file__, 459, 17), norm_254163, *[g_254164], **kwargs_254168)
    
    # Assigning a type to the variable 'g_norm' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'g_norm', norm_call_result_254169)
    
    
    # Getting the type of 'g_norm' (line 460)
    g_norm_254170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 11), 'g_norm')
    # Getting the type of 'gtol' (line 460)
    gtol_254171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 20), 'gtol')
    # Applying the binary operator '<' (line 460)
    result_lt_254172 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 11), '<', g_norm_254170, gtol_254171)
    
    # Testing the type of an if condition (line 460)
    if_condition_254173 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 460, 8), result_lt_254172)
    # Assigning a type to the variable 'if_condition_254173' (line 460)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'if_condition_254173', if_condition_254173)
    # SSA begins for if statement (line 460)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 461):
    
    # Assigning a Num to a Name (line 461):
    int_254174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 33), 'int')
    # Assigning a type to the variable 'termination_status' (line 461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'termination_status', int_254174)
    # SSA join for if statement (line 460)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'verbose' (line 463)
    verbose_254175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 11), 'verbose')
    int_254176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 22), 'int')
    # Applying the binary operator '==' (line 463)
    result_eq_254177 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 11), '==', verbose_254175, int_254176)
    
    # Testing the type of an if condition (line 463)
    if_condition_254178 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 463, 8), result_eq_254177)
    # Assigning a type to the variable 'if_condition_254178' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'if_condition_254178', if_condition_254178)
    # SSA begins for if statement (line 463)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print_iteration_nonlinear(...): (line 464)
    # Processing the call arguments (line 464)
    # Getting the type of 'iteration' (line 464)
    iteration_254180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 38), 'iteration', False)
    # Getting the type of 'nfev' (line 464)
    nfev_254181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 49), 'nfev', False)
    # Getting the type of 'cost' (line 464)
    cost_254182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 55), 'cost', False)
    # Getting the type of 'actual_reduction' (line 464)
    actual_reduction_254183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 61), 'actual_reduction', False)
    # Getting the type of 'step_norm' (line 465)
    step_norm_254184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 38), 'step_norm', False)
    # Getting the type of 'g_norm' (line 465)
    g_norm_254185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 49), 'g_norm', False)
    # Processing the call keyword arguments (line 464)
    kwargs_254186 = {}
    # Getting the type of 'print_iteration_nonlinear' (line 464)
    print_iteration_nonlinear_254179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 12), 'print_iteration_nonlinear', False)
    # Calling print_iteration_nonlinear(args, kwargs) (line 464)
    print_iteration_nonlinear_call_result_254187 = invoke(stypy.reporting.localization.Localization(__file__, 464, 12), print_iteration_nonlinear_254179, *[iteration_254180, nfev_254181, cost_254182, actual_reduction_254183, step_norm_254184, g_norm_254185], **kwargs_254186)
    
    # SSA join for if statement (line 463)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'termination_status' (line 467)
    termination_status_254188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 11), 'termination_status')
    # Getting the type of 'None' (line 467)
    None_254189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 37), 'None')
    # Applying the binary operator 'isnot' (line 467)
    result_is_not_254190 = python_operator(stypy.reporting.localization.Localization(__file__, 467, 11), 'isnot', termination_status_254188, None_254189)
    
    
    # Getting the type of 'nfev' (line 467)
    nfev_254191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 45), 'nfev')
    # Getting the type of 'max_nfev' (line 467)
    max_nfev_254192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 53), 'max_nfev')
    # Applying the binary operator '==' (line 467)
    result_eq_254193 = python_operator(stypy.reporting.localization.Localization(__file__, 467, 45), '==', nfev_254191, max_nfev_254192)
    
    # Applying the binary operator 'or' (line 467)
    result_or_keyword_254194 = python_operator(stypy.reporting.localization.Localization(__file__, 467, 11), 'or', result_is_not_254190, result_eq_254193)
    
    # Testing the type of an if condition (line 467)
    if_condition_254195 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 467, 8), result_or_keyword_254194)
    # Assigning a type to the variable 'if_condition_254195' (line 467)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'if_condition_254195', if_condition_254195)
    # SSA begins for if statement (line 467)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 467)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 470):
    
    # Assigning a Name to a Name (line 470):
    # Getting the type of 'scale' (line 470)
    scale_254196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'scale')
    # Assigning a type to the variable 'd' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'd', scale_254196)
    
    # Assigning a BinOp to a Name (line 471):
    
    # Assigning a BinOp to a Name (line 471):
    # Getting the type of 'd' (line 471)
    d_254197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 14), 'd')
    # Getting the type of 'g' (line 471)
    g_254198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 18), 'g')
    # Applying the binary operator '*' (line 471)
    result_mul_254199 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 14), '*', d_254197, g_254198)
    
    # Assigning a type to the variable 'g_h' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'g_h', result_mul_254199)
    
    
    # Getting the type of 'tr_solver' (line 473)
    tr_solver_254200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 11), 'tr_solver')
    str_254201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 24), 'str', 'exact')
    # Applying the binary operator '==' (line 473)
    result_eq_254202 = python_operator(stypy.reporting.localization.Localization(__file__, 473, 11), '==', tr_solver_254200, str_254201)
    
    # Testing the type of an if condition (line 473)
    if_condition_254203 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 473, 8), result_eq_254202)
    # Assigning a type to the variable 'if_condition_254203' (line 473)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'if_condition_254203', if_condition_254203)
    # SSA begins for if statement (line 473)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 474):
    
    # Assigning a BinOp to a Name (line 474):
    # Getting the type of 'J' (line 474)
    J_254204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 18), 'J')
    # Getting the type of 'd' (line 474)
    d_254205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 22), 'd')
    # Applying the binary operator '*' (line 474)
    result_mul_254206 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 18), '*', J_254204, d_254205)
    
    # Assigning a type to the variable 'J_h' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 12), 'J_h', result_mul_254206)
    
    # Assigning a Call to a Tuple (line 475):
    
    # Assigning a Subscript to a Name (line 475):
    
    # Obtaining the type of the subscript
    int_254207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 12), 'int')
    
    # Call to svd(...): (line 475)
    # Processing the call arguments (line 475)
    # Getting the type of 'J_h' (line 475)
    J_h_254209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 26), 'J_h', False)
    # Processing the call keyword arguments (line 475)
    # Getting the type of 'False' (line 475)
    False_254210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 45), 'False', False)
    keyword_254211 = False_254210
    kwargs_254212 = {'full_matrices': keyword_254211}
    # Getting the type of 'svd' (line 475)
    svd_254208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 22), 'svd', False)
    # Calling svd(args, kwargs) (line 475)
    svd_call_result_254213 = invoke(stypy.reporting.localization.Localization(__file__, 475, 22), svd_254208, *[J_h_254209], **kwargs_254212)
    
    # Obtaining the member '__getitem__' of a type (line 475)
    getitem___254214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 12), svd_call_result_254213, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 475)
    subscript_call_result_254215 = invoke(stypy.reporting.localization.Localization(__file__, 475, 12), getitem___254214, int_254207)
    
    # Assigning a type to the variable 'tuple_var_assignment_252643' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'tuple_var_assignment_252643', subscript_call_result_254215)
    
    # Assigning a Subscript to a Name (line 475):
    
    # Obtaining the type of the subscript
    int_254216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 12), 'int')
    
    # Call to svd(...): (line 475)
    # Processing the call arguments (line 475)
    # Getting the type of 'J_h' (line 475)
    J_h_254218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 26), 'J_h', False)
    # Processing the call keyword arguments (line 475)
    # Getting the type of 'False' (line 475)
    False_254219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 45), 'False', False)
    keyword_254220 = False_254219
    kwargs_254221 = {'full_matrices': keyword_254220}
    # Getting the type of 'svd' (line 475)
    svd_254217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 22), 'svd', False)
    # Calling svd(args, kwargs) (line 475)
    svd_call_result_254222 = invoke(stypy.reporting.localization.Localization(__file__, 475, 22), svd_254217, *[J_h_254218], **kwargs_254221)
    
    # Obtaining the member '__getitem__' of a type (line 475)
    getitem___254223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 12), svd_call_result_254222, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 475)
    subscript_call_result_254224 = invoke(stypy.reporting.localization.Localization(__file__, 475, 12), getitem___254223, int_254216)
    
    # Assigning a type to the variable 'tuple_var_assignment_252644' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'tuple_var_assignment_252644', subscript_call_result_254224)
    
    # Assigning a Subscript to a Name (line 475):
    
    # Obtaining the type of the subscript
    int_254225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 12), 'int')
    
    # Call to svd(...): (line 475)
    # Processing the call arguments (line 475)
    # Getting the type of 'J_h' (line 475)
    J_h_254227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 26), 'J_h', False)
    # Processing the call keyword arguments (line 475)
    # Getting the type of 'False' (line 475)
    False_254228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 45), 'False', False)
    keyword_254229 = False_254228
    kwargs_254230 = {'full_matrices': keyword_254229}
    # Getting the type of 'svd' (line 475)
    svd_254226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 22), 'svd', False)
    # Calling svd(args, kwargs) (line 475)
    svd_call_result_254231 = invoke(stypy.reporting.localization.Localization(__file__, 475, 22), svd_254226, *[J_h_254227], **kwargs_254230)
    
    # Obtaining the member '__getitem__' of a type (line 475)
    getitem___254232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 12), svd_call_result_254231, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 475)
    subscript_call_result_254233 = invoke(stypy.reporting.localization.Localization(__file__, 475, 12), getitem___254232, int_254225)
    
    # Assigning a type to the variable 'tuple_var_assignment_252645' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'tuple_var_assignment_252645', subscript_call_result_254233)
    
    # Assigning a Name to a Name (line 475):
    # Getting the type of 'tuple_var_assignment_252643' (line 475)
    tuple_var_assignment_252643_254234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'tuple_var_assignment_252643')
    # Assigning a type to the variable 'U' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'U', tuple_var_assignment_252643_254234)
    
    # Assigning a Name to a Name (line 475):
    # Getting the type of 'tuple_var_assignment_252644' (line 475)
    tuple_var_assignment_252644_254235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'tuple_var_assignment_252644')
    # Assigning a type to the variable 's' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 15), 's', tuple_var_assignment_252644_254235)
    
    # Assigning a Name to a Name (line 475):
    # Getting the type of 'tuple_var_assignment_252645' (line 475)
    tuple_var_assignment_252645_254236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'tuple_var_assignment_252645')
    # Assigning a type to the variable 'V' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 18), 'V', tuple_var_assignment_252645_254236)
    
    # Assigning a Attribute to a Name (line 476):
    
    # Assigning a Attribute to a Name (line 476):
    # Getting the type of 'V' (line 476)
    V_254237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 16), 'V')
    # Obtaining the member 'T' of a type (line 476)
    T_254238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 16), V_254237, 'T')
    # Assigning a type to the variable 'V' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 12), 'V', T_254238)
    
    # Assigning a Call to a Name (line 477):
    
    # Assigning a Call to a Name (line 477):
    
    # Call to dot(...): (line 477)
    # Processing the call arguments (line 477)
    # Getting the type of 'f' (line 477)
    f_254242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 25), 'f', False)
    # Processing the call keyword arguments (line 477)
    kwargs_254243 = {}
    # Getting the type of 'U' (line 477)
    U_254239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 17), 'U', False)
    # Obtaining the member 'T' of a type (line 477)
    T_254240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 17), U_254239, 'T')
    # Obtaining the member 'dot' of a type (line 477)
    dot_254241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 17), T_254240, 'dot')
    # Calling dot(args, kwargs) (line 477)
    dot_call_result_254244 = invoke(stypy.reporting.localization.Localization(__file__, 477, 17), dot_254241, *[f_254242], **kwargs_254243)
    
    # Assigning a type to the variable 'uf' (line 477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 12), 'uf', dot_call_result_254244)
    # SSA branch for the else part of an if statement (line 473)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'tr_solver' (line 478)
    tr_solver_254245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 13), 'tr_solver')
    str_254246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 26), 'str', 'lsmr')
    # Applying the binary operator '==' (line 478)
    result_eq_254247 = python_operator(stypy.reporting.localization.Localization(__file__, 478, 13), '==', tr_solver_254245, str_254246)
    
    # Testing the type of an if condition (line 478)
    if_condition_254248 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 478, 13), result_eq_254247)
    # Assigning a type to the variable 'if_condition_254248' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 13), 'if_condition_254248', if_condition_254248)
    # SSA begins for if statement (line 478)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 479):
    
    # Assigning a Call to a Name (line 479):
    
    # Call to right_multiplied_operator(...): (line 479)
    # Processing the call arguments (line 479)
    # Getting the type of 'J' (line 479)
    J_254250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 44), 'J', False)
    # Getting the type of 'd' (line 479)
    d_254251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 47), 'd', False)
    # Processing the call keyword arguments (line 479)
    kwargs_254252 = {}
    # Getting the type of 'right_multiplied_operator' (line 479)
    right_multiplied_operator_254249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 18), 'right_multiplied_operator', False)
    # Calling right_multiplied_operator(args, kwargs) (line 479)
    right_multiplied_operator_call_result_254253 = invoke(stypy.reporting.localization.Localization(__file__, 479, 18), right_multiplied_operator_254249, *[J_254250, d_254251], **kwargs_254252)
    
    # Assigning a type to the variable 'J_h' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'J_h', right_multiplied_operator_call_result_254253)
    
    # Getting the type of 'regularize' (line 481)
    regularize_254254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 15), 'regularize')
    # Testing the type of an if condition (line 481)
    if_condition_254255 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 481, 12), regularize_254254)
    # Assigning a type to the variable 'if_condition_254255' (line 481)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'if_condition_254255', if_condition_254255)
    # SSA begins for if statement (line 481)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 482):
    
    # Assigning a Subscript to a Name (line 482):
    
    # Obtaining the type of the subscript
    int_254256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 16), 'int')
    
    # Call to build_quadratic_1d(...): (line 482)
    # Processing the call arguments (line 482)
    # Getting the type of 'J_h' (line 482)
    J_h_254258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 42), 'J_h', False)
    # Getting the type of 'g_h' (line 482)
    g_h_254259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 47), 'g_h', False)
    
    # Getting the type of 'g_h' (line 482)
    g_h_254260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 53), 'g_h', False)
    # Applying the 'usub' unary operator (line 482)
    result___neg___254261 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 52), 'usub', g_h_254260)
    
    # Processing the call keyword arguments (line 482)
    kwargs_254262 = {}
    # Getting the type of 'build_quadratic_1d' (line 482)
    build_quadratic_1d_254257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 23), 'build_quadratic_1d', False)
    # Calling build_quadratic_1d(args, kwargs) (line 482)
    build_quadratic_1d_call_result_254263 = invoke(stypy.reporting.localization.Localization(__file__, 482, 23), build_quadratic_1d_254257, *[J_h_254258, g_h_254259, result___neg___254261], **kwargs_254262)
    
    # Obtaining the member '__getitem__' of a type (line 482)
    getitem___254264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 16), build_quadratic_1d_call_result_254263, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 482)
    subscript_call_result_254265 = invoke(stypy.reporting.localization.Localization(__file__, 482, 16), getitem___254264, int_254256)
    
    # Assigning a type to the variable 'tuple_var_assignment_252646' (line 482)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 16), 'tuple_var_assignment_252646', subscript_call_result_254265)
    
    # Assigning a Subscript to a Name (line 482):
    
    # Obtaining the type of the subscript
    int_254266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 16), 'int')
    
    # Call to build_quadratic_1d(...): (line 482)
    # Processing the call arguments (line 482)
    # Getting the type of 'J_h' (line 482)
    J_h_254268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 42), 'J_h', False)
    # Getting the type of 'g_h' (line 482)
    g_h_254269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 47), 'g_h', False)
    
    # Getting the type of 'g_h' (line 482)
    g_h_254270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 53), 'g_h', False)
    # Applying the 'usub' unary operator (line 482)
    result___neg___254271 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 52), 'usub', g_h_254270)
    
    # Processing the call keyword arguments (line 482)
    kwargs_254272 = {}
    # Getting the type of 'build_quadratic_1d' (line 482)
    build_quadratic_1d_254267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 23), 'build_quadratic_1d', False)
    # Calling build_quadratic_1d(args, kwargs) (line 482)
    build_quadratic_1d_call_result_254273 = invoke(stypy.reporting.localization.Localization(__file__, 482, 23), build_quadratic_1d_254267, *[J_h_254268, g_h_254269, result___neg___254271], **kwargs_254272)
    
    # Obtaining the member '__getitem__' of a type (line 482)
    getitem___254274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 16), build_quadratic_1d_call_result_254273, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 482)
    subscript_call_result_254275 = invoke(stypy.reporting.localization.Localization(__file__, 482, 16), getitem___254274, int_254266)
    
    # Assigning a type to the variable 'tuple_var_assignment_252647' (line 482)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 16), 'tuple_var_assignment_252647', subscript_call_result_254275)
    
    # Assigning a Name to a Name (line 482):
    # Getting the type of 'tuple_var_assignment_252646' (line 482)
    tuple_var_assignment_252646_254276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 16), 'tuple_var_assignment_252646')
    # Assigning a type to the variable 'a' (line 482)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 16), 'a', tuple_var_assignment_252646_254276)
    
    # Assigning a Name to a Name (line 482):
    # Getting the type of 'tuple_var_assignment_252647' (line 482)
    tuple_var_assignment_252647_254277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 16), 'tuple_var_assignment_252647')
    # Assigning a type to the variable 'b' (line 482)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 19), 'b', tuple_var_assignment_252647_254277)
    
    # Assigning a BinOp to a Name (line 483):
    
    # Assigning a BinOp to a Name (line 483):
    # Getting the type of 'Delta' (line 483)
    Delta_254278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 24), 'Delta')
    
    # Call to norm(...): (line 483)
    # Processing the call arguments (line 483)
    # Getting the type of 'g_h' (line 483)
    g_h_254280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 37), 'g_h', False)
    # Processing the call keyword arguments (line 483)
    kwargs_254281 = {}
    # Getting the type of 'norm' (line 483)
    norm_254279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 32), 'norm', False)
    # Calling norm(args, kwargs) (line 483)
    norm_call_result_254282 = invoke(stypy.reporting.localization.Localization(__file__, 483, 32), norm_254279, *[g_h_254280], **kwargs_254281)
    
    # Applying the binary operator 'div' (line 483)
    result_div_254283 = python_operator(stypy.reporting.localization.Localization(__file__, 483, 24), 'div', Delta_254278, norm_call_result_254282)
    
    # Assigning a type to the variable 'to_tr' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 16), 'to_tr', result_div_254283)
    
    # Assigning a Subscript to a Name (line 484):
    
    # Assigning a Subscript to a Name (line 484):
    
    # Obtaining the type of the subscript
    int_254284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 65), 'int')
    
    # Call to minimize_quadratic_1d(...): (line 484)
    # Processing the call arguments (line 484)
    # Getting the type of 'a' (line 484)
    a_254286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 49), 'a', False)
    # Getting the type of 'b' (line 484)
    b_254287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 52), 'b', False)
    int_254288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 55), 'int')
    # Getting the type of 'to_tr' (line 484)
    to_tr_254289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 58), 'to_tr', False)
    # Processing the call keyword arguments (line 484)
    kwargs_254290 = {}
    # Getting the type of 'minimize_quadratic_1d' (line 484)
    minimize_quadratic_1d_254285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 27), 'minimize_quadratic_1d', False)
    # Calling minimize_quadratic_1d(args, kwargs) (line 484)
    minimize_quadratic_1d_call_result_254291 = invoke(stypy.reporting.localization.Localization(__file__, 484, 27), minimize_quadratic_1d_254285, *[a_254286, b_254287, int_254288, to_tr_254289], **kwargs_254290)
    
    # Obtaining the member '__getitem__' of a type (line 484)
    getitem___254292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 27), minimize_quadratic_1d_call_result_254291, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 484)
    subscript_call_result_254293 = invoke(stypy.reporting.localization.Localization(__file__, 484, 27), getitem___254292, int_254284)
    
    # Assigning a type to the variable 'ag_value' (line 484)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 16), 'ag_value', subscript_call_result_254293)
    
    # Assigning a BinOp to a Name (line 485):
    
    # Assigning a BinOp to a Name (line 485):
    
    # Getting the type of 'ag_value' (line 485)
    ag_value_254294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 28), 'ag_value')
    # Applying the 'usub' unary operator (line 485)
    result___neg___254295 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 27), 'usub', ag_value_254294)
    
    # Getting the type of 'Delta' (line 485)
    Delta_254296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 39), 'Delta')
    int_254297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 46), 'int')
    # Applying the binary operator '**' (line 485)
    result_pow_254298 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 39), '**', Delta_254296, int_254297)
    
    # Applying the binary operator 'div' (line 485)
    result_div_254299 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 27), 'div', result___neg___254295, result_pow_254298)
    
    # Assigning a type to the variable 'reg_term' (line 485)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 16), 'reg_term', result_div_254299)
    # SSA join for if statement (line 481)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 487):
    
    # Assigning a BinOp to a Name (line 487):
    # Getting the type of 'damp' (line 487)
    damp_254300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 25), 'damp')
    int_254301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 31), 'int')
    # Applying the binary operator '**' (line 487)
    result_pow_254302 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 25), '**', damp_254300, int_254301)
    
    # Getting the type of 'reg_term' (line 487)
    reg_term_254303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 35), 'reg_term')
    # Applying the binary operator '+' (line 487)
    result_add_254304 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 25), '+', result_pow_254302, reg_term_254303)
    
    float_254305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 46), 'float')
    # Applying the binary operator '**' (line 487)
    result_pow_254306 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 24), '**', result_add_254304, float_254305)
    
    # Assigning a type to the variable 'damp_full' (line 487)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'damp_full', result_pow_254306)
    
    # Assigning a Subscript to a Name (line 488):
    
    # Assigning a Subscript to a Name (line 488):
    
    # Obtaining the type of the subscript
    int_254307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 62), 'int')
    
    # Call to lsmr(...): (line 488)
    # Processing the call arguments (line 488)
    # Getting the type of 'J_h' (line 488)
    J_h_254309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 24), 'J_h', False)
    # Getting the type of 'f' (line 488)
    f_254310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 29), 'f', False)
    # Processing the call keyword arguments (line 488)
    # Getting the type of 'damp_full' (line 488)
    damp_full_254311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 37), 'damp_full', False)
    keyword_254312 = damp_full_254311
    # Getting the type of 'tr_options' (line 488)
    tr_options_254313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 50), 'tr_options', False)
    kwargs_254314 = {'tr_options_254313': tr_options_254313, 'damp': keyword_254312}
    # Getting the type of 'lsmr' (line 488)
    lsmr_254308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 19), 'lsmr', False)
    # Calling lsmr(args, kwargs) (line 488)
    lsmr_call_result_254315 = invoke(stypy.reporting.localization.Localization(__file__, 488, 19), lsmr_254308, *[J_h_254309, f_254310], **kwargs_254314)
    
    # Obtaining the member '__getitem__' of a type (line 488)
    getitem___254316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 19), lsmr_call_result_254315, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 488)
    subscript_call_result_254317 = invoke(stypy.reporting.localization.Localization(__file__, 488, 19), getitem___254316, int_254307)
    
    # Assigning a type to the variable 'gn_h' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), 'gn_h', subscript_call_result_254317)
    
    # Assigning a Attribute to a Name (line 489):
    
    # Assigning a Attribute to a Name (line 489):
    
    # Call to vstack(...): (line 489)
    # Processing the call arguments (line 489)
    
    # Obtaining an instance of the builtin type 'tuple' (line 489)
    tuple_254320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 489)
    # Adding element type (line 489)
    # Getting the type of 'g_h' (line 489)
    g_h_254321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 27), 'g_h', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 27), tuple_254320, g_h_254321)
    # Adding element type (line 489)
    # Getting the type of 'gn_h' (line 489)
    gn_h_254322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 32), 'gn_h', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 27), tuple_254320, gn_h_254322)
    
    # Processing the call keyword arguments (line 489)
    kwargs_254323 = {}
    # Getting the type of 'np' (line 489)
    np_254318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 16), 'np', False)
    # Obtaining the member 'vstack' of a type (line 489)
    vstack_254319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 16), np_254318, 'vstack')
    # Calling vstack(args, kwargs) (line 489)
    vstack_call_result_254324 = invoke(stypy.reporting.localization.Localization(__file__, 489, 16), vstack_254319, *[tuple_254320], **kwargs_254323)
    
    # Obtaining the member 'T' of a type (line 489)
    T_254325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 16), vstack_call_result_254324, 'T')
    # Assigning a type to the variable 'S' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 12), 'S', T_254325)
    
    # Assigning a Call to a Tuple (line 490):
    
    # Assigning a Subscript to a Name (line 490):
    
    # Obtaining the type of the subscript
    int_254326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 12), 'int')
    
    # Call to qr(...): (line 490)
    # Processing the call arguments (line 490)
    # Getting the type of 'S' (line 490)
    S_254328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 22), 'S', False)
    # Processing the call keyword arguments (line 490)
    str_254329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 30), 'str', 'economic')
    keyword_254330 = str_254329
    kwargs_254331 = {'mode': keyword_254330}
    # Getting the type of 'qr' (line 490)
    qr_254327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 19), 'qr', False)
    # Calling qr(args, kwargs) (line 490)
    qr_call_result_254332 = invoke(stypy.reporting.localization.Localization(__file__, 490, 19), qr_254327, *[S_254328], **kwargs_254331)
    
    # Obtaining the member '__getitem__' of a type (line 490)
    getitem___254333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 12), qr_call_result_254332, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 490)
    subscript_call_result_254334 = invoke(stypy.reporting.localization.Localization(__file__, 490, 12), getitem___254333, int_254326)
    
    # Assigning a type to the variable 'tuple_var_assignment_252648' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'tuple_var_assignment_252648', subscript_call_result_254334)
    
    # Assigning a Subscript to a Name (line 490):
    
    # Obtaining the type of the subscript
    int_254335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 12), 'int')
    
    # Call to qr(...): (line 490)
    # Processing the call arguments (line 490)
    # Getting the type of 'S' (line 490)
    S_254337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 22), 'S', False)
    # Processing the call keyword arguments (line 490)
    str_254338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 30), 'str', 'economic')
    keyword_254339 = str_254338
    kwargs_254340 = {'mode': keyword_254339}
    # Getting the type of 'qr' (line 490)
    qr_254336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 19), 'qr', False)
    # Calling qr(args, kwargs) (line 490)
    qr_call_result_254341 = invoke(stypy.reporting.localization.Localization(__file__, 490, 19), qr_254336, *[S_254337], **kwargs_254340)
    
    # Obtaining the member '__getitem__' of a type (line 490)
    getitem___254342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 12), qr_call_result_254341, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 490)
    subscript_call_result_254343 = invoke(stypy.reporting.localization.Localization(__file__, 490, 12), getitem___254342, int_254335)
    
    # Assigning a type to the variable 'tuple_var_assignment_252649' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'tuple_var_assignment_252649', subscript_call_result_254343)
    
    # Assigning a Name to a Name (line 490):
    # Getting the type of 'tuple_var_assignment_252648' (line 490)
    tuple_var_assignment_252648_254344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'tuple_var_assignment_252648')
    # Assigning a type to the variable 'S' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'S', tuple_var_assignment_252648_254344)
    
    # Assigning a Name to a Name (line 490):
    # Getting the type of 'tuple_var_assignment_252649' (line 490)
    tuple_var_assignment_252649_254345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'tuple_var_assignment_252649')
    # Assigning a type to the variable '_' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 15), '_', tuple_var_assignment_252649_254345)
    
    # Assigning a Call to a Name (line 491):
    
    # Assigning a Call to a Name (line 491):
    
    # Call to dot(...): (line 491)
    # Processing the call arguments (line 491)
    # Getting the type of 'S' (line 491)
    S_254348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 25), 'S', False)
    # Processing the call keyword arguments (line 491)
    kwargs_254349 = {}
    # Getting the type of 'J_h' (line 491)
    J_h_254346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 17), 'J_h', False)
    # Obtaining the member 'dot' of a type (line 491)
    dot_254347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 17), J_h_254346, 'dot')
    # Calling dot(args, kwargs) (line 491)
    dot_call_result_254350 = invoke(stypy.reporting.localization.Localization(__file__, 491, 17), dot_254347, *[S_254348], **kwargs_254349)
    
    # Assigning a type to the variable 'JS' (line 491)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'JS', dot_call_result_254350)
    
    # Assigning a Call to a Name (line 492):
    
    # Assigning a Call to a Name (line 492):
    
    # Call to dot(...): (line 492)
    # Processing the call arguments (line 492)
    # Getting the type of 'JS' (line 492)
    JS_254353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 25), 'JS', False)
    # Obtaining the member 'T' of a type (line 492)
    T_254354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 25), JS_254353, 'T')
    # Getting the type of 'JS' (line 492)
    JS_254355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 31), 'JS', False)
    # Processing the call keyword arguments (line 492)
    kwargs_254356 = {}
    # Getting the type of 'np' (line 492)
    np_254351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 18), 'np', False)
    # Obtaining the member 'dot' of a type (line 492)
    dot_254352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 18), np_254351, 'dot')
    # Calling dot(args, kwargs) (line 492)
    dot_call_result_254357 = invoke(stypy.reporting.localization.Localization(__file__, 492, 18), dot_254352, *[T_254354, JS_254355], **kwargs_254356)
    
    # Assigning a type to the variable 'B_S' (line 492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 12), 'B_S', dot_call_result_254357)
    
    # Assigning a Call to a Name (line 493):
    
    # Assigning a Call to a Name (line 493):
    
    # Call to dot(...): (line 493)
    # Processing the call arguments (line 493)
    # Getting the type of 'g_h' (line 493)
    g_h_254361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 26), 'g_h', False)
    # Processing the call keyword arguments (line 493)
    kwargs_254362 = {}
    # Getting the type of 'S' (line 493)
    S_254358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 18), 'S', False)
    # Obtaining the member 'T' of a type (line 493)
    T_254359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 18), S_254358, 'T')
    # Obtaining the member 'dot' of a type (line 493)
    dot_254360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 18), T_254359, 'dot')
    # Calling dot(args, kwargs) (line 493)
    dot_call_result_254363 = invoke(stypy.reporting.localization.Localization(__file__, 493, 18), dot_254360, *[g_h_254361], **kwargs_254362)
    
    # Assigning a type to the variable 'g_S' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 12), 'g_S', dot_call_result_254363)
    # SSA join for if statement (line 478)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 473)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 495):
    
    # Assigning a Num to a Name (line 495):
    int_254364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 27), 'int')
    # Assigning a type to the variable 'actual_reduction' (line 495)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 8), 'actual_reduction', int_254364)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'actual_reduction' (line 496)
    actual_reduction_254365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 14), 'actual_reduction')
    int_254366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 34), 'int')
    # Applying the binary operator '<=' (line 496)
    result_le_254367 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 14), '<=', actual_reduction_254365, int_254366)
    
    
    # Getting the type of 'nfev' (line 496)
    nfev_254368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 40), 'nfev')
    # Getting the type of 'max_nfev' (line 496)
    max_nfev_254369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 47), 'max_nfev')
    # Applying the binary operator '<' (line 496)
    result_lt_254370 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 40), '<', nfev_254368, max_nfev_254369)
    
    # Applying the binary operator 'and' (line 496)
    result_and_keyword_254371 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 14), 'and', result_le_254367, result_lt_254370)
    
    # Testing the type of an if condition (line 496)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 496, 8), result_and_keyword_254371)
    # SSA begins for while statement (line 496)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    
    # Getting the type of 'tr_solver' (line 497)
    tr_solver_254372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 15), 'tr_solver')
    str_254373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 28), 'str', 'exact')
    # Applying the binary operator '==' (line 497)
    result_eq_254374 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 15), '==', tr_solver_254372, str_254373)
    
    # Testing the type of an if condition (line 497)
    if_condition_254375 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 497, 12), result_eq_254374)
    # Assigning a type to the variable 'if_condition_254375' (line 497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 12), 'if_condition_254375', if_condition_254375)
    # SSA begins for if statement (line 497)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 498):
    
    # Assigning a Subscript to a Name (line 498):
    
    # Obtaining the type of the subscript
    int_254376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 16), 'int')
    
    # Call to solve_lsq_trust_region(...): (line 498)
    # Processing the call arguments (line 498)
    # Getting the type of 'n' (line 499)
    n_254378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 20), 'n', False)
    # Getting the type of 'm' (line 499)
    m_254379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 23), 'm', False)
    # Getting the type of 'uf' (line 499)
    uf_254380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 26), 'uf', False)
    # Getting the type of 's' (line 499)
    s_254381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 30), 's', False)
    # Getting the type of 'V' (line 499)
    V_254382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 33), 'V', False)
    # Getting the type of 'Delta' (line 499)
    Delta_254383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 36), 'Delta', False)
    # Processing the call keyword arguments (line 498)
    # Getting the type of 'alpha' (line 499)
    alpha_254384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 57), 'alpha', False)
    keyword_254385 = alpha_254384
    kwargs_254386 = {'initial_alpha': keyword_254385}
    # Getting the type of 'solve_lsq_trust_region' (line 498)
    solve_lsq_trust_region_254377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 40), 'solve_lsq_trust_region', False)
    # Calling solve_lsq_trust_region(args, kwargs) (line 498)
    solve_lsq_trust_region_call_result_254387 = invoke(stypy.reporting.localization.Localization(__file__, 498, 40), solve_lsq_trust_region_254377, *[n_254378, m_254379, uf_254380, s_254381, V_254382, Delta_254383], **kwargs_254386)
    
    # Obtaining the member '__getitem__' of a type (line 498)
    getitem___254388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 16), solve_lsq_trust_region_call_result_254387, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 498)
    subscript_call_result_254389 = invoke(stypy.reporting.localization.Localization(__file__, 498, 16), getitem___254388, int_254376)
    
    # Assigning a type to the variable 'tuple_var_assignment_252650' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 16), 'tuple_var_assignment_252650', subscript_call_result_254389)
    
    # Assigning a Subscript to a Name (line 498):
    
    # Obtaining the type of the subscript
    int_254390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 16), 'int')
    
    # Call to solve_lsq_trust_region(...): (line 498)
    # Processing the call arguments (line 498)
    # Getting the type of 'n' (line 499)
    n_254392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 20), 'n', False)
    # Getting the type of 'm' (line 499)
    m_254393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 23), 'm', False)
    # Getting the type of 'uf' (line 499)
    uf_254394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 26), 'uf', False)
    # Getting the type of 's' (line 499)
    s_254395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 30), 's', False)
    # Getting the type of 'V' (line 499)
    V_254396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 33), 'V', False)
    # Getting the type of 'Delta' (line 499)
    Delta_254397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 36), 'Delta', False)
    # Processing the call keyword arguments (line 498)
    # Getting the type of 'alpha' (line 499)
    alpha_254398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 57), 'alpha', False)
    keyword_254399 = alpha_254398
    kwargs_254400 = {'initial_alpha': keyword_254399}
    # Getting the type of 'solve_lsq_trust_region' (line 498)
    solve_lsq_trust_region_254391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 40), 'solve_lsq_trust_region', False)
    # Calling solve_lsq_trust_region(args, kwargs) (line 498)
    solve_lsq_trust_region_call_result_254401 = invoke(stypy.reporting.localization.Localization(__file__, 498, 40), solve_lsq_trust_region_254391, *[n_254392, m_254393, uf_254394, s_254395, V_254396, Delta_254397], **kwargs_254400)
    
    # Obtaining the member '__getitem__' of a type (line 498)
    getitem___254402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 16), solve_lsq_trust_region_call_result_254401, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 498)
    subscript_call_result_254403 = invoke(stypy.reporting.localization.Localization(__file__, 498, 16), getitem___254402, int_254390)
    
    # Assigning a type to the variable 'tuple_var_assignment_252651' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 16), 'tuple_var_assignment_252651', subscript_call_result_254403)
    
    # Assigning a Subscript to a Name (line 498):
    
    # Obtaining the type of the subscript
    int_254404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 16), 'int')
    
    # Call to solve_lsq_trust_region(...): (line 498)
    # Processing the call arguments (line 498)
    # Getting the type of 'n' (line 499)
    n_254406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 20), 'n', False)
    # Getting the type of 'm' (line 499)
    m_254407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 23), 'm', False)
    # Getting the type of 'uf' (line 499)
    uf_254408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 26), 'uf', False)
    # Getting the type of 's' (line 499)
    s_254409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 30), 's', False)
    # Getting the type of 'V' (line 499)
    V_254410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 33), 'V', False)
    # Getting the type of 'Delta' (line 499)
    Delta_254411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 36), 'Delta', False)
    # Processing the call keyword arguments (line 498)
    # Getting the type of 'alpha' (line 499)
    alpha_254412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 57), 'alpha', False)
    keyword_254413 = alpha_254412
    kwargs_254414 = {'initial_alpha': keyword_254413}
    # Getting the type of 'solve_lsq_trust_region' (line 498)
    solve_lsq_trust_region_254405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 40), 'solve_lsq_trust_region', False)
    # Calling solve_lsq_trust_region(args, kwargs) (line 498)
    solve_lsq_trust_region_call_result_254415 = invoke(stypy.reporting.localization.Localization(__file__, 498, 40), solve_lsq_trust_region_254405, *[n_254406, m_254407, uf_254408, s_254409, V_254410, Delta_254411], **kwargs_254414)
    
    # Obtaining the member '__getitem__' of a type (line 498)
    getitem___254416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 16), solve_lsq_trust_region_call_result_254415, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 498)
    subscript_call_result_254417 = invoke(stypy.reporting.localization.Localization(__file__, 498, 16), getitem___254416, int_254404)
    
    # Assigning a type to the variable 'tuple_var_assignment_252652' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 16), 'tuple_var_assignment_252652', subscript_call_result_254417)
    
    # Assigning a Name to a Name (line 498):
    # Getting the type of 'tuple_var_assignment_252650' (line 498)
    tuple_var_assignment_252650_254418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 16), 'tuple_var_assignment_252650')
    # Assigning a type to the variable 'step_h' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 16), 'step_h', tuple_var_assignment_252650_254418)
    
    # Assigning a Name to a Name (line 498):
    # Getting the type of 'tuple_var_assignment_252651' (line 498)
    tuple_var_assignment_252651_254419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 16), 'tuple_var_assignment_252651')
    # Assigning a type to the variable 'alpha' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 24), 'alpha', tuple_var_assignment_252651_254419)
    
    # Assigning a Name to a Name (line 498):
    # Getting the type of 'tuple_var_assignment_252652' (line 498)
    tuple_var_assignment_252652_254420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 16), 'tuple_var_assignment_252652')
    # Assigning a type to the variable 'n_iter' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 31), 'n_iter', tuple_var_assignment_252652_254420)
    # SSA branch for the else part of an if statement (line 497)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'tr_solver' (line 500)
    tr_solver_254421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 17), 'tr_solver')
    str_254422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 30), 'str', 'lsmr')
    # Applying the binary operator '==' (line 500)
    result_eq_254423 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 17), '==', tr_solver_254421, str_254422)
    
    # Testing the type of an if condition (line 500)
    if_condition_254424 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 500, 17), result_eq_254423)
    # Assigning a type to the variable 'if_condition_254424' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 17), 'if_condition_254424', if_condition_254424)
    # SSA begins for if statement (line 500)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 501):
    
    # Assigning a Subscript to a Name (line 501):
    
    # Obtaining the type of the subscript
    int_254425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 16), 'int')
    
    # Call to solve_trust_region_2d(...): (line 501)
    # Processing the call arguments (line 501)
    # Getting the type of 'B_S' (line 501)
    B_S_254427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 47), 'B_S', False)
    # Getting the type of 'g_S' (line 501)
    g_S_254428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 52), 'g_S', False)
    # Getting the type of 'Delta' (line 501)
    Delta_254429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 57), 'Delta', False)
    # Processing the call keyword arguments (line 501)
    kwargs_254430 = {}
    # Getting the type of 'solve_trust_region_2d' (line 501)
    solve_trust_region_2d_254426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 25), 'solve_trust_region_2d', False)
    # Calling solve_trust_region_2d(args, kwargs) (line 501)
    solve_trust_region_2d_call_result_254431 = invoke(stypy.reporting.localization.Localization(__file__, 501, 25), solve_trust_region_2d_254426, *[B_S_254427, g_S_254428, Delta_254429], **kwargs_254430)
    
    # Obtaining the member '__getitem__' of a type (line 501)
    getitem___254432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 16), solve_trust_region_2d_call_result_254431, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 501)
    subscript_call_result_254433 = invoke(stypy.reporting.localization.Localization(__file__, 501, 16), getitem___254432, int_254425)
    
    # Assigning a type to the variable 'tuple_var_assignment_252653' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 16), 'tuple_var_assignment_252653', subscript_call_result_254433)
    
    # Assigning a Subscript to a Name (line 501):
    
    # Obtaining the type of the subscript
    int_254434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 16), 'int')
    
    # Call to solve_trust_region_2d(...): (line 501)
    # Processing the call arguments (line 501)
    # Getting the type of 'B_S' (line 501)
    B_S_254436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 47), 'B_S', False)
    # Getting the type of 'g_S' (line 501)
    g_S_254437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 52), 'g_S', False)
    # Getting the type of 'Delta' (line 501)
    Delta_254438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 57), 'Delta', False)
    # Processing the call keyword arguments (line 501)
    kwargs_254439 = {}
    # Getting the type of 'solve_trust_region_2d' (line 501)
    solve_trust_region_2d_254435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 25), 'solve_trust_region_2d', False)
    # Calling solve_trust_region_2d(args, kwargs) (line 501)
    solve_trust_region_2d_call_result_254440 = invoke(stypy.reporting.localization.Localization(__file__, 501, 25), solve_trust_region_2d_254435, *[B_S_254436, g_S_254437, Delta_254438], **kwargs_254439)
    
    # Obtaining the member '__getitem__' of a type (line 501)
    getitem___254441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 16), solve_trust_region_2d_call_result_254440, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 501)
    subscript_call_result_254442 = invoke(stypy.reporting.localization.Localization(__file__, 501, 16), getitem___254441, int_254434)
    
    # Assigning a type to the variable 'tuple_var_assignment_252654' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 16), 'tuple_var_assignment_252654', subscript_call_result_254442)
    
    # Assigning a Name to a Name (line 501):
    # Getting the type of 'tuple_var_assignment_252653' (line 501)
    tuple_var_assignment_252653_254443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 16), 'tuple_var_assignment_252653')
    # Assigning a type to the variable 'p_S' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 16), 'p_S', tuple_var_assignment_252653_254443)
    
    # Assigning a Name to a Name (line 501):
    # Getting the type of 'tuple_var_assignment_252654' (line 501)
    tuple_var_assignment_252654_254444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 16), 'tuple_var_assignment_252654')
    # Assigning a type to the variable '_' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 21), '_', tuple_var_assignment_252654_254444)
    
    # Assigning a Call to a Name (line 502):
    
    # Assigning a Call to a Name (line 502):
    
    # Call to dot(...): (line 502)
    # Processing the call arguments (line 502)
    # Getting the type of 'p_S' (line 502)
    p_S_254447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 31), 'p_S', False)
    # Processing the call keyword arguments (line 502)
    kwargs_254448 = {}
    # Getting the type of 'S' (line 502)
    S_254445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 25), 'S', False)
    # Obtaining the member 'dot' of a type (line 502)
    dot_254446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 25), S_254445, 'dot')
    # Calling dot(args, kwargs) (line 502)
    dot_call_result_254449 = invoke(stypy.reporting.localization.Localization(__file__, 502, 25), dot_254446, *[p_S_254447], **kwargs_254448)
    
    # Assigning a type to the variable 'step_h' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 16), 'step_h', dot_call_result_254449)
    # SSA join for if statement (line 500)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 497)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a UnaryOp to a Name (line 504):
    
    # Assigning a UnaryOp to a Name (line 504):
    
    
    # Call to evaluate_quadratic(...): (line 504)
    # Processing the call arguments (line 504)
    # Getting the type of 'J_h' (line 504)
    J_h_254451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 54), 'J_h', False)
    # Getting the type of 'g_h' (line 504)
    g_h_254452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 59), 'g_h', False)
    # Getting the type of 'step_h' (line 504)
    step_h_254453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 64), 'step_h', False)
    # Processing the call keyword arguments (line 504)
    kwargs_254454 = {}
    # Getting the type of 'evaluate_quadratic' (line 504)
    evaluate_quadratic_254450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 35), 'evaluate_quadratic', False)
    # Calling evaluate_quadratic(args, kwargs) (line 504)
    evaluate_quadratic_call_result_254455 = invoke(stypy.reporting.localization.Localization(__file__, 504, 35), evaluate_quadratic_254450, *[J_h_254451, g_h_254452, step_h_254453], **kwargs_254454)
    
    # Applying the 'usub' unary operator (line 504)
    result___neg___254456 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 34), 'usub', evaluate_quadratic_call_result_254455)
    
    # Assigning a type to the variable 'predicted_reduction' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'predicted_reduction', result___neg___254456)
    
    # Assigning a BinOp to a Name (line 505):
    
    # Assigning a BinOp to a Name (line 505):
    # Getting the type of 'd' (line 505)
    d_254457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 19), 'd')
    # Getting the type of 'step_h' (line 505)
    step_h_254458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 23), 'step_h')
    # Applying the binary operator '*' (line 505)
    result_mul_254459 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 19), '*', d_254457, step_h_254458)
    
    # Assigning a type to the variable 'step' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'step', result_mul_254459)
    
    # Assigning a BinOp to a Name (line 506):
    
    # Assigning a BinOp to a Name (line 506):
    # Getting the type of 'x' (line 506)
    x_254460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 20), 'x')
    # Getting the type of 'step' (line 506)
    step_254461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 24), 'step')
    # Applying the binary operator '+' (line 506)
    result_add_254462 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 20), '+', x_254460, step_254461)
    
    # Assigning a type to the variable 'x_new' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 12), 'x_new', result_add_254462)
    
    # Assigning a Call to a Name (line 507):
    
    # Assigning a Call to a Name (line 507):
    
    # Call to fun(...): (line 507)
    # Processing the call arguments (line 507)
    # Getting the type of 'x_new' (line 507)
    x_new_254464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 24), 'x_new', False)
    # Processing the call keyword arguments (line 507)
    kwargs_254465 = {}
    # Getting the type of 'fun' (line 507)
    fun_254463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 20), 'fun', False)
    # Calling fun(args, kwargs) (line 507)
    fun_call_result_254466 = invoke(stypy.reporting.localization.Localization(__file__, 507, 20), fun_254463, *[x_new_254464], **kwargs_254465)
    
    # Assigning a type to the variable 'f_new' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 12), 'f_new', fun_call_result_254466)
    
    # Getting the type of 'nfev' (line 508)
    nfev_254467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'nfev')
    int_254468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 20), 'int')
    # Applying the binary operator '+=' (line 508)
    result_iadd_254469 = python_operator(stypy.reporting.localization.Localization(__file__, 508, 12), '+=', nfev_254467, int_254468)
    # Assigning a type to the variable 'nfev' (line 508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'nfev', result_iadd_254469)
    
    
    # Assigning a Call to a Name (line 510):
    
    # Assigning a Call to a Name (line 510):
    
    # Call to norm(...): (line 510)
    # Processing the call arguments (line 510)
    # Getting the type of 'step_h' (line 510)
    step_h_254471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 31), 'step_h', False)
    # Processing the call keyword arguments (line 510)
    kwargs_254472 = {}
    # Getting the type of 'norm' (line 510)
    norm_254470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 26), 'norm', False)
    # Calling norm(args, kwargs) (line 510)
    norm_call_result_254473 = invoke(stypy.reporting.localization.Localization(__file__, 510, 26), norm_254470, *[step_h_254471], **kwargs_254472)
    
    # Assigning a type to the variable 'step_h_norm' (line 510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 12), 'step_h_norm', norm_call_result_254473)
    
    
    
    # Call to all(...): (line 512)
    # Processing the call arguments (line 512)
    
    # Call to isfinite(...): (line 512)
    # Processing the call arguments (line 512)
    # Getting the type of 'f_new' (line 512)
    f_new_254478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 38), 'f_new', False)
    # Processing the call keyword arguments (line 512)
    kwargs_254479 = {}
    # Getting the type of 'np' (line 512)
    np_254476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 26), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 512)
    isfinite_254477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 26), np_254476, 'isfinite')
    # Calling isfinite(args, kwargs) (line 512)
    isfinite_call_result_254480 = invoke(stypy.reporting.localization.Localization(__file__, 512, 26), isfinite_254477, *[f_new_254478], **kwargs_254479)
    
    # Processing the call keyword arguments (line 512)
    kwargs_254481 = {}
    # Getting the type of 'np' (line 512)
    np_254474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 19), 'np', False)
    # Obtaining the member 'all' of a type (line 512)
    all_254475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 19), np_254474, 'all')
    # Calling all(args, kwargs) (line 512)
    all_call_result_254482 = invoke(stypy.reporting.localization.Localization(__file__, 512, 19), all_254475, *[isfinite_call_result_254480], **kwargs_254481)
    
    # Applying the 'not' unary operator (line 512)
    result_not__254483 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 15), 'not', all_call_result_254482)
    
    # Testing the type of an if condition (line 512)
    if_condition_254484 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 512, 12), result_not__254483)
    # Assigning a type to the variable 'if_condition_254484' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'if_condition_254484', if_condition_254484)
    # SSA begins for if statement (line 512)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 513):
    
    # Assigning a BinOp to a Name (line 513):
    float_254485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 24), 'float')
    # Getting the type of 'step_h_norm' (line 513)
    step_h_norm_254486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 31), 'step_h_norm')
    # Applying the binary operator '*' (line 513)
    result_mul_254487 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 24), '*', float_254485, step_h_norm_254486)
    
    # Assigning a type to the variable 'Delta' (line 513)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 16), 'Delta', result_mul_254487)
    # SSA join for if statement (line 512)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 517)
    # Getting the type of 'loss_function' (line 517)
    loss_function_254488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 12), 'loss_function')
    # Getting the type of 'None' (line 517)
    None_254489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 36), 'None')
    
    (may_be_254490, more_types_in_union_254491) = may_not_be_none(loss_function_254488, None_254489)

    if may_be_254490:

        if more_types_in_union_254491:
            # Runtime conditional SSA (line 517)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 518):
        
        # Assigning a Call to a Name (line 518):
        
        # Call to loss_function(...): (line 518)
        # Processing the call arguments (line 518)
        # Getting the type of 'f_new' (line 518)
        f_new_254493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 41), 'f_new', False)
        # Processing the call keyword arguments (line 518)
        # Getting the type of 'True' (line 518)
        True_254494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 58), 'True', False)
        keyword_254495 = True_254494
        kwargs_254496 = {'cost_only': keyword_254495}
        # Getting the type of 'loss_function' (line 518)
        loss_function_254492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 27), 'loss_function', False)
        # Calling loss_function(args, kwargs) (line 518)
        loss_function_call_result_254497 = invoke(stypy.reporting.localization.Localization(__file__, 518, 27), loss_function_254492, *[f_new_254493], **kwargs_254496)
        
        # Assigning a type to the variable 'cost_new' (line 518)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 16), 'cost_new', loss_function_call_result_254497)

        if more_types_in_union_254491:
            # Runtime conditional SSA for else branch (line 517)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_254490) or more_types_in_union_254491):
        
        # Assigning a BinOp to a Name (line 520):
        
        # Assigning a BinOp to a Name (line 520):
        float_254498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 27), 'float')
        
        # Call to dot(...): (line 520)
        # Processing the call arguments (line 520)
        # Getting the type of 'f_new' (line 520)
        f_new_254501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 40), 'f_new', False)
        # Getting the type of 'f_new' (line 520)
        f_new_254502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 47), 'f_new', False)
        # Processing the call keyword arguments (line 520)
        kwargs_254503 = {}
        # Getting the type of 'np' (line 520)
        np_254499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 33), 'np', False)
        # Obtaining the member 'dot' of a type (line 520)
        dot_254500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 33), np_254499, 'dot')
        # Calling dot(args, kwargs) (line 520)
        dot_call_result_254504 = invoke(stypy.reporting.localization.Localization(__file__, 520, 33), dot_254500, *[f_new_254501, f_new_254502], **kwargs_254503)
        
        # Applying the binary operator '*' (line 520)
        result_mul_254505 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 27), '*', float_254498, dot_call_result_254504)
        
        # Assigning a type to the variable 'cost_new' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 16), 'cost_new', result_mul_254505)

        if (may_be_254490 and more_types_in_union_254491):
            # SSA join for if statement (line 517)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 521):
    
    # Assigning a BinOp to a Name (line 521):
    # Getting the type of 'cost' (line 521)
    cost_254506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 31), 'cost')
    # Getting the type of 'cost_new' (line 521)
    cost_new_254507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 38), 'cost_new')
    # Applying the binary operator '-' (line 521)
    result_sub_254508 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 31), '-', cost_254506, cost_new_254507)
    
    # Assigning a type to the variable 'actual_reduction' (line 521)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 12), 'actual_reduction', result_sub_254508)
    
    # Assigning a Call to a Tuple (line 523):
    
    # Assigning a Subscript to a Name (line 523):
    
    # Obtaining the type of the subscript
    int_254509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 12), 'int')
    
    # Call to update_tr_radius(...): (line 523)
    # Processing the call arguments (line 523)
    # Getting the type of 'Delta' (line 524)
    Delta_254511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 16), 'Delta', False)
    # Getting the type of 'actual_reduction' (line 524)
    actual_reduction_254512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 23), 'actual_reduction', False)
    # Getting the type of 'predicted_reduction' (line 524)
    predicted_reduction_254513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 41), 'predicted_reduction', False)
    # Getting the type of 'step_h_norm' (line 525)
    step_h_norm_254514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 16), 'step_h_norm', False)
    
    # Getting the type of 'step_h_norm' (line 525)
    step_h_norm_254515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 29), 'step_h_norm', False)
    float_254516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 43), 'float')
    # Getting the type of 'Delta' (line 525)
    Delta_254517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 50), 'Delta', False)
    # Applying the binary operator '*' (line 525)
    result_mul_254518 = python_operator(stypy.reporting.localization.Localization(__file__, 525, 43), '*', float_254516, Delta_254517)
    
    # Applying the binary operator '>' (line 525)
    result_gt_254519 = python_operator(stypy.reporting.localization.Localization(__file__, 525, 29), '>', step_h_norm_254515, result_mul_254518)
    
    # Processing the call keyword arguments (line 523)
    kwargs_254520 = {}
    # Getting the type of 'update_tr_radius' (line 523)
    update_tr_radius_254510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 31), 'update_tr_radius', False)
    # Calling update_tr_radius(args, kwargs) (line 523)
    update_tr_radius_call_result_254521 = invoke(stypy.reporting.localization.Localization(__file__, 523, 31), update_tr_radius_254510, *[Delta_254511, actual_reduction_254512, predicted_reduction_254513, step_h_norm_254514, result_gt_254519], **kwargs_254520)
    
    # Obtaining the member '__getitem__' of a type (line 523)
    getitem___254522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 12), update_tr_radius_call_result_254521, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 523)
    subscript_call_result_254523 = invoke(stypy.reporting.localization.Localization(__file__, 523, 12), getitem___254522, int_254509)
    
    # Assigning a type to the variable 'tuple_var_assignment_252655' (line 523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 12), 'tuple_var_assignment_252655', subscript_call_result_254523)
    
    # Assigning a Subscript to a Name (line 523):
    
    # Obtaining the type of the subscript
    int_254524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 12), 'int')
    
    # Call to update_tr_radius(...): (line 523)
    # Processing the call arguments (line 523)
    # Getting the type of 'Delta' (line 524)
    Delta_254526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 16), 'Delta', False)
    # Getting the type of 'actual_reduction' (line 524)
    actual_reduction_254527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 23), 'actual_reduction', False)
    # Getting the type of 'predicted_reduction' (line 524)
    predicted_reduction_254528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 41), 'predicted_reduction', False)
    # Getting the type of 'step_h_norm' (line 525)
    step_h_norm_254529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 16), 'step_h_norm', False)
    
    # Getting the type of 'step_h_norm' (line 525)
    step_h_norm_254530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 29), 'step_h_norm', False)
    float_254531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 43), 'float')
    # Getting the type of 'Delta' (line 525)
    Delta_254532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 50), 'Delta', False)
    # Applying the binary operator '*' (line 525)
    result_mul_254533 = python_operator(stypy.reporting.localization.Localization(__file__, 525, 43), '*', float_254531, Delta_254532)
    
    # Applying the binary operator '>' (line 525)
    result_gt_254534 = python_operator(stypy.reporting.localization.Localization(__file__, 525, 29), '>', step_h_norm_254530, result_mul_254533)
    
    # Processing the call keyword arguments (line 523)
    kwargs_254535 = {}
    # Getting the type of 'update_tr_radius' (line 523)
    update_tr_radius_254525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 31), 'update_tr_radius', False)
    # Calling update_tr_radius(args, kwargs) (line 523)
    update_tr_radius_call_result_254536 = invoke(stypy.reporting.localization.Localization(__file__, 523, 31), update_tr_radius_254525, *[Delta_254526, actual_reduction_254527, predicted_reduction_254528, step_h_norm_254529, result_gt_254534], **kwargs_254535)
    
    # Obtaining the member '__getitem__' of a type (line 523)
    getitem___254537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 12), update_tr_radius_call_result_254536, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 523)
    subscript_call_result_254538 = invoke(stypy.reporting.localization.Localization(__file__, 523, 12), getitem___254537, int_254524)
    
    # Assigning a type to the variable 'tuple_var_assignment_252656' (line 523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 12), 'tuple_var_assignment_252656', subscript_call_result_254538)
    
    # Assigning a Name to a Name (line 523):
    # Getting the type of 'tuple_var_assignment_252655' (line 523)
    tuple_var_assignment_252655_254539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 12), 'tuple_var_assignment_252655')
    # Assigning a type to the variable 'Delta_new' (line 523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 12), 'Delta_new', tuple_var_assignment_252655_254539)
    
    # Assigning a Name to a Name (line 523):
    # Getting the type of 'tuple_var_assignment_252656' (line 523)
    tuple_var_assignment_252656_254540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 12), 'tuple_var_assignment_252656')
    # Assigning a type to the variable 'ratio' (line 523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 23), 'ratio', tuple_var_assignment_252656_254540)
    
    # Getting the type of 'alpha' (line 526)
    alpha_254541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 12), 'alpha')
    # Getting the type of 'Delta' (line 526)
    Delta_254542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 21), 'Delta')
    # Getting the type of 'Delta_new' (line 526)
    Delta_new_254543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 29), 'Delta_new')
    # Applying the binary operator 'div' (line 526)
    result_div_254544 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 21), 'div', Delta_254542, Delta_new_254543)
    
    # Applying the binary operator '*=' (line 526)
    result_imul_254545 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 12), '*=', alpha_254541, result_div_254544)
    # Assigning a type to the variable 'alpha' (line 526)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 12), 'alpha', result_imul_254545)
    
    
    # Assigning a Name to a Name (line 527):
    
    # Assigning a Name to a Name (line 527):
    # Getting the type of 'Delta_new' (line 527)
    Delta_new_254546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 20), 'Delta_new')
    # Assigning a type to the variable 'Delta' (line 527)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 12), 'Delta', Delta_new_254546)
    
    # Assigning a Call to a Name (line 529):
    
    # Assigning a Call to a Name (line 529):
    
    # Call to norm(...): (line 529)
    # Processing the call arguments (line 529)
    # Getting the type of 'step' (line 529)
    step_254548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 29), 'step', False)
    # Processing the call keyword arguments (line 529)
    kwargs_254549 = {}
    # Getting the type of 'norm' (line 529)
    norm_254547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 24), 'norm', False)
    # Calling norm(args, kwargs) (line 529)
    norm_call_result_254550 = invoke(stypy.reporting.localization.Localization(__file__, 529, 24), norm_254547, *[step_254548], **kwargs_254549)
    
    # Assigning a type to the variable 'step_norm' (line 529)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'step_norm', norm_call_result_254550)
    
    # Assigning a Call to a Name (line 530):
    
    # Assigning a Call to a Name (line 530):
    
    # Call to check_termination(...): (line 530)
    # Processing the call arguments (line 530)
    # Getting the type of 'actual_reduction' (line 531)
    actual_reduction_254552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 16), 'actual_reduction', False)
    # Getting the type of 'cost' (line 531)
    cost_254553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 34), 'cost', False)
    # Getting the type of 'step_norm' (line 531)
    step_norm_254554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 40), 'step_norm', False)
    
    # Call to norm(...): (line 531)
    # Processing the call arguments (line 531)
    # Getting the type of 'x' (line 531)
    x_254556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 56), 'x', False)
    # Processing the call keyword arguments (line 531)
    kwargs_254557 = {}
    # Getting the type of 'norm' (line 531)
    norm_254555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 51), 'norm', False)
    # Calling norm(args, kwargs) (line 531)
    norm_call_result_254558 = invoke(stypy.reporting.localization.Localization(__file__, 531, 51), norm_254555, *[x_254556], **kwargs_254557)
    
    # Getting the type of 'ratio' (line 531)
    ratio_254559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 60), 'ratio', False)
    # Getting the type of 'ftol' (line 531)
    ftol_254560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 67), 'ftol', False)
    # Getting the type of 'xtol' (line 531)
    xtol_254561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 73), 'xtol', False)
    # Processing the call keyword arguments (line 530)
    kwargs_254562 = {}
    # Getting the type of 'check_termination' (line 530)
    check_termination_254551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 33), 'check_termination', False)
    # Calling check_termination(args, kwargs) (line 530)
    check_termination_call_result_254563 = invoke(stypy.reporting.localization.Localization(__file__, 530, 33), check_termination_254551, *[actual_reduction_254552, cost_254553, step_norm_254554, norm_call_result_254558, ratio_254559, ftol_254560, xtol_254561], **kwargs_254562)
    
    # Assigning a type to the variable 'termination_status' (line 530)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'termination_status', check_termination_call_result_254563)
    
    # Type idiom detected: calculating its left and rigth part (line 533)
    # Getting the type of 'termination_status' (line 533)
    termination_status_254564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'termination_status')
    # Getting the type of 'None' (line 533)
    None_254565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 41), 'None')
    
    (may_be_254566, more_types_in_union_254567) = may_not_be_none(termination_status_254564, None_254565)

    if may_be_254566:

        if more_types_in_union_254567:
            # Runtime conditional SSA (line 533)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        if more_types_in_union_254567:
            # SSA join for if statement (line 533)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for while statement (line 496)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'actual_reduction' (line 536)
    actual_reduction_254568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 11), 'actual_reduction')
    int_254569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 30), 'int')
    # Applying the binary operator '>' (line 536)
    result_gt_254570 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 11), '>', actual_reduction_254568, int_254569)
    
    # Testing the type of an if condition (line 536)
    if_condition_254571 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 536, 8), result_gt_254570)
    # Assigning a type to the variable 'if_condition_254571' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'if_condition_254571', if_condition_254571)
    # SSA begins for if statement (line 536)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 537):
    
    # Assigning a Name to a Name (line 537):
    # Getting the type of 'x_new' (line 537)
    x_new_254572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 16), 'x_new')
    # Assigning a type to the variable 'x' (line 537)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 12), 'x', x_new_254572)
    
    # Assigning a Name to a Name (line 539):
    
    # Assigning a Name to a Name (line 539):
    # Getting the type of 'f_new' (line 539)
    f_new_254573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 16), 'f_new')
    # Assigning a type to the variable 'f' (line 539)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 12), 'f', f_new_254573)
    
    # Assigning a Call to a Name (line 540):
    
    # Assigning a Call to a Name (line 540):
    
    # Call to copy(...): (line 540)
    # Processing the call keyword arguments (line 540)
    kwargs_254576 = {}
    # Getting the type of 'f' (line 540)
    f_254574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 21), 'f', False)
    # Obtaining the member 'copy' of a type (line 540)
    copy_254575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 21), f_254574, 'copy')
    # Calling copy(args, kwargs) (line 540)
    copy_call_result_254577 = invoke(stypy.reporting.localization.Localization(__file__, 540, 21), copy_254575, *[], **kwargs_254576)
    
    # Assigning a type to the variable 'f_true' (line 540)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 12), 'f_true', copy_call_result_254577)
    
    # Assigning a Name to a Name (line 542):
    
    # Assigning a Name to a Name (line 542):
    # Getting the type of 'cost_new' (line 542)
    cost_new_254578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 19), 'cost_new')
    # Assigning a type to the variable 'cost' (line 542)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 12), 'cost', cost_new_254578)
    
    # Assigning a Call to a Name (line 544):
    
    # Assigning a Call to a Name (line 544):
    
    # Call to jac(...): (line 544)
    # Processing the call arguments (line 544)
    # Getting the type of 'x' (line 544)
    x_254580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 20), 'x', False)
    # Getting the type of 'f' (line 544)
    f_254581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 23), 'f', False)
    # Processing the call keyword arguments (line 544)
    kwargs_254582 = {}
    # Getting the type of 'jac' (line 544)
    jac_254579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 16), 'jac', False)
    # Calling jac(args, kwargs) (line 544)
    jac_call_result_254583 = invoke(stypy.reporting.localization.Localization(__file__, 544, 16), jac_254579, *[x_254580, f_254581], **kwargs_254582)
    
    # Assigning a type to the variable 'J' (line 544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 12), 'J', jac_call_result_254583)
    
    # Getting the type of 'njev' (line 545)
    njev_254584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 12), 'njev')
    int_254585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 20), 'int')
    # Applying the binary operator '+=' (line 545)
    result_iadd_254586 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 12), '+=', njev_254584, int_254585)
    # Assigning a type to the variable 'njev' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 12), 'njev', result_iadd_254586)
    
    
    # Type idiom detected: calculating its left and rigth part (line 547)
    # Getting the type of 'loss_function' (line 547)
    loss_function_254587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 12), 'loss_function')
    # Getting the type of 'None' (line 547)
    None_254588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 36), 'None')
    
    (may_be_254589, more_types_in_union_254590) = may_not_be_none(loss_function_254587, None_254588)

    if may_be_254589:

        if more_types_in_union_254590:
            # Runtime conditional SSA (line 547)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 548):
        
        # Assigning a Call to a Name (line 548):
        
        # Call to loss_function(...): (line 548)
        # Processing the call arguments (line 548)
        # Getting the type of 'f' (line 548)
        f_254592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 36), 'f', False)
        # Processing the call keyword arguments (line 548)
        kwargs_254593 = {}
        # Getting the type of 'loss_function' (line 548)
        loss_function_254591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 22), 'loss_function', False)
        # Calling loss_function(args, kwargs) (line 548)
        loss_function_call_result_254594 = invoke(stypy.reporting.localization.Localization(__file__, 548, 22), loss_function_254591, *[f_254592], **kwargs_254593)
        
        # Assigning a type to the variable 'rho' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 16), 'rho', loss_function_call_result_254594)
        
        # Assigning a Call to a Tuple (line 549):
        
        # Assigning a Subscript to a Name (line 549):
        
        # Obtaining the type of the subscript
        int_254595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 16), 'int')
        
        # Call to scale_for_robust_loss_function(...): (line 549)
        # Processing the call arguments (line 549)
        # Getting the type of 'J' (line 549)
        J_254597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 54), 'J', False)
        # Getting the type of 'f' (line 549)
        f_254598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 57), 'f', False)
        # Getting the type of 'rho' (line 549)
        rho_254599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 60), 'rho', False)
        # Processing the call keyword arguments (line 549)
        kwargs_254600 = {}
        # Getting the type of 'scale_for_robust_loss_function' (line 549)
        scale_for_robust_loss_function_254596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 23), 'scale_for_robust_loss_function', False)
        # Calling scale_for_robust_loss_function(args, kwargs) (line 549)
        scale_for_robust_loss_function_call_result_254601 = invoke(stypy.reporting.localization.Localization(__file__, 549, 23), scale_for_robust_loss_function_254596, *[J_254597, f_254598, rho_254599], **kwargs_254600)
        
        # Obtaining the member '__getitem__' of a type (line 549)
        getitem___254602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 16), scale_for_robust_loss_function_call_result_254601, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 549)
        subscript_call_result_254603 = invoke(stypy.reporting.localization.Localization(__file__, 549, 16), getitem___254602, int_254595)
        
        # Assigning a type to the variable 'tuple_var_assignment_252657' (line 549)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 16), 'tuple_var_assignment_252657', subscript_call_result_254603)
        
        # Assigning a Subscript to a Name (line 549):
        
        # Obtaining the type of the subscript
        int_254604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 16), 'int')
        
        # Call to scale_for_robust_loss_function(...): (line 549)
        # Processing the call arguments (line 549)
        # Getting the type of 'J' (line 549)
        J_254606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 54), 'J', False)
        # Getting the type of 'f' (line 549)
        f_254607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 57), 'f', False)
        # Getting the type of 'rho' (line 549)
        rho_254608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 60), 'rho', False)
        # Processing the call keyword arguments (line 549)
        kwargs_254609 = {}
        # Getting the type of 'scale_for_robust_loss_function' (line 549)
        scale_for_robust_loss_function_254605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 23), 'scale_for_robust_loss_function', False)
        # Calling scale_for_robust_loss_function(args, kwargs) (line 549)
        scale_for_robust_loss_function_call_result_254610 = invoke(stypy.reporting.localization.Localization(__file__, 549, 23), scale_for_robust_loss_function_254605, *[J_254606, f_254607, rho_254608], **kwargs_254609)
        
        # Obtaining the member '__getitem__' of a type (line 549)
        getitem___254611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 16), scale_for_robust_loss_function_call_result_254610, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 549)
        subscript_call_result_254612 = invoke(stypy.reporting.localization.Localization(__file__, 549, 16), getitem___254611, int_254604)
        
        # Assigning a type to the variable 'tuple_var_assignment_252658' (line 549)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 16), 'tuple_var_assignment_252658', subscript_call_result_254612)
        
        # Assigning a Name to a Name (line 549):
        # Getting the type of 'tuple_var_assignment_252657' (line 549)
        tuple_var_assignment_252657_254613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 16), 'tuple_var_assignment_252657')
        # Assigning a type to the variable 'J' (line 549)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 16), 'J', tuple_var_assignment_252657_254613)
        
        # Assigning a Name to a Name (line 549):
        # Getting the type of 'tuple_var_assignment_252658' (line 549)
        tuple_var_assignment_252658_254614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 16), 'tuple_var_assignment_252658')
        # Assigning a type to the variable 'f' (line 549)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 19), 'f', tuple_var_assignment_252658_254614)

        if more_types_in_union_254590:
            # SSA join for if statement (line 547)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 551):
    
    # Assigning a Call to a Name (line 551):
    
    # Call to compute_grad(...): (line 551)
    # Processing the call arguments (line 551)
    # Getting the type of 'J' (line 551)
    J_254616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 29), 'J', False)
    # Getting the type of 'f' (line 551)
    f_254617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 32), 'f', False)
    # Processing the call keyword arguments (line 551)
    kwargs_254618 = {}
    # Getting the type of 'compute_grad' (line 551)
    compute_grad_254615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 16), 'compute_grad', False)
    # Calling compute_grad(args, kwargs) (line 551)
    compute_grad_call_result_254619 = invoke(stypy.reporting.localization.Localization(__file__, 551, 16), compute_grad_254615, *[J_254616, f_254617], **kwargs_254618)
    
    # Assigning a type to the variable 'g' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 12), 'g', compute_grad_call_result_254619)
    
    # Getting the type of 'jac_scale' (line 553)
    jac_scale_254620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 15), 'jac_scale')
    # Testing the type of an if condition (line 553)
    if_condition_254621 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 553, 12), jac_scale_254620)
    # Assigning a type to the variable 'if_condition_254621' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 12), 'if_condition_254621', if_condition_254621)
    # SSA begins for if statement (line 553)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 554):
    
    # Assigning a Subscript to a Name (line 554):
    
    # Obtaining the type of the subscript
    int_254622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 16), 'int')
    
    # Call to compute_jac_scale(...): (line 554)
    # Processing the call arguments (line 554)
    # Getting the type of 'J' (line 554)
    J_254624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 53), 'J', False)
    # Getting the type of 'scale_inv' (line 554)
    scale_inv_254625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 56), 'scale_inv', False)
    # Processing the call keyword arguments (line 554)
    kwargs_254626 = {}
    # Getting the type of 'compute_jac_scale' (line 554)
    compute_jac_scale_254623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 35), 'compute_jac_scale', False)
    # Calling compute_jac_scale(args, kwargs) (line 554)
    compute_jac_scale_call_result_254627 = invoke(stypy.reporting.localization.Localization(__file__, 554, 35), compute_jac_scale_254623, *[J_254624, scale_inv_254625], **kwargs_254626)
    
    # Obtaining the member '__getitem__' of a type (line 554)
    getitem___254628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 16), compute_jac_scale_call_result_254627, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 554)
    subscript_call_result_254629 = invoke(stypy.reporting.localization.Localization(__file__, 554, 16), getitem___254628, int_254622)
    
    # Assigning a type to the variable 'tuple_var_assignment_252659' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 16), 'tuple_var_assignment_252659', subscript_call_result_254629)
    
    # Assigning a Subscript to a Name (line 554):
    
    # Obtaining the type of the subscript
    int_254630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 16), 'int')
    
    # Call to compute_jac_scale(...): (line 554)
    # Processing the call arguments (line 554)
    # Getting the type of 'J' (line 554)
    J_254632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 53), 'J', False)
    # Getting the type of 'scale_inv' (line 554)
    scale_inv_254633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 56), 'scale_inv', False)
    # Processing the call keyword arguments (line 554)
    kwargs_254634 = {}
    # Getting the type of 'compute_jac_scale' (line 554)
    compute_jac_scale_254631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 35), 'compute_jac_scale', False)
    # Calling compute_jac_scale(args, kwargs) (line 554)
    compute_jac_scale_call_result_254635 = invoke(stypy.reporting.localization.Localization(__file__, 554, 35), compute_jac_scale_254631, *[J_254632, scale_inv_254633], **kwargs_254634)
    
    # Obtaining the member '__getitem__' of a type (line 554)
    getitem___254636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 16), compute_jac_scale_call_result_254635, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 554)
    subscript_call_result_254637 = invoke(stypy.reporting.localization.Localization(__file__, 554, 16), getitem___254636, int_254630)
    
    # Assigning a type to the variable 'tuple_var_assignment_252660' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 16), 'tuple_var_assignment_252660', subscript_call_result_254637)
    
    # Assigning a Name to a Name (line 554):
    # Getting the type of 'tuple_var_assignment_252659' (line 554)
    tuple_var_assignment_252659_254638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 16), 'tuple_var_assignment_252659')
    # Assigning a type to the variable 'scale' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 16), 'scale', tuple_var_assignment_252659_254638)
    
    # Assigning a Name to a Name (line 554):
    # Getting the type of 'tuple_var_assignment_252660' (line 554)
    tuple_var_assignment_252660_254639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 16), 'tuple_var_assignment_252660')
    # Assigning a type to the variable 'scale_inv' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 23), 'scale_inv', tuple_var_assignment_252660_254639)
    # SSA join for if statement (line 553)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 536)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 556):
    
    # Assigning a Num to a Name (line 556):
    int_254640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 24), 'int')
    # Assigning a type to the variable 'step_norm' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 12), 'step_norm', int_254640)
    
    # Assigning a Num to a Name (line 557):
    
    # Assigning a Num to a Name (line 557):
    int_254641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 31), 'int')
    # Assigning a type to the variable 'actual_reduction' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 12), 'actual_reduction', int_254641)
    # SSA join for if statement (line 536)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'iteration' (line 559)
    iteration_254642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'iteration')
    int_254643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 21), 'int')
    # Applying the binary operator '+=' (line 559)
    result_iadd_254644 = python_operator(stypy.reporting.localization.Localization(__file__, 559, 8), '+=', iteration_254642, int_254643)
    # Assigning a type to the variable 'iteration' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'iteration', result_iadd_254644)
    
    # SSA join for while statement (line 458)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 561)
    # Getting the type of 'termination_status' (line 561)
    termination_status_254645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 7), 'termination_status')
    # Getting the type of 'None' (line 561)
    None_254646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 29), 'None')
    
    (may_be_254647, more_types_in_union_254648) = may_be_none(termination_status_254645, None_254646)

    if may_be_254647:

        if more_types_in_union_254648:
            # Runtime conditional SSA (line 561)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Num to a Name (line 562):
        
        # Assigning a Num to a Name (line 562):
        int_254649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 29), 'int')
        # Assigning a type to the variable 'termination_status' (line 562)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'termination_status', int_254649)

        if more_types_in_union_254648:
            # SSA join for if statement (line 561)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 564):
    
    # Assigning a Call to a Name (line 564):
    
    # Call to zeros_like(...): (line 564)
    # Processing the call arguments (line 564)
    # Getting the type of 'x' (line 564)
    x_254652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 32), 'x', False)
    # Processing the call keyword arguments (line 564)
    kwargs_254653 = {}
    # Getting the type of 'np' (line 564)
    np_254650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 18), 'np', False)
    # Obtaining the member 'zeros_like' of a type (line 564)
    zeros_like_254651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 18), np_254650, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 564)
    zeros_like_call_result_254654 = invoke(stypy.reporting.localization.Localization(__file__, 564, 18), zeros_like_254651, *[x_254652], **kwargs_254653)
    
    # Assigning a type to the variable 'active_mask' (line 564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 4), 'active_mask', zeros_like_call_result_254654)
    
    # Call to OptimizeResult(...): (line 565)
    # Processing the call keyword arguments (line 565)
    # Getting the type of 'x' (line 566)
    x_254656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 10), 'x', False)
    keyword_254657 = x_254656
    # Getting the type of 'cost' (line 566)
    cost_254658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 18), 'cost', False)
    keyword_254659 = cost_254658
    # Getting the type of 'f_true' (line 566)
    f_true_254660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 28), 'f_true', False)
    keyword_254661 = f_true_254660
    # Getting the type of 'J' (line 566)
    J_254662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 40), 'J', False)
    keyword_254663 = J_254662
    # Getting the type of 'g' (line 566)
    g_254664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 48), 'g', False)
    keyword_254665 = g_254664
    # Getting the type of 'g_norm' (line 566)
    g_norm_254666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 62), 'g_norm', False)
    keyword_254667 = g_norm_254666
    # Getting the type of 'active_mask' (line 567)
    active_mask_254668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 20), 'active_mask', False)
    keyword_254669 = active_mask_254668
    # Getting the type of 'nfev' (line 567)
    nfev_254670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 38), 'nfev', False)
    keyword_254671 = nfev_254670
    # Getting the type of 'njev' (line 567)
    njev_254672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 49), 'njev', False)
    keyword_254673 = njev_254672
    # Getting the type of 'termination_status' (line 568)
    termination_status_254674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 15), 'termination_status', False)
    keyword_254675 = termination_status_254674
    kwargs_254676 = {'status': keyword_254675, 'njev': keyword_254673, 'nfev': keyword_254671, 'active_mask': keyword_254669, 'cost': keyword_254659, 'optimality': keyword_254667, 'fun': keyword_254661, 'x': keyword_254657, 'grad': keyword_254665, 'jac': keyword_254663}
    # Getting the type of 'OptimizeResult' (line 565)
    OptimizeResult_254655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 11), 'OptimizeResult', False)
    # Calling OptimizeResult(args, kwargs) (line 565)
    OptimizeResult_call_result_254677 = invoke(stypy.reporting.localization.Localization(__file__, 565, 11), OptimizeResult_254655, *[], **kwargs_254676)
    
    # Assigning a type to the variable 'stypy_return_type' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'stypy_return_type', OptimizeResult_call_result_254677)
    
    # ################# End of 'trf_no_bounds(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'trf_no_bounds' in the type store
    # Getting the type of 'stypy_return_type' (line 409)
    stypy_return_type_254678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_254678)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'trf_no_bounds'
    return stypy_return_type_254678

# Assigning a type to the variable 'trf_no_bounds' (line 409)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 0), 'trf_no_bounds', trf_no_bounds)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
