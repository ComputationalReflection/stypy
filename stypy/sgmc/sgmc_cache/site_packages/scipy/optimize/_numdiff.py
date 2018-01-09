
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Routines for numerical differentiation.'''
2: 
3: from __future__ import division
4: 
5: import numpy as np
6: 
7: from ..sparse import issparse, csc_matrix, csr_matrix, coo_matrix, find
8: from ._group_columns import group_dense, group_sparse
9: 
10: EPS = np.finfo(np.float64).eps
11: 
12: 
13: def _adjust_scheme_to_bounds(x0, h, num_steps, scheme, lb, ub):
14:     '''Adjust final difference scheme to the presence of bounds.
15: 
16:     Parameters
17:     ----------
18:     x0 : ndarray, shape (n,)
19:         Point at which we wish to estimate derivative.
20:     h : ndarray, shape (n,)
21:         Desired finite difference steps.
22:     num_steps : int
23:         Number of `h` steps in one direction required to implement finite
24:         difference scheme. For example, 2 means that we need to evaluate
25:         f(x0 + 2 * h) or f(x0 - 2 * h)
26:     scheme : {'1-sided', '2-sided'}
27:         Whether steps in one or both directions are required. In other
28:         words '1-sided' applies to forward and backward schemes, '2-sided'
29:         applies to center schemes.
30:     lb : ndarray, shape (n,)
31:         Lower bounds on independent variables.
32:     ub : ndarray, shape (n,)
33:         Upper bounds on independent variables.
34: 
35:     Returns
36:     -------
37:     h_adjusted : ndarray, shape (n,)
38:         Adjusted step sizes. Step size decreases only if a sign flip or
39:         switching to one-sided scheme doesn't allow to take a full step.
40:     use_one_sided : ndarray of bool, shape (n,)
41:         Whether to switch to one-sided scheme. Informative only for
42:         ``scheme='2-sided'``.
43:     '''
44:     if scheme == '1-sided':
45:         use_one_sided = np.ones_like(h, dtype=bool)
46:     elif scheme == '2-sided':
47:         h = np.abs(h)
48:         use_one_sided = np.zeros_like(h, dtype=bool)
49:     else:
50:         raise ValueError("`scheme` must be '1-sided' or '2-sided'.")
51: 
52:     if np.all((lb == -np.inf) & (ub == np.inf)):
53:         return h, use_one_sided
54: 
55:     h_total = h * num_steps
56:     h_adjusted = h.copy()
57: 
58:     lower_dist = x0 - lb
59:     upper_dist = ub - x0
60: 
61:     if scheme == '1-sided':
62:         x = x0 + h_total
63:         violated = (x < lb) | (x > ub)
64:         fitting = np.abs(h_total) <= np.maximum(lower_dist, upper_dist)
65:         h_adjusted[violated & fitting] *= -1
66: 
67:         forward = (upper_dist >= lower_dist) & ~fitting
68:         h_adjusted[forward] = upper_dist[forward] / num_steps
69:         backward = (upper_dist < lower_dist) & ~fitting
70:         h_adjusted[backward] = -lower_dist[backward] / num_steps
71:     elif scheme == '2-sided':
72:         central = (lower_dist >= h_total) & (upper_dist >= h_total)
73: 
74:         forward = (upper_dist >= lower_dist) & ~central
75:         h_adjusted[forward] = np.minimum(
76:             h[forward], 0.5 * upper_dist[forward] / num_steps)
77:         use_one_sided[forward] = True
78: 
79:         backward = (upper_dist < lower_dist) & ~central
80:         h_adjusted[backward] = -np.minimum(
81:             h[backward], 0.5 * lower_dist[backward] / num_steps)
82:         use_one_sided[backward] = True
83: 
84:         min_dist = np.minimum(upper_dist, lower_dist) / num_steps
85:         adjusted_central = (~central & (np.abs(h_adjusted) <= min_dist))
86:         h_adjusted[adjusted_central] = min_dist[adjusted_central]
87:         use_one_sided[adjusted_central] = False
88: 
89:     return h_adjusted, use_one_sided
90: 
91: 
92: def _compute_absolute_step(rel_step, x0, method):
93:     if rel_step is None:
94:         if method == '2-point':
95:             rel_step = EPS**0.5
96:         elif method == '3-point':
97:             rel_step = EPS**(1 / 3)
98:         elif method == 'cs':
99:             rel_step = EPS**(0.5)
100:         else:
101:             raise ValueError("`method` must be '2-point' or '3-point'.")
102: 
103:     sign_x0 = (x0 >= 0).astype(float) * 2 - 1
104:     return rel_step * sign_x0 * np.maximum(1.0, np.abs(x0))
105: 
106: 
107: def _prepare_bounds(bounds, x0):
108:     lb, ub = [np.asarray(b, dtype=float) for b in bounds]
109:     if lb.ndim == 0:
110:         lb = np.resize(lb, x0.shape)
111: 
112:     if ub.ndim == 0:
113:         ub = np.resize(ub, x0.shape)
114: 
115:     return lb, ub
116: 
117: 
118: def group_columns(A, order=0):
119:     '''Group columns of a 2-d matrix for sparse finite differencing [1]_.
120: 
121:     Two columns are in the same group if in each row at least one of them
122:     has zero. A greedy sequential algorithm is used to construct groups.
123: 
124:     Parameters
125:     ----------
126:     A : array_like or sparse matrix, shape (m, n)
127:         Matrix of which to group columns.
128:     order : int, iterable of int with shape (n,) or None
129:         Permutation array which defines the order of columns enumeration.
130:         If int or None, a random permutation is used with `order` used as
131:         a random seed. Default is 0, that is use a random permutation but
132:         guarantee repeatability.
133: 
134:     Returns
135:     -------
136:     groups : ndarray of int, shape (n,)
137:         Contains values from 0 to n_groups-1, where n_groups is the number
138:         of found groups. Each value ``groups[i]`` is an index of a group to
139:         which i-th column assigned. The procedure was helpful only if
140:         n_groups is significantly less than n.
141: 
142:     References
143:     ----------
144:     .. [1] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
145:            sparse Jacobian matrices", Journal of the Institute of Mathematics
146:            and its Applications, 13 (1974), pp. 117-120.
147:     '''
148:     if issparse(A):
149:         A = csc_matrix(A)
150:     else:
151:         A = np.atleast_2d(A)
152:         A = (A != 0).astype(np.int32)
153: 
154:     if A.ndim != 2:
155:         raise ValueError("`A` must be 2-dimensional.")
156: 
157:     m, n = A.shape
158: 
159:     if order is None or np.isscalar(order):
160:         rng = np.random.RandomState(order)
161:         order = rng.permutation(n)
162:     else:
163:         order = np.asarray(order)
164:         if order.shape != (n,):
165:             raise ValueError("`order` has incorrect shape.")
166: 
167:     A = A[:, order]
168: 
169:     if issparse(A):
170:         groups = group_sparse(m, n, A.indices, A.indptr)
171:     else:
172:         groups = group_dense(m, n, A)
173: 
174:     groups[order] = groups.copy()
175: 
176:     return groups
177: 
178: 
179: def approx_derivative(fun, x0, method='3-point', rel_step=None, f0=None,
180:                       bounds=(-np.inf, np.inf), sparsity=None, args=(),
181:                       kwargs={}):
182:     '''Compute finite difference approximation of the derivatives of a
183:     vector-valued function.
184: 
185:     If a function maps from R^n to R^m, its derivatives form m-by-n matrix
186:     called the Jacobian, where an element (i, j) is a partial derivative of
187:     f[i] with respect to x[j].
188: 
189:     Parameters
190:     ----------
191:     fun : callable
192:         Function of which to estimate the derivatives. The argument x
193:         passed to this function is ndarray of shape (n,) (never a scalar
194:         even if n=1). It must return 1-d array_like of shape (m,) or a scalar.
195:     x0 : array_like of shape (n,) or float
196:         Point at which to estimate the derivatives. Float will be converted
197:         to a 1-d array.
198:     method : {'3-point', '2-point'}, optional
199:         Finite difference method to use:
200:             - '2-point' - use the fist order accuracy forward or backward
201:                           difference.
202:             - '3-point' - use central difference in interior points and the
203:                           second order accuracy forward or backward difference
204:                           near the boundary.
205:             - 'cs' - use a complex-step finite difference scheme. This assumes
206:                      that the user function is real-valued and can be
207:                      analytically continued to the complex plane. Otherwise,
208:                      produces bogus results.
209:     rel_step : None or array_like, optional
210:         Relative step size to use. The absolute step size is computed as
211:         ``h = rel_step * sign(x0) * max(1, abs(x0))``, possibly adjusted to
212:         fit into the bounds. For ``method='3-point'`` the sign of `h` is
213:         ignored. If None (default) then step is selected automatically,
214:         see Notes.
215:     f0 : None or array_like, optional
216:         If not None it is assumed to be equal to ``fun(x0)``, in  this case
217:         the ``fun(x0)`` is not called. Default is None.
218:     bounds : tuple of array_like, optional
219:         Lower and upper bounds on independent variables. Defaults to no bounds.
220:         Each bound must match the size of `x0` or be a scalar, in the latter
221:         case the bound will be the same for all variables. Use it to limit the
222:         range of function evaluation.
223:     sparsity : {None, array_like, sparse matrix, 2-tuple}, optional
224:         Defines a sparsity structure of the Jacobian matrix. If the Jacobian
225:         matrix is known to have only few non-zero elements in each row, then
226:         it's possible to estimate its several columns by a single function
227:         evaluation [3]_. To perform such economic computations two ingredients
228:         are required:
229: 
230:         * structure : array_like or sparse matrix of shape (m, n). A zero
231:           element means that a corresponding element of the Jacobian
232:           identically equals to zero.
233:         * groups : array_like of shape (n,). A column grouping for a given
234:           sparsity structure, use `group_columns` to obtain it.
235: 
236:         A single array or a sparse matrix is interpreted as a sparsity
237:         structure, and groups are computed inside the function. A tuple is
238:         interpreted as (structure, groups). If None (default), a standard
239:         dense differencing will be used.
240: 
241:         Note, that sparse differencing makes sense only for large Jacobian
242:         matrices where each row contains few non-zero elements.
243:     args, kwargs : tuple and dict, optional
244:         Additional arguments passed to `fun`. Both empty by default.
245:         The calling signature is ``fun(x, *args, **kwargs)``.
246: 
247:     Returns
248:     -------
249:     J : ndarray or csr_matrix
250:         Finite difference approximation of the Jacobian matrix. If `sparsity`
251:         is None then ndarray with shape (m, n) is returned. Although if m=1 it
252:         is returned as a gradient with shape (n,). If `sparsity` is not None,
253:         csr_matrix with shape (m, n) is returned.
254: 
255:     See Also
256:     --------
257:     check_derivative : Check correctness of a function computing derivatives.
258: 
259:     Notes
260:     -----
261:     If `rel_step` is not provided, it assigned to ``EPS**(1/s)``, where EPS is
262:     machine epsilon for float64 numbers, s=2 for '2-point' method and s=3 for
263:     '3-point' method. Such relative step approximately minimizes a sum of
264:     truncation and round-off errors, see [1]_.
265: 
266:     A finite difference scheme for '3-point' method is selected automatically.
267:     The well-known central difference scheme is used for points sufficiently
268:     far from the boundary, and 3-point forward or backward scheme is used for
269:     points near the boundary. Both schemes have the second-order accuracy in
270:     terms of Taylor expansion. Refer to [2]_ for the formulas of 3-point
271:     forward and backward difference schemes.
272: 
273:     For dense differencing when m=1 Jacobian is returned with a shape (n,),
274:     on the other hand when n=1 Jacobian is returned with a shape (m, 1).
275:     Our motivation is the following: a) It handles a case of gradient
276:     computation (m=1) in a conventional way. b) It clearly separates these two
277:     different cases. b) In all cases np.atleast_2d can be called to get 2-d
278:     Jacobian with correct dimensions.
279: 
280:     References
281:     ----------
282:     .. [1] W. H. Press et. al. "Numerical Recipes. The Art of Scientific
283:            Computing. 3rd edition", sec. 5.7.
284: 
285:     .. [2] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
286:            sparse Jacobian matrices", Journal of the Institute of Mathematics
287:            and its Applications, 13 (1974), pp. 117-120.
288: 
289:     .. [3] B. Fornberg, "Generation of Finite Difference Formulas on
290:            Arbitrarily Spaced Grids", Mathematics of Computation 51, 1988.
291: 
292:     Examples
293:     --------
294:     >>> import numpy as np
295:     >>> from scipy.optimize import approx_derivative
296:     >>>
297:     >>> def f(x, c1, c2):
298:     ...     return np.array([x[0] * np.sin(c1 * x[1]),
299:     ...                      x[0] * np.cos(c2 * x[1])])
300:     ...
301:     >>> x0 = np.array([1.0, 0.5 * np.pi])
302:     >>> approx_derivative(f, x0, args=(1, 2))
303:     array([[ 1.,  0.],
304:            [-1.,  0.]])
305: 
306:     Bounds can be used to limit the region of function evaluation.
307:     In the example below we compute left and right derivative at point 1.0.
308: 
309:     >>> def g(x):
310:     ...     return x**2 if x >= 1 else x
311:     ...
312:     >>> x0 = 1.0
313:     >>> approx_derivative(g, x0, bounds=(-np.inf, 1.0))
314:     array([ 1.])
315:     >>> approx_derivative(g, x0, bounds=(1.0, np.inf))
316:     array([ 2.])
317:     '''
318:     if method not in ['2-point', '3-point', 'cs']:
319:         raise ValueError("Unknown method '%s'. " % method)
320: 
321:     x0 = np.atleast_1d(x0)
322:     if x0.ndim > 1:
323:         raise ValueError("`x0` must have at most 1 dimension.")
324: 
325:     lb, ub = _prepare_bounds(bounds, x0)
326: 
327:     if lb.shape != x0.shape or ub.shape != x0.shape:
328:         raise ValueError("Inconsistent shapes between bounds and `x0`.")
329: 
330:     def fun_wrapped(x):
331:         f = np.atleast_1d(fun(x, *args, **kwargs))
332:         if f.ndim > 1:
333:             raise RuntimeError(("`fun` return value has "
334:                                 "more than 1 dimension."))
335:         return f
336: 
337:     if f0 is None:
338:         f0 = fun_wrapped(x0)
339:     else:
340:         f0 = np.atleast_1d(f0)
341:         if f0.ndim > 1:
342:             raise ValueError("`f0` passed has more than 1 dimension.")
343: 
344:     if np.any((x0 < lb) | (x0 > ub)):
345:         raise ValueError("`x0` violates bound constraints.")
346: 
347:     h = _compute_absolute_step(rel_step, x0, method)
348: 
349:     if method == '2-point':
350:         h, use_one_sided = _adjust_scheme_to_bounds(
351:             x0, h, 1, '1-sided', lb, ub)
352:     elif method == '3-point':
353:         h, use_one_sided = _adjust_scheme_to_bounds(
354:             x0, h, 1, '2-sided', lb, ub)
355:     elif method == 'cs':
356:         use_one_sided = False
357: 
358:     if sparsity is None:
359:         return _dense_difference(fun_wrapped, x0, f0, h, use_one_sided, method)
360:     else:
361:         if not issparse(sparsity) and len(sparsity) == 2:
362:             structure, groups = sparsity
363:         else:
364:             structure = sparsity
365:             groups = group_columns(sparsity)
366: 
367:         if issparse(structure):
368:             structure = csc_matrix(structure)
369:         else:
370:             structure = np.atleast_2d(structure)
371: 
372:         groups = np.atleast_1d(groups)
373:         return _sparse_difference(fun_wrapped, x0, f0, h, use_one_sided,
374:                                   structure, groups, method)
375: 
376: 
377: def _dense_difference(fun, x0, f0, h, use_one_sided, method):
378:     m = f0.size
379:     n = x0.size
380:     J_transposed = np.empty((n, m))
381:     h_vecs = np.diag(h)
382: 
383:     for i in range(h.size):
384:         if method == '2-point':
385:             x = x0 + h_vecs[i]
386:             dx = x[i] - x0[i]  # Recompute dx as exactly representable number.
387:             df = fun(x) - f0
388:         elif method == '3-point' and use_one_sided[i]:
389:             x1 = x0 + h_vecs[i]
390:             x2 = x0 + 2 * h_vecs[i]
391:             dx = x2[i] - x0[i]
392:             f1 = fun(x1)
393:             f2 = fun(x2)
394:             df = -3.0 * f0 + 4 * f1 - f2
395:         elif method == '3-point' and not use_one_sided[i]:
396:             x1 = x0 - h_vecs[i]
397:             x2 = x0 + h_vecs[i]
398:             dx = x2[i] - x1[i]
399:             f1 = fun(x1)
400:             f2 = fun(x2)
401:             df = f2 - f1
402:         elif method == 'cs':
403:             f1 = fun(x0 + h_vecs[i]*1.j)
404:             df = f1.imag
405:             dx = h_vecs[i, i]
406:         else:
407:             raise RuntimeError("Never be here.")
408: 
409:         J_transposed[i] = df / dx
410: 
411:     if m == 1:
412:         J_transposed = np.ravel(J_transposed)
413: 
414:     return J_transposed.T
415: 
416: 
417: def _sparse_difference(fun, x0, f0, h, use_one_sided,
418:                        structure, groups, method):
419:     m = f0.size
420:     n = x0.size
421:     row_indices = []
422:     col_indices = []
423:     fractions = []
424: 
425:     n_groups = np.max(groups) + 1
426:     for group in range(n_groups):
427:         # Perturb variables which are in the same group simultaneously.
428:         e = np.equal(group, groups)
429:         h_vec = h * e
430:         if method == '2-point':
431:             x = x0 + h_vec
432:             dx = x - x0
433:             df = fun(x) - f0
434:             # The result is  written to columns which correspond to perturbed
435:             # variables.
436:             cols, = np.nonzero(e)
437:             # Find all non-zero elements in selected columns of Jacobian.
438:             i, j, _ = find(structure[:, cols])
439:             # Restore column indices in the full array.
440:             j = cols[j]
441:         elif method == '3-point':
442:             # Here we do conceptually the same but separate one-sided
443:             # and two-sided schemes.
444:             x1 = x0.copy()
445:             x2 = x0.copy()
446: 
447:             mask_1 = use_one_sided & e
448:             x1[mask_1] += h_vec[mask_1]
449:             x2[mask_1] += 2 * h_vec[mask_1]
450: 
451:             mask_2 = ~use_one_sided & e
452:             x1[mask_2] -= h_vec[mask_2]
453:             x2[mask_2] += h_vec[mask_2]
454: 
455:             dx = np.zeros(n)
456:             dx[mask_1] = x2[mask_1] - x0[mask_1]
457:             dx[mask_2] = x2[mask_2] - x1[mask_2]
458: 
459:             f1 = fun(x1)
460:             f2 = fun(x2)
461: 
462:             cols, = np.nonzero(e)
463:             i, j, _ = find(structure[:, cols])
464:             j = cols[j]
465: 
466:             mask = use_one_sided[j]
467:             df = np.empty(m)
468: 
469:             rows = i[mask]
470:             df[rows] = -3 * f0[rows] + 4 * f1[rows] - f2[rows]
471: 
472:             rows = i[~mask]
473:             df[rows] = f2[rows] - f1[rows]
474:         elif method == 'cs':
475:             f1 = fun(x0 + h_vec*1.j)
476:             df = f1.imag
477:             dx = h_vec
478:             cols, = np.nonzero(e)
479:             i, j, _ = find(structure[:, cols])
480:             j = cols[j]
481:         else:
482:             raise ValueError("Never be here.")
483: 
484:         # All that's left is to compute the fraction. We store i, j and
485:         # fractions as separate arrays and later construct coo_matrix.
486:         row_indices.append(i)
487:         col_indices.append(j)
488:         fractions.append(df[i] / dx[j])
489: 
490:     row_indices = np.hstack(row_indices)
491:     col_indices = np.hstack(col_indices)
492:     fractions = np.hstack(fractions)
493:     J = coo_matrix((fractions, (row_indices, col_indices)), shape=(m, n))
494:     return csr_matrix(J)
495: 
496: 
497: def check_derivative(fun, jac, x0, bounds=(-np.inf, np.inf), args=(),
498:                      kwargs={}):
499:     '''Check correctness of a function computing derivatives (Jacobian or
500:     gradient) by comparison with a finite difference approximation.
501: 
502:     Parameters
503:     ----------
504:     fun : callable
505:         Function of which to estimate the derivatives. The argument x
506:         passed to this function is ndarray of shape (n,) (never a scalar
507:         even if n=1). It must return 1-d array_like of shape (m,) or a scalar.
508:     jac : callable
509:         Function which computes Jacobian matrix of `fun`. It must work with
510:         argument x the same way as `fun`. The return value must be array_like
511:         or sparse matrix with an appropriate shape.
512:     x0 : array_like of shape (n,) or float
513:         Point at which to estimate the derivatives. Float will be converted
514:         to 1-d array.
515:     bounds : 2-tuple of array_like, optional
516:         Lower and upper bounds on independent variables. Defaults to no bounds.
517:         Each bound must match the size of `x0` or be a scalar, in the latter
518:         case the bound will be the same for all variables. Use it to limit the
519:         range of function evaluation.
520:     args, kwargs : tuple and dict, optional
521:         Additional arguments passed to `fun` and `jac`. Both empty by default.
522:         The calling signature is ``fun(x, *args, **kwargs)`` and the same
523:         for `jac`.
524: 
525:     Returns
526:     -------
527:     accuracy : float
528:         The maximum among all relative errors for elements with absolute values
529:         higher than 1 and absolute errors for elements with absolute values
530:         less or equal than 1. If `accuracy` is on the order of 1e-6 or lower,
531:         then it is likely that your `jac` implementation is correct.
532: 
533:     See Also
534:     --------
535:     approx_derivative : Compute finite difference approximation of derivative.
536: 
537:     Examples
538:     --------
539:     >>> import numpy as np
540:     >>> from scipy.optimize import check_derivative
541:     >>>
542:     >>>
543:     >>> def f(x, c1, c2):
544:     ...     return np.array([x[0] * np.sin(c1 * x[1]),
545:     ...                      x[0] * np.cos(c2 * x[1])])
546:     ...
547:     >>> def jac(x, c1, c2):
548:     ...     return np.array([
549:     ...         [np.sin(c1 * x[1]),  c1 * x[0] * np.cos(c1 * x[1])],
550:     ...         [np.cos(c2 * x[1]), -c2 * x[0] * np.sin(c2 * x[1])]
551:     ...     ])
552:     ...
553:     >>>
554:     >>> x0 = np.array([1.0, 0.5 * np.pi])
555:     >>> check_derivative(f, jac, x0, args=(1, 2))
556:     2.4492935982947064e-16
557:     '''
558:     J_to_test = jac(x0, *args, **kwargs)
559:     if issparse(J_to_test):
560:         J_diff = approx_derivative(fun, x0, bounds=bounds, sparsity=J_to_test,
561:                                    args=args, kwargs=kwargs)
562:         J_to_test = csr_matrix(J_to_test)
563:         abs_err = J_to_test - J_diff
564:         i, j, abs_err_data = find(abs_err)
565:         J_diff_data = np.asarray(J_diff[i, j]).ravel()
566:         return np.max(np.abs(abs_err_data) /
567:                       np.maximum(1, np.abs(J_diff_data)))
568:     else:
569:         J_diff = approx_derivative(fun, x0, bounds=bounds,
570:                                    args=args, kwargs=kwargs)
571:         abs_err = np.abs(J_to_test - J_diff)
572:         return np.max(abs_err / np.maximum(1, np.abs(J_diff)))
573: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_198624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Routines for numerical differentiation.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_198625 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_198625) is not StypyTypeError):

    if (import_198625 != 'pyd_module'):
        __import__(import_198625)
        sys_modules_198626 = sys.modules[import_198625]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_198626.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_198625)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.sparse import issparse, csc_matrix, csr_matrix, coo_matrix, find' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_198627 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse')

if (type(import_198627) is not StypyTypeError):

    if (import_198627 != 'pyd_module'):
        __import__(import_198627)
        sys_modules_198628 = sys.modules[import_198627]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse', sys_modules_198628.module_type_store, module_type_store, ['issparse', 'csc_matrix', 'csr_matrix', 'coo_matrix', 'find'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_198628, sys_modules_198628.module_type_store, module_type_store)
    else:
        from scipy.sparse import issparse, csc_matrix, csr_matrix, coo_matrix, find

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse', None, module_type_store, ['issparse', 'csc_matrix', 'csr_matrix', 'coo_matrix', 'find'], [issparse, csc_matrix, csr_matrix, coo_matrix, find])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse', import_198627)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.optimize._group_columns import group_dense, group_sparse' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_198629 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize._group_columns')

if (type(import_198629) is not StypyTypeError):

    if (import_198629 != 'pyd_module'):
        __import__(import_198629)
        sys_modules_198630 = sys.modules[import_198629]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize._group_columns', sys_modules_198630.module_type_store, module_type_store, ['group_dense', 'group_sparse'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_198630, sys_modules_198630.module_type_store, module_type_store)
    else:
        from scipy.optimize._group_columns import group_dense, group_sparse

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize._group_columns', None, module_type_store, ['group_dense', 'group_sparse'], [group_dense, group_sparse])

else:
    # Assigning a type to the variable 'scipy.optimize._group_columns' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize._group_columns', import_198629)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')


# Assigning a Attribute to a Name (line 10):

# Assigning a Attribute to a Name (line 10):

# Call to finfo(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'np' (line 10)
np_198633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 15), 'np', False)
# Obtaining the member 'float64' of a type (line 10)
float64_198634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 15), np_198633, 'float64')
# Processing the call keyword arguments (line 10)
kwargs_198635 = {}
# Getting the type of 'np' (line 10)
np_198631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 6), 'np', False)
# Obtaining the member 'finfo' of a type (line 10)
finfo_198632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 6), np_198631, 'finfo')
# Calling finfo(args, kwargs) (line 10)
finfo_call_result_198636 = invoke(stypy.reporting.localization.Localization(__file__, 10, 6), finfo_198632, *[float64_198634], **kwargs_198635)

# Obtaining the member 'eps' of a type (line 10)
eps_198637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 6), finfo_call_result_198636, 'eps')
# Assigning a type to the variable 'EPS' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'EPS', eps_198637)

@norecursion
def _adjust_scheme_to_bounds(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_adjust_scheme_to_bounds'
    module_type_store = module_type_store.open_function_context('_adjust_scheme_to_bounds', 13, 0, False)
    
    # Passed parameters checking function
    _adjust_scheme_to_bounds.stypy_localization = localization
    _adjust_scheme_to_bounds.stypy_type_of_self = None
    _adjust_scheme_to_bounds.stypy_type_store = module_type_store
    _adjust_scheme_to_bounds.stypy_function_name = '_adjust_scheme_to_bounds'
    _adjust_scheme_to_bounds.stypy_param_names_list = ['x0', 'h', 'num_steps', 'scheme', 'lb', 'ub']
    _adjust_scheme_to_bounds.stypy_varargs_param_name = None
    _adjust_scheme_to_bounds.stypy_kwargs_param_name = None
    _adjust_scheme_to_bounds.stypy_call_defaults = defaults
    _adjust_scheme_to_bounds.stypy_call_varargs = varargs
    _adjust_scheme_to_bounds.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_adjust_scheme_to_bounds', ['x0', 'h', 'num_steps', 'scheme', 'lb', 'ub'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_adjust_scheme_to_bounds', localization, ['x0', 'h', 'num_steps', 'scheme', 'lb', 'ub'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_adjust_scheme_to_bounds(...)' code ##################

    str_198638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, (-1)), 'str', "Adjust final difference scheme to the presence of bounds.\n\n    Parameters\n    ----------\n    x0 : ndarray, shape (n,)\n        Point at which we wish to estimate derivative.\n    h : ndarray, shape (n,)\n        Desired finite difference steps.\n    num_steps : int\n        Number of `h` steps in one direction required to implement finite\n        difference scheme. For example, 2 means that we need to evaluate\n        f(x0 + 2 * h) or f(x0 - 2 * h)\n    scheme : {'1-sided', '2-sided'}\n        Whether steps in one or both directions are required. In other\n        words '1-sided' applies to forward and backward schemes, '2-sided'\n        applies to center schemes.\n    lb : ndarray, shape (n,)\n        Lower bounds on independent variables.\n    ub : ndarray, shape (n,)\n        Upper bounds on independent variables.\n\n    Returns\n    -------\n    h_adjusted : ndarray, shape (n,)\n        Adjusted step sizes. Step size decreases only if a sign flip or\n        switching to one-sided scheme doesn't allow to take a full step.\n    use_one_sided : ndarray of bool, shape (n,)\n        Whether to switch to one-sided scheme. Informative only for\n        ``scheme='2-sided'``.\n    ")
    
    
    # Getting the type of 'scheme' (line 44)
    scheme_198639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 7), 'scheme')
    str_198640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 17), 'str', '1-sided')
    # Applying the binary operator '==' (line 44)
    result_eq_198641 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 7), '==', scheme_198639, str_198640)
    
    # Testing the type of an if condition (line 44)
    if_condition_198642 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 4), result_eq_198641)
    # Assigning a type to the variable 'if_condition_198642' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'if_condition_198642', if_condition_198642)
    # SSA begins for if statement (line 44)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 45):
    
    # Assigning a Call to a Name (line 45):
    
    # Call to ones_like(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'h' (line 45)
    h_198645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 37), 'h', False)
    # Processing the call keyword arguments (line 45)
    # Getting the type of 'bool' (line 45)
    bool_198646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 46), 'bool', False)
    keyword_198647 = bool_198646
    kwargs_198648 = {'dtype': keyword_198647}
    # Getting the type of 'np' (line 45)
    np_198643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 24), 'np', False)
    # Obtaining the member 'ones_like' of a type (line 45)
    ones_like_198644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 24), np_198643, 'ones_like')
    # Calling ones_like(args, kwargs) (line 45)
    ones_like_call_result_198649 = invoke(stypy.reporting.localization.Localization(__file__, 45, 24), ones_like_198644, *[h_198645], **kwargs_198648)
    
    # Assigning a type to the variable 'use_one_sided' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'use_one_sided', ones_like_call_result_198649)
    # SSA branch for the else part of an if statement (line 44)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'scheme' (line 46)
    scheme_198650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 9), 'scheme')
    str_198651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 19), 'str', '2-sided')
    # Applying the binary operator '==' (line 46)
    result_eq_198652 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 9), '==', scheme_198650, str_198651)
    
    # Testing the type of an if condition (line 46)
    if_condition_198653 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 9), result_eq_198652)
    # Assigning a type to the variable 'if_condition_198653' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 9), 'if_condition_198653', if_condition_198653)
    # SSA begins for if statement (line 46)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 47):
    
    # Assigning a Call to a Name (line 47):
    
    # Call to abs(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'h' (line 47)
    h_198656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'h', False)
    # Processing the call keyword arguments (line 47)
    kwargs_198657 = {}
    # Getting the type of 'np' (line 47)
    np_198654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'np', False)
    # Obtaining the member 'abs' of a type (line 47)
    abs_198655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), np_198654, 'abs')
    # Calling abs(args, kwargs) (line 47)
    abs_call_result_198658 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), abs_198655, *[h_198656], **kwargs_198657)
    
    # Assigning a type to the variable 'h' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'h', abs_call_result_198658)
    
    # Assigning a Call to a Name (line 48):
    
    # Assigning a Call to a Name (line 48):
    
    # Call to zeros_like(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'h' (line 48)
    h_198661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 38), 'h', False)
    # Processing the call keyword arguments (line 48)
    # Getting the type of 'bool' (line 48)
    bool_198662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 47), 'bool', False)
    keyword_198663 = bool_198662
    kwargs_198664 = {'dtype': keyword_198663}
    # Getting the type of 'np' (line 48)
    np_198659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 24), 'np', False)
    # Obtaining the member 'zeros_like' of a type (line 48)
    zeros_like_198660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 24), np_198659, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 48)
    zeros_like_call_result_198665 = invoke(stypy.reporting.localization.Localization(__file__, 48, 24), zeros_like_198660, *[h_198661], **kwargs_198664)
    
    # Assigning a type to the variable 'use_one_sided' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'use_one_sided', zeros_like_call_result_198665)
    # SSA branch for the else part of an if statement (line 46)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 50)
    # Processing the call arguments (line 50)
    str_198667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 25), 'str', "`scheme` must be '1-sided' or '2-sided'.")
    # Processing the call keyword arguments (line 50)
    kwargs_198668 = {}
    # Getting the type of 'ValueError' (line 50)
    ValueError_198666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 50)
    ValueError_call_result_198669 = invoke(stypy.reporting.localization.Localization(__file__, 50, 14), ValueError_198666, *[str_198667], **kwargs_198668)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 50, 8), ValueError_call_result_198669, 'raise parameter', BaseException)
    # SSA join for if statement (line 46)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 44)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to all(...): (line 52)
    # Processing the call arguments (line 52)
    
    # Getting the type of 'lb' (line 52)
    lb_198672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 15), 'lb', False)
    
    # Getting the type of 'np' (line 52)
    np_198673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 22), 'np', False)
    # Obtaining the member 'inf' of a type (line 52)
    inf_198674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 22), np_198673, 'inf')
    # Applying the 'usub' unary operator (line 52)
    result___neg___198675 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 21), 'usub', inf_198674)
    
    # Applying the binary operator '==' (line 52)
    result_eq_198676 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 15), '==', lb_198672, result___neg___198675)
    
    
    # Getting the type of 'ub' (line 52)
    ub_198677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 33), 'ub', False)
    # Getting the type of 'np' (line 52)
    np_198678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 39), 'np', False)
    # Obtaining the member 'inf' of a type (line 52)
    inf_198679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 39), np_198678, 'inf')
    # Applying the binary operator '==' (line 52)
    result_eq_198680 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 33), '==', ub_198677, inf_198679)
    
    # Applying the binary operator '&' (line 52)
    result_and__198681 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 14), '&', result_eq_198676, result_eq_198680)
    
    # Processing the call keyword arguments (line 52)
    kwargs_198682 = {}
    # Getting the type of 'np' (line 52)
    np_198670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 7), 'np', False)
    # Obtaining the member 'all' of a type (line 52)
    all_198671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 7), np_198670, 'all')
    # Calling all(args, kwargs) (line 52)
    all_call_result_198683 = invoke(stypy.reporting.localization.Localization(__file__, 52, 7), all_198671, *[result_and__198681], **kwargs_198682)
    
    # Testing the type of an if condition (line 52)
    if_condition_198684 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 4), all_call_result_198683)
    # Assigning a type to the variable 'if_condition_198684' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'if_condition_198684', if_condition_198684)
    # SSA begins for if statement (line 52)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 53)
    tuple_198685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 53)
    # Adding element type (line 53)
    # Getting the type of 'h' (line 53)
    h_198686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 15), 'h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 15), tuple_198685, h_198686)
    # Adding element type (line 53)
    # Getting the type of 'use_one_sided' (line 53)
    use_one_sided_198687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 18), 'use_one_sided')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 15), tuple_198685, use_one_sided_198687)
    
    # Assigning a type to the variable 'stypy_return_type' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'stypy_return_type', tuple_198685)
    # SSA join for if statement (line 52)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 55):
    
    # Assigning a BinOp to a Name (line 55):
    # Getting the type of 'h' (line 55)
    h_198688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 14), 'h')
    # Getting the type of 'num_steps' (line 55)
    num_steps_198689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 18), 'num_steps')
    # Applying the binary operator '*' (line 55)
    result_mul_198690 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 14), '*', h_198688, num_steps_198689)
    
    # Assigning a type to the variable 'h_total' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'h_total', result_mul_198690)
    
    # Assigning a Call to a Name (line 56):
    
    # Assigning a Call to a Name (line 56):
    
    # Call to copy(...): (line 56)
    # Processing the call keyword arguments (line 56)
    kwargs_198693 = {}
    # Getting the type of 'h' (line 56)
    h_198691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 17), 'h', False)
    # Obtaining the member 'copy' of a type (line 56)
    copy_198692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 17), h_198691, 'copy')
    # Calling copy(args, kwargs) (line 56)
    copy_call_result_198694 = invoke(stypy.reporting.localization.Localization(__file__, 56, 17), copy_198692, *[], **kwargs_198693)
    
    # Assigning a type to the variable 'h_adjusted' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'h_adjusted', copy_call_result_198694)
    
    # Assigning a BinOp to a Name (line 58):
    
    # Assigning a BinOp to a Name (line 58):
    # Getting the type of 'x0' (line 58)
    x0_198695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 17), 'x0')
    # Getting the type of 'lb' (line 58)
    lb_198696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 22), 'lb')
    # Applying the binary operator '-' (line 58)
    result_sub_198697 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 17), '-', x0_198695, lb_198696)
    
    # Assigning a type to the variable 'lower_dist' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'lower_dist', result_sub_198697)
    
    # Assigning a BinOp to a Name (line 59):
    
    # Assigning a BinOp to a Name (line 59):
    # Getting the type of 'ub' (line 59)
    ub_198698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 17), 'ub')
    # Getting the type of 'x0' (line 59)
    x0_198699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 22), 'x0')
    # Applying the binary operator '-' (line 59)
    result_sub_198700 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 17), '-', ub_198698, x0_198699)
    
    # Assigning a type to the variable 'upper_dist' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'upper_dist', result_sub_198700)
    
    
    # Getting the type of 'scheme' (line 61)
    scheme_198701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 7), 'scheme')
    str_198702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 17), 'str', '1-sided')
    # Applying the binary operator '==' (line 61)
    result_eq_198703 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 7), '==', scheme_198701, str_198702)
    
    # Testing the type of an if condition (line 61)
    if_condition_198704 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 4), result_eq_198703)
    # Assigning a type to the variable 'if_condition_198704' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'if_condition_198704', if_condition_198704)
    # SSA begins for if statement (line 61)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 62):
    
    # Assigning a BinOp to a Name (line 62):
    # Getting the type of 'x0' (line 62)
    x0_198705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'x0')
    # Getting the type of 'h_total' (line 62)
    h_total_198706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 17), 'h_total')
    # Applying the binary operator '+' (line 62)
    result_add_198707 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 12), '+', x0_198705, h_total_198706)
    
    # Assigning a type to the variable 'x' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'x', result_add_198707)
    
    # Assigning a BinOp to a Name (line 63):
    
    # Assigning a BinOp to a Name (line 63):
    
    # Getting the type of 'x' (line 63)
    x_198708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'x')
    # Getting the type of 'lb' (line 63)
    lb_198709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'lb')
    # Applying the binary operator '<' (line 63)
    result_lt_198710 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 20), '<', x_198708, lb_198709)
    
    
    # Getting the type of 'x' (line 63)
    x_198711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 31), 'x')
    # Getting the type of 'ub' (line 63)
    ub_198712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 35), 'ub')
    # Applying the binary operator '>' (line 63)
    result_gt_198713 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 31), '>', x_198711, ub_198712)
    
    # Applying the binary operator '|' (line 63)
    result_or__198714 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 19), '|', result_lt_198710, result_gt_198713)
    
    # Assigning a type to the variable 'violated' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'violated', result_or__198714)
    
    # Assigning a Compare to a Name (line 64):
    
    # Assigning a Compare to a Name (line 64):
    
    
    # Call to abs(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'h_total' (line 64)
    h_total_198717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 25), 'h_total', False)
    # Processing the call keyword arguments (line 64)
    kwargs_198718 = {}
    # Getting the type of 'np' (line 64)
    np_198715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 18), 'np', False)
    # Obtaining the member 'abs' of a type (line 64)
    abs_198716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 18), np_198715, 'abs')
    # Calling abs(args, kwargs) (line 64)
    abs_call_result_198719 = invoke(stypy.reporting.localization.Localization(__file__, 64, 18), abs_198716, *[h_total_198717], **kwargs_198718)
    
    
    # Call to maximum(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'lower_dist' (line 64)
    lower_dist_198722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 48), 'lower_dist', False)
    # Getting the type of 'upper_dist' (line 64)
    upper_dist_198723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 60), 'upper_dist', False)
    # Processing the call keyword arguments (line 64)
    kwargs_198724 = {}
    # Getting the type of 'np' (line 64)
    np_198720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 37), 'np', False)
    # Obtaining the member 'maximum' of a type (line 64)
    maximum_198721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 37), np_198720, 'maximum')
    # Calling maximum(args, kwargs) (line 64)
    maximum_call_result_198725 = invoke(stypy.reporting.localization.Localization(__file__, 64, 37), maximum_198721, *[lower_dist_198722, upper_dist_198723], **kwargs_198724)
    
    # Applying the binary operator '<=' (line 64)
    result_le_198726 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 18), '<=', abs_call_result_198719, maximum_call_result_198725)
    
    # Assigning a type to the variable 'fitting' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'fitting', result_le_198726)
    
    # Getting the type of 'h_adjusted' (line 65)
    h_adjusted_198727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'h_adjusted')
    
    # Obtaining the type of the subscript
    # Getting the type of 'violated' (line 65)
    violated_198728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'violated')
    # Getting the type of 'fitting' (line 65)
    fitting_198729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 30), 'fitting')
    # Applying the binary operator '&' (line 65)
    result_and__198730 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 19), '&', violated_198728, fitting_198729)
    
    # Getting the type of 'h_adjusted' (line 65)
    h_adjusted_198731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'h_adjusted')
    # Obtaining the member '__getitem__' of a type (line 65)
    getitem___198732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), h_adjusted_198731, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 65)
    subscript_call_result_198733 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), getitem___198732, result_and__198730)
    
    int_198734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 42), 'int')
    # Applying the binary operator '*=' (line 65)
    result_imul_198735 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 8), '*=', subscript_call_result_198733, int_198734)
    # Getting the type of 'h_adjusted' (line 65)
    h_adjusted_198736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'h_adjusted')
    # Getting the type of 'violated' (line 65)
    violated_198737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'violated')
    # Getting the type of 'fitting' (line 65)
    fitting_198738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 30), 'fitting')
    # Applying the binary operator '&' (line 65)
    result_and__198739 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 19), '&', violated_198737, fitting_198738)
    
    # Storing an element on a container (line 65)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 8), h_adjusted_198736, (result_and__198739, result_imul_198735))
    
    
    # Assigning a BinOp to a Name (line 67):
    
    # Assigning a BinOp to a Name (line 67):
    
    # Getting the type of 'upper_dist' (line 67)
    upper_dist_198740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 19), 'upper_dist')
    # Getting the type of 'lower_dist' (line 67)
    lower_dist_198741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 33), 'lower_dist')
    # Applying the binary operator '>=' (line 67)
    result_ge_198742 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 19), '>=', upper_dist_198740, lower_dist_198741)
    
    
    # Getting the type of 'fitting' (line 67)
    fitting_198743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 48), 'fitting')
    # Applying the '~' unary operator (line 67)
    result_inv_198744 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 47), '~', fitting_198743)
    
    # Applying the binary operator '&' (line 67)
    result_and__198745 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 18), '&', result_ge_198742, result_inv_198744)
    
    # Assigning a type to the variable 'forward' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'forward', result_and__198745)
    
    # Assigning a BinOp to a Subscript (line 68):
    
    # Assigning a BinOp to a Subscript (line 68):
    
    # Obtaining the type of the subscript
    # Getting the type of 'forward' (line 68)
    forward_198746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 41), 'forward')
    # Getting the type of 'upper_dist' (line 68)
    upper_dist_198747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 30), 'upper_dist')
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___198748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 30), upper_dist_198747, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_198749 = invoke(stypy.reporting.localization.Localization(__file__, 68, 30), getitem___198748, forward_198746)
    
    # Getting the type of 'num_steps' (line 68)
    num_steps_198750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 52), 'num_steps')
    # Applying the binary operator 'div' (line 68)
    result_div_198751 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 30), 'div', subscript_call_result_198749, num_steps_198750)
    
    # Getting the type of 'h_adjusted' (line 68)
    h_adjusted_198752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'h_adjusted')
    # Getting the type of 'forward' (line 68)
    forward_198753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 19), 'forward')
    # Storing an element on a container (line 68)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 8), h_adjusted_198752, (forward_198753, result_div_198751))
    
    # Assigning a BinOp to a Name (line 69):
    
    # Assigning a BinOp to a Name (line 69):
    
    # Getting the type of 'upper_dist' (line 69)
    upper_dist_198754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 20), 'upper_dist')
    # Getting the type of 'lower_dist' (line 69)
    lower_dist_198755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 33), 'lower_dist')
    # Applying the binary operator '<' (line 69)
    result_lt_198756 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 20), '<', upper_dist_198754, lower_dist_198755)
    
    
    # Getting the type of 'fitting' (line 69)
    fitting_198757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 48), 'fitting')
    # Applying the '~' unary operator (line 69)
    result_inv_198758 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 47), '~', fitting_198757)
    
    # Applying the binary operator '&' (line 69)
    result_and__198759 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 19), '&', result_lt_198756, result_inv_198758)
    
    # Assigning a type to the variable 'backward' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'backward', result_and__198759)
    
    # Assigning a BinOp to a Subscript (line 70):
    
    # Assigning a BinOp to a Subscript (line 70):
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'backward' (line 70)
    backward_198760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 43), 'backward')
    # Getting the type of 'lower_dist' (line 70)
    lower_dist_198761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 32), 'lower_dist')
    # Obtaining the member '__getitem__' of a type (line 70)
    getitem___198762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 32), lower_dist_198761, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 70)
    subscript_call_result_198763 = invoke(stypy.reporting.localization.Localization(__file__, 70, 32), getitem___198762, backward_198760)
    
    # Applying the 'usub' unary operator (line 70)
    result___neg___198764 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 31), 'usub', subscript_call_result_198763)
    
    # Getting the type of 'num_steps' (line 70)
    num_steps_198765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 55), 'num_steps')
    # Applying the binary operator 'div' (line 70)
    result_div_198766 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 31), 'div', result___neg___198764, num_steps_198765)
    
    # Getting the type of 'h_adjusted' (line 70)
    h_adjusted_198767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'h_adjusted')
    # Getting the type of 'backward' (line 70)
    backward_198768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), 'backward')
    # Storing an element on a container (line 70)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 8), h_adjusted_198767, (backward_198768, result_div_198766))
    # SSA branch for the else part of an if statement (line 61)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'scheme' (line 71)
    scheme_198769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 9), 'scheme')
    str_198770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 19), 'str', '2-sided')
    # Applying the binary operator '==' (line 71)
    result_eq_198771 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 9), '==', scheme_198769, str_198770)
    
    # Testing the type of an if condition (line 71)
    if_condition_198772 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 9), result_eq_198771)
    # Assigning a type to the variable 'if_condition_198772' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 9), 'if_condition_198772', if_condition_198772)
    # SSA begins for if statement (line 71)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 72):
    
    # Assigning a BinOp to a Name (line 72):
    
    # Getting the type of 'lower_dist' (line 72)
    lower_dist_198773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), 'lower_dist')
    # Getting the type of 'h_total' (line 72)
    h_total_198774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 33), 'h_total')
    # Applying the binary operator '>=' (line 72)
    result_ge_198775 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 19), '>=', lower_dist_198773, h_total_198774)
    
    
    # Getting the type of 'upper_dist' (line 72)
    upper_dist_198776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 45), 'upper_dist')
    # Getting the type of 'h_total' (line 72)
    h_total_198777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 59), 'h_total')
    # Applying the binary operator '>=' (line 72)
    result_ge_198778 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 45), '>=', upper_dist_198776, h_total_198777)
    
    # Applying the binary operator '&' (line 72)
    result_and__198779 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 18), '&', result_ge_198775, result_ge_198778)
    
    # Assigning a type to the variable 'central' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'central', result_and__198779)
    
    # Assigning a BinOp to a Name (line 74):
    
    # Assigning a BinOp to a Name (line 74):
    
    # Getting the type of 'upper_dist' (line 74)
    upper_dist_198780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 19), 'upper_dist')
    # Getting the type of 'lower_dist' (line 74)
    lower_dist_198781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 33), 'lower_dist')
    # Applying the binary operator '>=' (line 74)
    result_ge_198782 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 19), '>=', upper_dist_198780, lower_dist_198781)
    
    
    # Getting the type of 'central' (line 74)
    central_198783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 48), 'central')
    # Applying the '~' unary operator (line 74)
    result_inv_198784 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 47), '~', central_198783)
    
    # Applying the binary operator '&' (line 74)
    result_and__198785 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 18), '&', result_ge_198782, result_inv_198784)
    
    # Assigning a type to the variable 'forward' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'forward', result_and__198785)
    
    # Assigning a Call to a Subscript (line 75):
    
    # Assigning a Call to a Subscript (line 75):
    
    # Call to minimum(...): (line 75)
    # Processing the call arguments (line 75)
    
    # Obtaining the type of the subscript
    # Getting the type of 'forward' (line 76)
    forward_198788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 14), 'forward', False)
    # Getting the type of 'h' (line 76)
    h_198789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'h', False)
    # Obtaining the member '__getitem__' of a type (line 76)
    getitem___198790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), h_198789, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 76)
    subscript_call_result_198791 = invoke(stypy.reporting.localization.Localization(__file__, 76, 12), getitem___198790, forward_198788)
    
    float_198792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 24), 'float')
    
    # Obtaining the type of the subscript
    # Getting the type of 'forward' (line 76)
    forward_198793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 41), 'forward', False)
    # Getting the type of 'upper_dist' (line 76)
    upper_dist_198794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 30), 'upper_dist', False)
    # Obtaining the member '__getitem__' of a type (line 76)
    getitem___198795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 30), upper_dist_198794, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 76)
    subscript_call_result_198796 = invoke(stypy.reporting.localization.Localization(__file__, 76, 30), getitem___198795, forward_198793)
    
    # Applying the binary operator '*' (line 76)
    result_mul_198797 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 24), '*', float_198792, subscript_call_result_198796)
    
    # Getting the type of 'num_steps' (line 76)
    num_steps_198798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 52), 'num_steps', False)
    # Applying the binary operator 'div' (line 76)
    result_div_198799 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 50), 'div', result_mul_198797, num_steps_198798)
    
    # Processing the call keyword arguments (line 75)
    kwargs_198800 = {}
    # Getting the type of 'np' (line 75)
    np_198786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 30), 'np', False)
    # Obtaining the member 'minimum' of a type (line 75)
    minimum_198787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 30), np_198786, 'minimum')
    # Calling minimum(args, kwargs) (line 75)
    minimum_call_result_198801 = invoke(stypy.reporting.localization.Localization(__file__, 75, 30), minimum_198787, *[subscript_call_result_198791, result_div_198799], **kwargs_198800)
    
    # Getting the type of 'h_adjusted' (line 75)
    h_adjusted_198802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'h_adjusted')
    # Getting the type of 'forward' (line 75)
    forward_198803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 19), 'forward')
    # Storing an element on a container (line 75)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 8), h_adjusted_198802, (forward_198803, minimum_call_result_198801))
    
    # Assigning a Name to a Subscript (line 77):
    
    # Assigning a Name to a Subscript (line 77):
    # Getting the type of 'True' (line 77)
    True_198804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 33), 'True')
    # Getting the type of 'use_one_sided' (line 77)
    use_one_sided_198805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'use_one_sided')
    # Getting the type of 'forward' (line 77)
    forward_198806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 22), 'forward')
    # Storing an element on a container (line 77)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 8), use_one_sided_198805, (forward_198806, True_198804))
    
    # Assigning a BinOp to a Name (line 79):
    
    # Assigning a BinOp to a Name (line 79):
    
    # Getting the type of 'upper_dist' (line 79)
    upper_dist_198807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'upper_dist')
    # Getting the type of 'lower_dist' (line 79)
    lower_dist_198808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 33), 'lower_dist')
    # Applying the binary operator '<' (line 79)
    result_lt_198809 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 20), '<', upper_dist_198807, lower_dist_198808)
    
    
    # Getting the type of 'central' (line 79)
    central_198810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 48), 'central')
    # Applying the '~' unary operator (line 79)
    result_inv_198811 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 47), '~', central_198810)
    
    # Applying the binary operator '&' (line 79)
    result_and__198812 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 19), '&', result_lt_198809, result_inv_198811)
    
    # Assigning a type to the variable 'backward' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'backward', result_and__198812)
    
    # Assigning a UnaryOp to a Subscript (line 80):
    
    # Assigning a UnaryOp to a Subscript (line 80):
    
    
    # Call to minimum(...): (line 80)
    # Processing the call arguments (line 80)
    
    # Obtaining the type of the subscript
    # Getting the type of 'backward' (line 81)
    backward_198815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 14), 'backward', False)
    # Getting the type of 'h' (line 81)
    h_198816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'h', False)
    # Obtaining the member '__getitem__' of a type (line 81)
    getitem___198817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), h_198816, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 81)
    subscript_call_result_198818 = invoke(stypy.reporting.localization.Localization(__file__, 81, 12), getitem___198817, backward_198815)
    
    float_198819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 25), 'float')
    
    # Obtaining the type of the subscript
    # Getting the type of 'backward' (line 81)
    backward_198820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 42), 'backward', False)
    # Getting the type of 'lower_dist' (line 81)
    lower_dist_198821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 31), 'lower_dist', False)
    # Obtaining the member '__getitem__' of a type (line 81)
    getitem___198822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 31), lower_dist_198821, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 81)
    subscript_call_result_198823 = invoke(stypy.reporting.localization.Localization(__file__, 81, 31), getitem___198822, backward_198820)
    
    # Applying the binary operator '*' (line 81)
    result_mul_198824 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 25), '*', float_198819, subscript_call_result_198823)
    
    # Getting the type of 'num_steps' (line 81)
    num_steps_198825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 54), 'num_steps', False)
    # Applying the binary operator 'div' (line 81)
    result_div_198826 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 52), 'div', result_mul_198824, num_steps_198825)
    
    # Processing the call keyword arguments (line 80)
    kwargs_198827 = {}
    # Getting the type of 'np' (line 80)
    np_198813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 32), 'np', False)
    # Obtaining the member 'minimum' of a type (line 80)
    minimum_198814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 32), np_198813, 'minimum')
    # Calling minimum(args, kwargs) (line 80)
    minimum_call_result_198828 = invoke(stypy.reporting.localization.Localization(__file__, 80, 32), minimum_198814, *[subscript_call_result_198818, result_div_198826], **kwargs_198827)
    
    # Applying the 'usub' unary operator (line 80)
    result___neg___198829 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 31), 'usub', minimum_call_result_198828)
    
    # Getting the type of 'h_adjusted' (line 80)
    h_adjusted_198830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'h_adjusted')
    # Getting the type of 'backward' (line 80)
    backward_198831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 19), 'backward')
    # Storing an element on a container (line 80)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 8), h_adjusted_198830, (backward_198831, result___neg___198829))
    
    # Assigning a Name to a Subscript (line 82):
    
    # Assigning a Name to a Subscript (line 82):
    # Getting the type of 'True' (line 82)
    True_198832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 34), 'True')
    # Getting the type of 'use_one_sided' (line 82)
    use_one_sided_198833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'use_one_sided')
    # Getting the type of 'backward' (line 82)
    backward_198834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 22), 'backward')
    # Storing an element on a container (line 82)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 8), use_one_sided_198833, (backward_198834, True_198832))
    
    # Assigning a BinOp to a Name (line 84):
    
    # Assigning a BinOp to a Name (line 84):
    
    # Call to minimum(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'upper_dist' (line 84)
    upper_dist_198837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 30), 'upper_dist', False)
    # Getting the type of 'lower_dist' (line 84)
    lower_dist_198838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 42), 'lower_dist', False)
    # Processing the call keyword arguments (line 84)
    kwargs_198839 = {}
    # Getting the type of 'np' (line 84)
    np_198835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 19), 'np', False)
    # Obtaining the member 'minimum' of a type (line 84)
    minimum_198836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 19), np_198835, 'minimum')
    # Calling minimum(args, kwargs) (line 84)
    minimum_call_result_198840 = invoke(stypy.reporting.localization.Localization(__file__, 84, 19), minimum_198836, *[upper_dist_198837, lower_dist_198838], **kwargs_198839)
    
    # Getting the type of 'num_steps' (line 84)
    num_steps_198841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 56), 'num_steps')
    # Applying the binary operator 'div' (line 84)
    result_div_198842 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 19), 'div', minimum_call_result_198840, num_steps_198841)
    
    # Assigning a type to the variable 'min_dist' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'min_dist', result_div_198842)
    
    # Assigning a BinOp to a Name (line 85):
    
    # Assigning a BinOp to a Name (line 85):
    
    # Getting the type of 'central' (line 85)
    central_198843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 29), 'central')
    # Applying the '~' unary operator (line 85)
    result_inv_198844 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 28), '~', central_198843)
    
    
    
    # Call to abs(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'h_adjusted' (line 85)
    h_adjusted_198847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 47), 'h_adjusted', False)
    # Processing the call keyword arguments (line 85)
    kwargs_198848 = {}
    # Getting the type of 'np' (line 85)
    np_198845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 40), 'np', False)
    # Obtaining the member 'abs' of a type (line 85)
    abs_198846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 40), np_198845, 'abs')
    # Calling abs(args, kwargs) (line 85)
    abs_call_result_198849 = invoke(stypy.reporting.localization.Localization(__file__, 85, 40), abs_198846, *[h_adjusted_198847], **kwargs_198848)
    
    # Getting the type of 'min_dist' (line 85)
    min_dist_198850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 62), 'min_dist')
    # Applying the binary operator '<=' (line 85)
    result_le_198851 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 40), '<=', abs_call_result_198849, min_dist_198850)
    
    # Applying the binary operator '&' (line 85)
    result_and__198852 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 28), '&', result_inv_198844, result_le_198851)
    
    # Assigning a type to the variable 'adjusted_central' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'adjusted_central', result_and__198852)
    
    # Assigning a Subscript to a Subscript (line 86):
    
    # Assigning a Subscript to a Subscript (line 86):
    
    # Obtaining the type of the subscript
    # Getting the type of 'adjusted_central' (line 86)
    adjusted_central_198853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 48), 'adjusted_central')
    # Getting the type of 'min_dist' (line 86)
    min_dist_198854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 39), 'min_dist')
    # Obtaining the member '__getitem__' of a type (line 86)
    getitem___198855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 39), min_dist_198854, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 86)
    subscript_call_result_198856 = invoke(stypy.reporting.localization.Localization(__file__, 86, 39), getitem___198855, adjusted_central_198853)
    
    # Getting the type of 'h_adjusted' (line 86)
    h_adjusted_198857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'h_adjusted')
    # Getting the type of 'adjusted_central' (line 86)
    adjusted_central_198858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'adjusted_central')
    # Storing an element on a container (line 86)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 8), h_adjusted_198857, (adjusted_central_198858, subscript_call_result_198856))
    
    # Assigning a Name to a Subscript (line 87):
    
    # Assigning a Name to a Subscript (line 87):
    # Getting the type of 'False' (line 87)
    False_198859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 42), 'False')
    # Getting the type of 'use_one_sided' (line 87)
    use_one_sided_198860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'use_one_sided')
    # Getting the type of 'adjusted_central' (line 87)
    adjusted_central_198861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 22), 'adjusted_central')
    # Storing an element on a container (line 87)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 8), use_one_sided_198860, (adjusted_central_198861, False_198859))
    # SSA join for if statement (line 71)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 61)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 89)
    tuple_198862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 89)
    # Adding element type (line 89)
    # Getting the type of 'h_adjusted' (line 89)
    h_adjusted_198863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'h_adjusted')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 11), tuple_198862, h_adjusted_198863)
    # Adding element type (line 89)
    # Getting the type of 'use_one_sided' (line 89)
    use_one_sided_198864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 23), 'use_one_sided')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 11), tuple_198862, use_one_sided_198864)
    
    # Assigning a type to the variable 'stypy_return_type' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type', tuple_198862)
    
    # ################# End of '_adjust_scheme_to_bounds(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_adjust_scheme_to_bounds' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_198865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_198865)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_adjust_scheme_to_bounds'
    return stypy_return_type_198865

# Assigning a type to the variable '_adjust_scheme_to_bounds' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '_adjust_scheme_to_bounds', _adjust_scheme_to_bounds)

@norecursion
def _compute_absolute_step(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_compute_absolute_step'
    module_type_store = module_type_store.open_function_context('_compute_absolute_step', 92, 0, False)
    
    # Passed parameters checking function
    _compute_absolute_step.stypy_localization = localization
    _compute_absolute_step.stypy_type_of_self = None
    _compute_absolute_step.stypy_type_store = module_type_store
    _compute_absolute_step.stypy_function_name = '_compute_absolute_step'
    _compute_absolute_step.stypy_param_names_list = ['rel_step', 'x0', 'method']
    _compute_absolute_step.stypy_varargs_param_name = None
    _compute_absolute_step.stypy_kwargs_param_name = None
    _compute_absolute_step.stypy_call_defaults = defaults
    _compute_absolute_step.stypy_call_varargs = varargs
    _compute_absolute_step.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_compute_absolute_step', ['rel_step', 'x0', 'method'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_compute_absolute_step', localization, ['rel_step', 'x0', 'method'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_compute_absolute_step(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 93)
    # Getting the type of 'rel_step' (line 93)
    rel_step_198866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 7), 'rel_step')
    # Getting the type of 'None' (line 93)
    None_198867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 19), 'None')
    
    (may_be_198868, more_types_in_union_198869) = may_be_none(rel_step_198866, None_198867)

    if may_be_198868:

        if more_types_in_union_198869:
            # Runtime conditional SSA (line 93)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Getting the type of 'method' (line 94)
        method_198870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 11), 'method')
        str_198871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 21), 'str', '2-point')
        # Applying the binary operator '==' (line 94)
        result_eq_198872 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 11), '==', method_198870, str_198871)
        
        # Testing the type of an if condition (line 94)
        if_condition_198873 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 8), result_eq_198872)
        # Assigning a type to the variable 'if_condition_198873' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'if_condition_198873', if_condition_198873)
        # SSA begins for if statement (line 94)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 95):
        
        # Assigning a BinOp to a Name (line 95):
        # Getting the type of 'EPS' (line 95)
        EPS_198874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 23), 'EPS')
        float_198875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 28), 'float')
        # Applying the binary operator '**' (line 95)
        result_pow_198876 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 23), '**', EPS_198874, float_198875)
        
        # Assigning a type to the variable 'rel_step' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'rel_step', result_pow_198876)
        # SSA branch for the else part of an if statement (line 94)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'method' (line 96)
        method_198877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 13), 'method')
        str_198878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 23), 'str', '3-point')
        # Applying the binary operator '==' (line 96)
        result_eq_198879 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 13), '==', method_198877, str_198878)
        
        # Testing the type of an if condition (line 96)
        if_condition_198880 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 13), result_eq_198879)
        # Assigning a type to the variable 'if_condition_198880' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 13), 'if_condition_198880', if_condition_198880)
        # SSA begins for if statement (line 96)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 97):
        
        # Assigning a BinOp to a Name (line 97):
        # Getting the type of 'EPS' (line 97)
        EPS_198881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 23), 'EPS')
        int_198882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 29), 'int')
        int_198883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 33), 'int')
        # Applying the binary operator 'div' (line 97)
        result_div_198884 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 29), 'div', int_198882, int_198883)
        
        # Applying the binary operator '**' (line 97)
        result_pow_198885 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 23), '**', EPS_198881, result_div_198884)
        
        # Assigning a type to the variable 'rel_step' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'rel_step', result_pow_198885)
        # SSA branch for the else part of an if statement (line 96)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'method' (line 98)
        method_198886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 13), 'method')
        str_198887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 23), 'str', 'cs')
        # Applying the binary operator '==' (line 98)
        result_eq_198888 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 13), '==', method_198886, str_198887)
        
        # Testing the type of an if condition (line 98)
        if_condition_198889 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 13), result_eq_198888)
        # Assigning a type to the variable 'if_condition_198889' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 13), 'if_condition_198889', if_condition_198889)
        # SSA begins for if statement (line 98)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 99):
        
        # Assigning a BinOp to a Name (line 99):
        # Getting the type of 'EPS' (line 99)
        EPS_198890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 23), 'EPS')
        float_198891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 29), 'float')
        # Applying the binary operator '**' (line 99)
        result_pow_198892 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 23), '**', EPS_198890, float_198891)
        
        # Assigning a type to the variable 'rel_step' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'rel_step', result_pow_198892)
        # SSA branch for the else part of an if statement (line 98)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 101)
        # Processing the call arguments (line 101)
        str_198894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 29), 'str', "`method` must be '2-point' or '3-point'.")
        # Processing the call keyword arguments (line 101)
        kwargs_198895 = {}
        # Getting the type of 'ValueError' (line 101)
        ValueError_198893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 101)
        ValueError_call_result_198896 = invoke(stypy.reporting.localization.Localization(__file__, 101, 18), ValueError_198893, *[str_198894], **kwargs_198895)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 101, 12), ValueError_call_result_198896, 'raise parameter', BaseException)
        # SSA join for if statement (line 98)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 96)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 94)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_198869:
            # SSA join for if statement (line 93)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 103):
    
    # Assigning a BinOp to a Name (line 103):
    
    # Call to astype(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'float' (line 103)
    float_198901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 31), 'float', False)
    # Processing the call keyword arguments (line 103)
    kwargs_198902 = {}
    
    # Getting the type of 'x0' (line 103)
    x0_198897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'x0', False)
    int_198898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 21), 'int')
    # Applying the binary operator '>=' (line 103)
    result_ge_198899 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 15), '>=', x0_198897, int_198898)
    
    # Obtaining the member 'astype' of a type (line 103)
    astype_198900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 15), result_ge_198899, 'astype')
    # Calling astype(args, kwargs) (line 103)
    astype_call_result_198903 = invoke(stypy.reporting.localization.Localization(__file__, 103, 15), astype_198900, *[float_198901], **kwargs_198902)
    
    int_198904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 40), 'int')
    # Applying the binary operator '*' (line 103)
    result_mul_198905 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 14), '*', astype_call_result_198903, int_198904)
    
    int_198906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 44), 'int')
    # Applying the binary operator '-' (line 103)
    result_sub_198907 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 14), '-', result_mul_198905, int_198906)
    
    # Assigning a type to the variable 'sign_x0' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'sign_x0', result_sub_198907)
    # Getting the type of 'rel_step' (line 104)
    rel_step_198908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 11), 'rel_step')
    # Getting the type of 'sign_x0' (line 104)
    sign_x0_198909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 22), 'sign_x0')
    # Applying the binary operator '*' (line 104)
    result_mul_198910 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 11), '*', rel_step_198908, sign_x0_198909)
    
    
    # Call to maximum(...): (line 104)
    # Processing the call arguments (line 104)
    float_198913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 43), 'float')
    
    # Call to abs(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'x0' (line 104)
    x0_198916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 55), 'x0', False)
    # Processing the call keyword arguments (line 104)
    kwargs_198917 = {}
    # Getting the type of 'np' (line 104)
    np_198914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 48), 'np', False)
    # Obtaining the member 'abs' of a type (line 104)
    abs_198915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 48), np_198914, 'abs')
    # Calling abs(args, kwargs) (line 104)
    abs_call_result_198918 = invoke(stypy.reporting.localization.Localization(__file__, 104, 48), abs_198915, *[x0_198916], **kwargs_198917)
    
    # Processing the call keyword arguments (line 104)
    kwargs_198919 = {}
    # Getting the type of 'np' (line 104)
    np_198911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 32), 'np', False)
    # Obtaining the member 'maximum' of a type (line 104)
    maximum_198912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 32), np_198911, 'maximum')
    # Calling maximum(args, kwargs) (line 104)
    maximum_call_result_198920 = invoke(stypy.reporting.localization.Localization(__file__, 104, 32), maximum_198912, *[float_198913, abs_call_result_198918], **kwargs_198919)
    
    # Applying the binary operator '*' (line 104)
    result_mul_198921 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 30), '*', result_mul_198910, maximum_call_result_198920)
    
    # Assigning a type to the variable 'stypy_return_type' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'stypy_return_type', result_mul_198921)
    
    # ################# End of '_compute_absolute_step(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_compute_absolute_step' in the type store
    # Getting the type of 'stypy_return_type' (line 92)
    stypy_return_type_198922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_198922)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_compute_absolute_step'
    return stypy_return_type_198922

# Assigning a type to the variable '_compute_absolute_step' (line 92)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), '_compute_absolute_step', _compute_absolute_step)

@norecursion
def _prepare_bounds(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_prepare_bounds'
    module_type_store = module_type_store.open_function_context('_prepare_bounds', 107, 0, False)
    
    # Passed parameters checking function
    _prepare_bounds.stypy_localization = localization
    _prepare_bounds.stypy_type_of_self = None
    _prepare_bounds.stypy_type_store = module_type_store
    _prepare_bounds.stypy_function_name = '_prepare_bounds'
    _prepare_bounds.stypy_param_names_list = ['bounds', 'x0']
    _prepare_bounds.stypy_varargs_param_name = None
    _prepare_bounds.stypy_kwargs_param_name = None
    _prepare_bounds.stypy_call_defaults = defaults
    _prepare_bounds.stypy_call_varargs = varargs
    _prepare_bounds.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_prepare_bounds', ['bounds', 'x0'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_prepare_bounds', localization, ['bounds', 'x0'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_prepare_bounds(...)' code ##################

    
    # Assigning a ListComp to a Tuple (line 108):
    
    # Assigning a Subscript to a Name (line 108):
    
    # Obtaining the type of the subscript
    int_198923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'bounds' (line 108)
    bounds_198931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 50), 'bounds')
    comprehension_198932 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 14), bounds_198931)
    # Assigning a type to the variable 'b' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 'b', comprehension_198932)
    
    # Call to asarray(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'b' (line 108)
    b_198926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 25), 'b', False)
    # Processing the call keyword arguments (line 108)
    # Getting the type of 'float' (line 108)
    float_198927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 34), 'float', False)
    keyword_198928 = float_198927
    kwargs_198929 = {'dtype': keyword_198928}
    # Getting the type of 'np' (line 108)
    np_198924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 'np', False)
    # Obtaining the member 'asarray' of a type (line 108)
    asarray_198925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), np_198924, 'asarray')
    # Calling asarray(args, kwargs) (line 108)
    asarray_call_result_198930 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), asarray_198925, *[b_198926], **kwargs_198929)
    
    list_198933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 14), list_198933, asarray_call_result_198930)
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___198934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 4), list_198933, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_198935 = invoke(stypy.reporting.localization.Localization(__file__, 108, 4), getitem___198934, int_198923)
    
    # Assigning a type to the variable 'tuple_var_assignment_198597' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'tuple_var_assignment_198597', subscript_call_result_198935)
    
    # Assigning a Subscript to a Name (line 108):
    
    # Obtaining the type of the subscript
    int_198936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'bounds' (line 108)
    bounds_198944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 50), 'bounds')
    comprehension_198945 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 14), bounds_198944)
    # Assigning a type to the variable 'b' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 'b', comprehension_198945)
    
    # Call to asarray(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'b' (line 108)
    b_198939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 25), 'b', False)
    # Processing the call keyword arguments (line 108)
    # Getting the type of 'float' (line 108)
    float_198940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 34), 'float', False)
    keyword_198941 = float_198940
    kwargs_198942 = {'dtype': keyword_198941}
    # Getting the type of 'np' (line 108)
    np_198937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 'np', False)
    # Obtaining the member 'asarray' of a type (line 108)
    asarray_198938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), np_198937, 'asarray')
    # Calling asarray(args, kwargs) (line 108)
    asarray_call_result_198943 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), asarray_198938, *[b_198939], **kwargs_198942)
    
    list_198946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 14), list_198946, asarray_call_result_198943)
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___198947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 4), list_198946, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_198948 = invoke(stypy.reporting.localization.Localization(__file__, 108, 4), getitem___198947, int_198936)
    
    # Assigning a type to the variable 'tuple_var_assignment_198598' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'tuple_var_assignment_198598', subscript_call_result_198948)
    
    # Assigning a Name to a Name (line 108):
    # Getting the type of 'tuple_var_assignment_198597' (line 108)
    tuple_var_assignment_198597_198949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'tuple_var_assignment_198597')
    # Assigning a type to the variable 'lb' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'lb', tuple_var_assignment_198597_198949)
    
    # Assigning a Name to a Name (line 108):
    # Getting the type of 'tuple_var_assignment_198598' (line 108)
    tuple_var_assignment_198598_198950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'tuple_var_assignment_198598')
    # Assigning a type to the variable 'ub' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'ub', tuple_var_assignment_198598_198950)
    
    
    # Getting the type of 'lb' (line 109)
    lb_198951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 7), 'lb')
    # Obtaining the member 'ndim' of a type (line 109)
    ndim_198952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 7), lb_198951, 'ndim')
    int_198953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 18), 'int')
    # Applying the binary operator '==' (line 109)
    result_eq_198954 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 7), '==', ndim_198952, int_198953)
    
    # Testing the type of an if condition (line 109)
    if_condition_198955 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 4), result_eq_198954)
    # Assigning a type to the variable 'if_condition_198955' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'if_condition_198955', if_condition_198955)
    # SSA begins for if statement (line 109)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 110):
    
    # Assigning a Call to a Name (line 110):
    
    # Call to resize(...): (line 110)
    # Processing the call arguments (line 110)
    # Getting the type of 'lb' (line 110)
    lb_198958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 23), 'lb', False)
    # Getting the type of 'x0' (line 110)
    x0_198959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 27), 'x0', False)
    # Obtaining the member 'shape' of a type (line 110)
    shape_198960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 27), x0_198959, 'shape')
    # Processing the call keyword arguments (line 110)
    kwargs_198961 = {}
    # Getting the type of 'np' (line 110)
    np_198956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 13), 'np', False)
    # Obtaining the member 'resize' of a type (line 110)
    resize_198957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 13), np_198956, 'resize')
    # Calling resize(args, kwargs) (line 110)
    resize_call_result_198962 = invoke(stypy.reporting.localization.Localization(__file__, 110, 13), resize_198957, *[lb_198958, shape_198960], **kwargs_198961)
    
    # Assigning a type to the variable 'lb' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'lb', resize_call_result_198962)
    # SSA join for if statement (line 109)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'ub' (line 112)
    ub_198963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 7), 'ub')
    # Obtaining the member 'ndim' of a type (line 112)
    ndim_198964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 7), ub_198963, 'ndim')
    int_198965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 18), 'int')
    # Applying the binary operator '==' (line 112)
    result_eq_198966 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 7), '==', ndim_198964, int_198965)
    
    # Testing the type of an if condition (line 112)
    if_condition_198967 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 4), result_eq_198966)
    # Assigning a type to the variable 'if_condition_198967' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'if_condition_198967', if_condition_198967)
    # SSA begins for if statement (line 112)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 113):
    
    # Assigning a Call to a Name (line 113):
    
    # Call to resize(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'ub' (line 113)
    ub_198970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 23), 'ub', False)
    # Getting the type of 'x0' (line 113)
    x0_198971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 27), 'x0', False)
    # Obtaining the member 'shape' of a type (line 113)
    shape_198972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 27), x0_198971, 'shape')
    # Processing the call keyword arguments (line 113)
    kwargs_198973 = {}
    # Getting the type of 'np' (line 113)
    np_198968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 13), 'np', False)
    # Obtaining the member 'resize' of a type (line 113)
    resize_198969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 13), np_198968, 'resize')
    # Calling resize(args, kwargs) (line 113)
    resize_call_result_198974 = invoke(stypy.reporting.localization.Localization(__file__, 113, 13), resize_198969, *[ub_198970, shape_198972], **kwargs_198973)
    
    # Assigning a type to the variable 'ub' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'ub', resize_call_result_198974)
    # SSA join for if statement (line 112)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 115)
    tuple_198975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 115)
    # Adding element type (line 115)
    # Getting the type of 'lb' (line 115)
    lb_198976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 'lb')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 11), tuple_198975, lb_198976)
    # Adding element type (line 115)
    # Getting the type of 'ub' (line 115)
    ub_198977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 15), 'ub')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 11), tuple_198975, ub_198977)
    
    # Assigning a type to the variable 'stypy_return_type' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type', tuple_198975)
    
    # ################# End of '_prepare_bounds(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_prepare_bounds' in the type store
    # Getting the type of 'stypy_return_type' (line 107)
    stypy_return_type_198978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_198978)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_prepare_bounds'
    return stypy_return_type_198978

# Assigning a type to the variable '_prepare_bounds' (line 107)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), '_prepare_bounds', _prepare_bounds)

@norecursion
def group_columns(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_198979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 27), 'int')
    defaults = [int_198979]
    # Create a new context for function 'group_columns'
    module_type_store = module_type_store.open_function_context('group_columns', 118, 0, False)
    
    # Passed parameters checking function
    group_columns.stypy_localization = localization
    group_columns.stypy_type_of_self = None
    group_columns.stypy_type_store = module_type_store
    group_columns.stypy_function_name = 'group_columns'
    group_columns.stypy_param_names_list = ['A', 'order']
    group_columns.stypy_varargs_param_name = None
    group_columns.stypy_kwargs_param_name = None
    group_columns.stypy_call_defaults = defaults
    group_columns.stypy_call_varargs = varargs
    group_columns.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'group_columns', ['A', 'order'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'group_columns', localization, ['A', 'order'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'group_columns(...)' code ##################

    str_198980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, (-1)), 'str', 'Group columns of a 2-d matrix for sparse finite differencing [1]_.\n\n    Two columns are in the same group if in each row at least one of them\n    has zero. A greedy sequential algorithm is used to construct groups.\n\n    Parameters\n    ----------\n    A : array_like or sparse matrix, shape (m, n)\n        Matrix of which to group columns.\n    order : int, iterable of int with shape (n,) or None\n        Permutation array which defines the order of columns enumeration.\n        If int or None, a random permutation is used with `order` used as\n        a random seed. Default is 0, that is use a random permutation but\n        guarantee repeatability.\n\n    Returns\n    -------\n    groups : ndarray of int, shape (n,)\n        Contains values from 0 to n_groups-1, where n_groups is the number\n        of found groups. Each value ``groups[i]`` is an index of a group to\n        which i-th column assigned. The procedure was helpful only if\n        n_groups is significantly less than n.\n\n    References\n    ----------\n    .. [1] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of\n           sparse Jacobian matrices", Journal of the Institute of Mathematics\n           and its Applications, 13 (1974), pp. 117-120.\n    ')
    
    
    # Call to issparse(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'A' (line 148)
    A_198982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 16), 'A', False)
    # Processing the call keyword arguments (line 148)
    kwargs_198983 = {}
    # Getting the type of 'issparse' (line 148)
    issparse_198981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 7), 'issparse', False)
    # Calling issparse(args, kwargs) (line 148)
    issparse_call_result_198984 = invoke(stypy.reporting.localization.Localization(__file__, 148, 7), issparse_198981, *[A_198982], **kwargs_198983)
    
    # Testing the type of an if condition (line 148)
    if_condition_198985 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 4), issparse_call_result_198984)
    # Assigning a type to the variable 'if_condition_198985' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'if_condition_198985', if_condition_198985)
    # SSA begins for if statement (line 148)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 149):
    
    # Assigning a Call to a Name (line 149):
    
    # Call to csc_matrix(...): (line 149)
    # Processing the call arguments (line 149)
    # Getting the type of 'A' (line 149)
    A_198987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 23), 'A', False)
    # Processing the call keyword arguments (line 149)
    kwargs_198988 = {}
    # Getting the type of 'csc_matrix' (line 149)
    csc_matrix_198986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'csc_matrix', False)
    # Calling csc_matrix(args, kwargs) (line 149)
    csc_matrix_call_result_198989 = invoke(stypy.reporting.localization.Localization(__file__, 149, 12), csc_matrix_198986, *[A_198987], **kwargs_198988)
    
    # Assigning a type to the variable 'A' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'A', csc_matrix_call_result_198989)
    # SSA branch for the else part of an if statement (line 148)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 151):
    
    # Assigning a Call to a Name (line 151):
    
    # Call to atleast_2d(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'A' (line 151)
    A_198992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 26), 'A', False)
    # Processing the call keyword arguments (line 151)
    kwargs_198993 = {}
    # Getting the type of 'np' (line 151)
    np_198990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'np', False)
    # Obtaining the member 'atleast_2d' of a type (line 151)
    atleast_2d_198991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 12), np_198990, 'atleast_2d')
    # Calling atleast_2d(args, kwargs) (line 151)
    atleast_2d_call_result_198994 = invoke(stypy.reporting.localization.Localization(__file__, 151, 12), atleast_2d_198991, *[A_198992], **kwargs_198993)
    
    # Assigning a type to the variable 'A' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'A', atleast_2d_call_result_198994)
    
    # Assigning a Call to a Name (line 152):
    
    # Assigning a Call to a Name (line 152):
    
    # Call to astype(...): (line 152)
    # Processing the call arguments (line 152)
    # Getting the type of 'np' (line 152)
    np_198999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 28), 'np', False)
    # Obtaining the member 'int32' of a type (line 152)
    int32_199000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 28), np_198999, 'int32')
    # Processing the call keyword arguments (line 152)
    kwargs_199001 = {}
    
    # Getting the type of 'A' (line 152)
    A_198995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 13), 'A', False)
    int_198996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 18), 'int')
    # Applying the binary operator '!=' (line 152)
    result_ne_198997 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 13), '!=', A_198995, int_198996)
    
    # Obtaining the member 'astype' of a type (line 152)
    astype_198998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 13), result_ne_198997, 'astype')
    # Calling astype(args, kwargs) (line 152)
    astype_call_result_199002 = invoke(stypy.reporting.localization.Localization(__file__, 152, 13), astype_198998, *[int32_199000], **kwargs_199001)
    
    # Assigning a type to the variable 'A' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'A', astype_call_result_199002)
    # SSA join for if statement (line 148)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'A' (line 154)
    A_199003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 7), 'A')
    # Obtaining the member 'ndim' of a type (line 154)
    ndim_199004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 7), A_199003, 'ndim')
    int_199005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 17), 'int')
    # Applying the binary operator '!=' (line 154)
    result_ne_199006 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 7), '!=', ndim_199004, int_199005)
    
    # Testing the type of an if condition (line 154)
    if_condition_199007 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 154, 4), result_ne_199006)
    # Assigning a type to the variable 'if_condition_199007' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'if_condition_199007', if_condition_199007)
    # SSA begins for if statement (line 154)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 155)
    # Processing the call arguments (line 155)
    str_199009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 25), 'str', '`A` must be 2-dimensional.')
    # Processing the call keyword arguments (line 155)
    kwargs_199010 = {}
    # Getting the type of 'ValueError' (line 155)
    ValueError_199008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 155)
    ValueError_call_result_199011 = invoke(stypy.reporting.localization.Localization(__file__, 155, 14), ValueError_199008, *[str_199009], **kwargs_199010)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 155, 8), ValueError_call_result_199011, 'raise parameter', BaseException)
    # SSA join for if statement (line 154)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Tuple (line 157):
    
    # Assigning a Subscript to a Name (line 157):
    
    # Obtaining the type of the subscript
    int_199012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 4), 'int')
    # Getting the type of 'A' (line 157)
    A_199013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 11), 'A')
    # Obtaining the member 'shape' of a type (line 157)
    shape_199014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 11), A_199013, 'shape')
    # Obtaining the member '__getitem__' of a type (line 157)
    getitem___199015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 4), shape_199014, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 157)
    subscript_call_result_199016 = invoke(stypy.reporting.localization.Localization(__file__, 157, 4), getitem___199015, int_199012)
    
    # Assigning a type to the variable 'tuple_var_assignment_198599' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'tuple_var_assignment_198599', subscript_call_result_199016)
    
    # Assigning a Subscript to a Name (line 157):
    
    # Obtaining the type of the subscript
    int_199017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 4), 'int')
    # Getting the type of 'A' (line 157)
    A_199018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 11), 'A')
    # Obtaining the member 'shape' of a type (line 157)
    shape_199019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 11), A_199018, 'shape')
    # Obtaining the member '__getitem__' of a type (line 157)
    getitem___199020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 4), shape_199019, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 157)
    subscript_call_result_199021 = invoke(stypy.reporting.localization.Localization(__file__, 157, 4), getitem___199020, int_199017)
    
    # Assigning a type to the variable 'tuple_var_assignment_198600' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'tuple_var_assignment_198600', subscript_call_result_199021)
    
    # Assigning a Name to a Name (line 157):
    # Getting the type of 'tuple_var_assignment_198599' (line 157)
    tuple_var_assignment_198599_199022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'tuple_var_assignment_198599')
    # Assigning a type to the variable 'm' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'm', tuple_var_assignment_198599_199022)
    
    # Assigning a Name to a Name (line 157):
    # Getting the type of 'tuple_var_assignment_198600' (line 157)
    tuple_var_assignment_198600_199023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'tuple_var_assignment_198600')
    # Assigning a type to the variable 'n' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 7), 'n', tuple_var_assignment_198600_199023)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'order' (line 159)
    order_199024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 7), 'order')
    # Getting the type of 'None' (line 159)
    None_199025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'None')
    # Applying the binary operator 'is' (line 159)
    result_is__199026 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 7), 'is', order_199024, None_199025)
    
    
    # Call to isscalar(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of 'order' (line 159)
    order_199029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 36), 'order', False)
    # Processing the call keyword arguments (line 159)
    kwargs_199030 = {}
    # Getting the type of 'np' (line 159)
    np_199027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 24), 'np', False)
    # Obtaining the member 'isscalar' of a type (line 159)
    isscalar_199028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 24), np_199027, 'isscalar')
    # Calling isscalar(args, kwargs) (line 159)
    isscalar_call_result_199031 = invoke(stypy.reporting.localization.Localization(__file__, 159, 24), isscalar_199028, *[order_199029], **kwargs_199030)
    
    # Applying the binary operator 'or' (line 159)
    result_or_keyword_199032 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 7), 'or', result_is__199026, isscalar_call_result_199031)
    
    # Testing the type of an if condition (line 159)
    if_condition_199033 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 4), result_or_keyword_199032)
    # Assigning a type to the variable 'if_condition_199033' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'if_condition_199033', if_condition_199033)
    # SSA begins for if statement (line 159)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 160):
    
    # Assigning a Call to a Name (line 160):
    
    # Call to RandomState(...): (line 160)
    # Processing the call arguments (line 160)
    # Getting the type of 'order' (line 160)
    order_199037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 36), 'order', False)
    # Processing the call keyword arguments (line 160)
    kwargs_199038 = {}
    # Getting the type of 'np' (line 160)
    np_199034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 14), 'np', False)
    # Obtaining the member 'random' of a type (line 160)
    random_199035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 14), np_199034, 'random')
    # Obtaining the member 'RandomState' of a type (line 160)
    RandomState_199036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 14), random_199035, 'RandomState')
    # Calling RandomState(args, kwargs) (line 160)
    RandomState_call_result_199039 = invoke(stypy.reporting.localization.Localization(__file__, 160, 14), RandomState_199036, *[order_199037], **kwargs_199038)
    
    # Assigning a type to the variable 'rng' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'rng', RandomState_call_result_199039)
    
    # Assigning a Call to a Name (line 161):
    
    # Assigning a Call to a Name (line 161):
    
    # Call to permutation(...): (line 161)
    # Processing the call arguments (line 161)
    # Getting the type of 'n' (line 161)
    n_199042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 32), 'n', False)
    # Processing the call keyword arguments (line 161)
    kwargs_199043 = {}
    # Getting the type of 'rng' (line 161)
    rng_199040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'rng', False)
    # Obtaining the member 'permutation' of a type (line 161)
    permutation_199041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 16), rng_199040, 'permutation')
    # Calling permutation(args, kwargs) (line 161)
    permutation_call_result_199044 = invoke(stypy.reporting.localization.Localization(__file__, 161, 16), permutation_199041, *[n_199042], **kwargs_199043)
    
    # Assigning a type to the variable 'order' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'order', permutation_call_result_199044)
    # SSA branch for the else part of an if statement (line 159)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 163):
    
    # Assigning a Call to a Name (line 163):
    
    # Call to asarray(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'order' (line 163)
    order_199047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 27), 'order', False)
    # Processing the call keyword arguments (line 163)
    kwargs_199048 = {}
    # Getting the type of 'np' (line 163)
    np_199045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 16), 'np', False)
    # Obtaining the member 'asarray' of a type (line 163)
    asarray_199046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 16), np_199045, 'asarray')
    # Calling asarray(args, kwargs) (line 163)
    asarray_call_result_199049 = invoke(stypy.reporting.localization.Localization(__file__, 163, 16), asarray_199046, *[order_199047], **kwargs_199048)
    
    # Assigning a type to the variable 'order' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'order', asarray_call_result_199049)
    
    
    # Getting the type of 'order' (line 164)
    order_199050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 11), 'order')
    # Obtaining the member 'shape' of a type (line 164)
    shape_199051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 11), order_199050, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 164)
    tuple_199052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 164)
    # Adding element type (line 164)
    # Getting the type of 'n' (line 164)
    n_199053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 27), 'n')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 27), tuple_199052, n_199053)
    
    # Applying the binary operator '!=' (line 164)
    result_ne_199054 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 11), '!=', shape_199051, tuple_199052)
    
    # Testing the type of an if condition (line 164)
    if_condition_199055 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 8), result_ne_199054)
    # Assigning a type to the variable 'if_condition_199055' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'if_condition_199055', if_condition_199055)
    # SSA begins for if statement (line 164)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 165)
    # Processing the call arguments (line 165)
    str_199057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 29), 'str', '`order` has incorrect shape.')
    # Processing the call keyword arguments (line 165)
    kwargs_199058 = {}
    # Getting the type of 'ValueError' (line 165)
    ValueError_199056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 165)
    ValueError_call_result_199059 = invoke(stypy.reporting.localization.Localization(__file__, 165, 18), ValueError_199056, *[str_199057], **kwargs_199058)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 165, 12), ValueError_call_result_199059, 'raise parameter', BaseException)
    # SSA join for if statement (line 164)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 159)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 167):
    
    # Assigning a Subscript to a Name (line 167):
    
    # Obtaining the type of the subscript
    slice_199060 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 167, 8), None, None, None)
    # Getting the type of 'order' (line 167)
    order_199061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 13), 'order')
    # Getting the type of 'A' (line 167)
    A_199062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'A')
    # Obtaining the member '__getitem__' of a type (line 167)
    getitem___199063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), A_199062, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 167)
    subscript_call_result_199064 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), getitem___199063, (slice_199060, order_199061))
    
    # Assigning a type to the variable 'A' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'A', subscript_call_result_199064)
    
    
    # Call to issparse(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'A' (line 169)
    A_199066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'A', False)
    # Processing the call keyword arguments (line 169)
    kwargs_199067 = {}
    # Getting the type of 'issparse' (line 169)
    issparse_199065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 7), 'issparse', False)
    # Calling issparse(args, kwargs) (line 169)
    issparse_call_result_199068 = invoke(stypy.reporting.localization.Localization(__file__, 169, 7), issparse_199065, *[A_199066], **kwargs_199067)
    
    # Testing the type of an if condition (line 169)
    if_condition_199069 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 4), issparse_call_result_199068)
    # Assigning a type to the variable 'if_condition_199069' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'if_condition_199069', if_condition_199069)
    # SSA begins for if statement (line 169)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 170):
    
    # Assigning a Call to a Name (line 170):
    
    # Call to group_sparse(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'm' (line 170)
    m_199071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 30), 'm', False)
    # Getting the type of 'n' (line 170)
    n_199072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 33), 'n', False)
    # Getting the type of 'A' (line 170)
    A_199073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 36), 'A', False)
    # Obtaining the member 'indices' of a type (line 170)
    indices_199074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 36), A_199073, 'indices')
    # Getting the type of 'A' (line 170)
    A_199075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 47), 'A', False)
    # Obtaining the member 'indptr' of a type (line 170)
    indptr_199076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 47), A_199075, 'indptr')
    # Processing the call keyword arguments (line 170)
    kwargs_199077 = {}
    # Getting the type of 'group_sparse' (line 170)
    group_sparse_199070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 17), 'group_sparse', False)
    # Calling group_sparse(args, kwargs) (line 170)
    group_sparse_call_result_199078 = invoke(stypy.reporting.localization.Localization(__file__, 170, 17), group_sparse_199070, *[m_199071, n_199072, indices_199074, indptr_199076], **kwargs_199077)
    
    # Assigning a type to the variable 'groups' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'groups', group_sparse_call_result_199078)
    # SSA branch for the else part of an if statement (line 169)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 172):
    
    # Assigning a Call to a Name (line 172):
    
    # Call to group_dense(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'm' (line 172)
    m_199080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 29), 'm', False)
    # Getting the type of 'n' (line 172)
    n_199081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 32), 'n', False)
    # Getting the type of 'A' (line 172)
    A_199082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 35), 'A', False)
    # Processing the call keyword arguments (line 172)
    kwargs_199083 = {}
    # Getting the type of 'group_dense' (line 172)
    group_dense_199079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 17), 'group_dense', False)
    # Calling group_dense(args, kwargs) (line 172)
    group_dense_call_result_199084 = invoke(stypy.reporting.localization.Localization(__file__, 172, 17), group_dense_199079, *[m_199080, n_199081, A_199082], **kwargs_199083)
    
    # Assigning a type to the variable 'groups' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'groups', group_dense_call_result_199084)
    # SSA join for if statement (line 169)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Subscript (line 174):
    
    # Assigning a Call to a Subscript (line 174):
    
    # Call to copy(...): (line 174)
    # Processing the call keyword arguments (line 174)
    kwargs_199087 = {}
    # Getting the type of 'groups' (line 174)
    groups_199085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 20), 'groups', False)
    # Obtaining the member 'copy' of a type (line 174)
    copy_199086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 20), groups_199085, 'copy')
    # Calling copy(args, kwargs) (line 174)
    copy_call_result_199088 = invoke(stypy.reporting.localization.Localization(__file__, 174, 20), copy_199086, *[], **kwargs_199087)
    
    # Getting the type of 'groups' (line 174)
    groups_199089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'groups')
    # Getting the type of 'order' (line 174)
    order_199090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 11), 'order')
    # Storing an element on a container (line 174)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 4), groups_199089, (order_199090, copy_call_result_199088))
    # Getting the type of 'groups' (line 176)
    groups_199091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 11), 'groups')
    # Assigning a type to the variable 'stypy_return_type' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'stypy_return_type', groups_199091)
    
    # ################# End of 'group_columns(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'group_columns' in the type store
    # Getting the type of 'stypy_return_type' (line 118)
    stypy_return_type_199092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_199092)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'group_columns'
    return stypy_return_type_199092

# Assigning a type to the variable 'group_columns' (line 118)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'group_columns', group_columns)

@norecursion
def approx_derivative(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_199093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 38), 'str', '3-point')
    # Getting the type of 'None' (line 179)
    None_199094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 58), 'None')
    # Getting the type of 'None' (line 179)
    None_199095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 67), 'None')
    
    # Obtaining an instance of the builtin type 'tuple' (line 180)
    tuple_199096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 180)
    # Adding element type (line 180)
    
    # Getting the type of 'np' (line 180)
    np_199097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 31), 'np')
    # Obtaining the member 'inf' of a type (line 180)
    inf_199098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 31), np_199097, 'inf')
    # Applying the 'usub' unary operator (line 180)
    result___neg___199099 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 30), 'usub', inf_199098)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 30), tuple_199096, result___neg___199099)
    # Adding element type (line 180)
    # Getting the type of 'np' (line 180)
    np_199100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 39), 'np')
    # Obtaining the member 'inf' of a type (line 180)
    inf_199101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 39), np_199100, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 30), tuple_199096, inf_199101)
    
    # Getting the type of 'None' (line 180)
    None_199102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 57), 'None')
    
    # Obtaining an instance of the builtin type 'tuple' (line 180)
    tuple_199103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 68), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 180)
    
    
    # Obtaining an instance of the builtin type 'dict' (line 181)
    dict_199104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 29), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 181)
    
    defaults = [str_199093, None_199094, None_199095, tuple_199096, None_199102, tuple_199103, dict_199104]
    # Create a new context for function 'approx_derivative'
    module_type_store = module_type_store.open_function_context('approx_derivative', 179, 0, False)
    
    # Passed parameters checking function
    approx_derivative.stypy_localization = localization
    approx_derivative.stypy_type_of_self = None
    approx_derivative.stypy_type_store = module_type_store
    approx_derivative.stypy_function_name = 'approx_derivative'
    approx_derivative.stypy_param_names_list = ['fun', 'x0', 'method', 'rel_step', 'f0', 'bounds', 'sparsity', 'args', 'kwargs']
    approx_derivative.stypy_varargs_param_name = None
    approx_derivative.stypy_kwargs_param_name = None
    approx_derivative.stypy_call_defaults = defaults
    approx_derivative.stypy_call_varargs = varargs
    approx_derivative.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'approx_derivative', ['fun', 'x0', 'method', 'rel_step', 'f0', 'bounds', 'sparsity', 'args', 'kwargs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'approx_derivative', localization, ['fun', 'x0', 'method', 'rel_step', 'f0', 'bounds', 'sparsity', 'args', 'kwargs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'approx_derivative(...)' code ##################

    str_199105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, (-1)), 'str', 'Compute finite difference approximation of the derivatives of a\n    vector-valued function.\n\n    If a function maps from R^n to R^m, its derivatives form m-by-n matrix\n    called the Jacobian, where an element (i, j) is a partial derivative of\n    f[i] with respect to x[j].\n\n    Parameters\n    ----------\n    fun : callable\n        Function of which to estimate the derivatives. The argument x\n        passed to this function is ndarray of shape (n,) (never a scalar\n        even if n=1). It must return 1-d array_like of shape (m,) or a scalar.\n    x0 : array_like of shape (n,) or float\n        Point at which to estimate the derivatives. Float will be converted\n        to a 1-d array.\n    method : {\'3-point\', \'2-point\'}, optional\n        Finite difference method to use:\n            - \'2-point\' - use the fist order accuracy forward or backward\n                          difference.\n            - \'3-point\' - use central difference in interior points and the\n                          second order accuracy forward or backward difference\n                          near the boundary.\n            - \'cs\' - use a complex-step finite difference scheme. This assumes\n                     that the user function is real-valued and can be\n                     analytically continued to the complex plane. Otherwise,\n                     produces bogus results.\n    rel_step : None or array_like, optional\n        Relative step size to use. The absolute step size is computed as\n        ``h = rel_step * sign(x0) * max(1, abs(x0))``, possibly adjusted to\n        fit into the bounds. For ``method=\'3-point\'`` the sign of `h` is\n        ignored. If None (default) then step is selected automatically,\n        see Notes.\n    f0 : None or array_like, optional\n        If not None it is assumed to be equal to ``fun(x0)``, in  this case\n        the ``fun(x0)`` is not called. Default is None.\n    bounds : tuple of array_like, optional\n        Lower and upper bounds on independent variables. Defaults to no bounds.\n        Each bound must match the size of `x0` or be a scalar, in the latter\n        case the bound will be the same for all variables. Use it to limit the\n        range of function evaluation.\n    sparsity : {None, array_like, sparse matrix, 2-tuple}, optional\n        Defines a sparsity structure of the Jacobian matrix. If the Jacobian\n        matrix is known to have only few non-zero elements in each row, then\n        it\'s possible to estimate its several columns by a single function\n        evaluation [3]_. To perform such economic computations two ingredients\n        are required:\n\n        * structure : array_like or sparse matrix of shape (m, n). A zero\n          element means that a corresponding element of the Jacobian\n          identically equals to zero.\n        * groups : array_like of shape (n,). A column grouping for a given\n          sparsity structure, use `group_columns` to obtain it.\n\n        A single array or a sparse matrix is interpreted as a sparsity\n        structure, and groups are computed inside the function. A tuple is\n        interpreted as (structure, groups). If None (default), a standard\n        dense differencing will be used.\n\n        Note, that sparse differencing makes sense only for large Jacobian\n        matrices where each row contains few non-zero elements.\n    args, kwargs : tuple and dict, optional\n        Additional arguments passed to `fun`. Both empty by default.\n        The calling signature is ``fun(x, *args, **kwargs)``.\n\n    Returns\n    -------\n    J : ndarray or csr_matrix\n        Finite difference approximation of the Jacobian matrix. If `sparsity`\n        is None then ndarray with shape (m, n) is returned. Although if m=1 it\n        is returned as a gradient with shape (n,). If `sparsity` is not None,\n        csr_matrix with shape (m, n) is returned.\n\n    See Also\n    --------\n    check_derivative : Check correctness of a function computing derivatives.\n\n    Notes\n    -----\n    If `rel_step` is not provided, it assigned to ``EPS**(1/s)``, where EPS is\n    machine epsilon for float64 numbers, s=2 for \'2-point\' method and s=3 for\n    \'3-point\' method. Such relative step approximately minimizes a sum of\n    truncation and round-off errors, see [1]_.\n\n    A finite difference scheme for \'3-point\' method is selected automatically.\n    The well-known central difference scheme is used for points sufficiently\n    far from the boundary, and 3-point forward or backward scheme is used for\n    points near the boundary. Both schemes have the second-order accuracy in\n    terms of Taylor expansion. Refer to [2]_ for the formulas of 3-point\n    forward and backward difference schemes.\n\n    For dense differencing when m=1 Jacobian is returned with a shape (n,),\n    on the other hand when n=1 Jacobian is returned with a shape (m, 1).\n    Our motivation is the following: a) It handles a case of gradient\n    computation (m=1) in a conventional way. b) It clearly separates these two\n    different cases. b) In all cases np.atleast_2d can be called to get 2-d\n    Jacobian with correct dimensions.\n\n    References\n    ----------\n    .. [1] W. H. Press et. al. "Numerical Recipes. The Art of Scientific\n           Computing. 3rd edition", sec. 5.7.\n\n    .. [2] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of\n           sparse Jacobian matrices", Journal of the Institute of Mathematics\n           and its Applications, 13 (1974), pp. 117-120.\n\n    .. [3] B. Fornberg, "Generation of Finite Difference Formulas on\n           Arbitrarily Spaced Grids", Mathematics of Computation 51, 1988.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.optimize import approx_derivative\n    >>>\n    >>> def f(x, c1, c2):\n    ...     return np.array([x[0] * np.sin(c1 * x[1]),\n    ...                      x[0] * np.cos(c2 * x[1])])\n    ...\n    >>> x0 = np.array([1.0, 0.5 * np.pi])\n    >>> approx_derivative(f, x0, args=(1, 2))\n    array([[ 1.,  0.],\n           [-1.,  0.]])\n\n    Bounds can be used to limit the region of function evaluation.\n    In the example below we compute left and right derivative at point 1.0.\n\n    >>> def g(x):\n    ...     return x**2 if x >= 1 else x\n    ...\n    >>> x0 = 1.0\n    >>> approx_derivative(g, x0, bounds=(-np.inf, 1.0))\n    array([ 1.])\n    >>> approx_derivative(g, x0, bounds=(1.0, np.inf))\n    array([ 2.])\n    ')
    
    
    # Getting the type of 'method' (line 318)
    method_199106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 7), 'method')
    
    # Obtaining an instance of the builtin type 'list' (line 318)
    list_199107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 318)
    # Adding element type (line 318)
    str_199108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 22), 'str', '2-point')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 21), list_199107, str_199108)
    # Adding element type (line 318)
    str_199109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 33), 'str', '3-point')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 21), list_199107, str_199109)
    # Adding element type (line 318)
    str_199110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 44), 'str', 'cs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 21), list_199107, str_199110)
    
    # Applying the binary operator 'notin' (line 318)
    result_contains_199111 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 7), 'notin', method_199106, list_199107)
    
    # Testing the type of an if condition (line 318)
    if_condition_199112 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 318, 4), result_contains_199111)
    # Assigning a type to the variable 'if_condition_199112' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'if_condition_199112', if_condition_199112)
    # SSA begins for if statement (line 318)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 319)
    # Processing the call arguments (line 319)
    str_199114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 25), 'str', "Unknown method '%s'. ")
    # Getting the type of 'method' (line 319)
    method_199115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 51), 'method', False)
    # Applying the binary operator '%' (line 319)
    result_mod_199116 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 25), '%', str_199114, method_199115)
    
    # Processing the call keyword arguments (line 319)
    kwargs_199117 = {}
    # Getting the type of 'ValueError' (line 319)
    ValueError_199113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 319)
    ValueError_call_result_199118 = invoke(stypy.reporting.localization.Localization(__file__, 319, 14), ValueError_199113, *[result_mod_199116], **kwargs_199117)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 319, 8), ValueError_call_result_199118, 'raise parameter', BaseException)
    # SSA join for if statement (line 318)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 321):
    
    # Assigning a Call to a Name (line 321):
    
    # Call to atleast_1d(...): (line 321)
    # Processing the call arguments (line 321)
    # Getting the type of 'x0' (line 321)
    x0_199121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 23), 'x0', False)
    # Processing the call keyword arguments (line 321)
    kwargs_199122 = {}
    # Getting the type of 'np' (line 321)
    np_199119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 9), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 321)
    atleast_1d_199120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 9), np_199119, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 321)
    atleast_1d_call_result_199123 = invoke(stypy.reporting.localization.Localization(__file__, 321, 9), atleast_1d_199120, *[x0_199121], **kwargs_199122)
    
    # Assigning a type to the variable 'x0' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'x0', atleast_1d_call_result_199123)
    
    
    # Getting the type of 'x0' (line 322)
    x0_199124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 7), 'x0')
    # Obtaining the member 'ndim' of a type (line 322)
    ndim_199125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 7), x0_199124, 'ndim')
    int_199126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 17), 'int')
    # Applying the binary operator '>' (line 322)
    result_gt_199127 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 7), '>', ndim_199125, int_199126)
    
    # Testing the type of an if condition (line 322)
    if_condition_199128 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 4), result_gt_199127)
    # Assigning a type to the variable 'if_condition_199128' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'if_condition_199128', if_condition_199128)
    # SSA begins for if statement (line 322)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 323)
    # Processing the call arguments (line 323)
    str_199130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 25), 'str', '`x0` must have at most 1 dimension.')
    # Processing the call keyword arguments (line 323)
    kwargs_199131 = {}
    # Getting the type of 'ValueError' (line 323)
    ValueError_199129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 323)
    ValueError_call_result_199132 = invoke(stypy.reporting.localization.Localization(__file__, 323, 14), ValueError_199129, *[str_199130], **kwargs_199131)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 323, 8), ValueError_call_result_199132, 'raise parameter', BaseException)
    # SSA join for if statement (line 322)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 325):
    
    # Assigning a Subscript to a Name (line 325):
    
    # Obtaining the type of the subscript
    int_199133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 4), 'int')
    
    # Call to _prepare_bounds(...): (line 325)
    # Processing the call arguments (line 325)
    # Getting the type of 'bounds' (line 325)
    bounds_199135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 29), 'bounds', False)
    # Getting the type of 'x0' (line 325)
    x0_199136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 37), 'x0', False)
    # Processing the call keyword arguments (line 325)
    kwargs_199137 = {}
    # Getting the type of '_prepare_bounds' (line 325)
    _prepare_bounds_199134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 13), '_prepare_bounds', False)
    # Calling _prepare_bounds(args, kwargs) (line 325)
    _prepare_bounds_call_result_199138 = invoke(stypy.reporting.localization.Localization(__file__, 325, 13), _prepare_bounds_199134, *[bounds_199135, x0_199136], **kwargs_199137)
    
    # Obtaining the member '__getitem__' of a type (line 325)
    getitem___199139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 4), _prepare_bounds_call_result_199138, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 325)
    subscript_call_result_199140 = invoke(stypy.reporting.localization.Localization(__file__, 325, 4), getitem___199139, int_199133)
    
    # Assigning a type to the variable 'tuple_var_assignment_198601' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'tuple_var_assignment_198601', subscript_call_result_199140)
    
    # Assigning a Subscript to a Name (line 325):
    
    # Obtaining the type of the subscript
    int_199141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 4), 'int')
    
    # Call to _prepare_bounds(...): (line 325)
    # Processing the call arguments (line 325)
    # Getting the type of 'bounds' (line 325)
    bounds_199143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 29), 'bounds', False)
    # Getting the type of 'x0' (line 325)
    x0_199144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 37), 'x0', False)
    # Processing the call keyword arguments (line 325)
    kwargs_199145 = {}
    # Getting the type of '_prepare_bounds' (line 325)
    _prepare_bounds_199142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 13), '_prepare_bounds', False)
    # Calling _prepare_bounds(args, kwargs) (line 325)
    _prepare_bounds_call_result_199146 = invoke(stypy.reporting.localization.Localization(__file__, 325, 13), _prepare_bounds_199142, *[bounds_199143, x0_199144], **kwargs_199145)
    
    # Obtaining the member '__getitem__' of a type (line 325)
    getitem___199147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 4), _prepare_bounds_call_result_199146, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 325)
    subscript_call_result_199148 = invoke(stypy.reporting.localization.Localization(__file__, 325, 4), getitem___199147, int_199141)
    
    # Assigning a type to the variable 'tuple_var_assignment_198602' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'tuple_var_assignment_198602', subscript_call_result_199148)
    
    # Assigning a Name to a Name (line 325):
    # Getting the type of 'tuple_var_assignment_198601' (line 325)
    tuple_var_assignment_198601_199149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'tuple_var_assignment_198601')
    # Assigning a type to the variable 'lb' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'lb', tuple_var_assignment_198601_199149)
    
    # Assigning a Name to a Name (line 325):
    # Getting the type of 'tuple_var_assignment_198602' (line 325)
    tuple_var_assignment_198602_199150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'tuple_var_assignment_198602')
    # Assigning a type to the variable 'ub' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'ub', tuple_var_assignment_198602_199150)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'lb' (line 327)
    lb_199151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 7), 'lb')
    # Obtaining the member 'shape' of a type (line 327)
    shape_199152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 7), lb_199151, 'shape')
    # Getting the type of 'x0' (line 327)
    x0_199153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 19), 'x0')
    # Obtaining the member 'shape' of a type (line 327)
    shape_199154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 19), x0_199153, 'shape')
    # Applying the binary operator '!=' (line 327)
    result_ne_199155 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 7), '!=', shape_199152, shape_199154)
    
    
    # Getting the type of 'ub' (line 327)
    ub_199156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 31), 'ub')
    # Obtaining the member 'shape' of a type (line 327)
    shape_199157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 31), ub_199156, 'shape')
    # Getting the type of 'x0' (line 327)
    x0_199158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 43), 'x0')
    # Obtaining the member 'shape' of a type (line 327)
    shape_199159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 43), x0_199158, 'shape')
    # Applying the binary operator '!=' (line 327)
    result_ne_199160 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 31), '!=', shape_199157, shape_199159)
    
    # Applying the binary operator 'or' (line 327)
    result_or_keyword_199161 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 7), 'or', result_ne_199155, result_ne_199160)
    
    # Testing the type of an if condition (line 327)
    if_condition_199162 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 327, 4), result_or_keyword_199161)
    # Assigning a type to the variable 'if_condition_199162' (line 327)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'if_condition_199162', if_condition_199162)
    # SSA begins for if statement (line 327)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 328)
    # Processing the call arguments (line 328)
    str_199164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 25), 'str', 'Inconsistent shapes between bounds and `x0`.')
    # Processing the call keyword arguments (line 328)
    kwargs_199165 = {}
    # Getting the type of 'ValueError' (line 328)
    ValueError_199163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 328)
    ValueError_call_result_199166 = invoke(stypy.reporting.localization.Localization(__file__, 328, 14), ValueError_199163, *[str_199164], **kwargs_199165)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 328, 8), ValueError_call_result_199166, 'raise parameter', BaseException)
    # SSA join for if statement (line 327)
    module_type_store = module_type_store.join_ssa_context()
    

    @norecursion
    def fun_wrapped(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun_wrapped'
        module_type_store = module_type_store.open_function_context('fun_wrapped', 330, 4, False)
        
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

        
        # Assigning a Call to a Name (line 331):
        
        # Assigning a Call to a Name (line 331):
        
        # Call to atleast_1d(...): (line 331)
        # Processing the call arguments (line 331)
        
        # Call to fun(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 'x' (line 331)
        x_199170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 30), 'x', False)
        # Getting the type of 'args' (line 331)
        args_199171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 34), 'args', False)
        # Processing the call keyword arguments (line 331)
        # Getting the type of 'kwargs' (line 331)
        kwargs_199172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 42), 'kwargs', False)
        kwargs_199173 = {'kwargs_199172': kwargs_199172}
        # Getting the type of 'fun' (line 331)
        fun_199169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 26), 'fun', False)
        # Calling fun(args, kwargs) (line 331)
        fun_call_result_199174 = invoke(stypy.reporting.localization.Localization(__file__, 331, 26), fun_199169, *[x_199170, args_199171], **kwargs_199173)
        
        # Processing the call keyword arguments (line 331)
        kwargs_199175 = {}
        # Getting the type of 'np' (line 331)
        np_199167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'np', False)
        # Obtaining the member 'atleast_1d' of a type (line 331)
        atleast_1d_199168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 12), np_199167, 'atleast_1d')
        # Calling atleast_1d(args, kwargs) (line 331)
        atleast_1d_call_result_199176 = invoke(stypy.reporting.localization.Localization(__file__, 331, 12), atleast_1d_199168, *[fun_call_result_199174], **kwargs_199175)
        
        # Assigning a type to the variable 'f' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'f', atleast_1d_call_result_199176)
        
        
        # Getting the type of 'f' (line 332)
        f_199177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 11), 'f')
        # Obtaining the member 'ndim' of a type (line 332)
        ndim_199178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 11), f_199177, 'ndim')
        int_199179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 20), 'int')
        # Applying the binary operator '>' (line 332)
        result_gt_199180 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 11), '>', ndim_199178, int_199179)
        
        # Testing the type of an if condition (line 332)
        if_condition_199181 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 332, 8), result_gt_199180)
        # Assigning a type to the variable 'if_condition_199181' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'if_condition_199181', if_condition_199181)
        # SSA begins for if statement (line 332)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to RuntimeError(...): (line 333)
        # Processing the call arguments (line 333)
        str_199183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 32), 'str', '`fun` return value has more than 1 dimension.')
        # Processing the call keyword arguments (line 333)
        kwargs_199184 = {}
        # Getting the type of 'RuntimeError' (line 333)
        RuntimeError_199182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 18), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 333)
        RuntimeError_call_result_199185 = invoke(stypy.reporting.localization.Localization(__file__, 333, 18), RuntimeError_199182, *[str_199183], **kwargs_199184)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 333, 12), RuntimeError_call_result_199185, 'raise parameter', BaseException)
        # SSA join for if statement (line 332)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'f' (line 335)
        f_199186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 15), 'f')
        # Assigning a type to the variable 'stypy_return_type' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'stypy_return_type', f_199186)
        
        # ################# End of 'fun_wrapped(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun_wrapped' in the type store
        # Getting the type of 'stypy_return_type' (line 330)
        stypy_return_type_199187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_199187)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun_wrapped'
        return stypy_return_type_199187

    # Assigning a type to the variable 'fun_wrapped' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'fun_wrapped', fun_wrapped)
    
    # Type idiom detected: calculating its left and rigth part (line 337)
    # Getting the type of 'f0' (line 337)
    f0_199188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 7), 'f0')
    # Getting the type of 'None' (line 337)
    None_199189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 13), 'None')
    
    (may_be_199190, more_types_in_union_199191) = may_be_none(f0_199188, None_199189)

    if may_be_199190:

        if more_types_in_union_199191:
            # Runtime conditional SSA (line 337)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 338):
        
        # Assigning a Call to a Name (line 338):
        
        # Call to fun_wrapped(...): (line 338)
        # Processing the call arguments (line 338)
        # Getting the type of 'x0' (line 338)
        x0_199193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 25), 'x0', False)
        # Processing the call keyword arguments (line 338)
        kwargs_199194 = {}
        # Getting the type of 'fun_wrapped' (line 338)
        fun_wrapped_199192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 13), 'fun_wrapped', False)
        # Calling fun_wrapped(args, kwargs) (line 338)
        fun_wrapped_call_result_199195 = invoke(stypy.reporting.localization.Localization(__file__, 338, 13), fun_wrapped_199192, *[x0_199193], **kwargs_199194)
        
        # Assigning a type to the variable 'f0' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'f0', fun_wrapped_call_result_199195)

        if more_types_in_union_199191:
            # Runtime conditional SSA for else branch (line 337)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_199190) or more_types_in_union_199191):
        
        # Assigning a Call to a Name (line 340):
        
        # Assigning a Call to a Name (line 340):
        
        # Call to atleast_1d(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'f0' (line 340)
        f0_199198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 27), 'f0', False)
        # Processing the call keyword arguments (line 340)
        kwargs_199199 = {}
        # Getting the type of 'np' (line 340)
        np_199196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 13), 'np', False)
        # Obtaining the member 'atleast_1d' of a type (line 340)
        atleast_1d_199197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 13), np_199196, 'atleast_1d')
        # Calling atleast_1d(args, kwargs) (line 340)
        atleast_1d_call_result_199200 = invoke(stypy.reporting.localization.Localization(__file__, 340, 13), atleast_1d_199197, *[f0_199198], **kwargs_199199)
        
        # Assigning a type to the variable 'f0' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'f0', atleast_1d_call_result_199200)
        
        
        # Getting the type of 'f0' (line 341)
        f0_199201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 11), 'f0')
        # Obtaining the member 'ndim' of a type (line 341)
        ndim_199202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 11), f0_199201, 'ndim')
        int_199203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 21), 'int')
        # Applying the binary operator '>' (line 341)
        result_gt_199204 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 11), '>', ndim_199202, int_199203)
        
        # Testing the type of an if condition (line 341)
        if_condition_199205 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 341, 8), result_gt_199204)
        # Assigning a type to the variable 'if_condition_199205' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'if_condition_199205', if_condition_199205)
        # SSA begins for if statement (line 341)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 342)
        # Processing the call arguments (line 342)
        str_199207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 29), 'str', '`f0` passed has more than 1 dimension.')
        # Processing the call keyword arguments (line 342)
        kwargs_199208 = {}
        # Getting the type of 'ValueError' (line 342)
        ValueError_199206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 342)
        ValueError_call_result_199209 = invoke(stypy.reporting.localization.Localization(__file__, 342, 18), ValueError_199206, *[str_199207], **kwargs_199208)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 342, 12), ValueError_call_result_199209, 'raise parameter', BaseException)
        # SSA join for if statement (line 341)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_199190 and more_types_in_union_199191):
            # SSA join for if statement (line 337)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to any(...): (line 344)
    # Processing the call arguments (line 344)
    
    # Getting the type of 'x0' (line 344)
    x0_199212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 15), 'x0', False)
    # Getting the type of 'lb' (line 344)
    lb_199213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 20), 'lb', False)
    # Applying the binary operator '<' (line 344)
    result_lt_199214 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 15), '<', x0_199212, lb_199213)
    
    
    # Getting the type of 'x0' (line 344)
    x0_199215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 27), 'x0', False)
    # Getting the type of 'ub' (line 344)
    ub_199216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 32), 'ub', False)
    # Applying the binary operator '>' (line 344)
    result_gt_199217 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 27), '>', x0_199215, ub_199216)
    
    # Applying the binary operator '|' (line 344)
    result_or__199218 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 14), '|', result_lt_199214, result_gt_199217)
    
    # Processing the call keyword arguments (line 344)
    kwargs_199219 = {}
    # Getting the type of 'np' (line 344)
    np_199210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 7), 'np', False)
    # Obtaining the member 'any' of a type (line 344)
    any_199211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 7), np_199210, 'any')
    # Calling any(args, kwargs) (line 344)
    any_call_result_199220 = invoke(stypy.reporting.localization.Localization(__file__, 344, 7), any_199211, *[result_or__199218], **kwargs_199219)
    
    # Testing the type of an if condition (line 344)
    if_condition_199221 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 344, 4), any_call_result_199220)
    # Assigning a type to the variable 'if_condition_199221' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'if_condition_199221', if_condition_199221)
    # SSA begins for if statement (line 344)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 345)
    # Processing the call arguments (line 345)
    str_199223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 25), 'str', '`x0` violates bound constraints.')
    # Processing the call keyword arguments (line 345)
    kwargs_199224 = {}
    # Getting the type of 'ValueError' (line 345)
    ValueError_199222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 345)
    ValueError_call_result_199225 = invoke(stypy.reporting.localization.Localization(__file__, 345, 14), ValueError_199222, *[str_199223], **kwargs_199224)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 345, 8), ValueError_call_result_199225, 'raise parameter', BaseException)
    # SSA join for if statement (line 344)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 347):
    
    # Assigning a Call to a Name (line 347):
    
    # Call to _compute_absolute_step(...): (line 347)
    # Processing the call arguments (line 347)
    # Getting the type of 'rel_step' (line 347)
    rel_step_199227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 31), 'rel_step', False)
    # Getting the type of 'x0' (line 347)
    x0_199228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 41), 'x0', False)
    # Getting the type of 'method' (line 347)
    method_199229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 45), 'method', False)
    # Processing the call keyword arguments (line 347)
    kwargs_199230 = {}
    # Getting the type of '_compute_absolute_step' (line 347)
    _compute_absolute_step_199226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), '_compute_absolute_step', False)
    # Calling _compute_absolute_step(args, kwargs) (line 347)
    _compute_absolute_step_call_result_199231 = invoke(stypy.reporting.localization.Localization(__file__, 347, 8), _compute_absolute_step_199226, *[rel_step_199227, x0_199228, method_199229], **kwargs_199230)
    
    # Assigning a type to the variable 'h' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'h', _compute_absolute_step_call_result_199231)
    
    
    # Getting the type of 'method' (line 349)
    method_199232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 7), 'method')
    str_199233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 17), 'str', '2-point')
    # Applying the binary operator '==' (line 349)
    result_eq_199234 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 7), '==', method_199232, str_199233)
    
    # Testing the type of an if condition (line 349)
    if_condition_199235 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 349, 4), result_eq_199234)
    # Assigning a type to the variable 'if_condition_199235' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'if_condition_199235', if_condition_199235)
    # SSA begins for if statement (line 349)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 350):
    
    # Assigning a Subscript to a Name (line 350):
    
    # Obtaining the type of the subscript
    int_199236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 8), 'int')
    
    # Call to _adjust_scheme_to_bounds(...): (line 350)
    # Processing the call arguments (line 350)
    # Getting the type of 'x0' (line 351)
    x0_199238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'x0', False)
    # Getting the type of 'h' (line 351)
    h_199239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 16), 'h', False)
    int_199240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 19), 'int')
    str_199241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 22), 'str', '1-sided')
    # Getting the type of 'lb' (line 351)
    lb_199242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 33), 'lb', False)
    # Getting the type of 'ub' (line 351)
    ub_199243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 37), 'ub', False)
    # Processing the call keyword arguments (line 350)
    kwargs_199244 = {}
    # Getting the type of '_adjust_scheme_to_bounds' (line 350)
    _adjust_scheme_to_bounds_199237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 27), '_adjust_scheme_to_bounds', False)
    # Calling _adjust_scheme_to_bounds(args, kwargs) (line 350)
    _adjust_scheme_to_bounds_call_result_199245 = invoke(stypy.reporting.localization.Localization(__file__, 350, 27), _adjust_scheme_to_bounds_199237, *[x0_199238, h_199239, int_199240, str_199241, lb_199242, ub_199243], **kwargs_199244)
    
    # Obtaining the member '__getitem__' of a type (line 350)
    getitem___199246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), _adjust_scheme_to_bounds_call_result_199245, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 350)
    subscript_call_result_199247 = invoke(stypy.reporting.localization.Localization(__file__, 350, 8), getitem___199246, int_199236)
    
    # Assigning a type to the variable 'tuple_var_assignment_198603' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'tuple_var_assignment_198603', subscript_call_result_199247)
    
    # Assigning a Subscript to a Name (line 350):
    
    # Obtaining the type of the subscript
    int_199248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 8), 'int')
    
    # Call to _adjust_scheme_to_bounds(...): (line 350)
    # Processing the call arguments (line 350)
    # Getting the type of 'x0' (line 351)
    x0_199250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'x0', False)
    # Getting the type of 'h' (line 351)
    h_199251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 16), 'h', False)
    int_199252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 19), 'int')
    str_199253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 22), 'str', '1-sided')
    # Getting the type of 'lb' (line 351)
    lb_199254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 33), 'lb', False)
    # Getting the type of 'ub' (line 351)
    ub_199255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 37), 'ub', False)
    # Processing the call keyword arguments (line 350)
    kwargs_199256 = {}
    # Getting the type of '_adjust_scheme_to_bounds' (line 350)
    _adjust_scheme_to_bounds_199249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 27), '_adjust_scheme_to_bounds', False)
    # Calling _adjust_scheme_to_bounds(args, kwargs) (line 350)
    _adjust_scheme_to_bounds_call_result_199257 = invoke(stypy.reporting.localization.Localization(__file__, 350, 27), _adjust_scheme_to_bounds_199249, *[x0_199250, h_199251, int_199252, str_199253, lb_199254, ub_199255], **kwargs_199256)
    
    # Obtaining the member '__getitem__' of a type (line 350)
    getitem___199258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), _adjust_scheme_to_bounds_call_result_199257, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 350)
    subscript_call_result_199259 = invoke(stypy.reporting.localization.Localization(__file__, 350, 8), getitem___199258, int_199248)
    
    # Assigning a type to the variable 'tuple_var_assignment_198604' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'tuple_var_assignment_198604', subscript_call_result_199259)
    
    # Assigning a Name to a Name (line 350):
    # Getting the type of 'tuple_var_assignment_198603' (line 350)
    tuple_var_assignment_198603_199260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'tuple_var_assignment_198603')
    # Assigning a type to the variable 'h' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'h', tuple_var_assignment_198603_199260)
    
    # Assigning a Name to a Name (line 350):
    # Getting the type of 'tuple_var_assignment_198604' (line 350)
    tuple_var_assignment_198604_199261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'tuple_var_assignment_198604')
    # Assigning a type to the variable 'use_one_sided' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 11), 'use_one_sided', tuple_var_assignment_198604_199261)
    # SSA branch for the else part of an if statement (line 349)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'method' (line 352)
    method_199262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 9), 'method')
    str_199263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 19), 'str', '3-point')
    # Applying the binary operator '==' (line 352)
    result_eq_199264 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 9), '==', method_199262, str_199263)
    
    # Testing the type of an if condition (line 352)
    if_condition_199265 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 352, 9), result_eq_199264)
    # Assigning a type to the variable 'if_condition_199265' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 9), 'if_condition_199265', if_condition_199265)
    # SSA begins for if statement (line 352)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 353):
    
    # Assigning a Subscript to a Name (line 353):
    
    # Obtaining the type of the subscript
    int_199266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 8), 'int')
    
    # Call to _adjust_scheme_to_bounds(...): (line 353)
    # Processing the call arguments (line 353)
    # Getting the type of 'x0' (line 354)
    x0_199268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 12), 'x0', False)
    # Getting the type of 'h' (line 354)
    h_199269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 16), 'h', False)
    int_199270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 19), 'int')
    str_199271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 22), 'str', '2-sided')
    # Getting the type of 'lb' (line 354)
    lb_199272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 33), 'lb', False)
    # Getting the type of 'ub' (line 354)
    ub_199273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 37), 'ub', False)
    # Processing the call keyword arguments (line 353)
    kwargs_199274 = {}
    # Getting the type of '_adjust_scheme_to_bounds' (line 353)
    _adjust_scheme_to_bounds_199267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 27), '_adjust_scheme_to_bounds', False)
    # Calling _adjust_scheme_to_bounds(args, kwargs) (line 353)
    _adjust_scheme_to_bounds_call_result_199275 = invoke(stypy.reporting.localization.Localization(__file__, 353, 27), _adjust_scheme_to_bounds_199267, *[x0_199268, h_199269, int_199270, str_199271, lb_199272, ub_199273], **kwargs_199274)
    
    # Obtaining the member '__getitem__' of a type (line 353)
    getitem___199276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 8), _adjust_scheme_to_bounds_call_result_199275, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 353)
    subscript_call_result_199277 = invoke(stypy.reporting.localization.Localization(__file__, 353, 8), getitem___199276, int_199266)
    
    # Assigning a type to the variable 'tuple_var_assignment_198605' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'tuple_var_assignment_198605', subscript_call_result_199277)
    
    # Assigning a Subscript to a Name (line 353):
    
    # Obtaining the type of the subscript
    int_199278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 8), 'int')
    
    # Call to _adjust_scheme_to_bounds(...): (line 353)
    # Processing the call arguments (line 353)
    # Getting the type of 'x0' (line 354)
    x0_199280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 12), 'x0', False)
    # Getting the type of 'h' (line 354)
    h_199281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 16), 'h', False)
    int_199282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 19), 'int')
    str_199283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 22), 'str', '2-sided')
    # Getting the type of 'lb' (line 354)
    lb_199284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 33), 'lb', False)
    # Getting the type of 'ub' (line 354)
    ub_199285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 37), 'ub', False)
    # Processing the call keyword arguments (line 353)
    kwargs_199286 = {}
    # Getting the type of '_adjust_scheme_to_bounds' (line 353)
    _adjust_scheme_to_bounds_199279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 27), '_adjust_scheme_to_bounds', False)
    # Calling _adjust_scheme_to_bounds(args, kwargs) (line 353)
    _adjust_scheme_to_bounds_call_result_199287 = invoke(stypy.reporting.localization.Localization(__file__, 353, 27), _adjust_scheme_to_bounds_199279, *[x0_199280, h_199281, int_199282, str_199283, lb_199284, ub_199285], **kwargs_199286)
    
    # Obtaining the member '__getitem__' of a type (line 353)
    getitem___199288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 8), _adjust_scheme_to_bounds_call_result_199287, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 353)
    subscript_call_result_199289 = invoke(stypy.reporting.localization.Localization(__file__, 353, 8), getitem___199288, int_199278)
    
    # Assigning a type to the variable 'tuple_var_assignment_198606' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'tuple_var_assignment_198606', subscript_call_result_199289)
    
    # Assigning a Name to a Name (line 353):
    # Getting the type of 'tuple_var_assignment_198605' (line 353)
    tuple_var_assignment_198605_199290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'tuple_var_assignment_198605')
    # Assigning a type to the variable 'h' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'h', tuple_var_assignment_198605_199290)
    
    # Assigning a Name to a Name (line 353):
    # Getting the type of 'tuple_var_assignment_198606' (line 353)
    tuple_var_assignment_198606_199291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'tuple_var_assignment_198606')
    # Assigning a type to the variable 'use_one_sided' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 11), 'use_one_sided', tuple_var_assignment_198606_199291)
    # SSA branch for the else part of an if statement (line 352)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'method' (line 355)
    method_199292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 9), 'method')
    str_199293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 19), 'str', 'cs')
    # Applying the binary operator '==' (line 355)
    result_eq_199294 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 9), '==', method_199292, str_199293)
    
    # Testing the type of an if condition (line 355)
    if_condition_199295 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 355, 9), result_eq_199294)
    # Assigning a type to the variable 'if_condition_199295' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 9), 'if_condition_199295', if_condition_199295)
    # SSA begins for if statement (line 355)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 356):
    
    # Assigning a Name to a Name (line 356):
    # Getting the type of 'False' (line 356)
    False_199296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 24), 'False')
    # Assigning a type to the variable 'use_one_sided' (line 356)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'use_one_sided', False_199296)
    # SSA join for if statement (line 355)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 352)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 349)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 358)
    # Getting the type of 'sparsity' (line 358)
    sparsity_199297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 7), 'sparsity')
    # Getting the type of 'None' (line 358)
    None_199298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 19), 'None')
    
    (may_be_199299, more_types_in_union_199300) = may_be_none(sparsity_199297, None_199298)

    if may_be_199299:

        if more_types_in_union_199300:
            # Runtime conditional SSA (line 358)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to _dense_difference(...): (line 359)
        # Processing the call arguments (line 359)
        # Getting the type of 'fun_wrapped' (line 359)
        fun_wrapped_199302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 33), 'fun_wrapped', False)
        # Getting the type of 'x0' (line 359)
        x0_199303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 46), 'x0', False)
        # Getting the type of 'f0' (line 359)
        f0_199304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 50), 'f0', False)
        # Getting the type of 'h' (line 359)
        h_199305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 54), 'h', False)
        # Getting the type of 'use_one_sided' (line 359)
        use_one_sided_199306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 57), 'use_one_sided', False)
        # Getting the type of 'method' (line 359)
        method_199307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 72), 'method', False)
        # Processing the call keyword arguments (line 359)
        kwargs_199308 = {}
        # Getting the type of '_dense_difference' (line 359)
        _dense_difference_199301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 15), '_dense_difference', False)
        # Calling _dense_difference(args, kwargs) (line 359)
        _dense_difference_call_result_199309 = invoke(stypy.reporting.localization.Localization(__file__, 359, 15), _dense_difference_199301, *[fun_wrapped_199302, x0_199303, f0_199304, h_199305, use_one_sided_199306, method_199307], **kwargs_199308)
        
        # Assigning a type to the variable 'stypy_return_type' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'stypy_return_type', _dense_difference_call_result_199309)

        if more_types_in_union_199300:
            # Runtime conditional SSA for else branch (line 358)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_199299) or more_types_in_union_199300):
        
        
        # Evaluating a boolean operation
        
        
        # Call to issparse(...): (line 361)
        # Processing the call arguments (line 361)
        # Getting the type of 'sparsity' (line 361)
        sparsity_199311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 24), 'sparsity', False)
        # Processing the call keyword arguments (line 361)
        kwargs_199312 = {}
        # Getting the type of 'issparse' (line 361)
        issparse_199310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 15), 'issparse', False)
        # Calling issparse(args, kwargs) (line 361)
        issparse_call_result_199313 = invoke(stypy.reporting.localization.Localization(__file__, 361, 15), issparse_199310, *[sparsity_199311], **kwargs_199312)
        
        # Applying the 'not' unary operator (line 361)
        result_not__199314 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 11), 'not', issparse_call_result_199313)
        
        
        
        # Call to len(...): (line 361)
        # Processing the call arguments (line 361)
        # Getting the type of 'sparsity' (line 361)
        sparsity_199316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 42), 'sparsity', False)
        # Processing the call keyword arguments (line 361)
        kwargs_199317 = {}
        # Getting the type of 'len' (line 361)
        len_199315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 38), 'len', False)
        # Calling len(args, kwargs) (line 361)
        len_call_result_199318 = invoke(stypy.reporting.localization.Localization(__file__, 361, 38), len_199315, *[sparsity_199316], **kwargs_199317)
        
        int_199319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 55), 'int')
        # Applying the binary operator '==' (line 361)
        result_eq_199320 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 38), '==', len_call_result_199318, int_199319)
        
        # Applying the binary operator 'and' (line 361)
        result_and_keyword_199321 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 11), 'and', result_not__199314, result_eq_199320)
        
        # Testing the type of an if condition (line 361)
        if_condition_199322 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 361, 8), result_and_keyword_199321)
        # Assigning a type to the variable 'if_condition_199322' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'if_condition_199322', if_condition_199322)
        # SSA begins for if statement (line 361)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Tuple (line 362):
        
        # Assigning a Subscript to a Name (line 362):
        
        # Obtaining the type of the subscript
        int_199323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 12), 'int')
        # Getting the type of 'sparsity' (line 362)
        sparsity_199324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 32), 'sparsity')
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___199325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 12), sparsity_199324, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_199326 = invoke(stypy.reporting.localization.Localization(__file__, 362, 12), getitem___199325, int_199323)
        
        # Assigning a type to the variable 'tuple_var_assignment_198607' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'tuple_var_assignment_198607', subscript_call_result_199326)
        
        # Assigning a Subscript to a Name (line 362):
        
        # Obtaining the type of the subscript
        int_199327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 12), 'int')
        # Getting the type of 'sparsity' (line 362)
        sparsity_199328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 32), 'sparsity')
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___199329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 12), sparsity_199328, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_199330 = invoke(stypy.reporting.localization.Localization(__file__, 362, 12), getitem___199329, int_199327)
        
        # Assigning a type to the variable 'tuple_var_assignment_198608' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'tuple_var_assignment_198608', subscript_call_result_199330)
        
        # Assigning a Name to a Name (line 362):
        # Getting the type of 'tuple_var_assignment_198607' (line 362)
        tuple_var_assignment_198607_199331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'tuple_var_assignment_198607')
        # Assigning a type to the variable 'structure' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'structure', tuple_var_assignment_198607_199331)
        
        # Assigning a Name to a Name (line 362):
        # Getting the type of 'tuple_var_assignment_198608' (line 362)
        tuple_var_assignment_198608_199332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'tuple_var_assignment_198608')
        # Assigning a type to the variable 'groups' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 23), 'groups', tuple_var_assignment_198608_199332)
        # SSA branch for the else part of an if statement (line 361)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 364):
        
        # Assigning a Name to a Name (line 364):
        # Getting the type of 'sparsity' (line 364)
        sparsity_199333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 24), 'sparsity')
        # Assigning a type to the variable 'structure' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 12), 'structure', sparsity_199333)
        
        # Assigning a Call to a Name (line 365):
        
        # Assigning a Call to a Name (line 365):
        
        # Call to group_columns(...): (line 365)
        # Processing the call arguments (line 365)
        # Getting the type of 'sparsity' (line 365)
        sparsity_199335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 35), 'sparsity', False)
        # Processing the call keyword arguments (line 365)
        kwargs_199336 = {}
        # Getting the type of 'group_columns' (line 365)
        group_columns_199334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 21), 'group_columns', False)
        # Calling group_columns(args, kwargs) (line 365)
        group_columns_call_result_199337 = invoke(stypy.reporting.localization.Localization(__file__, 365, 21), group_columns_199334, *[sparsity_199335], **kwargs_199336)
        
        # Assigning a type to the variable 'groups' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'groups', group_columns_call_result_199337)
        # SSA join for if statement (line 361)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to issparse(...): (line 367)
        # Processing the call arguments (line 367)
        # Getting the type of 'structure' (line 367)
        structure_199339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 20), 'structure', False)
        # Processing the call keyword arguments (line 367)
        kwargs_199340 = {}
        # Getting the type of 'issparse' (line 367)
        issparse_199338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 11), 'issparse', False)
        # Calling issparse(args, kwargs) (line 367)
        issparse_call_result_199341 = invoke(stypy.reporting.localization.Localization(__file__, 367, 11), issparse_199338, *[structure_199339], **kwargs_199340)
        
        # Testing the type of an if condition (line 367)
        if_condition_199342 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 367, 8), issparse_call_result_199341)
        # Assigning a type to the variable 'if_condition_199342' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'if_condition_199342', if_condition_199342)
        # SSA begins for if statement (line 367)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 368):
        
        # Assigning a Call to a Name (line 368):
        
        # Call to csc_matrix(...): (line 368)
        # Processing the call arguments (line 368)
        # Getting the type of 'structure' (line 368)
        structure_199344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 35), 'structure', False)
        # Processing the call keyword arguments (line 368)
        kwargs_199345 = {}
        # Getting the type of 'csc_matrix' (line 368)
        csc_matrix_199343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 24), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 368)
        csc_matrix_call_result_199346 = invoke(stypy.reporting.localization.Localization(__file__, 368, 24), csc_matrix_199343, *[structure_199344], **kwargs_199345)
        
        # Assigning a type to the variable 'structure' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'structure', csc_matrix_call_result_199346)
        # SSA branch for the else part of an if statement (line 367)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 370):
        
        # Assigning a Call to a Name (line 370):
        
        # Call to atleast_2d(...): (line 370)
        # Processing the call arguments (line 370)
        # Getting the type of 'structure' (line 370)
        structure_199349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 38), 'structure', False)
        # Processing the call keyword arguments (line 370)
        kwargs_199350 = {}
        # Getting the type of 'np' (line 370)
        np_199347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 24), 'np', False)
        # Obtaining the member 'atleast_2d' of a type (line 370)
        atleast_2d_199348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 24), np_199347, 'atleast_2d')
        # Calling atleast_2d(args, kwargs) (line 370)
        atleast_2d_call_result_199351 = invoke(stypy.reporting.localization.Localization(__file__, 370, 24), atleast_2d_199348, *[structure_199349], **kwargs_199350)
        
        # Assigning a type to the variable 'structure' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'structure', atleast_2d_call_result_199351)
        # SSA join for if statement (line 367)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 372):
        
        # Assigning a Call to a Name (line 372):
        
        # Call to atleast_1d(...): (line 372)
        # Processing the call arguments (line 372)
        # Getting the type of 'groups' (line 372)
        groups_199354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 31), 'groups', False)
        # Processing the call keyword arguments (line 372)
        kwargs_199355 = {}
        # Getting the type of 'np' (line 372)
        np_199352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 17), 'np', False)
        # Obtaining the member 'atleast_1d' of a type (line 372)
        atleast_1d_199353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 17), np_199352, 'atleast_1d')
        # Calling atleast_1d(args, kwargs) (line 372)
        atleast_1d_call_result_199356 = invoke(stypy.reporting.localization.Localization(__file__, 372, 17), atleast_1d_199353, *[groups_199354], **kwargs_199355)
        
        # Assigning a type to the variable 'groups' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'groups', atleast_1d_call_result_199356)
        
        # Call to _sparse_difference(...): (line 373)
        # Processing the call arguments (line 373)
        # Getting the type of 'fun_wrapped' (line 373)
        fun_wrapped_199358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 34), 'fun_wrapped', False)
        # Getting the type of 'x0' (line 373)
        x0_199359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 47), 'x0', False)
        # Getting the type of 'f0' (line 373)
        f0_199360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 51), 'f0', False)
        # Getting the type of 'h' (line 373)
        h_199361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 55), 'h', False)
        # Getting the type of 'use_one_sided' (line 373)
        use_one_sided_199362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 58), 'use_one_sided', False)
        # Getting the type of 'structure' (line 374)
        structure_199363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 34), 'structure', False)
        # Getting the type of 'groups' (line 374)
        groups_199364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 45), 'groups', False)
        # Getting the type of 'method' (line 374)
        method_199365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 53), 'method', False)
        # Processing the call keyword arguments (line 373)
        kwargs_199366 = {}
        # Getting the type of '_sparse_difference' (line 373)
        _sparse_difference_199357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 15), '_sparse_difference', False)
        # Calling _sparse_difference(args, kwargs) (line 373)
        _sparse_difference_call_result_199367 = invoke(stypy.reporting.localization.Localization(__file__, 373, 15), _sparse_difference_199357, *[fun_wrapped_199358, x0_199359, f0_199360, h_199361, use_one_sided_199362, structure_199363, groups_199364, method_199365], **kwargs_199366)
        
        # Assigning a type to the variable 'stypy_return_type' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'stypy_return_type', _sparse_difference_call_result_199367)

        if (may_be_199299 and more_types_in_union_199300):
            # SSA join for if statement (line 358)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'approx_derivative(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'approx_derivative' in the type store
    # Getting the type of 'stypy_return_type' (line 179)
    stypy_return_type_199368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_199368)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'approx_derivative'
    return stypy_return_type_199368

# Assigning a type to the variable 'approx_derivative' (line 179)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 0), 'approx_derivative', approx_derivative)

@norecursion
def _dense_difference(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_dense_difference'
    module_type_store = module_type_store.open_function_context('_dense_difference', 377, 0, False)
    
    # Passed parameters checking function
    _dense_difference.stypy_localization = localization
    _dense_difference.stypy_type_of_self = None
    _dense_difference.stypy_type_store = module_type_store
    _dense_difference.stypy_function_name = '_dense_difference'
    _dense_difference.stypy_param_names_list = ['fun', 'x0', 'f0', 'h', 'use_one_sided', 'method']
    _dense_difference.stypy_varargs_param_name = None
    _dense_difference.stypy_kwargs_param_name = None
    _dense_difference.stypy_call_defaults = defaults
    _dense_difference.stypy_call_varargs = varargs
    _dense_difference.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_dense_difference', ['fun', 'x0', 'f0', 'h', 'use_one_sided', 'method'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_dense_difference', localization, ['fun', 'x0', 'f0', 'h', 'use_one_sided', 'method'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_dense_difference(...)' code ##################

    
    # Assigning a Attribute to a Name (line 378):
    
    # Assigning a Attribute to a Name (line 378):
    # Getting the type of 'f0' (line 378)
    f0_199369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'f0')
    # Obtaining the member 'size' of a type (line 378)
    size_199370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 8), f0_199369, 'size')
    # Assigning a type to the variable 'm' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'm', size_199370)
    
    # Assigning a Attribute to a Name (line 379):
    
    # Assigning a Attribute to a Name (line 379):
    # Getting the type of 'x0' (line 379)
    x0_199371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'x0')
    # Obtaining the member 'size' of a type (line 379)
    size_199372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 8), x0_199371, 'size')
    # Assigning a type to the variable 'n' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'n', size_199372)
    
    # Assigning a Call to a Name (line 380):
    
    # Assigning a Call to a Name (line 380):
    
    # Call to empty(...): (line 380)
    # Processing the call arguments (line 380)
    
    # Obtaining an instance of the builtin type 'tuple' (line 380)
    tuple_199375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 380)
    # Adding element type (line 380)
    # Getting the type of 'n' (line 380)
    n_199376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 29), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 29), tuple_199375, n_199376)
    # Adding element type (line 380)
    # Getting the type of 'm' (line 380)
    m_199377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 32), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 29), tuple_199375, m_199377)
    
    # Processing the call keyword arguments (line 380)
    kwargs_199378 = {}
    # Getting the type of 'np' (line 380)
    np_199373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 19), 'np', False)
    # Obtaining the member 'empty' of a type (line 380)
    empty_199374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 19), np_199373, 'empty')
    # Calling empty(args, kwargs) (line 380)
    empty_call_result_199379 = invoke(stypy.reporting.localization.Localization(__file__, 380, 19), empty_199374, *[tuple_199375], **kwargs_199378)
    
    # Assigning a type to the variable 'J_transposed' (line 380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 4), 'J_transposed', empty_call_result_199379)
    
    # Assigning a Call to a Name (line 381):
    
    # Assigning a Call to a Name (line 381):
    
    # Call to diag(...): (line 381)
    # Processing the call arguments (line 381)
    # Getting the type of 'h' (line 381)
    h_199382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 21), 'h', False)
    # Processing the call keyword arguments (line 381)
    kwargs_199383 = {}
    # Getting the type of 'np' (line 381)
    np_199380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 13), 'np', False)
    # Obtaining the member 'diag' of a type (line 381)
    diag_199381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 13), np_199380, 'diag')
    # Calling diag(args, kwargs) (line 381)
    diag_call_result_199384 = invoke(stypy.reporting.localization.Localization(__file__, 381, 13), diag_199381, *[h_199382], **kwargs_199383)
    
    # Assigning a type to the variable 'h_vecs' (line 381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'h_vecs', diag_call_result_199384)
    
    
    # Call to range(...): (line 383)
    # Processing the call arguments (line 383)
    # Getting the type of 'h' (line 383)
    h_199386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 19), 'h', False)
    # Obtaining the member 'size' of a type (line 383)
    size_199387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 19), h_199386, 'size')
    # Processing the call keyword arguments (line 383)
    kwargs_199388 = {}
    # Getting the type of 'range' (line 383)
    range_199385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 13), 'range', False)
    # Calling range(args, kwargs) (line 383)
    range_call_result_199389 = invoke(stypy.reporting.localization.Localization(__file__, 383, 13), range_199385, *[size_199387], **kwargs_199388)
    
    # Testing the type of a for loop iterable (line 383)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 383, 4), range_call_result_199389)
    # Getting the type of the for loop variable (line 383)
    for_loop_var_199390 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 383, 4), range_call_result_199389)
    # Assigning a type to the variable 'i' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'i', for_loop_var_199390)
    # SSA begins for a for statement (line 383)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'method' (line 384)
    method_199391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 11), 'method')
    str_199392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 21), 'str', '2-point')
    # Applying the binary operator '==' (line 384)
    result_eq_199393 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 11), '==', method_199391, str_199392)
    
    # Testing the type of an if condition (line 384)
    if_condition_199394 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 384, 8), result_eq_199393)
    # Assigning a type to the variable 'if_condition_199394' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'if_condition_199394', if_condition_199394)
    # SSA begins for if statement (line 384)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 385):
    
    # Assigning a BinOp to a Name (line 385):
    # Getting the type of 'x0' (line 385)
    x0_199395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 16), 'x0')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 385)
    i_199396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 28), 'i')
    # Getting the type of 'h_vecs' (line 385)
    h_vecs_199397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 21), 'h_vecs')
    # Obtaining the member '__getitem__' of a type (line 385)
    getitem___199398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 21), h_vecs_199397, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 385)
    subscript_call_result_199399 = invoke(stypy.reporting.localization.Localization(__file__, 385, 21), getitem___199398, i_199396)
    
    # Applying the binary operator '+' (line 385)
    result_add_199400 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 16), '+', x0_199395, subscript_call_result_199399)
    
    # Assigning a type to the variable 'x' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'x', result_add_199400)
    
    # Assigning a BinOp to a Name (line 386):
    
    # Assigning a BinOp to a Name (line 386):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 386)
    i_199401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 19), 'i')
    # Getting the type of 'x' (line 386)
    x_199402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 17), 'x')
    # Obtaining the member '__getitem__' of a type (line 386)
    getitem___199403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 17), x_199402, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 386)
    subscript_call_result_199404 = invoke(stypy.reporting.localization.Localization(__file__, 386, 17), getitem___199403, i_199401)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 386)
    i_199405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 27), 'i')
    # Getting the type of 'x0' (line 386)
    x0_199406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 24), 'x0')
    # Obtaining the member '__getitem__' of a type (line 386)
    getitem___199407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 24), x0_199406, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 386)
    subscript_call_result_199408 = invoke(stypy.reporting.localization.Localization(__file__, 386, 24), getitem___199407, i_199405)
    
    # Applying the binary operator '-' (line 386)
    result_sub_199409 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 17), '-', subscript_call_result_199404, subscript_call_result_199408)
    
    # Assigning a type to the variable 'dx' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'dx', result_sub_199409)
    
    # Assigning a BinOp to a Name (line 387):
    
    # Assigning a BinOp to a Name (line 387):
    
    # Call to fun(...): (line 387)
    # Processing the call arguments (line 387)
    # Getting the type of 'x' (line 387)
    x_199411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 21), 'x', False)
    # Processing the call keyword arguments (line 387)
    kwargs_199412 = {}
    # Getting the type of 'fun' (line 387)
    fun_199410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 17), 'fun', False)
    # Calling fun(args, kwargs) (line 387)
    fun_call_result_199413 = invoke(stypy.reporting.localization.Localization(__file__, 387, 17), fun_199410, *[x_199411], **kwargs_199412)
    
    # Getting the type of 'f0' (line 387)
    f0_199414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 26), 'f0')
    # Applying the binary operator '-' (line 387)
    result_sub_199415 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 17), '-', fun_call_result_199413, f0_199414)
    
    # Assigning a type to the variable 'df' (line 387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'df', result_sub_199415)
    # SSA branch for the else part of an if statement (line 384)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'method' (line 388)
    method_199416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 13), 'method')
    str_199417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 23), 'str', '3-point')
    # Applying the binary operator '==' (line 388)
    result_eq_199418 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 13), '==', method_199416, str_199417)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 388)
    i_199419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 51), 'i')
    # Getting the type of 'use_one_sided' (line 388)
    use_one_sided_199420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 37), 'use_one_sided')
    # Obtaining the member '__getitem__' of a type (line 388)
    getitem___199421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 37), use_one_sided_199420, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 388)
    subscript_call_result_199422 = invoke(stypy.reporting.localization.Localization(__file__, 388, 37), getitem___199421, i_199419)
    
    # Applying the binary operator 'and' (line 388)
    result_and_keyword_199423 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 13), 'and', result_eq_199418, subscript_call_result_199422)
    
    # Testing the type of an if condition (line 388)
    if_condition_199424 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 388, 13), result_and_keyword_199423)
    # Assigning a type to the variable 'if_condition_199424' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 13), 'if_condition_199424', if_condition_199424)
    # SSA begins for if statement (line 388)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 389):
    
    # Assigning a BinOp to a Name (line 389):
    # Getting the type of 'x0' (line 389)
    x0_199425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 17), 'x0')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 389)
    i_199426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 29), 'i')
    # Getting the type of 'h_vecs' (line 389)
    h_vecs_199427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 22), 'h_vecs')
    # Obtaining the member '__getitem__' of a type (line 389)
    getitem___199428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 22), h_vecs_199427, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 389)
    subscript_call_result_199429 = invoke(stypy.reporting.localization.Localization(__file__, 389, 22), getitem___199428, i_199426)
    
    # Applying the binary operator '+' (line 389)
    result_add_199430 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 17), '+', x0_199425, subscript_call_result_199429)
    
    # Assigning a type to the variable 'x1' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'x1', result_add_199430)
    
    # Assigning a BinOp to a Name (line 390):
    
    # Assigning a BinOp to a Name (line 390):
    # Getting the type of 'x0' (line 390)
    x0_199431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 17), 'x0')
    int_199432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 22), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 390)
    i_199433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 33), 'i')
    # Getting the type of 'h_vecs' (line 390)
    h_vecs_199434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 26), 'h_vecs')
    # Obtaining the member '__getitem__' of a type (line 390)
    getitem___199435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 26), h_vecs_199434, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 390)
    subscript_call_result_199436 = invoke(stypy.reporting.localization.Localization(__file__, 390, 26), getitem___199435, i_199433)
    
    # Applying the binary operator '*' (line 390)
    result_mul_199437 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 22), '*', int_199432, subscript_call_result_199436)
    
    # Applying the binary operator '+' (line 390)
    result_add_199438 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 17), '+', x0_199431, result_mul_199437)
    
    # Assigning a type to the variable 'x2' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'x2', result_add_199438)
    
    # Assigning a BinOp to a Name (line 391):
    
    # Assigning a BinOp to a Name (line 391):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 391)
    i_199439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 20), 'i')
    # Getting the type of 'x2' (line 391)
    x2_199440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 17), 'x2')
    # Obtaining the member '__getitem__' of a type (line 391)
    getitem___199441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 17), x2_199440, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 391)
    subscript_call_result_199442 = invoke(stypy.reporting.localization.Localization(__file__, 391, 17), getitem___199441, i_199439)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 391)
    i_199443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 28), 'i')
    # Getting the type of 'x0' (line 391)
    x0_199444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 25), 'x0')
    # Obtaining the member '__getitem__' of a type (line 391)
    getitem___199445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 25), x0_199444, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 391)
    subscript_call_result_199446 = invoke(stypy.reporting.localization.Localization(__file__, 391, 25), getitem___199445, i_199443)
    
    # Applying the binary operator '-' (line 391)
    result_sub_199447 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 17), '-', subscript_call_result_199442, subscript_call_result_199446)
    
    # Assigning a type to the variable 'dx' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'dx', result_sub_199447)
    
    # Assigning a Call to a Name (line 392):
    
    # Assigning a Call to a Name (line 392):
    
    # Call to fun(...): (line 392)
    # Processing the call arguments (line 392)
    # Getting the type of 'x1' (line 392)
    x1_199449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 21), 'x1', False)
    # Processing the call keyword arguments (line 392)
    kwargs_199450 = {}
    # Getting the type of 'fun' (line 392)
    fun_199448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 17), 'fun', False)
    # Calling fun(args, kwargs) (line 392)
    fun_call_result_199451 = invoke(stypy.reporting.localization.Localization(__file__, 392, 17), fun_199448, *[x1_199449], **kwargs_199450)
    
    # Assigning a type to the variable 'f1' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'f1', fun_call_result_199451)
    
    # Assigning a Call to a Name (line 393):
    
    # Assigning a Call to a Name (line 393):
    
    # Call to fun(...): (line 393)
    # Processing the call arguments (line 393)
    # Getting the type of 'x2' (line 393)
    x2_199453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 21), 'x2', False)
    # Processing the call keyword arguments (line 393)
    kwargs_199454 = {}
    # Getting the type of 'fun' (line 393)
    fun_199452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 17), 'fun', False)
    # Calling fun(args, kwargs) (line 393)
    fun_call_result_199455 = invoke(stypy.reporting.localization.Localization(__file__, 393, 17), fun_199452, *[x2_199453], **kwargs_199454)
    
    # Assigning a type to the variable 'f2' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'f2', fun_call_result_199455)
    
    # Assigning a BinOp to a Name (line 394):
    
    # Assigning a BinOp to a Name (line 394):
    float_199456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 17), 'float')
    # Getting the type of 'f0' (line 394)
    f0_199457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 24), 'f0')
    # Applying the binary operator '*' (line 394)
    result_mul_199458 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 17), '*', float_199456, f0_199457)
    
    int_199459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 29), 'int')
    # Getting the type of 'f1' (line 394)
    f1_199460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 33), 'f1')
    # Applying the binary operator '*' (line 394)
    result_mul_199461 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 29), '*', int_199459, f1_199460)
    
    # Applying the binary operator '+' (line 394)
    result_add_199462 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 17), '+', result_mul_199458, result_mul_199461)
    
    # Getting the type of 'f2' (line 394)
    f2_199463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 38), 'f2')
    # Applying the binary operator '-' (line 394)
    result_sub_199464 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 36), '-', result_add_199462, f2_199463)
    
    # Assigning a type to the variable 'df' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'df', result_sub_199464)
    # SSA branch for the else part of an if statement (line 388)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'method' (line 395)
    method_199465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 13), 'method')
    str_199466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 23), 'str', '3-point')
    # Applying the binary operator '==' (line 395)
    result_eq_199467 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 13), '==', method_199465, str_199466)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 395)
    i_199468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 55), 'i')
    # Getting the type of 'use_one_sided' (line 395)
    use_one_sided_199469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 41), 'use_one_sided')
    # Obtaining the member '__getitem__' of a type (line 395)
    getitem___199470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 41), use_one_sided_199469, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 395)
    subscript_call_result_199471 = invoke(stypy.reporting.localization.Localization(__file__, 395, 41), getitem___199470, i_199468)
    
    # Applying the 'not' unary operator (line 395)
    result_not__199472 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 37), 'not', subscript_call_result_199471)
    
    # Applying the binary operator 'and' (line 395)
    result_and_keyword_199473 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 13), 'and', result_eq_199467, result_not__199472)
    
    # Testing the type of an if condition (line 395)
    if_condition_199474 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 395, 13), result_and_keyword_199473)
    # Assigning a type to the variable 'if_condition_199474' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 13), 'if_condition_199474', if_condition_199474)
    # SSA begins for if statement (line 395)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 396):
    
    # Assigning a BinOp to a Name (line 396):
    # Getting the type of 'x0' (line 396)
    x0_199475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 17), 'x0')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 396)
    i_199476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 29), 'i')
    # Getting the type of 'h_vecs' (line 396)
    h_vecs_199477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 22), 'h_vecs')
    # Obtaining the member '__getitem__' of a type (line 396)
    getitem___199478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 22), h_vecs_199477, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 396)
    subscript_call_result_199479 = invoke(stypy.reporting.localization.Localization(__file__, 396, 22), getitem___199478, i_199476)
    
    # Applying the binary operator '-' (line 396)
    result_sub_199480 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 17), '-', x0_199475, subscript_call_result_199479)
    
    # Assigning a type to the variable 'x1' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'x1', result_sub_199480)
    
    # Assigning a BinOp to a Name (line 397):
    
    # Assigning a BinOp to a Name (line 397):
    # Getting the type of 'x0' (line 397)
    x0_199481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 17), 'x0')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 397)
    i_199482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 29), 'i')
    # Getting the type of 'h_vecs' (line 397)
    h_vecs_199483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 22), 'h_vecs')
    # Obtaining the member '__getitem__' of a type (line 397)
    getitem___199484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 22), h_vecs_199483, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 397)
    subscript_call_result_199485 = invoke(stypy.reporting.localization.Localization(__file__, 397, 22), getitem___199484, i_199482)
    
    # Applying the binary operator '+' (line 397)
    result_add_199486 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 17), '+', x0_199481, subscript_call_result_199485)
    
    # Assigning a type to the variable 'x2' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'x2', result_add_199486)
    
    # Assigning a BinOp to a Name (line 398):
    
    # Assigning a BinOp to a Name (line 398):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 398)
    i_199487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 20), 'i')
    # Getting the type of 'x2' (line 398)
    x2_199488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 17), 'x2')
    # Obtaining the member '__getitem__' of a type (line 398)
    getitem___199489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 17), x2_199488, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 398)
    subscript_call_result_199490 = invoke(stypy.reporting.localization.Localization(__file__, 398, 17), getitem___199489, i_199487)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 398)
    i_199491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 28), 'i')
    # Getting the type of 'x1' (line 398)
    x1_199492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 25), 'x1')
    # Obtaining the member '__getitem__' of a type (line 398)
    getitem___199493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 25), x1_199492, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 398)
    subscript_call_result_199494 = invoke(stypy.reporting.localization.Localization(__file__, 398, 25), getitem___199493, i_199491)
    
    # Applying the binary operator '-' (line 398)
    result_sub_199495 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 17), '-', subscript_call_result_199490, subscript_call_result_199494)
    
    # Assigning a type to the variable 'dx' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'dx', result_sub_199495)
    
    # Assigning a Call to a Name (line 399):
    
    # Assigning a Call to a Name (line 399):
    
    # Call to fun(...): (line 399)
    # Processing the call arguments (line 399)
    # Getting the type of 'x1' (line 399)
    x1_199497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 21), 'x1', False)
    # Processing the call keyword arguments (line 399)
    kwargs_199498 = {}
    # Getting the type of 'fun' (line 399)
    fun_199496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 17), 'fun', False)
    # Calling fun(args, kwargs) (line 399)
    fun_call_result_199499 = invoke(stypy.reporting.localization.Localization(__file__, 399, 17), fun_199496, *[x1_199497], **kwargs_199498)
    
    # Assigning a type to the variable 'f1' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'f1', fun_call_result_199499)
    
    # Assigning a Call to a Name (line 400):
    
    # Assigning a Call to a Name (line 400):
    
    # Call to fun(...): (line 400)
    # Processing the call arguments (line 400)
    # Getting the type of 'x2' (line 400)
    x2_199501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 21), 'x2', False)
    # Processing the call keyword arguments (line 400)
    kwargs_199502 = {}
    # Getting the type of 'fun' (line 400)
    fun_199500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 17), 'fun', False)
    # Calling fun(args, kwargs) (line 400)
    fun_call_result_199503 = invoke(stypy.reporting.localization.Localization(__file__, 400, 17), fun_199500, *[x2_199501], **kwargs_199502)
    
    # Assigning a type to the variable 'f2' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'f2', fun_call_result_199503)
    
    # Assigning a BinOp to a Name (line 401):
    
    # Assigning a BinOp to a Name (line 401):
    # Getting the type of 'f2' (line 401)
    f2_199504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 17), 'f2')
    # Getting the type of 'f1' (line 401)
    f1_199505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 22), 'f1')
    # Applying the binary operator '-' (line 401)
    result_sub_199506 = python_operator(stypy.reporting.localization.Localization(__file__, 401, 17), '-', f2_199504, f1_199505)
    
    # Assigning a type to the variable 'df' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'df', result_sub_199506)
    # SSA branch for the else part of an if statement (line 395)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'method' (line 402)
    method_199507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 13), 'method')
    str_199508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 23), 'str', 'cs')
    # Applying the binary operator '==' (line 402)
    result_eq_199509 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 13), '==', method_199507, str_199508)
    
    # Testing the type of an if condition (line 402)
    if_condition_199510 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 402, 13), result_eq_199509)
    # Assigning a type to the variable 'if_condition_199510' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 13), 'if_condition_199510', if_condition_199510)
    # SSA begins for if statement (line 402)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 403):
    
    # Assigning a Call to a Name (line 403):
    
    # Call to fun(...): (line 403)
    # Processing the call arguments (line 403)
    # Getting the type of 'x0' (line 403)
    x0_199512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 21), 'x0', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 403)
    i_199513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 33), 'i', False)
    # Getting the type of 'h_vecs' (line 403)
    h_vecs_199514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 26), 'h_vecs', False)
    # Obtaining the member '__getitem__' of a type (line 403)
    getitem___199515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 26), h_vecs_199514, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 403)
    subscript_call_result_199516 = invoke(stypy.reporting.localization.Localization(__file__, 403, 26), getitem___199515, i_199513)
    
    complex_199517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 36), 'complex')
    # Applying the binary operator '*' (line 403)
    result_mul_199518 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 26), '*', subscript_call_result_199516, complex_199517)
    
    # Applying the binary operator '+' (line 403)
    result_add_199519 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 21), '+', x0_199512, result_mul_199518)
    
    # Processing the call keyword arguments (line 403)
    kwargs_199520 = {}
    # Getting the type of 'fun' (line 403)
    fun_199511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 17), 'fun', False)
    # Calling fun(args, kwargs) (line 403)
    fun_call_result_199521 = invoke(stypy.reporting.localization.Localization(__file__, 403, 17), fun_199511, *[result_add_199519], **kwargs_199520)
    
    # Assigning a type to the variable 'f1' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 12), 'f1', fun_call_result_199521)
    
    # Assigning a Attribute to a Name (line 404):
    
    # Assigning a Attribute to a Name (line 404):
    # Getting the type of 'f1' (line 404)
    f1_199522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 17), 'f1')
    # Obtaining the member 'imag' of a type (line 404)
    imag_199523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 17), f1_199522, 'imag')
    # Assigning a type to the variable 'df' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 12), 'df', imag_199523)
    
    # Assigning a Subscript to a Name (line 405):
    
    # Assigning a Subscript to a Name (line 405):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 405)
    tuple_199524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 405)
    # Adding element type (line 405)
    # Getting the type of 'i' (line 405)
    i_199525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 24), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 24), tuple_199524, i_199525)
    # Adding element type (line 405)
    # Getting the type of 'i' (line 405)
    i_199526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 27), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 24), tuple_199524, i_199526)
    
    # Getting the type of 'h_vecs' (line 405)
    h_vecs_199527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 17), 'h_vecs')
    # Obtaining the member '__getitem__' of a type (line 405)
    getitem___199528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 17), h_vecs_199527, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 405)
    subscript_call_result_199529 = invoke(stypy.reporting.localization.Localization(__file__, 405, 17), getitem___199528, tuple_199524)
    
    # Assigning a type to the variable 'dx' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'dx', subscript_call_result_199529)
    # SSA branch for the else part of an if statement (line 402)
    module_type_store.open_ssa_branch('else')
    
    # Call to RuntimeError(...): (line 407)
    # Processing the call arguments (line 407)
    str_199531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 31), 'str', 'Never be here.')
    # Processing the call keyword arguments (line 407)
    kwargs_199532 = {}
    # Getting the type of 'RuntimeError' (line 407)
    RuntimeError_199530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 18), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 407)
    RuntimeError_call_result_199533 = invoke(stypy.reporting.localization.Localization(__file__, 407, 18), RuntimeError_199530, *[str_199531], **kwargs_199532)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 407, 12), RuntimeError_call_result_199533, 'raise parameter', BaseException)
    # SSA join for if statement (line 402)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 395)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 388)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 384)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Subscript (line 409):
    
    # Assigning a BinOp to a Subscript (line 409):
    # Getting the type of 'df' (line 409)
    df_199534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 26), 'df')
    # Getting the type of 'dx' (line 409)
    dx_199535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 31), 'dx')
    # Applying the binary operator 'div' (line 409)
    result_div_199536 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 26), 'div', df_199534, dx_199535)
    
    # Getting the type of 'J_transposed' (line 409)
    J_transposed_199537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'J_transposed')
    # Getting the type of 'i' (line 409)
    i_199538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 21), 'i')
    # Storing an element on a container (line 409)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 8), J_transposed_199537, (i_199538, result_div_199536))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'm' (line 411)
    m_199539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 7), 'm')
    int_199540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 12), 'int')
    # Applying the binary operator '==' (line 411)
    result_eq_199541 = python_operator(stypy.reporting.localization.Localization(__file__, 411, 7), '==', m_199539, int_199540)
    
    # Testing the type of an if condition (line 411)
    if_condition_199542 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 411, 4), result_eq_199541)
    # Assigning a type to the variable 'if_condition_199542' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'if_condition_199542', if_condition_199542)
    # SSA begins for if statement (line 411)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 412):
    
    # Assigning a Call to a Name (line 412):
    
    # Call to ravel(...): (line 412)
    # Processing the call arguments (line 412)
    # Getting the type of 'J_transposed' (line 412)
    J_transposed_199545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 32), 'J_transposed', False)
    # Processing the call keyword arguments (line 412)
    kwargs_199546 = {}
    # Getting the type of 'np' (line 412)
    np_199543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 23), 'np', False)
    # Obtaining the member 'ravel' of a type (line 412)
    ravel_199544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 23), np_199543, 'ravel')
    # Calling ravel(args, kwargs) (line 412)
    ravel_call_result_199547 = invoke(stypy.reporting.localization.Localization(__file__, 412, 23), ravel_199544, *[J_transposed_199545], **kwargs_199546)
    
    # Assigning a type to the variable 'J_transposed' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'J_transposed', ravel_call_result_199547)
    # SSA join for if statement (line 411)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'J_transposed' (line 414)
    J_transposed_199548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 11), 'J_transposed')
    # Obtaining the member 'T' of a type (line 414)
    T_199549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 11), J_transposed_199548, 'T')
    # Assigning a type to the variable 'stypy_return_type' (line 414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'stypy_return_type', T_199549)
    
    # ################# End of '_dense_difference(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_dense_difference' in the type store
    # Getting the type of 'stypy_return_type' (line 377)
    stypy_return_type_199550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_199550)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_dense_difference'
    return stypy_return_type_199550

# Assigning a type to the variable '_dense_difference' (line 377)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 0), '_dense_difference', _dense_difference)

@norecursion
def _sparse_difference(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_sparse_difference'
    module_type_store = module_type_store.open_function_context('_sparse_difference', 417, 0, False)
    
    # Passed parameters checking function
    _sparse_difference.stypy_localization = localization
    _sparse_difference.stypy_type_of_self = None
    _sparse_difference.stypy_type_store = module_type_store
    _sparse_difference.stypy_function_name = '_sparse_difference'
    _sparse_difference.stypy_param_names_list = ['fun', 'x0', 'f0', 'h', 'use_one_sided', 'structure', 'groups', 'method']
    _sparse_difference.stypy_varargs_param_name = None
    _sparse_difference.stypy_kwargs_param_name = None
    _sparse_difference.stypy_call_defaults = defaults
    _sparse_difference.stypy_call_varargs = varargs
    _sparse_difference.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_sparse_difference', ['fun', 'x0', 'f0', 'h', 'use_one_sided', 'structure', 'groups', 'method'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_sparse_difference', localization, ['fun', 'x0', 'f0', 'h', 'use_one_sided', 'structure', 'groups', 'method'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_sparse_difference(...)' code ##################

    
    # Assigning a Attribute to a Name (line 419):
    
    # Assigning a Attribute to a Name (line 419):
    # Getting the type of 'f0' (line 419)
    f0_199551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'f0')
    # Obtaining the member 'size' of a type (line 419)
    size_199552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 8), f0_199551, 'size')
    # Assigning a type to the variable 'm' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'm', size_199552)
    
    # Assigning a Attribute to a Name (line 420):
    
    # Assigning a Attribute to a Name (line 420):
    # Getting the type of 'x0' (line 420)
    x0_199553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'x0')
    # Obtaining the member 'size' of a type (line 420)
    size_199554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 8), x0_199553, 'size')
    # Assigning a type to the variable 'n' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'n', size_199554)
    
    # Assigning a List to a Name (line 421):
    
    # Assigning a List to a Name (line 421):
    
    # Obtaining an instance of the builtin type 'list' (line 421)
    list_199555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 421)
    
    # Assigning a type to the variable 'row_indices' (line 421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 4), 'row_indices', list_199555)
    
    # Assigning a List to a Name (line 422):
    
    # Assigning a List to a Name (line 422):
    
    # Obtaining an instance of the builtin type 'list' (line 422)
    list_199556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 422)
    
    # Assigning a type to the variable 'col_indices' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'col_indices', list_199556)
    
    # Assigning a List to a Name (line 423):
    
    # Assigning a List to a Name (line 423):
    
    # Obtaining an instance of the builtin type 'list' (line 423)
    list_199557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 423)
    
    # Assigning a type to the variable 'fractions' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'fractions', list_199557)
    
    # Assigning a BinOp to a Name (line 425):
    
    # Assigning a BinOp to a Name (line 425):
    
    # Call to max(...): (line 425)
    # Processing the call arguments (line 425)
    # Getting the type of 'groups' (line 425)
    groups_199560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 22), 'groups', False)
    # Processing the call keyword arguments (line 425)
    kwargs_199561 = {}
    # Getting the type of 'np' (line 425)
    np_199558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 15), 'np', False)
    # Obtaining the member 'max' of a type (line 425)
    max_199559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 15), np_199558, 'max')
    # Calling max(args, kwargs) (line 425)
    max_call_result_199562 = invoke(stypy.reporting.localization.Localization(__file__, 425, 15), max_199559, *[groups_199560], **kwargs_199561)
    
    int_199563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 32), 'int')
    # Applying the binary operator '+' (line 425)
    result_add_199564 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 15), '+', max_call_result_199562, int_199563)
    
    # Assigning a type to the variable 'n_groups' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'n_groups', result_add_199564)
    
    
    # Call to range(...): (line 426)
    # Processing the call arguments (line 426)
    # Getting the type of 'n_groups' (line 426)
    n_groups_199566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 23), 'n_groups', False)
    # Processing the call keyword arguments (line 426)
    kwargs_199567 = {}
    # Getting the type of 'range' (line 426)
    range_199565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 17), 'range', False)
    # Calling range(args, kwargs) (line 426)
    range_call_result_199568 = invoke(stypy.reporting.localization.Localization(__file__, 426, 17), range_199565, *[n_groups_199566], **kwargs_199567)
    
    # Testing the type of a for loop iterable (line 426)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 426, 4), range_call_result_199568)
    # Getting the type of the for loop variable (line 426)
    for_loop_var_199569 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 426, 4), range_call_result_199568)
    # Assigning a type to the variable 'group' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'group', for_loop_var_199569)
    # SSA begins for a for statement (line 426)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 428):
    
    # Assigning a Call to a Name (line 428):
    
    # Call to equal(...): (line 428)
    # Processing the call arguments (line 428)
    # Getting the type of 'group' (line 428)
    group_199572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 21), 'group', False)
    # Getting the type of 'groups' (line 428)
    groups_199573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 28), 'groups', False)
    # Processing the call keyword arguments (line 428)
    kwargs_199574 = {}
    # Getting the type of 'np' (line 428)
    np_199570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 12), 'np', False)
    # Obtaining the member 'equal' of a type (line 428)
    equal_199571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 12), np_199570, 'equal')
    # Calling equal(args, kwargs) (line 428)
    equal_call_result_199575 = invoke(stypy.reporting.localization.Localization(__file__, 428, 12), equal_199571, *[group_199572, groups_199573], **kwargs_199574)
    
    # Assigning a type to the variable 'e' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'e', equal_call_result_199575)
    
    # Assigning a BinOp to a Name (line 429):
    
    # Assigning a BinOp to a Name (line 429):
    # Getting the type of 'h' (line 429)
    h_199576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 16), 'h')
    # Getting the type of 'e' (line 429)
    e_199577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 20), 'e')
    # Applying the binary operator '*' (line 429)
    result_mul_199578 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 16), '*', h_199576, e_199577)
    
    # Assigning a type to the variable 'h_vec' (line 429)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'h_vec', result_mul_199578)
    
    
    # Getting the type of 'method' (line 430)
    method_199579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 11), 'method')
    str_199580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 21), 'str', '2-point')
    # Applying the binary operator '==' (line 430)
    result_eq_199581 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 11), '==', method_199579, str_199580)
    
    # Testing the type of an if condition (line 430)
    if_condition_199582 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 430, 8), result_eq_199581)
    # Assigning a type to the variable 'if_condition_199582' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'if_condition_199582', if_condition_199582)
    # SSA begins for if statement (line 430)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 431):
    
    # Assigning a BinOp to a Name (line 431):
    # Getting the type of 'x0' (line 431)
    x0_199583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 16), 'x0')
    # Getting the type of 'h_vec' (line 431)
    h_vec_199584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 21), 'h_vec')
    # Applying the binary operator '+' (line 431)
    result_add_199585 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 16), '+', x0_199583, h_vec_199584)
    
    # Assigning a type to the variable 'x' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 12), 'x', result_add_199585)
    
    # Assigning a BinOp to a Name (line 432):
    
    # Assigning a BinOp to a Name (line 432):
    # Getting the type of 'x' (line 432)
    x_199586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 17), 'x')
    # Getting the type of 'x0' (line 432)
    x0_199587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 21), 'x0')
    # Applying the binary operator '-' (line 432)
    result_sub_199588 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 17), '-', x_199586, x0_199587)
    
    # Assigning a type to the variable 'dx' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'dx', result_sub_199588)
    
    # Assigning a BinOp to a Name (line 433):
    
    # Assigning a BinOp to a Name (line 433):
    
    # Call to fun(...): (line 433)
    # Processing the call arguments (line 433)
    # Getting the type of 'x' (line 433)
    x_199590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 21), 'x', False)
    # Processing the call keyword arguments (line 433)
    kwargs_199591 = {}
    # Getting the type of 'fun' (line 433)
    fun_199589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 17), 'fun', False)
    # Calling fun(args, kwargs) (line 433)
    fun_call_result_199592 = invoke(stypy.reporting.localization.Localization(__file__, 433, 17), fun_199589, *[x_199590], **kwargs_199591)
    
    # Getting the type of 'f0' (line 433)
    f0_199593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 26), 'f0')
    # Applying the binary operator '-' (line 433)
    result_sub_199594 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 17), '-', fun_call_result_199592, f0_199593)
    
    # Assigning a type to the variable 'df' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 12), 'df', result_sub_199594)
    
    # Assigning a Call to a Tuple (line 436):
    
    # Assigning a Subscript to a Name (line 436):
    
    # Obtaining the type of the subscript
    int_199595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 12), 'int')
    
    # Call to nonzero(...): (line 436)
    # Processing the call arguments (line 436)
    # Getting the type of 'e' (line 436)
    e_199598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 31), 'e', False)
    # Processing the call keyword arguments (line 436)
    kwargs_199599 = {}
    # Getting the type of 'np' (line 436)
    np_199596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 20), 'np', False)
    # Obtaining the member 'nonzero' of a type (line 436)
    nonzero_199597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 20), np_199596, 'nonzero')
    # Calling nonzero(args, kwargs) (line 436)
    nonzero_call_result_199600 = invoke(stypy.reporting.localization.Localization(__file__, 436, 20), nonzero_199597, *[e_199598], **kwargs_199599)
    
    # Obtaining the member '__getitem__' of a type (line 436)
    getitem___199601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 12), nonzero_call_result_199600, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 436)
    subscript_call_result_199602 = invoke(stypy.reporting.localization.Localization(__file__, 436, 12), getitem___199601, int_199595)
    
    # Assigning a type to the variable 'tuple_var_assignment_198609' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'tuple_var_assignment_198609', subscript_call_result_199602)
    
    # Assigning a Name to a Name (line 436):
    # Getting the type of 'tuple_var_assignment_198609' (line 436)
    tuple_var_assignment_198609_199603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'tuple_var_assignment_198609')
    # Assigning a type to the variable 'cols' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'cols', tuple_var_assignment_198609_199603)
    
    # Assigning a Call to a Tuple (line 438):
    
    # Assigning a Subscript to a Name (line 438):
    
    # Obtaining the type of the subscript
    int_199604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 12), 'int')
    
    # Call to find(...): (line 438)
    # Processing the call arguments (line 438)
    
    # Obtaining the type of the subscript
    slice_199606 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 438, 27), None, None, None)
    # Getting the type of 'cols' (line 438)
    cols_199607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 40), 'cols', False)
    # Getting the type of 'structure' (line 438)
    structure_199608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 27), 'structure', False)
    # Obtaining the member '__getitem__' of a type (line 438)
    getitem___199609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 27), structure_199608, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 438)
    subscript_call_result_199610 = invoke(stypy.reporting.localization.Localization(__file__, 438, 27), getitem___199609, (slice_199606, cols_199607))
    
    # Processing the call keyword arguments (line 438)
    kwargs_199611 = {}
    # Getting the type of 'find' (line 438)
    find_199605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 22), 'find', False)
    # Calling find(args, kwargs) (line 438)
    find_call_result_199612 = invoke(stypy.reporting.localization.Localization(__file__, 438, 22), find_199605, *[subscript_call_result_199610], **kwargs_199611)
    
    # Obtaining the member '__getitem__' of a type (line 438)
    getitem___199613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 12), find_call_result_199612, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 438)
    subscript_call_result_199614 = invoke(stypy.reporting.localization.Localization(__file__, 438, 12), getitem___199613, int_199604)
    
    # Assigning a type to the variable 'tuple_var_assignment_198610' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'tuple_var_assignment_198610', subscript_call_result_199614)
    
    # Assigning a Subscript to a Name (line 438):
    
    # Obtaining the type of the subscript
    int_199615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 12), 'int')
    
    # Call to find(...): (line 438)
    # Processing the call arguments (line 438)
    
    # Obtaining the type of the subscript
    slice_199617 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 438, 27), None, None, None)
    # Getting the type of 'cols' (line 438)
    cols_199618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 40), 'cols', False)
    # Getting the type of 'structure' (line 438)
    structure_199619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 27), 'structure', False)
    # Obtaining the member '__getitem__' of a type (line 438)
    getitem___199620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 27), structure_199619, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 438)
    subscript_call_result_199621 = invoke(stypy.reporting.localization.Localization(__file__, 438, 27), getitem___199620, (slice_199617, cols_199618))
    
    # Processing the call keyword arguments (line 438)
    kwargs_199622 = {}
    # Getting the type of 'find' (line 438)
    find_199616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 22), 'find', False)
    # Calling find(args, kwargs) (line 438)
    find_call_result_199623 = invoke(stypy.reporting.localization.Localization(__file__, 438, 22), find_199616, *[subscript_call_result_199621], **kwargs_199622)
    
    # Obtaining the member '__getitem__' of a type (line 438)
    getitem___199624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 12), find_call_result_199623, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 438)
    subscript_call_result_199625 = invoke(stypy.reporting.localization.Localization(__file__, 438, 12), getitem___199624, int_199615)
    
    # Assigning a type to the variable 'tuple_var_assignment_198611' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'tuple_var_assignment_198611', subscript_call_result_199625)
    
    # Assigning a Subscript to a Name (line 438):
    
    # Obtaining the type of the subscript
    int_199626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 12), 'int')
    
    # Call to find(...): (line 438)
    # Processing the call arguments (line 438)
    
    # Obtaining the type of the subscript
    slice_199628 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 438, 27), None, None, None)
    # Getting the type of 'cols' (line 438)
    cols_199629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 40), 'cols', False)
    # Getting the type of 'structure' (line 438)
    structure_199630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 27), 'structure', False)
    # Obtaining the member '__getitem__' of a type (line 438)
    getitem___199631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 27), structure_199630, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 438)
    subscript_call_result_199632 = invoke(stypy.reporting.localization.Localization(__file__, 438, 27), getitem___199631, (slice_199628, cols_199629))
    
    # Processing the call keyword arguments (line 438)
    kwargs_199633 = {}
    # Getting the type of 'find' (line 438)
    find_199627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 22), 'find', False)
    # Calling find(args, kwargs) (line 438)
    find_call_result_199634 = invoke(stypy.reporting.localization.Localization(__file__, 438, 22), find_199627, *[subscript_call_result_199632], **kwargs_199633)
    
    # Obtaining the member '__getitem__' of a type (line 438)
    getitem___199635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 12), find_call_result_199634, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 438)
    subscript_call_result_199636 = invoke(stypy.reporting.localization.Localization(__file__, 438, 12), getitem___199635, int_199626)
    
    # Assigning a type to the variable 'tuple_var_assignment_198612' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'tuple_var_assignment_198612', subscript_call_result_199636)
    
    # Assigning a Name to a Name (line 438):
    # Getting the type of 'tuple_var_assignment_198610' (line 438)
    tuple_var_assignment_198610_199637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'tuple_var_assignment_198610')
    # Assigning a type to the variable 'i' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'i', tuple_var_assignment_198610_199637)
    
    # Assigning a Name to a Name (line 438):
    # Getting the type of 'tuple_var_assignment_198611' (line 438)
    tuple_var_assignment_198611_199638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'tuple_var_assignment_198611')
    # Assigning a type to the variable 'j' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 15), 'j', tuple_var_assignment_198611_199638)
    
    # Assigning a Name to a Name (line 438):
    # Getting the type of 'tuple_var_assignment_198612' (line 438)
    tuple_var_assignment_198612_199639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'tuple_var_assignment_198612')
    # Assigning a type to the variable '_' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 18), '_', tuple_var_assignment_198612_199639)
    
    # Assigning a Subscript to a Name (line 440):
    
    # Assigning a Subscript to a Name (line 440):
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 440)
    j_199640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 21), 'j')
    # Getting the type of 'cols' (line 440)
    cols_199641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 16), 'cols')
    # Obtaining the member '__getitem__' of a type (line 440)
    getitem___199642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 16), cols_199641, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 440)
    subscript_call_result_199643 = invoke(stypy.reporting.localization.Localization(__file__, 440, 16), getitem___199642, j_199640)
    
    # Assigning a type to the variable 'j' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'j', subscript_call_result_199643)
    # SSA branch for the else part of an if statement (line 430)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'method' (line 441)
    method_199644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 13), 'method')
    str_199645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 23), 'str', '3-point')
    # Applying the binary operator '==' (line 441)
    result_eq_199646 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 13), '==', method_199644, str_199645)
    
    # Testing the type of an if condition (line 441)
    if_condition_199647 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 441, 13), result_eq_199646)
    # Assigning a type to the variable 'if_condition_199647' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 13), 'if_condition_199647', if_condition_199647)
    # SSA begins for if statement (line 441)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 444):
    
    # Assigning a Call to a Name (line 444):
    
    # Call to copy(...): (line 444)
    # Processing the call keyword arguments (line 444)
    kwargs_199650 = {}
    # Getting the type of 'x0' (line 444)
    x0_199648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 17), 'x0', False)
    # Obtaining the member 'copy' of a type (line 444)
    copy_199649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 17), x0_199648, 'copy')
    # Calling copy(args, kwargs) (line 444)
    copy_call_result_199651 = invoke(stypy.reporting.localization.Localization(__file__, 444, 17), copy_199649, *[], **kwargs_199650)
    
    # Assigning a type to the variable 'x1' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'x1', copy_call_result_199651)
    
    # Assigning a Call to a Name (line 445):
    
    # Assigning a Call to a Name (line 445):
    
    # Call to copy(...): (line 445)
    # Processing the call keyword arguments (line 445)
    kwargs_199654 = {}
    # Getting the type of 'x0' (line 445)
    x0_199652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 17), 'x0', False)
    # Obtaining the member 'copy' of a type (line 445)
    copy_199653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 17), x0_199652, 'copy')
    # Calling copy(args, kwargs) (line 445)
    copy_call_result_199655 = invoke(stypy.reporting.localization.Localization(__file__, 445, 17), copy_199653, *[], **kwargs_199654)
    
    # Assigning a type to the variable 'x2' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'x2', copy_call_result_199655)
    
    # Assigning a BinOp to a Name (line 447):
    
    # Assigning a BinOp to a Name (line 447):
    # Getting the type of 'use_one_sided' (line 447)
    use_one_sided_199656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 21), 'use_one_sided')
    # Getting the type of 'e' (line 447)
    e_199657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 37), 'e')
    # Applying the binary operator '&' (line 447)
    result_and__199658 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 21), '&', use_one_sided_199656, e_199657)
    
    # Assigning a type to the variable 'mask_1' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'mask_1', result_and__199658)
    
    # Getting the type of 'x1' (line 448)
    x1_199659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 12), 'x1')
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask_1' (line 448)
    mask_1_199660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 15), 'mask_1')
    # Getting the type of 'x1' (line 448)
    x1_199661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 12), 'x1')
    # Obtaining the member '__getitem__' of a type (line 448)
    getitem___199662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 12), x1_199661, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 448)
    subscript_call_result_199663 = invoke(stypy.reporting.localization.Localization(__file__, 448, 12), getitem___199662, mask_1_199660)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask_1' (line 448)
    mask_1_199664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 32), 'mask_1')
    # Getting the type of 'h_vec' (line 448)
    h_vec_199665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 26), 'h_vec')
    # Obtaining the member '__getitem__' of a type (line 448)
    getitem___199666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 26), h_vec_199665, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 448)
    subscript_call_result_199667 = invoke(stypy.reporting.localization.Localization(__file__, 448, 26), getitem___199666, mask_1_199664)
    
    # Applying the binary operator '+=' (line 448)
    result_iadd_199668 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 12), '+=', subscript_call_result_199663, subscript_call_result_199667)
    # Getting the type of 'x1' (line 448)
    x1_199669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 12), 'x1')
    # Getting the type of 'mask_1' (line 448)
    mask_1_199670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 15), 'mask_1')
    # Storing an element on a container (line 448)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 448, 12), x1_199669, (mask_1_199670, result_iadd_199668))
    
    
    # Getting the type of 'x2' (line 449)
    x2_199671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 'x2')
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask_1' (line 449)
    mask_1_199672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 15), 'mask_1')
    # Getting the type of 'x2' (line 449)
    x2_199673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 'x2')
    # Obtaining the member '__getitem__' of a type (line 449)
    getitem___199674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 12), x2_199673, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 449)
    subscript_call_result_199675 = invoke(stypy.reporting.localization.Localization(__file__, 449, 12), getitem___199674, mask_1_199672)
    
    int_199676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 26), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask_1' (line 449)
    mask_1_199677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 36), 'mask_1')
    # Getting the type of 'h_vec' (line 449)
    h_vec_199678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 30), 'h_vec')
    # Obtaining the member '__getitem__' of a type (line 449)
    getitem___199679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 30), h_vec_199678, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 449)
    subscript_call_result_199680 = invoke(stypy.reporting.localization.Localization(__file__, 449, 30), getitem___199679, mask_1_199677)
    
    # Applying the binary operator '*' (line 449)
    result_mul_199681 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 26), '*', int_199676, subscript_call_result_199680)
    
    # Applying the binary operator '+=' (line 449)
    result_iadd_199682 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 12), '+=', subscript_call_result_199675, result_mul_199681)
    # Getting the type of 'x2' (line 449)
    x2_199683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 'x2')
    # Getting the type of 'mask_1' (line 449)
    mask_1_199684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 15), 'mask_1')
    # Storing an element on a container (line 449)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 12), x2_199683, (mask_1_199684, result_iadd_199682))
    
    
    # Assigning a BinOp to a Name (line 451):
    
    # Assigning a BinOp to a Name (line 451):
    
    # Getting the type of 'use_one_sided' (line 451)
    use_one_sided_199685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 22), 'use_one_sided')
    # Applying the '~' unary operator (line 451)
    result_inv_199686 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 21), '~', use_one_sided_199685)
    
    # Getting the type of 'e' (line 451)
    e_199687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 38), 'e')
    # Applying the binary operator '&' (line 451)
    result_and__199688 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 21), '&', result_inv_199686, e_199687)
    
    # Assigning a type to the variable 'mask_2' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'mask_2', result_and__199688)
    
    # Getting the type of 'x1' (line 452)
    x1_199689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 12), 'x1')
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask_2' (line 452)
    mask_2_199690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 15), 'mask_2')
    # Getting the type of 'x1' (line 452)
    x1_199691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 12), 'x1')
    # Obtaining the member '__getitem__' of a type (line 452)
    getitem___199692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 12), x1_199691, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 452)
    subscript_call_result_199693 = invoke(stypy.reporting.localization.Localization(__file__, 452, 12), getitem___199692, mask_2_199690)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask_2' (line 452)
    mask_2_199694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 32), 'mask_2')
    # Getting the type of 'h_vec' (line 452)
    h_vec_199695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 26), 'h_vec')
    # Obtaining the member '__getitem__' of a type (line 452)
    getitem___199696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 26), h_vec_199695, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 452)
    subscript_call_result_199697 = invoke(stypy.reporting.localization.Localization(__file__, 452, 26), getitem___199696, mask_2_199694)
    
    # Applying the binary operator '-=' (line 452)
    result_isub_199698 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 12), '-=', subscript_call_result_199693, subscript_call_result_199697)
    # Getting the type of 'x1' (line 452)
    x1_199699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 12), 'x1')
    # Getting the type of 'mask_2' (line 452)
    mask_2_199700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 15), 'mask_2')
    # Storing an element on a container (line 452)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 12), x1_199699, (mask_2_199700, result_isub_199698))
    
    
    # Getting the type of 'x2' (line 453)
    x2_199701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 12), 'x2')
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask_2' (line 453)
    mask_2_199702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 15), 'mask_2')
    # Getting the type of 'x2' (line 453)
    x2_199703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 12), 'x2')
    # Obtaining the member '__getitem__' of a type (line 453)
    getitem___199704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 12), x2_199703, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 453)
    subscript_call_result_199705 = invoke(stypy.reporting.localization.Localization(__file__, 453, 12), getitem___199704, mask_2_199702)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask_2' (line 453)
    mask_2_199706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 32), 'mask_2')
    # Getting the type of 'h_vec' (line 453)
    h_vec_199707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 26), 'h_vec')
    # Obtaining the member '__getitem__' of a type (line 453)
    getitem___199708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 26), h_vec_199707, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 453)
    subscript_call_result_199709 = invoke(stypy.reporting.localization.Localization(__file__, 453, 26), getitem___199708, mask_2_199706)
    
    # Applying the binary operator '+=' (line 453)
    result_iadd_199710 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 12), '+=', subscript_call_result_199705, subscript_call_result_199709)
    # Getting the type of 'x2' (line 453)
    x2_199711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 12), 'x2')
    # Getting the type of 'mask_2' (line 453)
    mask_2_199712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 15), 'mask_2')
    # Storing an element on a container (line 453)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 453, 12), x2_199711, (mask_2_199712, result_iadd_199710))
    
    
    # Assigning a Call to a Name (line 455):
    
    # Assigning a Call to a Name (line 455):
    
    # Call to zeros(...): (line 455)
    # Processing the call arguments (line 455)
    # Getting the type of 'n' (line 455)
    n_199715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 26), 'n', False)
    # Processing the call keyword arguments (line 455)
    kwargs_199716 = {}
    # Getting the type of 'np' (line 455)
    np_199713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 17), 'np', False)
    # Obtaining the member 'zeros' of a type (line 455)
    zeros_199714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 17), np_199713, 'zeros')
    # Calling zeros(args, kwargs) (line 455)
    zeros_call_result_199717 = invoke(stypy.reporting.localization.Localization(__file__, 455, 17), zeros_199714, *[n_199715], **kwargs_199716)
    
    # Assigning a type to the variable 'dx' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 12), 'dx', zeros_call_result_199717)
    
    # Assigning a BinOp to a Subscript (line 456):
    
    # Assigning a BinOp to a Subscript (line 456):
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask_1' (line 456)
    mask_1_199718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 28), 'mask_1')
    # Getting the type of 'x2' (line 456)
    x2_199719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 25), 'x2')
    # Obtaining the member '__getitem__' of a type (line 456)
    getitem___199720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 25), x2_199719, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 456)
    subscript_call_result_199721 = invoke(stypy.reporting.localization.Localization(__file__, 456, 25), getitem___199720, mask_1_199718)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask_1' (line 456)
    mask_1_199722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 41), 'mask_1')
    # Getting the type of 'x0' (line 456)
    x0_199723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 38), 'x0')
    # Obtaining the member '__getitem__' of a type (line 456)
    getitem___199724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 38), x0_199723, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 456)
    subscript_call_result_199725 = invoke(stypy.reporting.localization.Localization(__file__, 456, 38), getitem___199724, mask_1_199722)
    
    # Applying the binary operator '-' (line 456)
    result_sub_199726 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 25), '-', subscript_call_result_199721, subscript_call_result_199725)
    
    # Getting the type of 'dx' (line 456)
    dx_199727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'dx')
    # Getting the type of 'mask_1' (line 456)
    mask_1_199728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 15), 'mask_1')
    # Storing an element on a container (line 456)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 456, 12), dx_199727, (mask_1_199728, result_sub_199726))
    
    # Assigning a BinOp to a Subscript (line 457):
    
    # Assigning a BinOp to a Subscript (line 457):
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask_2' (line 457)
    mask_2_199729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 28), 'mask_2')
    # Getting the type of 'x2' (line 457)
    x2_199730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 25), 'x2')
    # Obtaining the member '__getitem__' of a type (line 457)
    getitem___199731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 25), x2_199730, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 457)
    subscript_call_result_199732 = invoke(stypy.reporting.localization.Localization(__file__, 457, 25), getitem___199731, mask_2_199729)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask_2' (line 457)
    mask_2_199733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 41), 'mask_2')
    # Getting the type of 'x1' (line 457)
    x1_199734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 38), 'x1')
    # Obtaining the member '__getitem__' of a type (line 457)
    getitem___199735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 38), x1_199734, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 457)
    subscript_call_result_199736 = invoke(stypy.reporting.localization.Localization(__file__, 457, 38), getitem___199735, mask_2_199733)
    
    # Applying the binary operator '-' (line 457)
    result_sub_199737 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 25), '-', subscript_call_result_199732, subscript_call_result_199736)
    
    # Getting the type of 'dx' (line 457)
    dx_199738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'dx')
    # Getting the type of 'mask_2' (line 457)
    mask_2_199739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 15), 'mask_2')
    # Storing an element on a container (line 457)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 12), dx_199738, (mask_2_199739, result_sub_199737))
    
    # Assigning a Call to a Name (line 459):
    
    # Assigning a Call to a Name (line 459):
    
    # Call to fun(...): (line 459)
    # Processing the call arguments (line 459)
    # Getting the type of 'x1' (line 459)
    x1_199741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 21), 'x1', False)
    # Processing the call keyword arguments (line 459)
    kwargs_199742 = {}
    # Getting the type of 'fun' (line 459)
    fun_199740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 17), 'fun', False)
    # Calling fun(args, kwargs) (line 459)
    fun_call_result_199743 = invoke(stypy.reporting.localization.Localization(__file__, 459, 17), fun_199740, *[x1_199741], **kwargs_199742)
    
    # Assigning a type to the variable 'f1' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'f1', fun_call_result_199743)
    
    # Assigning a Call to a Name (line 460):
    
    # Assigning a Call to a Name (line 460):
    
    # Call to fun(...): (line 460)
    # Processing the call arguments (line 460)
    # Getting the type of 'x2' (line 460)
    x2_199745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 21), 'x2', False)
    # Processing the call keyword arguments (line 460)
    kwargs_199746 = {}
    # Getting the type of 'fun' (line 460)
    fun_199744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 17), 'fun', False)
    # Calling fun(args, kwargs) (line 460)
    fun_call_result_199747 = invoke(stypy.reporting.localization.Localization(__file__, 460, 17), fun_199744, *[x2_199745], **kwargs_199746)
    
    # Assigning a type to the variable 'f2' (line 460)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'f2', fun_call_result_199747)
    
    # Assigning a Call to a Tuple (line 462):
    
    # Assigning a Subscript to a Name (line 462):
    
    # Obtaining the type of the subscript
    int_199748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 12), 'int')
    
    # Call to nonzero(...): (line 462)
    # Processing the call arguments (line 462)
    # Getting the type of 'e' (line 462)
    e_199751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 31), 'e', False)
    # Processing the call keyword arguments (line 462)
    kwargs_199752 = {}
    # Getting the type of 'np' (line 462)
    np_199749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 20), 'np', False)
    # Obtaining the member 'nonzero' of a type (line 462)
    nonzero_199750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 20), np_199749, 'nonzero')
    # Calling nonzero(args, kwargs) (line 462)
    nonzero_call_result_199753 = invoke(stypy.reporting.localization.Localization(__file__, 462, 20), nonzero_199750, *[e_199751], **kwargs_199752)
    
    # Obtaining the member '__getitem__' of a type (line 462)
    getitem___199754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 12), nonzero_call_result_199753, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 462)
    subscript_call_result_199755 = invoke(stypy.reporting.localization.Localization(__file__, 462, 12), getitem___199754, int_199748)
    
    # Assigning a type to the variable 'tuple_var_assignment_198613' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 12), 'tuple_var_assignment_198613', subscript_call_result_199755)
    
    # Assigning a Name to a Name (line 462):
    # Getting the type of 'tuple_var_assignment_198613' (line 462)
    tuple_var_assignment_198613_199756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 12), 'tuple_var_assignment_198613')
    # Assigning a type to the variable 'cols' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 12), 'cols', tuple_var_assignment_198613_199756)
    
    # Assigning a Call to a Tuple (line 463):
    
    # Assigning a Subscript to a Name (line 463):
    
    # Obtaining the type of the subscript
    int_199757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 12), 'int')
    
    # Call to find(...): (line 463)
    # Processing the call arguments (line 463)
    
    # Obtaining the type of the subscript
    slice_199759 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 463, 27), None, None, None)
    # Getting the type of 'cols' (line 463)
    cols_199760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 40), 'cols', False)
    # Getting the type of 'structure' (line 463)
    structure_199761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 27), 'structure', False)
    # Obtaining the member '__getitem__' of a type (line 463)
    getitem___199762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 27), structure_199761, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 463)
    subscript_call_result_199763 = invoke(stypy.reporting.localization.Localization(__file__, 463, 27), getitem___199762, (slice_199759, cols_199760))
    
    # Processing the call keyword arguments (line 463)
    kwargs_199764 = {}
    # Getting the type of 'find' (line 463)
    find_199758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 22), 'find', False)
    # Calling find(args, kwargs) (line 463)
    find_call_result_199765 = invoke(stypy.reporting.localization.Localization(__file__, 463, 22), find_199758, *[subscript_call_result_199763], **kwargs_199764)
    
    # Obtaining the member '__getitem__' of a type (line 463)
    getitem___199766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 12), find_call_result_199765, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 463)
    subscript_call_result_199767 = invoke(stypy.reporting.localization.Localization(__file__, 463, 12), getitem___199766, int_199757)
    
    # Assigning a type to the variable 'tuple_var_assignment_198614' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 12), 'tuple_var_assignment_198614', subscript_call_result_199767)
    
    # Assigning a Subscript to a Name (line 463):
    
    # Obtaining the type of the subscript
    int_199768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 12), 'int')
    
    # Call to find(...): (line 463)
    # Processing the call arguments (line 463)
    
    # Obtaining the type of the subscript
    slice_199770 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 463, 27), None, None, None)
    # Getting the type of 'cols' (line 463)
    cols_199771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 40), 'cols', False)
    # Getting the type of 'structure' (line 463)
    structure_199772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 27), 'structure', False)
    # Obtaining the member '__getitem__' of a type (line 463)
    getitem___199773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 27), structure_199772, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 463)
    subscript_call_result_199774 = invoke(stypy.reporting.localization.Localization(__file__, 463, 27), getitem___199773, (slice_199770, cols_199771))
    
    # Processing the call keyword arguments (line 463)
    kwargs_199775 = {}
    # Getting the type of 'find' (line 463)
    find_199769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 22), 'find', False)
    # Calling find(args, kwargs) (line 463)
    find_call_result_199776 = invoke(stypy.reporting.localization.Localization(__file__, 463, 22), find_199769, *[subscript_call_result_199774], **kwargs_199775)
    
    # Obtaining the member '__getitem__' of a type (line 463)
    getitem___199777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 12), find_call_result_199776, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 463)
    subscript_call_result_199778 = invoke(stypy.reporting.localization.Localization(__file__, 463, 12), getitem___199777, int_199768)
    
    # Assigning a type to the variable 'tuple_var_assignment_198615' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 12), 'tuple_var_assignment_198615', subscript_call_result_199778)
    
    # Assigning a Subscript to a Name (line 463):
    
    # Obtaining the type of the subscript
    int_199779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 12), 'int')
    
    # Call to find(...): (line 463)
    # Processing the call arguments (line 463)
    
    # Obtaining the type of the subscript
    slice_199781 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 463, 27), None, None, None)
    # Getting the type of 'cols' (line 463)
    cols_199782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 40), 'cols', False)
    # Getting the type of 'structure' (line 463)
    structure_199783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 27), 'structure', False)
    # Obtaining the member '__getitem__' of a type (line 463)
    getitem___199784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 27), structure_199783, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 463)
    subscript_call_result_199785 = invoke(stypy.reporting.localization.Localization(__file__, 463, 27), getitem___199784, (slice_199781, cols_199782))
    
    # Processing the call keyword arguments (line 463)
    kwargs_199786 = {}
    # Getting the type of 'find' (line 463)
    find_199780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 22), 'find', False)
    # Calling find(args, kwargs) (line 463)
    find_call_result_199787 = invoke(stypy.reporting.localization.Localization(__file__, 463, 22), find_199780, *[subscript_call_result_199785], **kwargs_199786)
    
    # Obtaining the member '__getitem__' of a type (line 463)
    getitem___199788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 12), find_call_result_199787, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 463)
    subscript_call_result_199789 = invoke(stypy.reporting.localization.Localization(__file__, 463, 12), getitem___199788, int_199779)
    
    # Assigning a type to the variable 'tuple_var_assignment_198616' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 12), 'tuple_var_assignment_198616', subscript_call_result_199789)
    
    # Assigning a Name to a Name (line 463):
    # Getting the type of 'tuple_var_assignment_198614' (line 463)
    tuple_var_assignment_198614_199790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 12), 'tuple_var_assignment_198614')
    # Assigning a type to the variable 'i' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 12), 'i', tuple_var_assignment_198614_199790)
    
    # Assigning a Name to a Name (line 463):
    # Getting the type of 'tuple_var_assignment_198615' (line 463)
    tuple_var_assignment_198615_199791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 12), 'tuple_var_assignment_198615')
    # Assigning a type to the variable 'j' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 15), 'j', tuple_var_assignment_198615_199791)
    
    # Assigning a Name to a Name (line 463):
    # Getting the type of 'tuple_var_assignment_198616' (line 463)
    tuple_var_assignment_198616_199792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 12), 'tuple_var_assignment_198616')
    # Assigning a type to the variable '_' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 18), '_', tuple_var_assignment_198616_199792)
    
    # Assigning a Subscript to a Name (line 464):
    
    # Assigning a Subscript to a Name (line 464):
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 464)
    j_199793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 21), 'j')
    # Getting the type of 'cols' (line 464)
    cols_199794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 16), 'cols')
    # Obtaining the member '__getitem__' of a type (line 464)
    getitem___199795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 16), cols_199794, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 464)
    subscript_call_result_199796 = invoke(stypy.reporting.localization.Localization(__file__, 464, 16), getitem___199795, j_199793)
    
    # Assigning a type to the variable 'j' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 12), 'j', subscript_call_result_199796)
    
    # Assigning a Subscript to a Name (line 466):
    
    # Assigning a Subscript to a Name (line 466):
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 466)
    j_199797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 33), 'j')
    # Getting the type of 'use_one_sided' (line 466)
    use_one_sided_199798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 19), 'use_one_sided')
    # Obtaining the member '__getitem__' of a type (line 466)
    getitem___199799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 19), use_one_sided_199798, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 466)
    subscript_call_result_199800 = invoke(stypy.reporting.localization.Localization(__file__, 466, 19), getitem___199799, j_199797)
    
    # Assigning a type to the variable 'mask' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'mask', subscript_call_result_199800)
    
    # Assigning a Call to a Name (line 467):
    
    # Assigning a Call to a Name (line 467):
    
    # Call to empty(...): (line 467)
    # Processing the call arguments (line 467)
    # Getting the type of 'm' (line 467)
    m_199803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 26), 'm', False)
    # Processing the call keyword arguments (line 467)
    kwargs_199804 = {}
    # Getting the type of 'np' (line 467)
    np_199801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 17), 'np', False)
    # Obtaining the member 'empty' of a type (line 467)
    empty_199802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 17), np_199801, 'empty')
    # Calling empty(args, kwargs) (line 467)
    empty_call_result_199805 = invoke(stypy.reporting.localization.Localization(__file__, 467, 17), empty_199802, *[m_199803], **kwargs_199804)
    
    # Assigning a type to the variable 'df' (line 467)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), 'df', empty_call_result_199805)
    
    # Assigning a Subscript to a Name (line 469):
    
    # Assigning a Subscript to a Name (line 469):
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 469)
    mask_199806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 21), 'mask')
    # Getting the type of 'i' (line 469)
    i_199807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 19), 'i')
    # Obtaining the member '__getitem__' of a type (line 469)
    getitem___199808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 19), i_199807, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 469)
    subscript_call_result_199809 = invoke(stypy.reporting.localization.Localization(__file__, 469, 19), getitem___199808, mask_199806)
    
    # Assigning a type to the variable 'rows' (line 469)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 12), 'rows', subscript_call_result_199809)
    
    # Assigning a BinOp to a Subscript (line 470):
    
    # Assigning a BinOp to a Subscript (line 470):
    int_199810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 23), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'rows' (line 470)
    rows_199811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 31), 'rows')
    # Getting the type of 'f0' (line 470)
    f0_199812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 28), 'f0')
    # Obtaining the member '__getitem__' of a type (line 470)
    getitem___199813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 28), f0_199812, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 470)
    subscript_call_result_199814 = invoke(stypy.reporting.localization.Localization(__file__, 470, 28), getitem___199813, rows_199811)
    
    # Applying the binary operator '*' (line 470)
    result_mul_199815 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 23), '*', int_199810, subscript_call_result_199814)
    
    int_199816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 39), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'rows' (line 470)
    rows_199817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 46), 'rows')
    # Getting the type of 'f1' (line 470)
    f1_199818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 43), 'f1')
    # Obtaining the member '__getitem__' of a type (line 470)
    getitem___199819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 43), f1_199818, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 470)
    subscript_call_result_199820 = invoke(stypy.reporting.localization.Localization(__file__, 470, 43), getitem___199819, rows_199817)
    
    # Applying the binary operator '*' (line 470)
    result_mul_199821 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 39), '*', int_199816, subscript_call_result_199820)
    
    # Applying the binary operator '+' (line 470)
    result_add_199822 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 23), '+', result_mul_199815, result_mul_199821)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'rows' (line 470)
    rows_199823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 57), 'rows')
    # Getting the type of 'f2' (line 470)
    f2_199824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 54), 'f2')
    # Obtaining the member '__getitem__' of a type (line 470)
    getitem___199825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 54), f2_199824, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 470)
    subscript_call_result_199826 = invoke(stypy.reporting.localization.Localization(__file__, 470, 54), getitem___199825, rows_199823)
    
    # Applying the binary operator '-' (line 470)
    result_sub_199827 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 52), '-', result_add_199822, subscript_call_result_199826)
    
    # Getting the type of 'df' (line 470)
    df_199828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'df')
    # Getting the type of 'rows' (line 470)
    rows_199829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 15), 'rows')
    # Storing an element on a container (line 470)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 12), df_199828, (rows_199829, result_sub_199827))
    
    # Assigning a Subscript to a Name (line 472):
    
    # Assigning a Subscript to a Name (line 472):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'mask' (line 472)
    mask_199830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 22), 'mask')
    # Applying the '~' unary operator (line 472)
    result_inv_199831 = python_operator(stypy.reporting.localization.Localization(__file__, 472, 21), '~', mask_199830)
    
    # Getting the type of 'i' (line 472)
    i_199832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 19), 'i')
    # Obtaining the member '__getitem__' of a type (line 472)
    getitem___199833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 19), i_199832, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 472)
    subscript_call_result_199834 = invoke(stypy.reporting.localization.Localization(__file__, 472, 19), getitem___199833, result_inv_199831)
    
    # Assigning a type to the variable 'rows' (line 472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 12), 'rows', subscript_call_result_199834)
    
    # Assigning a BinOp to a Subscript (line 473):
    
    # Assigning a BinOp to a Subscript (line 473):
    
    # Obtaining the type of the subscript
    # Getting the type of 'rows' (line 473)
    rows_199835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 26), 'rows')
    # Getting the type of 'f2' (line 473)
    f2_199836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 23), 'f2')
    # Obtaining the member '__getitem__' of a type (line 473)
    getitem___199837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 23), f2_199836, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 473)
    subscript_call_result_199838 = invoke(stypy.reporting.localization.Localization(__file__, 473, 23), getitem___199837, rows_199835)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'rows' (line 473)
    rows_199839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 37), 'rows')
    # Getting the type of 'f1' (line 473)
    f1_199840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 34), 'f1')
    # Obtaining the member '__getitem__' of a type (line 473)
    getitem___199841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 34), f1_199840, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 473)
    subscript_call_result_199842 = invoke(stypy.reporting.localization.Localization(__file__, 473, 34), getitem___199841, rows_199839)
    
    # Applying the binary operator '-' (line 473)
    result_sub_199843 = python_operator(stypy.reporting.localization.Localization(__file__, 473, 23), '-', subscript_call_result_199838, subscript_call_result_199842)
    
    # Getting the type of 'df' (line 473)
    df_199844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'df')
    # Getting the type of 'rows' (line 473)
    rows_199845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 15), 'rows')
    # Storing an element on a container (line 473)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 12), df_199844, (rows_199845, result_sub_199843))
    # SSA branch for the else part of an if statement (line 441)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'method' (line 474)
    method_199846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 13), 'method')
    str_199847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 23), 'str', 'cs')
    # Applying the binary operator '==' (line 474)
    result_eq_199848 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 13), '==', method_199846, str_199847)
    
    # Testing the type of an if condition (line 474)
    if_condition_199849 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 474, 13), result_eq_199848)
    # Assigning a type to the variable 'if_condition_199849' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 13), 'if_condition_199849', if_condition_199849)
    # SSA begins for if statement (line 474)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 475):
    
    # Assigning a Call to a Name (line 475):
    
    # Call to fun(...): (line 475)
    # Processing the call arguments (line 475)
    # Getting the type of 'x0' (line 475)
    x0_199851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 21), 'x0', False)
    # Getting the type of 'h_vec' (line 475)
    h_vec_199852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 26), 'h_vec', False)
    complex_199853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 32), 'complex')
    # Applying the binary operator '*' (line 475)
    result_mul_199854 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 26), '*', h_vec_199852, complex_199853)
    
    # Applying the binary operator '+' (line 475)
    result_add_199855 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 21), '+', x0_199851, result_mul_199854)
    
    # Processing the call keyword arguments (line 475)
    kwargs_199856 = {}
    # Getting the type of 'fun' (line 475)
    fun_199850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 17), 'fun', False)
    # Calling fun(args, kwargs) (line 475)
    fun_call_result_199857 = invoke(stypy.reporting.localization.Localization(__file__, 475, 17), fun_199850, *[result_add_199855], **kwargs_199856)
    
    # Assigning a type to the variable 'f1' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'f1', fun_call_result_199857)
    
    # Assigning a Attribute to a Name (line 476):
    
    # Assigning a Attribute to a Name (line 476):
    # Getting the type of 'f1' (line 476)
    f1_199858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 17), 'f1')
    # Obtaining the member 'imag' of a type (line 476)
    imag_199859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 17), f1_199858, 'imag')
    # Assigning a type to the variable 'df' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 12), 'df', imag_199859)
    
    # Assigning a Name to a Name (line 477):
    
    # Assigning a Name to a Name (line 477):
    # Getting the type of 'h_vec' (line 477)
    h_vec_199860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 17), 'h_vec')
    # Assigning a type to the variable 'dx' (line 477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 12), 'dx', h_vec_199860)
    
    # Assigning a Call to a Tuple (line 478):
    
    # Assigning a Subscript to a Name (line 478):
    
    # Obtaining the type of the subscript
    int_199861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 12), 'int')
    
    # Call to nonzero(...): (line 478)
    # Processing the call arguments (line 478)
    # Getting the type of 'e' (line 478)
    e_199864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 31), 'e', False)
    # Processing the call keyword arguments (line 478)
    kwargs_199865 = {}
    # Getting the type of 'np' (line 478)
    np_199862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 20), 'np', False)
    # Obtaining the member 'nonzero' of a type (line 478)
    nonzero_199863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 20), np_199862, 'nonzero')
    # Calling nonzero(args, kwargs) (line 478)
    nonzero_call_result_199866 = invoke(stypy.reporting.localization.Localization(__file__, 478, 20), nonzero_199863, *[e_199864], **kwargs_199865)
    
    # Obtaining the member '__getitem__' of a type (line 478)
    getitem___199867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 12), nonzero_call_result_199866, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 478)
    subscript_call_result_199868 = invoke(stypy.reporting.localization.Localization(__file__, 478, 12), getitem___199867, int_199861)
    
    # Assigning a type to the variable 'tuple_var_assignment_198617' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'tuple_var_assignment_198617', subscript_call_result_199868)
    
    # Assigning a Name to a Name (line 478):
    # Getting the type of 'tuple_var_assignment_198617' (line 478)
    tuple_var_assignment_198617_199869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'tuple_var_assignment_198617')
    # Assigning a type to the variable 'cols' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'cols', tuple_var_assignment_198617_199869)
    
    # Assigning a Call to a Tuple (line 479):
    
    # Assigning a Subscript to a Name (line 479):
    
    # Obtaining the type of the subscript
    int_199870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 12), 'int')
    
    # Call to find(...): (line 479)
    # Processing the call arguments (line 479)
    
    # Obtaining the type of the subscript
    slice_199872 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 479, 27), None, None, None)
    # Getting the type of 'cols' (line 479)
    cols_199873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 40), 'cols', False)
    # Getting the type of 'structure' (line 479)
    structure_199874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 27), 'structure', False)
    # Obtaining the member '__getitem__' of a type (line 479)
    getitem___199875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 27), structure_199874, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 479)
    subscript_call_result_199876 = invoke(stypy.reporting.localization.Localization(__file__, 479, 27), getitem___199875, (slice_199872, cols_199873))
    
    # Processing the call keyword arguments (line 479)
    kwargs_199877 = {}
    # Getting the type of 'find' (line 479)
    find_199871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 22), 'find', False)
    # Calling find(args, kwargs) (line 479)
    find_call_result_199878 = invoke(stypy.reporting.localization.Localization(__file__, 479, 22), find_199871, *[subscript_call_result_199876], **kwargs_199877)
    
    # Obtaining the member '__getitem__' of a type (line 479)
    getitem___199879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 12), find_call_result_199878, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 479)
    subscript_call_result_199880 = invoke(stypy.reporting.localization.Localization(__file__, 479, 12), getitem___199879, int_199870)
    
    # Assigning a type to the variable 'tuple_var_assignment_198618' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'tuple_var_assignment_198618', subscript_call_result_199880)
    
    # Assigning a Subscript to a Name (line 479):
    
    # Obtaining the type of the subscript
    int_199881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 12), 'int')
    
    # Call to find(...): (line 479)
    # Processing the call arguments (line 479)
    
    # Obtaining the type of the subscript
    slice_199883 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 479, 27), None, None, None)
    # Getting the type of 'cols' (line 479)
    cols_199884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 40), 'cols', False)
    # Getting the type of 'structure' (line 479)
    structure_199885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 27), 'structure', False)
    # Obtaining the member '__getitem__' of a type (line 479)
    getitem___199886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 27), structure_199885, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 479)
    subscript_call_result_199887 = invoke(stypy.reporting.localization.Localization(__file__, 479, 27), getitem___199886, (slice_199883, cols_199884))
    
    # Processing the call keyword arguments (line 479)
    kwargs_199888 = {}
    # Getting the type of 'find' (line 479)
    find_199882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 22), 'find', False)
    # Calling find(args, kwargs) (line 479)
    find_call_result_199889 = invoke(stypy.reporting.localization.Localization(__file__, 479, 22), find_199882, *[subscript_call_result_199887], **kwargs_199888)
    
    # Obtaining the member '__getitem__' of a type (line 479)
    getitem___199890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 12), find_call_result_199889, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 479)
    subscript_call_result_199891 = invoke(stypy.reporting.localization.Localization(__file__, 479, 12), getitem___199890, int_199881)
    
    # Assigning a type to the variable 'tuple_var_assignment_198619' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'tuple_var_assignment_198619', subscript_call_result_199891)
    
    # Assigning a Subscript to a Name (line 479):
    
    # Obtaining the type of the subscript
    int_199892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 12), 'int')
    
    # Call to find(...): (line 479)
    # Processing the call arguments (line 479)
    
    # Obtaining the type of the subscript
    slice_199894 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 479, 27), None, None, None)
    # Getting the type of 'cols' (line 479)
    cols_199895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 40), 'cols', False)
    # Getting the type of 'structure' (line 479)
    structure_199896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 27), 'structure', False)
    # Obtaining the member '__getitem__' of a type (line 479)
    getitem___199897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 27), structure_199896, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 479)
    subscript_call_result_199898 = invoke(stypy.reporting.localization.Localization(__file__, 479, 27), getitem___199897, (slice_199894, cols_199895))
    
    # Processing the call keyword arguments (line 479)
    kwargs_199899 = {}
    # Getting the type of 'find' (line 479)
    find_199893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 22), 'find', False)
    # Calling find(args, kwargs) (line 479)
    find_call_result_199900 = invoke(stypy.reporting.localization.Localization(__file__, 479, 22), find_199893, *[subscript_call_result_199898], **kwargs_199899)
    
    # Obtaining the member '__getitem__' of a type (line 479)
    getitem___199901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 12), find_call_result_199900, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 479)
    subscript_call_result_199902 = invoke(stypy.reporting.localization.Localization(__file__, 479, 12), getitem___199901, int_199892)
    
    # Assigning a type to the variable 'tuple_var_assignment_198620' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'tuple_var_assignment_198620', subscript_call_result_199902)
    
    # Assigning a Name to a Name (line 479):
    # Getting the type of 'tuple_var_assignment_198618' (line 479)
    tuple_var_assignment_198618_199903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'tuple_var_assignment_198618')
    # Assigning a type to the variable 'i' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'i', tuple_var_assignment_198618_199903)
    
    # Assigning a Name to a Name (line 479):
    # Getting the type of 'tuple_var_assignment_198619' (line 479)
    tuple_var_assignment_198619_199904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'tuple_var_assignment_198619')
    # Assigning a type to the variable 'j' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 15), 'j', tuple_var_assignment_198619_199904)
    
    # Assigning a Name to a Name (line 479):
    # Getting the type of 'tuple_var_assignment_198620' (line 479)
    tuple_var_assignment_198620_199905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'tuple_var_assignment_198620')
    # Assigning a type to the variable '_' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 18), '_', tuple_var_assignment_198620_199905)
    
    # Assigning a Subscript to a Name (line 480):
    
    # Assigning a Subscript to a Name (line 480):
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 480)
    j_199906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 21), 'j')
    # Getting the type of 'cols' (line 480)
    cols_199907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 16), 'cols')
    # Obtaining the member '__getitem__' of a type (line 480)
    getitem___199908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 16), cols_199907, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 480)
    subscript_call_result_199909 = invoke(stypy.reporting.localization.Localization(__file__, 480, 16), getitem___199908, j_199906)
    
    # Assigning a type to the variable 'j' (line 480)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 12), 'j', subscript_call_result_199909)
    # SSA branch for the else part of an if statement (line 474)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 482)
    # Processing the call arguments (line 482)
    str_199911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 29), 'str', 'Never be here.')
    # Processing the call keyword arguments (line 482)
    kwargs_199912 = {}
    # Getting the type of 'ValueError' (line 482)
    ValueError_199910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 482)
    ValueError_call_result_199913 = invoke(stypy.reporting.localization.Localization(__file__, 482, 18), ValueError_199910, *[str_199911], **kwargs_199912)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 482, 12), ValueError_call_result_199913, 'raise parameter', BaseException)
    # SSA join for if statement (line 474)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 441)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 430)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 486)
    # Processing the call arguments (line 486)
    # Getting the type of 'i' (line 486)
    i_199916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 27), 'i', False)
    # Processing the call keyword arguments (line 486)
    kwargs_199917 = {}
    # Getting the type of 'row_indices' (line 486)
    row_indices_199914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 8), 'row_indices', False)
    # Obtaining the member 'append' of a type (line 486)
    append_199915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 8), row_indices_199914, 'append')
    # Calling append(args, kwargs) (line 486)
    append_call_result_199918 = invoke(stypy.reporting.localization.Localization(__file__, 486, 8), append_199915, *[i_199916], **kwargs_199917)
    
    
    # Call to append(...): (line 487)
    # Processing the call arguments (line 487)
    # Getting the type of 'j' (line 487)
    j_199921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 27), 'j', False)
    # Processing the call keyword arguments (line 487)
    kwargs_199922 = {}
    # Getting the type of 'col_indices' (line 487)
    col_indices_199919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'col_indices', False)
    # Obtaining the member 'append' of a type (line 487)
    append_199920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 8), col_indices_199919, 'append')
    # Calling append(args, kwargs) (line 487)
    append_call_result_199923 = invoke(stypy.reporting.localization.Localization(__file__, 487, 8), append_199920, *[j_199921], **kwargs_199922)
    
    
    # Call to append(...): (line 488)
    # Processing the call arguments (line 488)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 488)
    i_199926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 28), 'i', False)
    # Getting the type of 'df' (line 488)
    df_199927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 25), 'df', False)
    # Obtaining the member '__getitem__' of a type (line 488)
    getitem___199928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 25), df_199927, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 488)
    subscript_call_result_199929 = invoke(stypy.reporting.localization.Localization(__file__, 488, 25), getitem___199928, i_199926)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 488)
    j_199930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 36), 'j', False)
    # Getting the type of 'dx' (line 488)
    dx_199931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 33), 'dx', False)
    # Obtaining the member '__getitem__' of a type (line 488)
    getitem___199932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 33), dx_199931, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 488)
    subscript_call_result_199933 = invoke(stypy.reporting.localization.Localization(__file__, 488, 33), getitem___199932, j_199930)
    
    # Applying the binary operator 'div' (line 488)
    result_div_199934 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 25), 'div', subscript_call_result_199929, subscript_call_result_199933)
    
    # Processing the call keyword arguments (line 488)
    kwargs_199935 = {}
    # Getting the type of 'fractions' (line 488)
    fractions_199924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'fractions', False)
    # Obtaining the member 'append' of a type (line 488)
    append_199925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 8), fractions_199924, 'append')
    # Calling append(args, kwargs) (line 488)
    append_call_result_199936 = invoke(stypy.reporting.localization.Localization(__file__, 488, 8), append_199925, *[result_div_199934], **kwargs_199935)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 490):
    
    # Assigning a Call to a Name (line 490):
    
    # Call to hstack(...): (line 490)
    # Processing the call arguments (line 490)
    # Getting the type of 'row_indices' (line 490)
    row_indices_199939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 28), 'row_indices', False)
    # Processing the call keyword arguments (line 490)
    kwargs_199940 = {}
    # Getting the type of 'np' (line 490)
    np_199937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 18), 'np', False)
    # Obtaining the member 'hstack' of a type (line 490)
    hstack_199938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 18), np_199937, 'hstack')
    # Calling hstack(args, kwargs) (line 490)
    hstack_call_result_199941 = invoke(stypy.reporting.localization.Localization(__file__, 490, 18), hstack_199938, *[row_indices_199939], **kwargs_199940)
    
    # Assigning a type to the variable 'row_indices' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'row_indices', hstack_call_result_199941)
    
    # Assigning a Call to a Name (line 491):
    
    # Assigning a Call to a Name (line 491):
    
    # Call to hstack(...): (line 491)
    # Processing the call arguments (line 491)
    # Getting the type of 'col_indices' (line 491)
    col_indices_199944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 28), 'col_indices', False)
    # Processing the call keyword arguments (line 491)
    kwargs_199945 = {}
    # Getting the type of 'np' (line 491)
    np_199942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 18), 'np', False)
    # Obtaining the member 'hstack' of a type (line 491)
    hstack_199943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 18), np_199942, 'hstack')
    # Calling hstack(args, kwargs) (line 491)
    hstack_call_result_199946 = invoke(stypy.reporting.localization.Localization(__file__, 491, 18), hstack_199943, *[col_indices_199944], **kwargs_199945)
    
    # Assigning a type to the variable 'col_indices' (line 491)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 4), 'col_indices', hstack_call_result_199946)
    
    # Assigning a Call to a Name (line 492):
    
    # Assigning a Call to a Name (line 492):
    
    # Call to hstack(...): (line 492)
    # Processing the call arguments (line 492)
    # Getting the type of 'fractions' (line 492)
    fractions_199949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 26), 'fractions', False)
    # Processing the call keyword arguments (line 492)
    kwargs_199950 = {}
    # Getting the type of 'np' (line 492)
    np_199947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 16), 'np', False)
    # Obtaining the member 'hstack' of a type (line 492)
    hstack_199948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 16), np_199947, 'hstack')
    # Calling hstack(args, kwargs) (line 492)
    hstack_call_result_199951 = invoke(stypy.reporting.localization.Localization(__file__, 492, 16), hstack_199948, *[fractions_199949], **kwargs_199950)
    
    # Assigning a type to the variable 'fractions' (line 492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 4), 'fractions', hstack_call_result_199951)
    
    # Assigning a Call to a Name (line 493):
    
    # Assigning a Call to a Name (line 493):
    
    # Call to coo_matrix(...): (line 493)
    # Processing the call arguments (line 493)
    
    # Obtaining an instance of the builtin type 'tuple' (line 493)
    tuple_199953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 493)
    # Adding element type (line 493)
    # Getting the type of 'fractions' (line 493)
    fractions_199954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 20), 'fractions', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 20), tuple_199953, fractions_199954)
    # Adding element type (line 493)
    
    # Obtaining an instance of the builtin type 'tuple' (line 493)
    tuple_199955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 493)
    # Adding element type (line 493)
    # Getting the type of 'row_indices' (line 493)
    row_indices_199956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 32), 'row_indices', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 32), tuple_199955, row_indices_199956)
    # Adding element type (line 493)
    # Getting the type of 'col_indices' (line 493)
    col_indices_199957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 45), 'col_indices', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 32), tuple_199955, col_indices_199957)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 20), tuple_199953, tuple_199955)
    
    # Processing the call keyword arguments (line 493)
    
    # Obtaining an instance of the builtin type 'tuple' (line 493)
    tuple_199958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 67), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 493)
    # Adding element type (line 493)
    # Getting the type of 'm' (line 493)
    m_199959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 67), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 67), tuple_199958, m_199959)
    # Adding element type (line 493)
    # Getting the type of 'n' (line 493)
    n_199960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 70), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 67), tuple_199958, n_199960)
    
    keyword_199961 = tuple_199958
    kwargs_199962 = {'shape': keyword_199961}
    # Getting the type of 'coo_matrix' (line 493)
    coo_matrix_199952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 493)
    coo_matrix_call_result_199963 = invoke(stypy.reporting.localization.Localization(__file__, 493, 8), coo_matrix_199952, *[tuple_199953], **kwargs_199962)
    
    # Assigning a type to the variable 'J' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'J', coo_matrix_call_result_199963)
    
    # Call to csr_matrix(...): (line 494)
    # Processing the call arguments (line 494)
    # Getting the type of 'J' (line 494)
    J_199965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 22), 'J', False)
    # Processing the call keyword arguments (line 494)
    kwargs_199966 = {}
    # Getting the type of 'csr_matrix' (line 494)
    csr_matrix_199964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 11), 'csr_matrix', False)
    # Calling csr_matrix(args, kwargs) (line 494)
    csr_matrix_call_result_199967 = invoke(stypy.reporting.localization.Localization(__file__, 494, 11), csr_matrix_199964, *[J_199965], **kwargs_199966)
    
    # Assigning a type to the variable 'stypy_return_type' (line 494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 4), 'stypy_return_type', csr_matrix_call_result_199967)
    
    # ################# End of '_sparse_difference(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_sparse_difference' in the type store
    # Getting the type of 'stypy_return_type' (line 417)
    stypy_return_type_199968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_199968)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_sparse_difference'
    return stypy_return_type_199968

# Assigning a type to the variable '_sparse_difference' (line 417)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 0), '_sparse_difference', _sparse_difference)

@norecursion
def check_derivative(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 497)
    tuple_199969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 497)
    # Adding element type (line 497)
    
    # Getting the type of 'np' (line 497)
    np_199970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 44), 'np')
    # Obtaining the member 'inf' of a type (line 497)
    inf_199971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 44), np_199970, 'inf')
    # Applying the 'usub' unary operator (line 497)
    result___neg___199972 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 43), 'usub', inf_199971)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 43), tuple_199969, result___neg___199972)
    # Adding element type (line 497)
    # Getting the type of 'np' (line 497)
    np_199973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 52), 'np')
    # Obtaining the member 'inf' of a type (line 497)
    inf_199974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 52), np_199973, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 43), tuple_199969, inf_199974)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 497)
    tuple_199975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 66), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 497)
    
    
    # Obtaining an instance of the builtin type 'dict' (line 498)
    dict_199976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 28), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 498)
    
    defaults = [tuple_199969, tuple_199975, dict_199976]
    # Create a new context for function 'check_derivative'
    module_type_store = module_type_store.open_function_context('check_derivative', 497, 0, False)
    
    # Passed parameters checking function
    check_derivative.stypy_localization = localization
    check_derivative.stypy_type_of_self = None
    check_derivative.stypy_type_store = module_type_store
    check_derivative.stypy_function_name = 'check_derivative'
    check_derivative.stypy_param_names_list = ['fun', 'jac', 'x0', 'bounds', 'args', 'kwargs']
    check_derivative.stypy_varargs_param_name = None
    check_derivative.stypy_kwargs_param_name = None
    check_derivative.stypy_call_defaults = defaults
    check_derivative.stypy_call_varargs = varargs
    check_derivative.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_derivative', ['fun', 'jac', 'x0', 'bounds', 'args', 'kwargs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_derivative', localization, ['fun', 'jac', 'x0', 'bounds', 'args', 'kwargs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_derivative(...)' code ##################

    str_199977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, (-1)), 'str', 'Check correctness of a function computing derivatives (Jacobian or\n    gradient) by comparison with a finite difference approximation.\n\n    Parameters\n    ----------\n    fun : callable\n        Function of which to estimate the derivatives. The argument x\n        passed to this function is ndarray of shape (n,) (never a scalar\n        even if n=1). It must return 1-d array_like of shape (m,) or a scalar.\n    jac : callable\n        Function which computes Jacobian matrix of `fun`. It must work with\n        argument x the same way as `fun`. The return value must be array_like\n        or sparse matrix with an appropriate shape.\n    x0 : array_like of shape (n,) or float\n        Point at which to estimate the derivatives. Float will be converted\n        to 1-d array.\n    bounds : 2-tuple of array_like, optional\n        Lower and upper bounds on independent variables. Defaults to no bounds.\n        Each bound must match the size of `x0` or be a scalar, in the latter\n        case the bound will be the same for all variables. Use it to limit the\n        range of function evaluation.\n    args, kwargs : tuple and dict, optional\n        Additional arguments passed to `fun` and `jac`. Both empty by default.\n        The calling signature is ``fun(x, *args, **kwargs)`` and the same\n        for `jac`.\n\n    Returns\n    -------\n    accuracy : float\n        The maximum among all relative errors for elements with absolute values\n        higher than 1 and absolute errors for elements with absolute values\n        less or equal than 1. If `accuracy` is on the order of 1e-6 or lower,\n        then it is likely that your `jac` implementation is correct.\n\n    See Also\n    --------\n    approx_derivative : Compute finite difference approximation of derivative.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.optimize import check_derivative\n    >>>\n    >>>\n    >>> def f(x, c1, c2):\n    ...     return np.array([x[0] * np.sin(c1 * x[1]),\n    ...                      x[0] * np.cos(c2 * x[1])])\n    ...\n    >>> def jac(x, c1, c2):\n    ...     return np.array([\n    ...         [np.sin(c1 * x[1]),  c1 * x[0] * np.cos(c1 * x[1])],\n    ...         [np.cos(c2 * x[1]), -c2 * x[0] * np.sin(c2 * x[1])]\n    ...     ])\n    ...\n    >>>\n    >>> x0 = np.array([1.0, 0.5 * np.pi])\n    >>> check_derivative(f, jac, x0, args=(1, 2))\n    2.4492935982947064e-16\n    ')
    
    # Assigning a Call to a Name (line 558):
    
    # Assigning a Call to a Name (line 558):
    
    # Call to jac(...): (line 558)
    # Processing the call arguments (line 558)
    # Getting the type of 'x0' (line 558)
    x0_199979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 20), 'x0', False)
    # Getting the type of 'args' (line 558)
    args_199980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 25), 'args', False)
    # Processing the call keyword arguments (line 558)
    # Getting the type of 'kwargs' (line 558)
    kwargs_199981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 33), 'kwargs', False)
    kwargs_199982 = {'kwargs_199981': kwargs_199981}
    # Getting the type of 'jac' (line 558)
    jac_199978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 16), 'jac', False)
    # Calling jac(args, kwargs) (line 558)
    jac_call_result_199983 = invoke(stypy.reporting.localization.Localization(__file__, 558, 16), jac_199978, *[x0_199979, args_199980], **kwargs_199982)
    
    # Assigning a type to the variable 'J_to_test' (line 558)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 4), 'J_to_test', jac_call_result_199983)
    
    
    # Call to issparse(...): (line 559)
    # Processing the call arguments (line 559)
    # Getting the type of 'J_to_test' (line 559)
    J_to_test_199985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 16), 'J_to_test', False)
    # Processing the call keyword arguments (line 559)
    kwargs_199986 = {}
    # Getting the type of 'issparse' (line 559)
    issparse_199984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 7), 'issparse', False)
    # Calling issparse(args, kwargs) (line 559)
    issparse_call_result_199987 = invoke(stypy.reporting.localization.Localization(__file__, 559, 7), issparse_199984, *[J_to_test_199985], **kwargs_199986)
    
    # Testing the type of an if condition (line 559)
    if_condition_199988 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 559, 4), issparse_call_result_199987)
    # Assigning a type to the variable 'if_condition_199988' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'if_condition_199988', if_condition_199988)
    # SSA begins for if statement (line 559)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 560):
    
    # Assigning a Call to a Name (line 560):
    
    # Call to approx_derivative(...): (line 560)
    # Processing the call arguments (line 560)
    # Getting the type of 'fun' (line 560)
    fun_199990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 35), 'fun', False)
    # Getting the type of 'x0' (line 560)
    x0_199991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 40), 'x0', False)
    # Processing the call keyword arguments (line 560)
    # Getting the type of 'bounds' (line 560)
    bounds_199992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 51), 'bounds', False)
    keyword_199993 = bounds_199992
    # Getting the type of 'J_to_test' (line 560)
    J_to_test_199994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 68), 'J_to_test', False)
    keyword_199995 = J_to_test_199994
    # Getting the type of 'args' (line 561)
    args_199996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 40), 'args', False)
    keyword_199997 = args_199996
    # Getting the type of 'kwargs' (line 561)
    kwargs_199998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 53), 'kwargs', False)
    keyword_199999 = kwargs_199998
    kwargs_200000 = {'sparsity': keyword_199995, 'args': keyword_199997, 'bounds': keyword_199993, 'kwargs': keyword_199999}
    # Getting the type of 'approx_derivative' (line 560)
    approx_derivative_199989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 17), 'approx_derivative', False)
    # Calling approx_derivative(args, kwargs) (line 560)
    approx_derivative_call_result_200001 = invoke(stypy.reporting.localization.Localization(__file__, 560, 17), approx_derivative_199989, *[fun_199990, x0_199991], **kwargs_200000)
    
    # Assigning a type to the variable 'J_diff' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'J_diff', approx_derivative_call_result_200001)
    
    # Assigning a Call to a Name (line 562):
    
    # Assigning a Call to a Name (line 562):
    
    # Call to csr_matrix(...): (line 562)
    # Processing the call arguments (line 562)
    # Getting the type of 'J_to_test' (line 562)
    J_to_test_200003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 31), 'J_to_test', False)
    # Processing the call keyword arguments (line 562)
    kwargs_200004 = {}
    # Getting the type of 'csr_matrix' (line 562)
    csr_matrix_200002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 20), 'csr_matrix', False)
    # Calling csr_matrix(args, kwargs) (line 562)
    csr_matrix_call_result_200005 = invoke(stypy.reporting.localization.Localization(__file__, 562, 20), csr_matrix_200002, *[J_to_test_200003], **kwargs_200004)
    
    # Assigning a type to the variable 'J_to_test' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'J_to_test', csr_matrix_call_result_200005)
    
    # Assigning a BinOp to a Name (line 563):
    
    # Assigning a BinOp to a Name (line 563):
    # Getting the type of 'J_to_test' (line 563)
    J_to_test_200006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 18), 'J_to_test')
    # Getting the type of 'J_diff' (line 563)
    J_diff_200007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 30), 'J_diff')
    # Applying the binary operator '-' (line 563)
    result_sub_200008 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 18), '-', J_to_test_200006, J_diff_200007)
    
    # Assigning a type to the variable 'abs_err' (line 563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 8), 'abs_err', result_sub_200008)
    
    # Assigning a Call to a Tuple (line 564):
    
    # Assigning a Subscript to a Name (line 564):
    
    # Obtaining the type of the subscript
    int_200009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 8), 'int')
    
    # Call to find(...): (line 564)
    # Processing the call arguments (line 564)
    # Getting the type of 'abs_err' (line 564)
    abs_err_200011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 34), 'abs_err', False)
    # Processing the call keyword arguments (line 564)
    kwargs_200012 = {}
    # Getting the type of 'find' (line 564)
    find_200010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 29), 'find', False)
    # Calling find(args, kwargs) (line 564)
    find_call_result_200013 = invoke(stypy.reporting.localization.Localization(__file__, 564, 29), find_200010, *[abs_err_200011], **kwargs_200012)
    
    # Obtaining the member '__getitem__' of a type (line 564)
    getitem___200014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 8), find_call_result_200013, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 564)
    subscript_call_result_200015 = invoke(stypy.reporting.localization.Localization(__file__, 564, 8), getitem___200014, int_200009)
    
    # Assigning a type to the variable 'tuple_var_assignment_198621' (line 564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'tuple_var_assignment_198621', subscript_call_result_200015)
    
    # Assigning a Subscript to a Name (line 564):
    
    # Obtaining the type of the subscript
    int_200016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 8), 'int')
    
    # Call to find(...): (line 564)
    # Processing the call arguments (line 564)
    # Getting the type of 'abs_err' (line 564)
    abs_err_200018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 34), 'abs_err', False)
    # Processing the call keyword arguments (line 564)
    kwargs_200019 = {}
    # Getting the type of 'find' (line 564)
    find_200017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 29), 'find', False)
    # Calling find(args, kwargs) (line 564)
    find_call_result_200020 = invoke(stypy.reporting.localization.Localization(__file__, 564, 29), find_200017, *[abs_err_200018], **kwargs_200019)
    
    # Obtaining the member '__getitem__' of a type (line 564)
    getitem___200021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 8), find_call_result_200020, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 564)
    subscript_call_result_200022 = invoke(stypy.reporting.localization.Localization(__file__, 564, 8), getitem___200021, int_200016)
    
    # Assigning a type to the variable 'tuple_var_assignment_198622' (line 564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'tuple_var_assignment_198622', subscript_call_result_200022)
    
    # Assigning a Subscript to a Name (line 564):
    
    # Obtaining the type of the subscript
    int_200023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 8), 'int')
    
    # Call to find(...): (line 564)
    # Processing the call arguments (line 564)
    # Getting the type of 'abs_err' (line 564)
    abs_err_200025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 34), 'abs_err', False)
    # Processing the call keyword arguments (line 564)
    kwargs_200026 = {}
    # Getting the type of 'find' (line 564)
    find_200024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 29), 'find', False)
    # Calling find(args, kwargs) (line 564)
    find_call_result_200027 = invoke(stypy.reporting.localization.Localization(__file__, 564, 29), find_200024, *[abs_err_200025], **kwargs_200026)
    
    # Obtaining the member '__getitem__' of a type (line 564)
    getitem___200028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 8), find_call_result_200027, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 564)
    subscript_call_result_200029 = invoke(stypy.reporting.localization.Localization(__file__, 564, 8), getitem___200028, int_200023)
    
    # Assigning a type to the variable 'tuple_var_assignment_198623' (line 564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'tuple_var_assignment_198623', subscript_call_result_200029)
    
    # Assigning a Name to a Name (line 564):
    # Getting the type of 'tuple_var_assignment_198621' (line 564)
    tuple_var_assignment_198621_200030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'tuple_var_assignment_198621')
    # Assigning a type to the variable 'i' (line 564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'i', tuple_var_assignment_198621_200030)
    
    # Assigning a Name to a Name (line 564):
    # Getting the type of 'tuple_var_assignment_198622' (line 564)
    tuple_var_assignment_198622_200031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'tuple_var_assignment_198622')
    # Assigning a type to the variable 'j' (line 564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 11), 'j', tuple_var_assignment_198622_200031)
    
    # Assigning a Name to a Name (line 564):
    # Getting the type of 'tuple_var_assignment_198623' (line 564)
    tuple_var_assignment_198623_200032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'tuple_var_assignment_198623')
    # Assigning a type to the variable 'abs_err_data' (line 564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 14), 'abs_err_data', tuple_var_assignment_198623_200032)
    
    # Assigning a Call to a Name (line 565):
    
    # Assigning a Call to a Name (line 565):
    
    # Call to ravel(...): (line 565)
    # Processing the call keyword arguments (line 565)
    kwargs_200044 = {}
    
    # Call to asarray(...): (line 565)
    # Processing the call arguments (line 565)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 565)
    tuple_200035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 565)
    # Adding element type (line 565)
    # Getting the type of 'i' (line 565)
    i_200036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 40), 'i', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 565, 40), tuple_200035, i_200036)
    # Adding element type (line 565)
    # Getting the type of 'j' (line 565)
    j_200037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 43), 'j', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 565, 40), tuple_200035, j_200037)
    
    # Getting the type of 'J_diff' (line 565)
    J_diff_200038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 33), 'J_diff', False)
    # Obtaining the member '__getitem__' of a type (line 565)
    getitem___200039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 33), J_diff_200038, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 565)
    subscript_call_result_200040 = invoke(stypy.reporting.localization.Localization(__file__, 565, 33), getitem___200039, tuple_200035)
    
    # Processing the call keyword arguments (line 565)
    kwargs_200041 = {}
    # Getting the type of 'np' (line 565)
    np_200033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 22), 'np', False)
    # Obtaining the member 'asarray' of a type (line 565)
    asarray_200034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 22), np_200033, 'asarray')
    # Calling asarray(args, kwargs) (line 565)
    asarray_call_result_200042 = invoke(stypy.reporting.localization.Localization(__file__, 565, 22), asarray_200034, *[subscript_call_result_200040], **kwargs_200041)
    
    # Obtaining the member 'ravel' of a type (line 565)
    ravel_200043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 22), asarray_call_result_200042, 'ravel')
    # Calling ravel(args, kwargs) (line 565)
    ravel_call_result_200045 = invoke(stypy.reporting.localization.Localization(__file__, 565, 22), ravel_200043, *[], **kwargs_200044)
    
    # Assigning a type to the variable 'J_diff_data' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 8), 'J_diff_data', ravel_call_result_200045)
    
    # Call to max(...): (line 566)
    # Processing the call arguments (line 566)
    
    # Call to abs(...): (line 566)
    # Processing the call arguments (line 566)
    # Getting the type of 'abs_err_data' (line 566)
    abs_err_data_200050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 29), 'abs_err_data', False)
    # Processing the call keyword arguments (line 566)
    kwargs_200051 = {}
    # Getting the type of 'np' (line 566)
    np_200048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 22), 'np', False)
    # Obtaining the member 'abs' of a type (line 566)
    abs_200049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 22), np_200048, 'abs')
    # Calling abs(args, kwargs) (line 566)
    abs_call_result_200052 = invoke(stypy.reporting.localization.Localization(__file__, 566, 22), abs_200049, *[abs_err_data_200050], **kwargs_200051)
    
    
    # Call to maximum(...): (line 567)
    # Processing the call arguments (line 567)
    int_200055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 33), 'int')
    
    # Call to abs(...): (line 567)
    # Processing the call arguments (line 567)
    # Getting the type of 'J_diff_data' (line 567)
    J_diff_data_200058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 43), 'J_diff_data', False)
    # Processing the call keyword arguments (line 567)
    kwargs_200059 = {}
    # Getting the type of 'np' (line 567)
    np_200056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 36), 'np', False)
    # Obtaining the member 'abs' of a type (line 567)
    abs_200057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 36), np_200056, 'abs')
    # Calling abs(args, kwargs) (line 567)
    abs_call_result_200060 = invoke(stypy.reporting.localization.Localization(__file__, 567, 36), abs_200057, *[J_diff_data_200058], **kwargs_200059)
    
    # Processing the call keyword arguments (line 567)
    kwargs_200061 = {}
    # Getting the type of 'np' (line 567)
    np_200053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 22), 'np', False)
    # Obtaining the member 'maximum' of a type (line 567)
    maximum_200054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 22), np_200053, 'maximum')
    # Calling maximum(args, kwargs) (line 567)
    maximum_call_result_200062 = invoke(stypy.reporting.localization.Localization(__file__, 567, 22), maximum_200054, *[int_200055, abs_call_result_200060], **kwargs_200061)
    
    # Applying the binary operator 'div' (line 566)
    result_div_200063 = python_operator(stypy.reporting.localization.Localization(__file__, 566, 22), 'div', abs_call_result_200052, maximum_call_result_200062)
    
    # Processing the call keyword arguments (line 566)
    kwargs_200064 = {}
    # Getting the type of 'np' (line 566)
    np_200046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 15), 'np', False)
    # Obtaining the member 'max' of a type (line 566)
    max_200047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 15), np_200046, 'max')
    # Calling max(args, kwargs) (line 566)
    max_call_result_200065 = invoke(stypy.reporting.localization.Localization(__file__, 566, 15), max_200047, *[result_div_200063], **kwargs_200064)
    
    # Assigning a type to the variable 'stypy_return_type' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'stypy_return_type', max_call_result_200065)
    # SSA branch for the else part of an if statement (line 559)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 569):
    
    # Assigning a Call to a Name (line 569):
    
    # Call to approx_derivative(...): (line 569)
    # Processing the call arguments (line 569)
    # Getting the type of 'fun' (line 569)
    fun_200067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 35), 'fun', False)
    # Getting the type of 'x0' (line 569)
    x0_200068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 40), 'x0', False)
    # Processing the call keyword arguments (line 569)
    # Getting the type of 'bounds' (line 569)
    bounds_200069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 51), 'bounds', False)
    keyword_200070 = bounds_200069
    # Getting the type of 'args' (line 570)
    args_200071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 40), 'args', False)
    keyword_200072 = args_200071
    # Getting the type of 'kwargs' (line 570)
    kwargs_200073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 53), 'kwargs', False)
    keyword_200074 = kwargs_200073
    kwargs_200075 = {'args': keyword_200072, 'bounds': keyword_200070, 'kwargs': keyword_200074}
    # Getting the type of 'approx_derivative' (line 569)
    approx_derivative_200066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 17), 'approx_derivative', False)
    # Calling approx_derivative(args, kwargs) (line 569)
    approx_derivative_call_result_200076 = invoke(stypy.reporting.localization.Localization(__file__, 569, 17), approx_derivative_200066, *[fun_200067, x0_200068], **kwargs_200075)
    
    # Assigning a type to the variable 'J_diff' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 8), 'J_diff', approx_derivative_call_result_200076)
    
    # Assigning a Call to a Name (line 571):
    
    # Assigning a Call to a Name (line 571):
    
    # Call to abs(...): (line 571)
    # Processing the call arguments (line 571)
    # Getting the type of 'J_to_test' (line 571)
    J_to_test_200079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 25), 'J_to_test', False)
    # Getting the type of 'J_diff' (line 571)
    J_diff_200080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 37), 'J_diff', False)
    # Applying the binary operator '-' (line 571)
    result_sub_200081 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 25), '-', J_to_test_200079, J_diff_200080)
    
    # Processing the call keyword arguments (line 571)
    kwargs_200082 = {}
    # Getting the type of 'np' (line 571)
    np_200077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 18), 'np', False)
    # Obtaining the member 'abs' of a type (line 571)
    abs_200078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 18), np_200077, 'abs')
    # Calling abs(args, kwargs) (line 571)
    abs_call_result_200083 = invoke(stypy.reporting.localization.Localization(__file__, 571, 18), abs_200078, *[result_sub_200081], **kwargs_200082)
    
    # Assigning a type to the variable 'abs_err' (line 571)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'abs_err', abs_call_result_200083)
    
    # Call to max(...): (line 572)
    # Processing the call arguments (line 572)
    # Getting the type of 'abs_err' (line 572)
    abs_err_200086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 22), 'abs_err', False)
    
    # Call to maximum(...): (line 572)
    # Processing the call arguments (line 572)
    int_200089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 43), 'int')
    
    # Call to abs(...): (line 572)
    # Processing the call arguments (line 572)
    # Getting the type of 'J_diff' (line 572)
    J_diff_200092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 53), 'J_diff', False)
    # Processing the call keyword arguments (line 572)
    kwargs_200093 = {}
    # Getting the type of 'np' (line 572)
    np_200090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 46), 'np', False)
    # Obtaining the member 'abs' of a type (line 572)
    abs_200091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 46), np_200090, 'abs')
    # Calling abs(args, kwargs) (line 572)
    abs_call_result_200094 = invoke(stypy.reporting.localization.Localization(__file__, 572, 46), abs_200091, *[J_diff_200092], **kwargs_200093)
    
    # Processing the call keyword arguments (line 572)
    kwargs_200095 = {}
    # Getting the type of 'np' (line 572)
    np_200087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 32), 'np', False)
    # Obtaining the member 'maximum' of a type (line 572)
    maximum_200088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 32), np_200087, 'maximum')
    # Calling maximum(args, kwargs) (line 572)
    maximum_call_result_200096 = invoke(stypy.reporting.localization.Localization(__file__, 572, 32), maximum_200088, *[int_200089, abs_call_result_200094], **kwargs_200095)
    
    # Applying the binary operator 'div' (line 572)
    result_div_200097 = python_operator(stypy.reporting.localization.Localization(__file__, 572, 22), 'div', abs_err_200086, maximum_call_result_200096)
    
    # Processing the call keyword arguments (line 572)
    kwargs_200098 = {}
    # Getting the type of 'np' (line 572)
    np_200084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 15), 'np', False)
    # Obtaining the member 'max' of a type (line 572)
    max_200085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 15), np_200084, 'max')
    # Calling max(args, kwargs) (line 572)
    max_call_result_200099 = invoke(stypy.reporting.localization.Localization(__file__, 572, 15), max_200085, *[result_div_200097], **kwargs_200098)
    
    # Assigning a type to the variable 'stypy_return_type' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'stypy_return_type', max_call_result_200099)
    # SSA join for if statement (line 559)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'check_derivative(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_derivative' in the type store
    # Getting the type of 'stypy_return_type' (line 497)
    stypy_return_type_200100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_200100)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_derivative'
    return stypy_return_type_200100

# Assigning a type to the variable 'check_derivative' (line 497)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 0), 'check_derivative', check_derivative)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
